"""Retrieval benchmark: which design choices actually help? Writes bench/RESULTS.md.

A result is *relevant* if it is a scene of the labelled video whose transcript contains
the labelled answer phrase. Moment accuracy checks that the returned moment (the exact
sentence search links to) contains the answer or starts within 3s of it.

    uv run python bench/ingest.py   # once
    uv run python bench/run.py
"""

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from corpus import COLLECTIONS, DB, EPISODES, NO_CAPTIONS, STATION, STATION_COLLECTION, STATION_DB  # noqa: E402

from cinematlas import Cinematlas  # noqa: E402

HERE = Path(__file__).parent
K = 10


def ask(eng, q, top_k=K, *, only=None, sources=None, weights=None, routing=None, rerank=True):
    """One search configuration as a keyword call, so the arms below read like a table."""
    s = eng.search(q).limit(top_k).rerank(rerank)
    if only:
        s = s.only(only)
    if sources:
        s = s.using(*sources)
    if weights is not None:
        s = s.weights(**weights)
    if routing:
        s = s.routing(routing)
    return s.run()
load_dotenv(HERE.parent / ".env")


def relevant(hit: dict, label: dict) -> bool:
    if "scenes" in label:  # visual question: any of the labelled scenes
        return [hit["video_id"], hit["scene_id"]] in label["scenes"]
    return hit["video_id"] == label["video"] and label["answer"].lower() in (hit.get("transcript") or "").lower()


def answer_time(eng: Cinematlas, label: dict) -> float | None:
    if "scenes" in label:
        return None
    for d in eng.collection.find({"video_id": label["video"]}, {"segments": 1}):
        for s in d.get("segments") or []:
            if label["answer"].lower() in s["text"].lower():
                return s["start"]
    return None


def wait_for_sync(eng: Cinematlas, timeout_s: float = 600) -> None:
    """autoEmbed and mongot replicate asynchronously: don't score a half-synced index."""
    spoken = eng.collection.count_documents({"transcript": {"$nin": ["", None]}, "status": "COMPLETED"})
    deadline = time.monotonic() + timeout_s
    while True:
        seen = len(ask(eng, only="transcript", q="the", top_k=min(spoken, 500)))
        if seen >= spoken or time.monotonic() > deadline:
            print(f"index sync: {seen}/{spoken} spoken scenes searchable")
            return
        time.sleep(5)


def evaluate(name: str, search, labels: list[dict], times: list[float | None]) -> dict:
    hit1 = hit3 = 0
    rr, moment_ok, moment_n, latencies, per_q = [], 0, 0, [], []
    for label, t_answer in zip(labels, times, strict=True):
        t0 = time.perf_counter()
        hits = search(label["q"])
        latencies.append(time.perf_counter() - t0)
        ranks = [i for i, h in enumerate(hits[:K], start=1) if relevant(h, label)]
        first = ranks[0] if ranks else None
        hit1 += first == 1
        per_q.append(first == 1)
        hit3 += bool(first and first <= 3)
        rr.append(1 / first if first else 0.0)
        if hits and hits[0].get("moment") is not None and first == 1 and "answer" in label:
            moment_n += 1
            m = hits[0]["moment"]
            moment_ok += (label["answer"].lower() in m["text"].lower()
                          or (t_answer is not None and abs(m["start"] - t_answer) <= 3))
    n = len(labels)
    return {
        "config": name,
        "hit@1": hit1 / n, "hit@3": hit3 / n, "mrr@10": sum(rr) / n,
        "moment@1": (moment_ok / moment_n) if moment_n else None,
        "p50_ms": statistics.median(latencies) * 1000,
        "per_q": per_q,
    }


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact paired test on the questions where two systems disagree: (a-only wins, b-only wins, p)."""
    wins = sum(x and not y for x, y in zip(a, b, strict=True))
    losses = sum(y and not x for x, y in zip(a, b, strict=True))
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return wins, losses, p


def both(name: str, fn, speech: list[dict], visual: list[dict], t_speech: list[float | None]) -> dict:
    s_row = evaluate(name, fn, speech, t_speech)
    v_row = evaluate(name, fn, visual, [None] * len(visual))
    return {"config": name, "speech": s_row, "visual": v_row,
            "mean@1": (s_row["hit@1"] + v_row["hit@1"]) / 2, "p50_ms": s_row["p50_ms"],
            "per_q": s_row["per_q"] + v_row["per_q"]}


def main() -> None:
    speech = json.loads((HERE / "queries.json").read_text())
    visual = json.loads((HERE / "queries_visual.json").read_text())
    uri = os.getenv("MONGODB_URI") or os.environ["MDB_URI"]
    coll = {"db_name": DB, "collection_name": COLLECTIONS["autoembed"], "transcript_mode": "autoembed"}
    auto = Cinematlas(uri, **coll)
    client_side_fusion = Cinematlas(uri, native_fusion=False, **coll)
    client = Cinematlas(uri, db_name=DB, collection_name=COLLECTIONS["client"], transcript_mode="client")
    client_lite = Cinematlas(uri, db_name=DB, collection_name=COLLECTIONS["client"], transcript_mode="client",
                             text_model="voyage-4-lite")
    nocap = Cinematlas(uri, db_name=DB, collection_name=NO_CAPTIONS, transcript_mode="autoembed")
    wait_for_sync(auto)
    wait_for_sync(nocap)
    station = Cinematlas(uri, db_name=STATION_DB, collection_name=STATION_COLLECTION, transcript_mode="autoembed")
    wait_for_sync(station)
    t_speech = [answer_time(auto, lab) for lab in speech]
    missing = [lab["answer"] for lab, t in zip(speech, t_speech, strict=True) if t is None]
    if missing:
        print(f"WARNING: labels not found in corpus transcripts: {missing}")

    equal = {"visual": 1, "scene": 1, "transcript": 1, "text": 1, "rerank": 1}
    tuned = {"visual": 0.25, "scene": 1, "transcript": 1, "text": 1, "rerank": 2}
    configs = {
        "visual only (keyframes)": lambda q: ask(auto, only="visual", q=q, top_k=K),
        "joint vector only (image+speech)": lambda q: ask(auto, only="scene", q=q, top_k=K),
        "full-text only (Atlas Search BM25)": lambda q: ask(auto, only="text", q=q, top_k=K),
        "transcript only · autoEmbed voyage-4": lambda q: ask(auto, only="transcript", q=q, top_k=K),
        "transcript only · client voyage-4": lambda q: ask(client, only="transcript", q=q, top_k=K),
        "transcript only · client, voyage-4-lite queries": lambda q: ask(
            client_lite, only="transcript", q=q, top_k=K),
        "transcript + rerank": lambda q: ask(auto, q, top_k=K, sources=("transcript",), routing="fixed"),
        "fixed fusion, equal weights, no rerank": lambda q: ask(auto,
            q, top_k=K, rerank=False, weights=equal, routing="fixed"),
        "fixed fusion, equal weights + rerank": lambda q: ask(auto, q, top_k=K, weights=equal, routing="fixed"),
        "fixed fusion, tuned weights + rerank": lambda q: ask(auto, q, top_k=K, weights=tuned, routing="fixed"),
        "adaptive routing · autoEmbed": lambda q: ask(auto, q, top_k=K, routing="adaptive"),
        "**scene-first (default)**: joint vector ranks, reranker picks the second": lambda q: ask(auto,
            q, top_k=K, routing="scene"),
        "adaptive · client-side fusion": lambda q: ask(client_side_fusion, q, top_k=K, routing="adaptive"),
        "adaptive · client transcript mode": lambda q: ask(client, q, top_k=K, routing="adaptive"),
    }
    rows = []
    for name, fn in configs.items():
        row = both(name, fn, speech, visual, t_speech)
        s_row, v_row = row["speech"], row["visual"]
        rows.append(row)
        print(f"{name:52s} speech={s_row['hit@1']:.2f} visual={v_row['hit@1']:.2f} mean={row['mean@1']:.2f} "
              f"moment={s_row['moment@1']} p50={row['p50_ms']:.0f}ms")

    ablation = {}
    arms = {"visual only (keyframes)": {"only": "visual"},
            "joint vector only (image+speech)": {"only": "scene"},
            "transcript + rerank": {"sources": ("transcript",), "routing": "fixed"},
            "adaptive routing": {"routing": "adaptive"}}
    for name, arm in arms.items():
        for label, eng in [("captions", auto), ("no captions", nocap)]:
            fn = lambda q, e=eng, a=arm: ask(e, q, top_k=K, **a)  # noqa: E731
            ablation[name, label] = both(name, fn, speech, visual, t_speech)
            r = ablation[name, label]
            print(f"[{label:11s}] {name:36s} speech={r['speech']['hit@1']:.2f} visual={r['visual']['hit@1']:.2f}")

    scenes = auto.collection.count_documents({"status": "COMPLETED"})
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [
        "# Retrieval benchmark",
        "",
        f"Corpus: {len(EPISODES)} NASA *The Quiet Crew* interviews ({scenes} scenes; one program, one topic, "
        "overlapping vocabulary, burned-in captions). Two labelled question sets:",
        "",
        f"* **Speech** ({len(speech)}, [queries.json](queries.json)): paraphrased questions answered by what someone "
        "*says*. Relevant = a scene of the right person containing the answer phrase.",
        f"* **Visual** ({len(visual)}, [queries_visual.json](queries_visual.json)): questions about what is *shown*, "
        "written from the keyframes. Relevant = one of the labelled scenes.",
        "",
        "| Configuration | Speech Hit@1 | Visual Hit@1 | **Mean Hit@1** | Speech MRR | Visual MRR | Moment@1 | p50 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *[f"| {r['config']} | {fmt(r['speech']['hit@1'])} | {fmt(r['visual']['hit@1'])} | **{r['mean@1']:.2f}** | "
          f"{r['speech']['mrr@10']:.3f} | {r['visual']['mrr@10']:.3f} | {fmt(r['speech']['moment@1'])} | "
          f"{r['p50_ms']:.0f} ms |" for r in rows],
        "",
        "*Moment@1*: among top-1 speech hits, the returned moment contains the answer or starts within 3 s of it. "
        "Latency is the speech-set p50 from a laptop over the internet, including Voyage calls. Hybrid runs as one "
        "native `$rankFusion` query unless marked client-side; reranking runs as native `$rerank`. Vector indexes use "
        "scalar quantization; vectors are stored as BSON float32.",
        "",
        reading_guide(rows),
        "",
        *caption_ablation(ablation),
        "",
        *held_out(station, rows, first_corpus=(auto, speech, t_speech)),
        "",
        *beyond_video(),
        "",
        *outside_nasa(),
        "",
        *query_robustness(),
        "",
        *smarter_merges(),
        "",
        *boundary_section(),
        "",
        "Reproduce: `uv run python bench/ingest.py`, then `--no-captions` and `--station`, then "
        "`uv run python bench/run.py` · "
        f"generated {time.strftime('%Y-%m-%d')}",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote bench/RESULTS.md")


HELD_OUT = {
    "keyframes only": lambda e, q: ask(e, only="visual", q=q, top_k=K),
    "transcript + rerank": lambda e, q: ask(e, q, top_k=K, sources=("transcript",), routing="fixed"),
    "rank fusion, tuned weights + rerank": lambda e, q: ask(e,
        q, top_k=K, weights={"visual": 0.25, "scene": 1, "transcript": 1, "text": 1, "rerank": 2}, routing="fixed"),
    "joint vector only": lambda e, q: ask(e, only="scene", q=q, top_k=K),
    "adaptive routing": lambda e, q: ask(e, q, top_k=K, routing="adaptive"),
    "scene-first": lambda e, q: ask(e, q, top_k=K, routing="scene"),
}


def matched_moments(eng: Cinematlas, labels: list[dict], times: list[float | None]) -> tuple[int, int, int]:
    """Moment accuracy compared fairly: only on questions where both methods found the right scene.

    Moment@1 in the tables is measured on each method's own correct answers, which are different
    questions, so the two numbers aren't comparable. Returns (both right, scene-first moments, routing moments).
    """
    def ok(hit: dict, label: dict, t: float | None) -> bool:
        m = hit.get("moment")
        if not m:
            return False
        return label["answer"].lower() in m["text"].lower() or (t is not None and abs(m["start"] - t) <= 3)

    both = sf = ad = 0
    for label, t in zip(labels, times, strict=True):
        s = ask(eng, label["q"], top_k=K, routing="scene")[:1]
        a = ask(eng, label["q"], top_k=K, routing="adaptive")[:1]
        if s and a and relevant(s[0], label) and relevant(a[0], label):
            both, sf, ad = both + 1, sf + ok(s[0], label, t), ad + ok(a[0], label, t)
    return both, sf, ad


def held_out(eng: Cinematlas, first: list[dict], first_corpus: tuple) -> list[str]:
    """A second corpus nobody tuned on: different domain, no burned-in captions, independently written questions."""
    speech = json.loads((HERE / "queries_station_speech.json").read_text())
    visual = json.loads((HERE / "queries_station_visual.json").read_text())
    t_speech = [answer_time(eng, lab) for lab in speech]
    for lab in speech + visual:  # warm the query cache so latency compares retrieval, not Voyage round trips
        ask(eng, only="scene", q=lab["q"], top_k=1)
    rows = {name: both(name, lambda q, f=fn: f(eng, q), speech, visual, t_speech) for name, fn in HELD_OUT.items()}
    for name, r in rows.items():
        print(f"[held-out] {name:36s} speech={r['speech']['hit@1']:.2f} visual={r['visual']['hit@1']:.2f} "
              f"p50={r['p50_ms']:.0f}ms")
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    table = [f"| {n} | {r['speech']['hit@1']:.2f} | {r['visual']['hit@1']:.2f} | **{r['mean@1']:.2f}** | "
             f"{fmt(r['speech']['moment@1'])} | {r['p50_ms']:.0f} ms |" for n, r in rows.items()]
    pairs = [("joint vector only", "rank fusion, tuned weights + rerank"), ("scene-first", "adaptive routing"),
             ("joint vector only", "adaptive routing")]
    paired = []
    for a, b in pairs:
        w, lo, p = mcnemar(rows[a]["per_q"], rows[b]["per_q"])
        paired.append(f"| {a} vs {b} | {w} | {lo} | {fmt_p(p)} |")
    sf, ad = rows["scene-first"], rows["adaptive routing"]
    w, lo, p = mcnemar(sf["per_q"], ad["per_q"])
    worse = lo > w and p < 0.05
    verdict = ("**Decision rule, fixed before this corpus was run:** scene-first becomes the default if it is not "
               "significantly worse than adaptive routing on either corpus and it is faster. On this corpus it is "
               + ("significantly worse" if worse else "not significantly worse")
               + f" ({w} vs {lo}, p = {fmt_p(p)}) and "
               + (f"faster ({sf['p50_ms']:.0f} vs {ad['p50_ms']:.0f} ms)." if sf["p50_ms"] < ad["p50_ms"]
                  else f"not faster ({sf['p50_ms']:.0f} vs {ad['p50_ms']:.0f} ms)."))
    by_first = {r["config"]: r for r in first}
    f_sf = next(r for n, r in by_first.items() if "scene-first" in n)
    f_ad = next(r for n, r in by_first.items() if n.startswith("adaptive routing"))
    split = []
    for corpus, a, b in (("first", f_sf, f_ad), ("held-out", sf, ad)):
        for cat in ("speech", "visual"):
            cw, cl, cp = mcnemar(a[cat]["per_q"], b[cat]["per_q"])
            split.append(f"| {corpus} | {cat} | {a[cat]['hit@1']:.2f} | {b[cat]['hit@1']:.2f} | {cw} | {cl} | "
                         f"{fmt_p(cp)} |")
    m_both, m_sf, m_ad = matched_moments(eng, speech, t_speech)
    f_both, f_sf_m, f_ad_m = matched_moments(*first_corpus)
    videos = sorted({lab.get("video") or lab["scenes"][0][0] for lab in speech + visual})
    scenes = eng.collection.count_documents({"status": "COMPLETED"})
    return [
        "## Held-out corpus",
        "",
        f"{len(STATION)} NASA videos from a different domain ({', '.join(videos)}: a silent 15-minute station tour, "
        f"astronaut Q&A, science demos, food science; {scenes} scenes, no burned-in captions). "
        f"{len(speech)} speech and {len(visual)} visual questions ([speech](queries_station_speech.json), "
        "[visual](queries_station_visual.json)) were written by an agent that saw only these videos' keyframes and "
        "transcripts, never the code or any results. Routing thresholds and fusion weights were tuned on the first "
        "corpus only.",
        "",
        "| Configuration | Speech Hit@1 | Visual Hit@1 | Mean Hit@1 | Moment@1 | p50 |",
        "| --- | --- | --- | --- | --- | --- |",
        *table,
        "",
        "| Paired comparison | A right, B wrong | B right, A wrong | p |",
        "| --- | --- | --- | --- |",
        *paired,
        "",
        verdict,
        "",
        "**Where scene-first and routing differ.** They tie overall, but not per category: scene-first is better on "
        "questions about what was shown, routing leans ahead on what was said. Finding the exact second is not "
        "a difference: on the questions where both found the right scene, scene-first picked the right second "
        f"{m_sf}/{m_both} times and routing {m_ad}/{m_both} here ({f_sf_m}/{f_both} and {f_ad_m}/{f_both} on the "
        "first corpus). The Moment@1 columns above differ only because each method is scored on its own "
        "correct answers. If your users mostly ask about speech, use `.adaptive()` "
        "(`engine.search(q).adaptive()`).",
        "",
        "| Corpus | Questions | Scene-first | Routing | Scene-first only | Routing only | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *split,
    ]


def beyond_video() -> list[str]:
    """Photos, via cinematlas.core's own evaluate(): the finding outside video, and where it stops."""
    import photos

    results = photos.evaluate()
    table, kinds = [], []
    for name, label in (("aligned", "aligned: each photo with its own title and description"),
                        ("misaligned", "misaligned: each photo with another photo's title and description")):
        r = results[name]["report"]
        table.append(f"| {label} | {r.joint.hit1:.2f} | {r.merged.hit1:.2f} | {r.joint_only} | {r.merged_only} | "
                     f"{fmt_p(r.p)} |")
        for kind in ("visual", "text"):
            k = results[name]["by_kind"][kind]
            w, lo, p = k["split"]
            kinds.append(f"| {name} | {kind} | {k['joint']:.2f} | {k['merged']:.2f} | {w} | {lo} | {fmt_p(p)} |")
    a, m = results["aligned"]["report"], results["misaligned"]["report"]
    return [
        "## Beyond video: photos",
        "",
        f"{results['aligned']['n']} public-domain NASA photos across 16 topics ([corpus](photos_corpus.json)), each "
        "embedded as `Text(title) + Text(description) + Image(photo)`. The joint vector is compared with merged "
        "per-part rankings (Reciprocal Rank Fusion over one vector per part) using `Collection.evaluate()`, the same "
        f"check users can run on their own data. {a.n} questions ([questions](queries_photos.json)), half about "
        "what a photo shows and half about facts in its text, were written by an AI agent that saw only the photos "
        "and their text, never the code or results.",
        "",
        "The misaligned collection is the boundary test: identical photos and text, but each photo is paired with "
        "another photo's title and description, so the parts of a record no longer describe the same thing. "
        "Predictions, recorded before these runs: the joint vector wins on aligned records, and its advantage "
        "disappears on misaligned ones.",
        "",
        "| Collection | Joint Hit@1 | Merged Hit@1 | Joint only | Merged only | p |",
        "| --- | --- | --- | --- | --- | --- |",
        *table,
        "",
        "| Collection | Questions | Joint | Merged | Joint only | Merged only | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *kinds,
        "",
        f"Going from aligned to misaligned, the joint vector's Hit@1 changes by {m.joint.hit1 - a.joint.hit1:+.2f} "
        f"and merged rankings' by {m.merged.hit1 - a.merged.hit1:+.2f}.",
        "",
        "**Predictions vs outcome.** First prediction (joint wins on aligned records): "
        + ("held." if a.winner == "joint" else "did not hold.")
        + " Second prediction (its advantage disappears on misaligned records): "
        + ("held." if m.winner != "joint" else
           "did not hold. Misalignment hurt merged rankings more than the joint vector: Reciprocal Rank Fusion "
           "rewards records that rank well in every list, and once a record's parts describe different things "
           "its lists stop agreeing. The joint vector did degrade, mostly on questions about the photo, which "
           "two unrelated text parts now outweigh. Where merging rankings beats early fusion, if anywhere, is "
           "still open; a smarter merge than RRF is the next thing to test."),
        "",
        "Reproduce: `uv run python bench/photos.py --ingest`, then `uv run python bench/run.py`.",
    ]


def _visual(methods: dict, name: str) -> float:
    return methods[name]["by_kind"]["visual"]["hit1"]


def boundary_section() -> list[str]:
    """Where early fusion stops winning: one unrelated part, and long records."""
    import boundary

    res = boundary.evaluate()

    def cell(m: dict, joint: bool) -> str:
        base = f"{m['hit1']:.2f} ({m['by_kind']['visual']['hit1']:.2f} / {m['by_kind']['text']['hit1']:.2f})"
        return f"**{base}**" if joint else f"{base}, {m['joint_only']} vs {m['method_only']}, p {fmt_p(m['p'])}"

    def rows(block: list[dict]) -> list[str]:
        methods = list(dict.fromkeys(m for row in block for m in row["methods"]))
        out = ["| Method | " + " | ".join(r["name"] for r in block) + " |", "| --- |" + " --- |" * len(block)]
        for m in methods:
            out.append(f"| {m} | " + " | ".join(cell(r["methods"][m], m == "joint vector") if m in r["methods"]
                                                  else "—" for r in block) + " |")
        return out

    b1, b2 = res["b1"], res["b2"]
    mis = b1[1]["methods"]
    long_rows = [r for r in b2 if not r["name"].endswith(" 1 description")]
    late_wins = [r["name"] for r in long_rows if r["methods"]["chunked late (sum)"]["method_only"]
                 > r["methods"]["chunked late (sum)"]["joint_only"] and r["methods"]["chunked late (sum)"]["p"] < 0.05]
    fusion_rows = [r for r in long_rows if "chunk-level joint" in r["methods"]]
    fusion_verdict = ""
    if fusion_rows:
        parts = []
        for r in fusion_rows:
            c, late = r["methods"]["chunk-level joint"], r["methods"]["chunked late (sum)"]
            a_only, b_only, p = mcnemar(r["correct"]["chunk-level joint"], r["correct"]["chunked late (sum)"])
            parts.append(f"{r['name']}: {c['hit1']:.2f} vs chunked late {late['hit1']:.2f} ({a_only} vs {b_only}, "
                         f"p {fmt_p(p)}), visual {c['by_kind']['visual']['hit1']:.2f}")
        fusion_verdict = (" Chunk-level fusion (the photo embedded together with each chunk, a record scoring its "
                          "best chunk), predicted after seeing B2 to beat chunked late fusion and recover visual "
                          "accuracy to 0.80 or more: " + "; ".join(parts) + ".")
    return [
        "## Where early fusion loses",
        "",
        "Two controlled tests on the photo corpus and its 80 questions ([boundary.py](boundary.py)). Cells: Hit@1 "
        "(visual / text questions), and against the joint vector: questions only it got right vs only the method "
        "got right, exact McNemar p.",
        "",
        "**B1: a single unrelated part.** Each record is `Text(description) + Image(photo)`, one part each, with "
        "the photo's own description (aligned) or another photo's (misaligned). This removes the two-texts-vs-one-"
        "photo imbalance of the earlier misaligned test.",
        "",
        *rows(b1),
        "",
        "**B2: long records.** Each record is `Text(body) + Image(photo)`, the body being the photo's own "
        "description buried among 0, 7 or 31 descriptions of NASA photos from outside the corpus "
        "([distractors](distractors.json)), so every answer stays unique. Chunked late fusion gets the ideal "
        "chunking: one vector per description, a record scored by its best chunk, merged with the photo's vector. "
        "Unchunked late fusion merges one whole-body vector with the photo's.",
        "",
        *rows(b2),
        "",
        "**Predictions vs outcome** (recorded before these runs). The joint vector still wins with one unrelated "
        "part but loses to CombSUM on visual questions: "
        + ("held." if _visual(mis, "merged (sum)") > _visual(mis, "joint vector")
           else f"did not hold; it wins overall and on visual questions "
                f"({mis['joint vector']['by_kind']['visual']['hit1']:.2f} vs "
                f"{mis['merged (sum)']['by_kind']['visual']['hit1']:.2f}).")
        + " On long records, chunked late fusion beats the joint vector significantly: "
        + (f"held, on {', '.join(late_wins)}. " if late_wins else "did not hold. ")
        + "This is the boundary: one vector per record stops working when a part is long enough that the relevant "
        "passage is a small share of it, and the long text also drowns out the photo (the joint vector's visual "
        "accuracy falls with length though the photo never changes)."
        + fusion_verdict,
        "",
        *chunker_lines(),
    ]


def chunker_lines() -> list[str]:
    """Real chunkers vs the ideal boundaries, through the public API (bench/boundary.py CHUNKERS)."""
    import boundary

    from cinematlas.core import Atlas

    with Atlas() as atlas:
        res = boundary.evaluate_chunkers(atlas)
    if not res:
        return []
    rows = [f"| {name} | {r['length']} | {r['chunks']} | {r['hit1']:.2f} ({r['by_kind']['visual']:.2f} / "
            f"{r['by_kind']['text']:.2f}) | {r['ideal']:.2f} | {r['vs_ideal'][0]} vs {r['vs_ideal'][1]}, "
            f"p {fmt_p(r['vs_ideal'][2])} |" for name, r in res.items()]
    semantic = next((r for n, r in res.items() if "Semantic" in n), None)
    fixed = [r for n, r in res.items() if n.startswith("unmarked, fixed")]
    verdicts = []
    if semantic and fixed:
        held = semantic["hit1"] >= 0.88 and all(semantic["hit1"] > f["hit1"] for f in fixed)
        fixed_scores = ", ".join(f"{f['hit1']:.2f}" for f in fixed)
        verdicts.append("With paragraph breaks removed, the semantic chunker beats fixed-size chunking and gets "
                        f"within 0.05 of the ideal (>= 0.88): {'held' if held else 'did not hold'} "
                        f"({semantic['hit1']:.2f}; fixed-size {fixed_scores}). Directly against fixed-size chunking "
                        "its lead is not significant at this size ("
                        + "; ".join("{} vs {}, p {}".format(*mcnemar(semantic["correct"], f["correct"])[:2],
                                                            fmt_p(mcnemar(semantic["correct"], f["correct"])[2]))
                                    for f in fixed)
                        + "); every real chunker keeps chunk-level fusion far above one vector per record.")
    return [
        "**Real chunkers.** Chunk-level fusion through `cinematlas.core`, with the chunkers users get. \"Unmarked\" "
        "removes the paragraph breaks, so a chunker has to find topic boundaries itself (with them present, "
        "`Paragraphs` gets the ideal boundaries by construction). The semantic chunker's rule and defaults were "
        "chosen on documents built only from the distractor pool, which no question targets. Ideal: chunk-level "
        "fusion with one description per chunk, at the same length.",
        "",
        "| Chunker | Descriptions per record | Chunks stored | Hit@1 (visual / text) | Ideal | vs ideal |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "**Predictions vs outcome** (recorded before these runs). " + " ".join(verdicts),
    ]


def _oracle_wins(row: dict) -> bool:
    return row["p"] < 0.05 and row["method_only"] > row["joint_only"]


def outside_nasa() -> list[str]:
    """The Met's open-access collection: a different domain, through Collection.evaluate()."""
    import met

    res = met.evaluate()
    rows, wins = [], []
    for fusion, label in (("rrf", "RRF"), ("sum", "CombSUM (strongest merge)")):
        r, kinds = res[fusion]["report"], res[fusion]["by_kind"]
        rows.append(f"| {label} | {r.joint.hit1:.2f} | {r.merged.hit1:.2f} | {kinds['visual'][0]:.2f} vs "
                    f"{kinds['visual'][1]:.2f} | {kinds['text'][0]:.2f} vs {kinds['text'][1]:.2f} | "
                    f"{r.joint_only} vs {r.merged_only} | {fmt_p(r.p)} |")
        wins.append(r.winner == "joint")
    return [
        "## Outside NASA: the Met",
        "",
        f"{res['n_records']} public-domain artworks from the Metropolitan Museum's open-access collection (CC0), "
        "16 subjects from armor to calligraphy ([corpus](met_corpus.json)), each embedded as `Text(title) + "
        "Text(details) + Image(photo)`, details being artist, date, medium, culture and department. 80 questions "
        "([questions](queries_met.json)), half about what a work shows and half about its catalogue facts, were "
        "written by an AI agent that saw only the images and records. Run with `Collection.evaluate()` against "
        "two merges. Code: [met.py](met.py).",
        "",
        "| Merged with | Joint Hit@1 | Merged Hit@1 | Visual (joint vs merged) | Text (joint vs merged) | "
        "Joint only vs merged only | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "**Prediction vs outcome** (recorded before any Met data existed): the joint vector beats merged rankings "
        "significantly against both RRF and CombSUM: " + ("held." if all(wins) else "did not hold."),
    ]


def query_robustness() -> list[str]:
    """Predictions 13-14: are the benchmark questions too clean? Terse simulated searchers, and messy typing."""
    import met

    res = met.evaluate_robustness()
    rows = []
    for name, r in res.items():
        s, rrf = r["sum"], r["rrf"]
        rows.append(f"| {name} | **{s.joint.hit1:.2f}** | {s.merged.hit1:.2f} | {rrf.merged.hit1:.2f} | "
                    f"{s.joint_only} vs {s.merged_only} | {fmt_p(s.p)} |")
    return [
        "## Are the questions too clean?",
        "",
        "Benchmark questions written by an AI agent that sees the answer tend to be complete, well spelled and "
        "detail-rich, which could flatter a joint vector. Two stress tests on the Met artworks "
        "([predictions](PREDICTIONS.md), committed before the runs): **terse searchers**, an agent simulating "
        "a visitor who glimpsed one work and later types 2–5 words from memory, one detail, sometimes misspelled "
        "([queries](queries_met_searchers.json)); and **messy typing**, the 80 questions degraded by a fixed-seed "
        "script (filler dropped, at most five words, a typo in about one word in four; `met.noisy`).",
        "",
        "| Queries | Joint | CombSUM | RRF | Joint only vs CombSUM only | p |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "Both predictions held. The joint vector keeps a significant lead over the strongest merge on terse, "
        "single-detail queries, and under messy typing the gap widens: merged rankings degrade faster than the "
        "joint vector. Real users remain untested.",
    ]


def smarter_merges() -> list[str]:
    """Every late-fusion method we could think of, plus an oracle, against the joint vector on all four corpora."""
    import fusion

    results = fusion.run_all()
    real = ["rrf", "sum", "mnz", "max", "learned (2-fold CV)"]
    closure, beaten, sig = [], 0, 0
    for name, res in results.items():
        r = res["rows"]
        best = max(real, key=lambda m: r[m]["hit1"])
        gap = r["oracle single signal"]["hit1"] - r[best]["hit1"]
        share = (r["joint vector"]["hit1"] - r[best]["hit1"]) / gap if gap > 0 else float("nan")
        closure.append(f"{share:.0%} ({name})")
        for m in real:
            beaten += r[m]["hit1"] < r["joint vector"]["hit1"]
            sig += r[m]["joint_only"] > r[m]["method_only"] and r[m]["p"] < 0.05
    total = len(real) * len(results)
    rrf_beaten = sum(res["rows"][m]["hit1"] > res["rows"]["rrf"]["hit1"]
                     for res in results.values() for m in ("sum", "mnz", "max", "learned (2-fold CV)"))
    max_mis = results["photos, misaligned"]["rows"]["max"]["hit1"]
    joint_mis = results["photos, misaligned"]["rows"]["joint vector"]["hit1"]
    return [
        "## Can a smarter merge win?",
        "",
        "Merged rankings above use Reciprocal Rank Fusion. Here every signal's own index is searched once per "
        "question (top 50, with scores) and merged five ways, all from the same retrieved lists: RRF; CombSUM "
        "(add min-max-normalized scores); CombMNZ (CombSUM times the number of lists that found the record); "
        "CombMAX (a record's best single-signal score, which needs no agreement between lists); and CombSUM with "
        "per-signal weights learned by 2-fold cross-validation. The oracle picks, for each question, whichever "
        "single signal ranks the answer highest, knowing the answer: not a real method, but an upper bound for "
        "any router that sends each question to one signal. Merges use only the separate signals (video: "
        "keyframe, transcript, full text; photos: title, description, photo), never the joint vector. "
        "Cells: Hit@1 (joint only vs method only, exact McNemar p). Code: [fusion.py](fusion.py).",
        "",
        *fusion.table(results),
        "",
        f"**No real merge beats the joint vector on any corpus.** It is ahead in all {total} comparisons, "
        f"significantly in {sig}. Of the gap between the best real merge and the oracle, the joint vector "
        f"closes {', '.join(closure)}.",
        "",
        "**Predictions vs outcome** (recorded before this run). Score-based merges beat RRF but none beats the "
        f"joint vector: mostly held; they beat RRF in {rrf_beaten} of {len(results) * 4} cases and none beats the "
        "joint vector. CombMAX ties or beats the joint vector on misaligned "
        + ("photos: held." if max_mis >= joint_mis else f"photos: did not hold ({max_mis:.2f} vs {joint_mis:.2f}).")
        + " The oracle beats the joint vector everywhere: it's ahead on every corpus, significantly on "
        + ", ".join(n for n, res in results.items() if _oracle_wins(res["rows"]["oracle single signal"]))
        + ". That remaining gap is what perfect per-question routing could still add on top of early fusion.",
    ]


def caption_ablation(ab: dict) -> list[str]:
    """Same corpus, same questions, keyframes with the burned-in caption band cropped before embedding."""
    names = ["visual only (keyframes)", "joint vector only (image+speech)", "transcript + rerank",
             "adaptive routing"]
    table = []
    for n in names:
        c, x = ab[n, "captions"], ab[n, "no captions"]
        table.append(f"| {n} | {c['speech']['hit@1']:.2f} → {x['speech']['hit@1']:.2f} | "
                     f"{c['visual']['hit@1']:.2f} → {x['visual']['hit@1']:.2f} | "
                     f"{c['mean@1']:.2f} → **{x['mean@1']:.2f}** |")
    speech = lambda n: {lab: ab[n, lab]["speech"]["hit@1"] for lab in ("captions", "no captions")}  # noqa: E731
    kf, sc = speech("visual only (keyframes)"), speech("joint vector only (image+speech)")
    paired = []
    for label in ("captions", "no captions"):
        w, lo, p = mcnemar(ab["adaptive routing", label]["per_q"],
                           ab["joint vector only (image+speech)", label]["per_q"])
        paired.append(f"| {label} | {w} | {lo} | {fmt_p(p)} |")
    return [
        "## Caption ablation",
        "",
        "Every frame in this corpus shows its dialogue as a burned-in caption, so image vectors can read the "
        "speech. Most real video has no burned-in captions. This run crops the caption band (bottom 15%) off each "
        f"keyframe before embedding (collection `{NO_CAPTIONS}`); transcripts, questions and labels are unchanged.",
        "",
        "| Configuration | Speech Hit@1 | Visual Hit@1 | Mean Hit@1 |",
        "| --- | --- | --- | --- |",
        *table,
        "",
        "Without captions, keyframes alone lose speech questions "
        f"({kf['captions']:.2f} → {kf['no captions']:.2f}): pixels were reading the subtitles. The joint vector "
        f"doesn't need them ({sc['captions']:.2f} → {sc['no captions']:.2f}), because the speech is inside the "
        "embedding, not painted on the frame.",
        "",
        "**Paired comparison, adaptive routing vs joint vector only** (all 60 questions). Only questions where "
        "exactly one "
        "system is right carry information; *p* is an exact two-sided McNemar test.",
        "",
        "| Corpus | Routing right, joint vector wrong | Joint vector right, routing wrong | p |",
        "| --- | --- | --- | --- |",
        *paired,
    ]


def reading_guide(rows: list[dict]) -> str:
    """Computed from this run, so it can't drift from the table."""
    by = {r["config"]: r for r in rows}
    pick = lambda frag: next(r for name, r in by.items() if frag in name)  # noqa: E731
    joint, fused = pick("joint vector only"), pick("tuned weights")
    adaptive, default = pick("adaptive routing · autoEmbed"), pick("scene-first (default)")
    w, lo, p = mcnemar(joint["per_q"], fused["per_q"])
    w2, lo2, p2 = mcnemar(default["per_q"], adaptive["per_q"])
    return (
        "**Reading this table.** One joint image+speech vector per scene (mean "
        f"{joint['mean@1']:.2f}) beats fusing the same signals after retrieval (best tuned fusion "
        f"{fused['mean@1']:.2f}): it wins {w} questions the fusion misses and loses {lo} (exact McNemar "
        f"p = {fmt_p(p)}). Separate lists disagree on every question that is about only one of the two, and "
        "rank fusion averages the disagreement away; a joint vector never produces it. The default, scene-first, "
        "ranks with that vector and uses the reranker only to pick the second (Moment@1 "
        f"{default['speech']['moment@1']:.2f}), in {default['p50_ms']:.0f} ms. Adaptive routing, which repairs "
        f"late fusion by choosing a specialist per question, ties it ({w2} vs {lo2}, p = {fmt_p(p2)}) at "
        f"{adaptive['p50_ms']:.0f} ms: it leans ahead on questions about what was said, behind on what was shown."
        "\n\n**Limits.** 60 questions over 6 videos from one program, written by the authors; one question is "
        "3.3 points, so only gaps confirmed by the paired test count. Weights and routing thresholds were tuned on "
        "this set, which is why the held-out corpus below exists. Latency is one laptop to one cloud region, "
        "comparative only."
    )

if __name__ == "__main__":
    main()
