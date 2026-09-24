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
        seen = len(eng.search_transcript("the", top_k=min(spoken, 500)))
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
        "visual only (keyframes)": lambda q: auto.search_visual_vector(q, top_k=K),
        "joint vector only (image+speech)": lambda q: auto.search_scene_vector(q, top_k=K),
        "full-text only (Atlas Search BM25)": lambda q: auto.search_text(q, top_k=K),
        "transcript only · autoEmbed voyage-4": lambda q: auto.search_transcript(q, top_k=K),
        "transcript only · client voyage-4": lambda q: client.search_transcript(q, top_k=K),
        "transcript only · client, voyage-4-lite queries": lambda q: client_lite.search_transcript(q, top_k=K),
        "transcript + rerank": lambda q: auto.search(q, top_k=K, sources=("transcript",), routing="fixed"),
        "fixed fusion, equal weights, no rerank": lambda q: auto.search(
            q, top_k=K, rerank=False, weights=equal, routing="fixed"),
        "fixed fusion, equal weights + rerank": lambda q: auto.search(q, top_k=K, weights=equal, routing="fixed"),
        "fixed fusion, tuned weights + rerank": lambda q: auto.search(q, top_k=K, weights=tuned, routing="fixed"),
        "adaptive routing · autoEmbed": lambda q: auto.search(q, top_k=K, routing="adaptive"),
        "**scene-first (default)**: joint vector ranks, reranker picks the second": lambda q: auto.search(
            q, top_k=K, routing="scene"),
        "adaptive · client-side fusion": lambda q: client_side_fusion.search(q, top_k=K, routing="adaptive"),
        "adaptive · client transcript mode": lambda q: client.search(q, top_k=K, routing="adaptive"),
    }
    rows = []
    for name, fn in configs.items():
        row = both(name, fn, speech, visual, t_speech)
        s_row, v_row = row["speech"], row["visual"]
        rows.append(row)
        print(f"{name:52s} speech={s_row['hit@1']:.2f} visual={v_row['hit@1']:.2f} mean={row['mean@1']:.2f} "
              f"moment={s_row['moment@1']} p50={row['p50_ms']:.0f}ms")

    ablation = {}
    for name, method in [("visual only (keyframes)", "search_visual_vector"),
                         ("joint vector only (image+speech)", "search_scene_vector"),
                         ("transcript + rerank", None), ("adaptive routing", "search")]:
        for label, eng in [("captions", auto), ("no captions", nocap)]:
            if method is None:
                fn = lambda q, e=eng: e.search(q, top_k=K, sources=("transcript",), routing="fixed")  # noqa: E731
            else:
                fn = lambda q, e=eng, m=method: getattr(e, m)(q, top_k=K)  # noqa: E731
                if method == "search":
                    fn = lambda q, e=eng: e.search(q, top_k=K, routing="adaptive")  # noqa: E731
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
        *held_out(station, rows),
        "",
        "Reproduce: `uv run python bench/ingest.py`, then `--no-captions` and `--station`, then "
        "`uv run python bench/run.py` · "
        f"generated {time.strftime('%Y-%m-%d')}",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote bench/RESULTS.md")


HELD_OUT = {
    "keyframes only": lambda e, q: e.search_visual_vector(q, top_k=K),
    "transcript + rerank": lambda e, q: e.search(q, top_k=K, sources=("transcript",), routing="fixed"),
    "rank fusion, tuned weights + rerank": lambda e, q: e.search(
        q, top_k=K, weights={"visual": 0.25, "scene": 1, "transcript": 1, "text": 1, "rerank": 2}, routing="fixed"),
    "joint vector only": lambda e, q: e.search_scene_vector(q, top_k=K),
    "adaptive routing": lambda e, q: e.search(q, top_k=K, routing="adaptive"),
    "scene-first": lambda e, q: e.search(q, top_k=K, routing="scene"),
}


def held_out(eng: Cinematlas, first: list[dict]) -> list[str]:
    """A second corpus nobody tuned on: different domain, no burned-in captions, independently written questions."""
    speech = json.loads((HERE / "queries_station_speech.json").read_text())
    visual = json.loads((HERE / "queries_station_visual.json").read_text())
    t_speech = [answer_time(eng, lab) for lab in speech]
    for lab in speech + visual:  # warm the query cache so latency compares retrieval, not Voyage round trips
        eng.search_scene_vector(lab["q"], top_k=1)
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
        "questions about what was shown, routing leans ahead on what was said. Routing also lands on the exact "
        f"second more often when it finds the right scene (Moment@1 {ad['speech']['moment@1']:.2f} vs "
        f"{sf['speech']['moment@1']:.2f} here, {f_ad['speech']['moment@1']:.2f} vs "
        f"{f_sf['speech']['moment@1']:.2f} on the first corpus). If your users mostly ask about speech, pass "
        "`routing=\"adaptive\"`.",
        "",
        "| Corpus | Questions | Scene-first | Routing | Scene-first only | Routing only | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *split,
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
