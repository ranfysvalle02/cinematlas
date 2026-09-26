"""Can any late-fusion method beat the joint vector? Smarter merges, learned weights, and an oracle.

For every question on every corpus, each signal's own index is searched once (top 50, with scores), and
the joint vector is searched once. Every merge below is then computed offline from those same lists, so
all methods see exactly the same retrieval:

* rrf       Reciprocal Rank Fusion, k = 60 (what Atlas $rankFusion does; the baseline so far)
* sum       CombSUM: add min-max-normalized scores
* mnz       CombMNZ: CombSUM x number of lists that found the record
* max       CombMAX: a record's best single-signal score (needs no agreement between lists)
* learned   CombSUM with per-signal weights chosen by 2-fold cross-validation on the questions
* oracle    per question, the single signal that ranks the answer highest, chosen knowing the answer.
            Not a real method: an upper bound for any router that picks one signal per question.

    uv run python bench/fusion.py      # prints the table; bench/run.py includes it in RESULTS.md
"""

import itertools
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import photos  # noqa: E402
from corpus import COLLECTIONS, DB, STATION_COLLECTION, STATION_DB  # noqa: E402
from run import relevant  # noqa: E402

from cinematlas import Cinematlas  # noqa: E402
from cinematlas.core import Atlas, mcnemar  # noqa: E402
from cinematlas.core.collection import _query_inputs, part_path  # noqa: E402
from cinematlas.core.evaluate import fmt_p  # noqa: E402
from cinematlas.core.fusion import comb_max, comb_sum, rrf  # noqa: E402

HERE = Path(__file__).parent
DEPTH = 50
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
load_dotenv(HERE.parent / ".env")

# Merges live in the library so users' evaluate() and this experiment share one implementation.


# ------------------------------------------------------------------ corpora
def photo_items(misaligned: bool):
    """Per question: the joint list, each part's list, and the relevant ids."""
    with Atlas() as atlas:
        coll = photos.collections(atlas)[1 if misaligned else 0]
        names = [repr(p) for p in coll.embed.parts]
        for q in photos.questions(misaligned):
            vector = coll._query_vector(_query_inputs(q["q"]))
            lists = {name: [(r["_key"], r["score"]) for r in coll._vector_search(part_path(i), vector, DEPTH, None)]
                     for i, name in enumerate(names)}
            joint = [r["_key"] for r in coll._vector_search("embedding", vector, DEPTH, None)]
            yield joint, lists, set(q["relevant"])


def video_items(db: str, collection: str, speech_file: str, visual_file: str):
    uri = os.getenv("MONGODB_URI") or os.environ["MDB_URI"]
    eng = Cinematlas(uri, db_name=db, collection_name=collection, transcript_mode="autoembed")
    labels = json.loads((HERE / speech_file).read_text()) + json.loads((HERE / visual_file).read_text())
    for label in labels:
        q, key = label["q"], (lambda h: (h["video_id"], h["scene_id"]))
        found = {"keyframe": eng.search(q).only("visual").limit(DEPTH).run(),
                 "transcript": eng.search(q).only("transcript").limit(DEPTH).run(),
                 "full text": eng.search(q).only("text").limit(DEPTH).run()}
        joint_hits = eng.search(q).only("scene").limit(DEPTH).run()
        docs = {key(h): h for hits in [*found.values(), joint_hits] for h in hits}
        rel = {k for k, h in docs.items() if relevant(h, label)}
        # A labelled scene that no list returned still counts as relevant (it's a miss for everyone).
        if "scenes" in label:
            rel |= {tuple(s) for s in label["scenes"]}
        yield ([key(h) for h in joint_hits], {n: [(key(h), float(h.get("score") or 0)) for h in hits]
                                              for n, hits in found.items()}, rel)


CORPORA: dict[str, Callable] = {
    "video (interviews)": lambda: video_items(DB, COLLECTIONS["autoembed"], "queries.json", "queries_visual.json"),
    "held-out video (station)": lambda: video_items(STATION_DB, STATION_COLLECTION, "queries_station_speech.json",
                                                    "queries_station_visual.json"),
    "photos, aligned": lambda: photo_items(False),
    "photos, misaligned": lambda: photo_items(True),
}


# ------------------------------------------------------------------ evaluation
def top1(ranking: list, rel: set) -> bool:
    return bool(ranking) and ranking[0] in rel


def learned_cv(items: list, names: list[str]) -> list[bool]:
    """CombSUM weights fitted on one half of the questions, scored on the other half (2-fold)."""
    folds = [items[0::2], items[1::2]]
    grid = [dict(zip(names, ws, strict=True)) for ws in itertools.product(WEIGHT_GRID, repeat=len(names))
            if any(ws)]
    out: dict[int, bool] = {}
    for test in (0, 1):
        train = folds[1 - test]
        best = max(grid, key=lambda w: sum(top1(comb_sum(lists, w), rel) for _, lists, rel in train))
        for i, (_, lists, rel) in enumerate(folds[test]):
            out[2 * i + test] = top1(comb_sum(lists, best), rel)
    return [out[i] for i in range(len(items))]


def run_corpus(name: str) -> dict:
    items = list(CORPORA[name]())
    names = list(items[0][1])
    correct = {
        "joint vector": [top1(j, rel) for j, _, rel in items],
        "rrf": [top1(rrf(lists), rel) for _, lists, rel in items],
        "sum": [top1(comb_sum(lists), rel) for _, lists, rel in items],
        "mnz": [top1(comb_sum(lists, mnz=True), rel) for _, lists, rel in items],
        "max": [top1(comb_max(lists), rel) for _, lists, rel in items],
        "learned (2-fold CV)": learned_cv(items, names),
        "oracle single signal": [any(top1([d for d, _ in lists[n]], rel) for n in names) for _, lists, rel in items],
    }
    rows = {}
    for method, hits in correct.items():
        a, b, p = mcnemar(correct["joint vector"], hits)
        rows[method] = {"hit1": sum(hits) / len(hits), "joint_only": a, "method_only": b, "p": p}
    return {"n": len(items), "signals": names, "rows": rows}


def run_all() -> dict:
    results = {}
    for name in CORPORA:
        results[name] = run_corpus(name)
        r = results[name]["rows"]
        print(f"{name} (n={results[name]['n']}): " + ", ".join(f"{m} {v['hit1']:.2f}" for m, v in r.items()))
    return results


def table(results: dict) -> list[str]:
    methods = list(next(iter(results.values()))["rows"])
    head = "| Method | " + " | ".join(results) + " |"
    lines = [head, "| --- |" + " --- |" * len(results)]
    for m in methods:
        cells = []
        for res in results.values():
            v = res["rows"][m]
            mark = "" if m == "joint vector" else f" ({v['joint_only']} vs {v['method_only']}, p {fmt_p(v['p'])})"
            cells.append(f"**{v['hit1']:.2f}**{mark}" if m == "joint vector" else f"{v['hit1']:.2f}{mark}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines


if __name__ == "__main__":
    print("\n".join(table(run_all())))
