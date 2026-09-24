"""Where does early fusion stop winning? Two controlled boundary tests on the photo corpus.

B1  A single unrelated part. Each record is Text(description) + Image(photo), one part each. Aligned: the
    photo's own description. Misaligned: another photo's. (The earlier misaligned test had two text parts
    against one photo; this removes that imbalance.)

B2  Long documents. Each record is Text(body) + Image(photo), where body is the photo's own description
    buried among L-1 descriptions of *other* NASA photos from outside the corpus (bench/distractors.json),
    so every answer stays unique. L = 1, 8, 32. The joint vector must compress the whole body into one
    vector; late fusion gets the ideal chunking, one vector per description, scored by each record's best
    chunk, merged with the photo's own vector. Chunk-level fusion does both: the photo is embedded
    together with each chunk, and a record scores its best chunk.

    uv run python bench/boundary.py --build     # distractor pool (once; it's committed)
    uv run python bench/boundary.py --ingest    # embed every collection
    uv run python bench/boundary.py             # evaluate; bench/run.py includes it in RESULTS.md
"""

import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import photos  # noqa: E402

from cinematlas.core import Atlas, Image, Paragraphs, Semantic, Text, mcnemar  # noqa: E402
from cinematlas.core.collection import _query_inputs, part_path  # noqa: E402
from cinematlas.core.fusion import comb_sum, rrf  # noqa: E402

HERE = Path(__file__).parent
POOL = HERE / "distractors.json"
DB = "cinematlas_bench_boundary"
LENGTHS = (1, 8, 32)
DEPTH = 50
POOL_TOPICS = ["hurricane from space", "aeronautics research", "telescope mirror", "rover prototype testing",
               "clean room", "balloon launch", "wildfire satellite", "moon rock laboratory", "space suit",
               "parachute test", "ice sheet", "comet", "student challenge", "supercomputer", "robotic arm",
               "airborne observatory", "sounding rocket", "asteroid", "glacier", "ocean color",
               "lightning", "aurora", "heliophysics", "exoplanet illustration"]


def build(per_topic: int = 40) -> None:
    corpus = {p["nasa_id"] for p in json.loads(photos.MANIFEST.read_text())}
    titles = {p["title"] for p in json.loads(photos.MANIFEST.read_text())}
    pool, seen = [], set()
    for topic in POOL_TOPICS:
        url = "https://images-api.nasa.gov/search?" + urllib.parse.urlencode(
            {"q": topic, "media_type": "image", "page_size": per_topic})
        with urllib.request.urlopen(url, timeout=30) as r:
            for item in json.load(r)["collection"]["items"]:
                d = item["data"][0]
                text = (d.get("description") or "").strip()
                if d["nasa_id"] in corpus | seen or d.get("title") in titles or len(text) < 80:
                    continue
                seen.add(d["nasa_id"])
                pool.append({"nasa_id": d["nasa_id"], "description": text[:1500]})
    POOL.write_text(json.dumps(pool, indent=1))
    print(f"{len(pool)} distractor descriptions → {POOL}")


# ------------------------------------------------------------------ records
def b1_records(misaligned: bool) -> list[dict]:
    return photos.records(misaligned)  # description (own or another photo's) + photo


def b2_records(length: int) -> tuple[list[dict], list[dict]]:
    """(records with a long body, one chunk record per description in each body)."""
    pool = [d["description"] for d in json.loads(POOL.read_text())]
    rng = random.Random(length)
    records, chunks = [], []
    for r in photos.records(False):
        parts = rng.sample(pool, length - 1)
        parts.insert(rng.randrange(length), r["description"])
        records.append({**r, "body": "\n\n".join(parts)})
        chunks += [{"cid": f"{r['nasa_id']}#{i}", "parent": r["nasa_id"], "chunk": text, "image": r["image"]}
                   for i, text in enumerate(parts)]
    return records, chunks


def b1_collection(atlas: Atlas, misaligned: bool):
    return atlas.collection(f"{DB}.single_{'misaligned' if misaligned else 'aligned'}",
                            embed=Text("description") + Image("image"), key="nasa_id", late=True)


def b2_collections(atlas: Atlas, length: int):
    joint = atlas.collection(f"{DB}.long{length}", embed=Text("body", max_chars=100_000) + Image("image"),
                             key="nasa_id", late=True)
    chunks = atlas.collection(f"{DB}.long{length}_chunks", embed=Text("chunk"), key="cid")
    return joint, chunks


def chunk_fusion_collection(atlas: Atlas, length: int):
    """Early fusion at chunk granularity: each chunk embedded together with its record's photo."""
    return atlas.collection(f"{DB}.long{length}_jointchunks", embed=Text("chunk") + Image("image"), key="cid")


def ingest_chunk_fusion(lengths=(8, 32)) -> None:
    progress = lambda s, a: print(f"\r  {s}", end="", flush=True)  # noqa: E731
    with Atlas() as atlas:
        for length in lengths:
            coll = chunk_fusion_collection(atlas, length).setup(timeout_s=900)
            print(coll, coll.add(b2_records(length)[1], batch_size=32, progress=progress))
            coll.wait_until_searchable(timeout_s=900)


# Chunkers, through the public API. "unmarked" strips paragraph breaks, so a chunker has to find topic
# boundaries itself (with them present, a paragraph chunker gets the ideal boundaries by construction).
CHUNKERS = {
    "unmarked, fixed 1600 chars": (8, False, lambda: Paragraphs(1600)),
    "unmarked, fixed 500 chars": (8, False, lambda: Paragraphs(500)),
    "unmarked, Semantic(800)": (8, False, lambda: Semantic(800)),
}


def chunker_records(length: int, marked: bool) -> list[dict]:
    records = b2_records(length)[0]
    return records if marked else [{**r, "body": r["body"].replace("\n\n", " ")} for r in records]


def chunker_collection(atlas: Atlas, name: str):
    length, _, make = CHUNKERS[name]
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")[:40]
    return atlas.collection(f"{DB}.chunker_{slug}", embed=Text("body", chunk=make()) + Image("image"),
                            key="nasa_id")


def ingest_chunkers(names=None) -> None:
    with Atlas() as atlas:
        for name in names or CHUNKERS:
            length, marked, _ = CHUNKERS[name]
            coll = chunker_collection(atlas, name).setup(timeout_s=900)
            print(name, coll.add(chunker_records(length, marked), batch_size=32,
                                 progress=lambda s, a: print(f"\r  {s}", end="", flush=True)), flush=True)
            coll.wait_until_searchable(timeout_s=900)


def evaluate_chunkers(atlas: Atlas) -> dict:
    """Hit@1 per chunker, plus each one's paired test against the ideal-boundary result at the same length."""
    qs, out = photos.questions(False), {}
    for name, (length, _, _) in CHUNKERS.items():
        coll = chunker_collection(atlas, name)
        if not coll.count():
            continue
        ideal = chunk_fusion_collection(atlas, length)
        hits, ref = [], []
        for q in qs:
            rel = set(q["relevant"])
            hits.append(top1([h["_key"] for h in coll.search(q["q"], k=5, moment=False)], rel))
            ref.append(top1(best_chunk(ideal, ideal._query_vector(_query_inputs(q["q"])), DEPTH), rel))
        by_kind = {k: sum(h for h, q in zip(hits, qs, strict=True) if q["kind"] == k) / 40 for k in ("visual", "text")}
        out[name] = {"length": length, "hit1": sum(hits) / len(hits), "by_kind": by_kind,
                     "ideal": sum(ref) / len(ref), "vs_ideal": mcnemar(hits, ref),
                     "chunks": coll.mongo.count_documents({}), "correct": hits}
        print(name, out[name])
    return out


def best_chunk(coll, vec, depth: int) -> list:
    best: dict = {}
    for r in coll._vector_search("embedding", vec, depth * 4, None):
        best[r["parent"]] = max(best.get(r["parent"], 0.0), r["score"])
    return [doc for doc, _ in sorted(best.items(), key=lambda kv: -kv[1])]


def ingest() -> None:
    progress = lambda s, a: print(f"\r  {s}", end="", flush=True)  # noqa: E731
    with Atlas() as atlas:
        for misaligned in (False, True):
            coll = b1_collection(atlas, misaligned).setup(timeout_s=900)
            print(coll, coll.add(b1_records(misaligned), progress=progress))
            coll.wait_until_searchable(timeout_s=600)
        for length in LENGTHS:
            joint, chunks = b2_collections(atlas, length)
            records, chunk_records = b2_records(length)
            joint.setup(timeout_s=900)
            print(joint, joint.add(records, batch_size=4 if length > 8 else 16, progress=progress))
            chunks.setup(timeout_s=900)
            print(chunks, chunks.add(chunk_records, batch_size=64, progress=progress))
            joint.wait_until_searchable(timeout_s=900)
            chunks.wait_until_searchable(timeout_s=900)


# ------------------------------------------------------------------ evaluation
def _row(name: str, qs: list[dict], correct: dict[str, list[bool]]) -> dict:
    out = {"name": name, "n": len(qs), "methods": {}, "correct": correct}
    for method, hits in correct.items():
        by_kind = {}
        for kind in ("visual", "text"):
            idx = [i for i, q in enumerate(qs) if q["kind"] == kind]
            a, b, p = mcnemar([correct["joint vector"][i] for i in idx], [hits[i] for i in idx])
            by_kind[kind] = {"hit1": sum(hits[i] for i in idx) / len(idx), "joint_only": a, "method_only": b, "p": p}
        a, b, p = mcnemar(correct["joint vector"], hits)
        out["methods"][method] = {"hit1": sum(hits) / len(hits), "joint_only": a, "method_only": b, "p": p,
                                  "by_kind": by_kind}
    return out


def top1(ranking: list, rel: set) -> bool:
    return bool(ranking) and ranking[0] in rel


def evaluate_b1(atlas: Atlas) -> list[dict]:
    rows = []
    for misaligned in (False, True):
        coll, qs = b1_collection(atlas, misaligned), photos.questions(misaligned)
        correct: dict[str, list[bool]] = {"joint vector": [], "merged (rrf)": [], "merged (sum)": []}
        for q in qs:
            vec = coll._query_vector(_query_inputs(q["q"]))
            lists = {f"part{i}": [(r["_key"], r["score"])
                                  for r in coll._vector_search(part_path(i), vec, DEPTH, None)] for i in range(2)}
            rel = set(q["relevant"])
            joint = [r["_key"] for r in coll._vector_search("embedding", vec, 5, None)]
            correct["joint vector"].append(top1(joint, rel))
            correct["merged (rrf)"].append(top1(rrf(lists), rel))
            correct["merged (sum)"].append(top1(comb_sum(lists), rel))
        rows.append(_row(f"single part, {'misaligned' if misaligned else 'aligned'}", qs, correct))
    return rows


def evaluate_b2(atlas: Atlas) -> list[dict]:
    rows, qs = [], photos.questions(False)
    for length in LENGTHS:
        joint, chunks = b2_collections(atlas, length)
        correct: dict[str, list[bool]] = {"joint vector": [], "chunked late (rrf)": [], "chunked late (sum)": [],
                                          "unchunked late (sum)": []}
        fused = chunk_fusion_collection(atlas, length)
        if length > 1 and fused.count():
            correct["chunk-level joint"] = []
        for q in qs:
            vec = joint._query_vector(_query_inputs(q["q"]))
            image = [(r["_key"], r["score"]) for r in joint._vector_search(part_path(1), vec, DEPTH, None)]
            whole = [(r["_key"], r["score"]) for r in joint._vector_search(part_path(0), vec, DEPTH, None)]
            best: dict = {}
            for r in chunks._vector_search("embedding", vec, DEPTH * 4, None):  # a record's best chunk
                best[r["parent"]] = max(best.get(r["parent"], 0.0), r["score"])
            chunked = sorted(best.items(), key=lambda kv: -kv[1])[:DEPTH]
            rel = set(q["relevant"])
            correct["joint vector"].append(top1([r["_key"] for r in joint._vector_search("embedding", vec, 5, None)],
                                                rel))
            correct["chunked late (rrf)"].append(top1(rrf({"image": image, "text": chunked}), rel))
            correct["chunked late (sum)"].append(top1(comb_sum({"image": image, "text": chunked}), rel))
            correct["unchunked late (sum)"].append(top1(comb_sum({"image": image, "text": whole}), rel))
            if "chunk-level joint" in correct:
                correct["chunk-level joint"].append(top1(best_chunk(fused, vec, DEPTH), rel))
        rows.append(_row(f"long text, {length} description{'s' if length > 1 else ''}", qs, correct))
    return rows


def evaluate() -> dict:
    with Atlas() as atlas:
        results = {"b1": evaluate_b1(atlas), "b2": evaluate_b2(atlas)}
    for row in results["b1"] + results["b2"]:
        print(row["name"], {m: (round(v["hit1"], 2), round(v["by_kind"]["visual"]["hit1"], 2),
                                round(v["by_kind"]["text"]["hit1"], 2)) for m, v in row["methods"].items()})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--ingest-chunk-fusion", type=int, nargs="*", help="lengths, e.g. 8 32")
    ap.add_argument("--ingest-chunkers", action="store_true", help="the chunker comparison")
    ap.add_argument("--chunkers", action="store_true", help="evaluate only the chunker comparison")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.ingest:
        ingest()
    elif args.ingest_chunkers:
        ingest_chunkers()
    elif args.chunkers:
        with Atlas() as atlas:
            evaluate_chunkers(atlas)
    elif args.ingest_chunk_fusion is not None:
        ingest_chunk_fusion(tuple(args.ingest_chunk_fusion) or (8, 32))
    else:
        evaluate()
