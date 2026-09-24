"""Beyond video: does the joint vector beat merged rankings on photos, and where does that stop?

Corpus: ~400 public-domain NASA photos (bench/photos_corpus.json), each with a title, a description
and the photo. Two collections, both storing per-part vectors (late=True) so the same records can be
searched both ways:

* aligned:    Text(title) + Text(description) + Image(photo), all about the same photo
* misaligned: the same photos, but each paired with *another* photo's title and description, so the
              parts describe different things. The boundary test: does early fusion still win?

    uv run python bench/photos.py --build     # fetch the corpus manifest (once; it's committed)
    uv run python bench/photos.py --ingest    # embed both collections
    uv run python bench/photos.py             # evaluate; bench/run.py includes this in RESULTS.md
"""

import argparse
import json
import os
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from cinematlas.core import Atlas, Image, Text
from cinematlas.core.parts import safe_url

HERE = Path(__file__).parent
MANIFEST = HERE / "photos_corpus.json"
QUESTIONS = HERE / "queries_photos.json"
CACHE = Path(os.getenv("CINEMATLAS_BENCH_CACHE", Path.home() / ".cache" / "cinematlas-bench" / "photos"))
DB = "cinematlas_bench_photos"
TOPICS = ["hubble servicing mission", "apollo lunar surface", "mars rover", "space station interior",
          "shuttle launch", "spacewalk", "earth from orbit", "saturn cassini", "jupiter juno",
          "astronaut training", "rocket engine test", "wind tunnel", "aircraft flight research",
          "nebula", "solar flare", "mission control"]
EMBED = Text("title") + Text("description") + Image("image")
load_dotenv(HERE.parent / ".env")


def build(per_topic: int = 25) -> None:
    records, seen = [], set()
    for topic in TOPICS:
        url = "https://images-api.nasa.gov/search?" + urllib.parse.urlencode(
            {"q": topic, "media_type": "image", "page_size": per_topic})
        with urllib.request.urlopen(url, timeout=30) as r:
            items = json.load(r)["collection"]["items"]
        for item in items:
            data = item["data"][0]
            preview = next((x["href"] for x in item.get("links", []) if x.get("rel") == "preview"), None)
            if not preview or data["nasa_id"] in seen or not data.get("description"):
                continue
            seen.add(data["nasa_id"])
            records.append({"nasa_id": data["nasa_id"], "title": data.get("title", ""),
                            "description": data["description"][:1500], "center": data.get("center"),
                            "topic": topic, "image_url": preview})
    MANIFEST.write_text(json.dumps(records, indent=1))
    print(f"{len(records)} photos → {MANIFEST}")


def cached(record: dict) -> str:
    path = CACHE / f"{record['nasa_id']}.jpg"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(safe_url(record["image_url"]), timeout=30) as r:
            path.write_bytes(r.read())
    return str(path)


def records(misaligned: bool = False) -> list[dict]:
    photos = json.loads(MANIFEST.read_text())
    rows = [{**p, "image": cached(p)} for p in photos]
    if misaligned:
        # A fixed derangement: record i keeps its photo but takes the text of record order[i] != i.
        order = list(range(len(rows)))
        rng = random.Random(59)
        while any(i == j for i, j in enumerate(order)):
            rng.shuffle(order)
        texts = [(r["title"], r["description"], r["nasa_id"]) for r in rows]
        rows = [{**r, "title": texts[j][0], "description": texts[j][1], "text_of": texts[j][2]}
                for r, j in zip(rows, order, strict=True)]
    return rows


def collections(atlas: Atlas):
    common = {"embed": EMBED, "key": "nasa_id", "moment": "description", "late": True, "display": ("title",)}
    return (atlas.collection(f"{DB}.aligned", **common), atlas.collection(f"{DB}.misaligned", **common))


def ingest() -> None:
    with Atlas() as atlas:
        for coll, misaligned in zip(collections(atlas), (False, True), strict=True):
            coll.setup(timeout_s=900)
            print(coll, coll.add(records(misaligned), progress=lambda s, a: print(f"\r  {s}", end="", flush=True)))
            coll.wait_until_searchable(timeout_s=600)


def questions(misaligned: bool) -> list[dict]:
    """Visual questions point at a photo, text questions at a title/description. When they're split
    across records, each question's answer is the record that now holds that photo or that text."""
    qs = json.loads(QUESTIONS.read_text())
    if not misaligned:
        return [{"q": q["q"], "relevant": q["relevant"], "kind": q["kind"]} for q in qs]
    holder = {r["text_of"]: r["nasa_id"] for r in records(misaligned=True)}
    return [{"q": q["q"], "kind": q["kind"],
             "relevant": q["relevant"] if q["kind"] == "visual" else [holder[x] for x in q["relevant"]]} for q in qs]


def evaluate() -> dict:
    out = {}
    with Atlas() as atlas:
        for coll, misaligned in zip(collections(atlas), (False, True), strict=True):
            qs = questions(misaligned)
            report = coll.evaluate(qs, k=10)
            by_kind = {}
            for kind in ("visual", "text"):
                idx = [i for i, q in enumerate(qs) if q["kind"] == kind]
                by_kind[kind] = {
                    "joint": sum(report.joint.correct[i] for i in idx) / len(idx),
                    "merged": sum(report.merged.correct[i] for i in idx) / len(idx),
                    "split": __import__("cinematlas.core", fromlist=["mcnemar"]).mcnemar(
                        [report.joint.correct[i] for i in idx], [report.merged.correct[i] for i in idx]),
                }
            out["misaligned" if misaligned else "aligned"] = {"report": report, "by_kind": by_kind,
                                                              "n": coll.count()}
            print(f"[{'misaligned' if misaligned else 'aligned'}]\n{report}\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ingest", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.ingest:
        ingest()
    else:
        if not QUESTIONS.is_file():
            sys.exit(f"Missing {QUESTIONS}: see the module docstring.")
        evaluate()
