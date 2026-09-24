"""Outside NASA: the Metropolitan Museum's open-access collection (CC0), through cinematlas.core.evaluate().

~400 public-domain artworks across 16 subjects, each embedded as Text(title) + Text(details) + Image(photo),
where details are the catalogue fields (artist, date, medium, culture, department). The joint vector is
compared with merged per-part rankings, against RRF and against CombSUM (the strongest merge found on the
other corpora). Questions were written by an AI agent that saw only the artworks' images and catalogue
records, never the code or results.

    uv run python bench/met.py --build     # fetch the corpus manifest (once; it's committed)
    uv run python bench/met.py --ingest    # embed
    uv run python bench/met.py             # evaluate; bench/run.py includes it in RESULTS.md
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from cinematlas.core import Atlas, Image, Text
from cinematlas.core.parts import safe_url

HERE = Path(__file__).parent
MANIFEST = HERE / "met_corpus.json"
QUESTIONS = HERE / "queries_met.json"
CACHE = Path(os.getenv("CINEMATLAS_BENCH_CACHE", Path.home() / ".cache" / "cinematlas-bench")) / "met"
API = "https://collectionapi.metmuseum.org/public/collection/v1"
SUBJECTS = ["portrait", "landscape", "armor", "vase", "textile", "sculpture", "mask", "jewelry", "ship", "horse",
            "flowers", "musical instrument", "clock", "furniture", "calligraphy", "coin"]
EMBED = Text("title") + Text("details") + Image("image")
load_dotenv(HERE.parent / ".env")


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "cinematlas-bench"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 ** attempt)
    return {}


def build(per_subject: int = 25) -> None:
    records, seen = [], set()
    for subject in SUBJECTS:
        query = urllib.parse.urlencode({"hasImages": "true", "q": subject})
        ids = _get(f"{API}/search?{query}").get("objectIDs") or []
        added = 0
        for object_id in ids[:200]:
            if added >= per_subject or object_id in seen:
                continue
            o = _get(f"{API}/objects/{object_id}")
            if not (o.get("isPublicDomain") and o.get("primaryImageSmall") and o.get("title")):
                continue
            details = ". ".join(v for v in [o.get("artistDisplayName"), o.get("objectDate"), o.get("medium"),
                                            o.get("culture"), o.get("department")] if v)
            seen.add(object_id)
            added += 1
            records.append({"object_id": str(object_id), "title": o["title"], "details": details,
                            "subject": subject, "image_url": o["primaryImageSmall"]})
            time.sleep(0.05)  # stay well under the API's rate limit
        print(f"{subject}: {added}", flush=True)
    MANIFEST.write_text(json.dumps(records, indent=1))
    print(f"{len(records)} artworks → {MANIFEST}")


def cached(record: dict) -> str:
    path = CACHE / f"{record['object_id']}.jpg"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(safe_url(record["image_url"]), headers={"User-Agent": "cinematlas-bench"})
        with urllib.request.urlopen(request, timeout=60) as r:
            path.write_bytes(r.read())
    return str(path)


def records() -> list[dict]:
    return [{**r, "image": cached(r)} for r in json.loads(MANIFEST.read_text())]


def collection(atlas: Atlas):
    return atlas.collection("cinematlas_bench_met.artworks", embed=EMBED, key="object_id", late=True,
                            moment="details", display=("title",))


def ingest() -> None:
    with Atlas() as atlas:
        coll = collection(atlas).setup(timeout_s=900)
        print(coll, coll.add(records(), progress=lambda s, a: print(f"\r  {s}", end="", flush=True)), flush=True)
        coll.wait_until_searchable(timeout_s=600)


def evaluate() -> dict:
    questions = json.loads(QUESTIONS.read_text())
    out = {}
    with Atlas() as atlas:
        coll = collection(atlas)
        for fusion in ("rrf", "sum"):
            report = coll.evaluate(questions, k=10, fusion=fusion)
            by_kind = {}
            for kind in ("visual", "text"):
                idx = [i for i, q in enumerate(questions) if q["kind"] == kind]
                by_kind[kind] = (sum(report.joint.correct[i] for i in idx) / len(idx),
                                 sum(report.merged.correct[i] for i in idx) / len(idx))
            out[fusion] = {"report": report, "by_kind": by_kind}
            print(f"[vs {fusion}]\n{report}\n", flush=True)
        out["n_records"] = coll.count()
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
            sys.exit(f"Missing {QUESTIONS}")
        evaluate()
