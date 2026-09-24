"""Ingest the benchmark corpus (idempotent: re-ingest replaces).

    uv run python bench/ingest.py                # both transcript backends, as shipped
    uv run python bench/ingest.py --no-captions  # caption ablation: keyframes with the caption band cropped
    uv run python bench/ingest.py --station      # held-out corpus: ISS videos, different domain, no captions
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from corpus import (  # noqa: E402
    CAPTION_BAND,
    COLLECTIONS,
    DB,
    EPISODES,
    NO_CAPTIONS,
    STATION,
    STATION_COLLECTION,
    STATION_DB,
    url,
)

from cinematlas import Cinematlas  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class CaptionFree(Cinematlas):
    """Crops the burned-in caption band off each keyframe, so image vectors can't read the speech."""

    def _extract_keyframes(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        scenes = super()._extract_keyframes(*args, **kwargs)
        for s in scenes:
            img = s.get("keyframe_image")
            if img is not None:
                s["keyframe_image"] = img.crop((0, 0, img.width, round(img.height * (1 - CAPTION_BAND))))
        return scenes


def ingest(cls: type[Cinematlas], uri: str, mode: str, coll: str, db: str = DB,
           episodes: dict[str, str] = EPISODES) -> None:
    with cls(uri, db_name=db, collection_name=coll, transcript_mode=mode) as eng:
        print(f"[{coll}] indexes ->", eng.ensure_indexes(timeout_s=900))
        for vid, nasa_id in episodes.items():
            t = time.time()
            n = eng.ingest_video(url(nasa_id), video_id=vid)
            print(f"[{coll}] {vid}: {n} scenes in {time.time() - t:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-captions", action="store_true", help="ingest only the caption-ablation collection")
    ap.add_argument("--station", action="store_true", help="ingest only the held-out ISS corpus")
    args = ap.parse_args()
    uri = os.getenv("MONGODB_URI") or os.environ["MDB_URI"]
    if args.station:
        ingest(Cinematlas, uri, "autoembed", STATION_COLLECTION, STATION_DB, STATION)
        return
    if args.no_captions:
        ingest(CaptionFree, uri, "autoembed", NO_CAPTIONS)
        return
    for mode, coll in COLLECTIONS.items():
        ingest(Cinematlas, uri, mode, coll)


if __name__ == "__main__":
    main()
