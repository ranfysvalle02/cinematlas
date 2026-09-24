"""Shared setup for the examples: load .env and fail early with a clear message."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from cinematlas import Cinematlas

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# The demo corpus: six public-domain NASA interviews, already indexed (see bench/).
DEMO = {"db_name": "cinematlas_bench", "collection_name": "scenes_autoembed"}
# Where examples write your own videos. Never the demo corpus.
YOURS = {"db_name": "cinematlas_examples", "collection_name": "scenes"}


def engine(target: dict = DEMO) -> Cinematlas:
    missing = [k for k in ("VOYAGE_API_KEY",) if not os.getenv(k)]
    if not (os.getenv("MONGODB_URI") or os.getenv("MDB_URI")):
        missing.insert(0, "MONGODB_URI")
    if missing:
        sys.exit(f"Set {' and '.join(missing)} in .env (see the README quickstart).")
    return Cinematlas(**target)
