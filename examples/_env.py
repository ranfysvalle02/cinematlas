"""Shared setup: read .env and point at the live demo corpus (six NASA interviews, already indexed)."""

from pathlib import Path

from dotenv import load_dotenv

from cinematlas import Cinematlas

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # MONGODB_URI, VOYAGE_API_KEY


def demo_engine() -> Cinematlas:
    return Cinematlas(db_name="cinematlas_bench", collection_name="scenes_autoembed")
