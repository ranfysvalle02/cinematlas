"""Advanced: index your own video, watch each stage, then search it.

Writes to its own collection (cinematlas_examples.scenes), never the demo corpus. Uses a
public-domain NASA clip by default; pass any URL or local path.

    uv run python examples/05_ingest_your_video.py [url-or-path]
"""

import sys
import time

from _env import demo_engine  # noqa: F401  (loads .env)

from cinematlas import Cinematlas

SOURCE = sys.argv[1] if len(sys.argv) > 1 else (
    "https://images-assets.nasa.gov/video/NHQ20210805ARMD01/NHQ20210805ARMD01~small.mp4")


def progress(stage: str, info: dict) -> None:
    print(f"  {stage:<12} {info}")


with Cinematlas(db_name="cinematlas_examples", collection_name="scenes") as engine:
    print("indexes:", engine.ensure_indexes())       # idempotent; the first run waits for Atlas to build them
    print(f"\ningesting {SOURCE}")
    result = engine.ingest(SOURCE, progress=progress)
    print(f"\n{result}")
    print(f"stage timings: {result.stages}\n")

    # Atlas syncs new documents into search indexes asynchronously (usually seconds).
    question = "what does he enjoy about flying?"
    for _ in range(30):
        results = engine.search(question, top_k=2, video_id=result.video_id)
        if results:
            break
        time.sleep(2)
    print(f"{question!r}\n{results}")
