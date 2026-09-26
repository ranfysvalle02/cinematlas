"""Index your own video, watch every stage, then search it.

    uv run python examples/index_and_search.py                                # a NASA clip
    uv run python examples/index_and_search.py lecture.mp4 "when is the exam?"
    uv run python examples/index_and_search.py "youtube.com/watch?v=…" "the demo"

Writes to cinematlas_examples.scenes, never the demo corpus. Re-running replaces the video.
"""

import sys
import time

from _env import YOURS, engine

SOURCE = sys.argv[1] if len(sys.argv) > 1 else (
    "https://images-assets.nasa.gov/video/NHQ20210805ARMD01/NHQ20210805ARMD01~small.mp4")
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "what did he enjoy about flying?"

with engine(YOURS) as eng:
    print("Search indexes:", eng.ensure_indexes())  # idempotent; the first run waits for Atlas to build them

    print(f"\nIndexing {SOURCE}")
    result = eng.ingest(SOURCE, progress=lambda stage, info: print(f"  ✓ {stage:<12} {info}"))
    print(f"\n{result}\n{eng.usage}")

    # Atlas syncs new documents into its search indexes asynchronously, usually within seconds.
    # Wait until every spoken scene is searchable, or early answers come from a partial index.
    deadline = time.monotonic() + 120
    while len(eng.search("the").only("transcript").video(result.video_id).limit(500)) < result.spoken_scenes:
        if time.monotonic() > deadline:
            raise SystemExit("Indexed, but Atlas is still syncing its search indexes. Try again in a minute.")
        time.sleep(2)

    print(f"\nQ: {QUESTION}")
    results = eng.search(QUESTION).video(result.video_id).limit(3).run()

for i, hit in enumerate(results, 1):
    where = hit.link or f"{SOURCE}, at {hit.timestamp}"  # local files have no URL to deep-link
    print(f"{i}. [{hit.timestamp}] {hit.text[:110]}\n   {where}")
