"""Basic: the joint image+speech vector finds the scene in one query, no reranker.

The benchmark finding in one script: this single vector ties the full pipeline on finding the
right scene. search() adds the exact second on top.

    uv run python examples/02_fast_scene_search.py
"""

import time

from _env import demo_engine

QUESTIONS = [
    "a baby's hand holding a finger",          # about what's shown
    "why can't supersonic jets fly over land",  # about what's said
]

with demo_engine() as engine:
    for q in QUESTIONS:
        engine.search_scene_vector(q, top_k=1)  # warm the query-embedding cache so timings compare search, not Voyage
        t = time.perf_counter()
        scene = engine.search_scene_vector(q, top_k=1).top
        fast_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        moment = engine.search(q, top_k=1).top
        full_ms = (time.perf_counter() - t) * 1000

        print(f"\n{q!r}")
        print(f"  scene vector  {fast_ms:5.0f} ms  {scene.video_id}#{scene.scene_id} starts {scene.timestamp}")
        print(f"  search()      {full_ms:5.0f} ms  {moment.video_id}#{moment.scene_id} at {moment.timestamp}: "
              f"{moment.text[:60]!r}")
