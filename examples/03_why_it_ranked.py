"""Advanced: see how search reads a question: said vs shown, and why each hit ranked.

    uv run python examples/03_why_it_ranked.py
"""

from _env import demo_engine

QUESTIONS = [
    "what did he say about his first flight",   # said
    "little girl standing on top of hay bales",  # shown
]

with demo_engine() as engine:
    for q in QUESTIONS:
        results = engine.search(q, top_k=3)
        conf = results.speech_confidence
        kind = "said" if conf is not None and conf >= 0.5 else "shown"
        print(f"\n{q!r}")
        print(f"  speech confidence {conf}  → treated as about what was {kind}")
        print(f"  effective weights {results.weights}")
        for i, hit in enumerate(results, 1):
            print(f"  {i}. {hit.video_id}#{hit.scene_id} @ {hit.timestamp}  ({hit.explain()})")
