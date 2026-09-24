"""Ask a question, get the second in the video that answers it.

    uv run python examples/search.py "how loud is a sonic boom?"
    uv run python examples/search.py "a little girl standing on hay bales"
    uv run python examples/search.py --fast "a pilot flying over a bridge"

Default: the scene and the exact second, and whether the question was read as being about
what was said or what was shown. --fast: one joint image+speech vector query, scenes only.
"""

import argparse

from _env import engine

from cinematlas import build_deep_link

p = argparse.ArgumentParser()
p.add_argument("question", nargs="+")
p.add_argument("--fast", action="store_true", help="joint-vector scene search only (no reranker)")
p.add_argument("-k", type=int, default=3)
args = p.parse_args()
question = " ".join(args.question)

with engine() as eng:
    results = eng.search_scene_vector(question, top_k=args.k) if args.fast else eng.search(question, top_k=args.k)

if not results:
    raise SystemExit("No matches. Is anything indexed? Try examples/index_and_search.py first.")


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


# Questions about what was *said* get the exact sentence; questions about what was *shown* get the scene.
about_speech = results.speech_confidence is not None and results.speech_confidence >= 0.5
if results.speech_confidence is None:
    print("Joint image+speech vector: one query, scenes only.\n")
else:
    kind = "said" if about_speech else "shown"
    print(f"Read as a question about what was {kind} (speech confidence {results.speech_confidence:.2f}).\n")

for i, hit in enumerate(results, 1):
    if about_speech:
        print(f"{i}. {hit.video_id} @ {hit.timestamp}  \"{hit.text[:100]}\"")
    else:
        span = f"{clock(hit.timestamp_start)}–{clock(hit.timestamp_end)}"
        said = (hit.get("transcript") or "").strip()
        print(f"{i}. {hit.video_id}, scene {span}" + (f"  (said here: \"{said[:70]}…\")" if said else ""))
    link = hit.link if about_speech else build_deep_link(hit.video_url, hit.timestamp_start)
    print(f"   {link}\n   why: {hit.explain()}\n")
