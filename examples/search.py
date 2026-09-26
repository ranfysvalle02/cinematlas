"""Ask a question, get the second in the video that answers it.

    uv run python examples/search.py "where did the pilot grow up?"
    uv run python examples/search.py "a little girl standing on hay bales"
    uv run python examples/search.py --adaptive "what did he say about his first flight?"

Default: the joint image+speech vector ranks scenes in one query and the reranker picks the exact
second inside each. --adaptive: fuses every source, weighted by whether the question sounds like
it's about what was said or what was shown (it leans ahead on speech questions).
"""

import argparse

from _env import engine

p = argparse.ArgumentParser()
p.add_argument("question", nargs="+")
p.add_argument("--adaptive", action="store_true", help="fuse all sources, routed per question")
p.add_argument("-k", type=int, default=3)
args = p.parse_args()
question = " ".join(args.question)

with engine() as eng:
    results = eng.search(question).limit(args.k)
    results = (results.adaptive() if args.adaptive else results).run()

if not results:
    raise SystemExit("No matches. Is anything indexed? Try examples/index_and_search.py first.")


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


if results.speech_confidence is not None:
    kind = "said" if results.speech_confidence >= 0.5 else "shown"
    print(f"Read as a question about what was {kind} (speech confidence {results.speech_confidence:.2f}).\n")

for i, hit in enumerate(results, 1):
    span = f"{clock(hit.timestamp_start)}–{clock(hit.timestamp_end)}"
    print(f"{i}. {hit.video_id}, scene {span}")
    if hit.moment:
        print(f"   said at {hit.timestamp}: \"{hit.text[:100]}\"")
    print(f"   {hit.link}\n   why: {hit.explain()}\n")
