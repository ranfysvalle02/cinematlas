"""Basic: ask a question, get the second that answers it.

    uv run python examples/01_search.py "how loud is a sonic boom?"
"""

import sys

from _env import demo_engine

question = " ".join(sys.argv[1:]) or "how loud is a sonic boom?"

with demo_engine() as engine:
    results = engine.search(question, top_k=3)

print(results)                      # ranked moments with deep links
print()
print("best answer:", results.top.text)
print("jump to    :", results.top.link)
