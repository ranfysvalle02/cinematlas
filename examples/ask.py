"""Answer a question from video, with every citation a link to the exact second.

Cinematlas finds the moments; a local model writes the answer. No extra API key:
the model runs on your machine with Ollama (https://ollama.com).

    ollama pull qwen3:14b
    uv run python examples/ask.py "What first got these people interested in aviation?"

Each excerpt is a whole scene's transcript, so the model has enough to reason with, and every
citation still links to the exact second.

Env: OLLAMA_MODEL (default qwen3:14b; llama3.2 runs on small machines but answers less well),
OLLAMA_HOST (default http://localhost:11434).
"""

import json
import os
import sys
import urllib.error
import urllib.request

from _env import engine

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
question = " ".join(sys.argv[1:]) or "What first got these people interested in aviation?"

with engine() as eng:
    hits = eng.search(question).limit(4).run()
if not hits:
    raise SystemExit("No matching moments to answer from.")

prompt = (
    "Answer the question using only the numbered video excerpts below. Cite every claim with its "
    "excerpt number, like [2]. If the excerpts don't contain the answer, say so plainly.\n\n"
    f"{hits.to_context(full_scene=True)}\n\nQuestion: {question}\nAnswer:"
)
request = urllib.request.Request(
    f"{HOST}/api/generate",
    data=json.dumps({"model": MODEL, "prompt": prompt, "stream": True}).encode(),
    headers={"Content-Type": "application/json"},
)

print(f"Q: {question}\n")
try:
    with urllib.request.urlopen(request, timeout=300) as response:  # one JSON object per line
        for line in response:
            print(json.loads(line).get("response", ""), end="", flush=True)
except urllib.error.HTTPError as e:
    raise SystemExit(f"\nOllama error {e.code}: {e.read().decode()[:200]}  (try: ollama pull {MODEL})") from None
except urllib.error.URLError:
    raise SystemExit(f"Ollama isn't reachable at {HOST}. Get it at https://ollama.com.") from None

print("\n\nSources:")
for i, hit in enumerate(hits, 1):
    print(f"  [{i}] {hit.video_id} @ {hit.timestamp}  {hit.link}")
