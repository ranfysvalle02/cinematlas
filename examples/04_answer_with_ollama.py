"""Advanced: answer a question from video, with citations that jump to the second.

Cinematlas finds the moments; a local model writes the answer. No extra API key:
the answer comes from Ollama on your machine (https://ollama.com).

    ollama pull llama3.2
    uv run python examples/04_answer_with_ollama.py "why is supersonic flight banned over land?"
"""

import json
import os
import sys
import urllib.request

from _env import demo_engine

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
question = " ".join(sys.argv[1:]) or "why is supersonic flight banned over land, and what is NASA doing about it?"

with demo_engine() as engine:
    hits = engine.search(question, top_k=4)

prompt = (
    "Answer the question using only these video excerpts. Cite each claim with its number, like [2]. "
    "If the excerpts don't answer it, say so.\n\n"
    f"{hits.to_context()}\n\nQuestion: {question}"
)
body = json.dumps({"model": MODEL, "prompt": prompt, "stream": True}).encode()
request = urllib.request.Request(f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"})

print(f"Q: {question}\n")
with urllib.request.urlopen(request) as response:  # streams one JSON object per line
    for line in response:
        print(json.loads(line).get("response", ""), end="", flush=True)

print("\n\nSources:")
for i, hit in enumerate(hits, 1):
    print(f"  [{i}] {hit.link}")
