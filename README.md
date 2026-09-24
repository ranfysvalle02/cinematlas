# Cinematlas

**Ask a question. Get the second in the video that answers it.**

```bash
pip install "cinematlas[whisper]"
cinematlas doctor
cinematlas ingest "www.b.com/keynote.mp4"
cinematlas search "when do they announce pricing?"
```

```text
 1. vid_3f2a…#12 @    7:11  Pricing starts at ten dollars a seat.
    https://www.b.com/keynote.mp4#t=431
```

---

## The finding: fuse in the embedding, not in the ranking

Video search has two kinds of question. *"How many medals has his beer won?"* is about what was **said**.
*"The one with the girl on hay bales"* is about what was **shown**.

The standard design indexes speech and pictures separately, retrieves from each, and merges the ranked
lists. That fails, because the lists disagree on every question that's about only one of the two, and
merging averages the disagreement away. Cinematlas embeds each scene's keyframe **and** its transcript
into **one** vector, so there's nothing to reconcile.

| Same signals, fused… | Said | Shown | Mean Hit@1 |
| --- | --- | --- | --- |
| after retrieval (rank fusion, tuned weights, reranked) | 0.80 | 0.50 | 0.65 |
| **inside one joint image+speech vector** | 0.73 | 0.93 | **0.83** |

On the questions where the two disagree, the joint vector wins 14 and loses 3 (exact McNemar p = 0.013).

**It isn't reading the subtitles.** Every frame in the benchmark has burned-in captions, so we cropped
them off and re-embedded. Keyframes alone dropped on speech questions (0.53 → 0.40), because the pixels
had been reading the subtitles. The joint vector didn't drop (0.73 → 0.77): the speech is in the
embedding, not painted on the frame.

**The router we built to fix rank fusion is now optional.** Routing each question to a specialist
recovers late fusion to 0.82, statistically tied with the joint vector alone (p = 1.0). The joint vector
finds the scene in one query (~80 ms). `search()` adds a sentence reranker on top to return the exact
second.

60 questions over 6 videos from one program, written by the authors. The paired test is how we tell
signal from noise. [Full results, caption ablation and limits](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md) ·
[the story: fuse in the embedding, not in the ranking](https://github.com/ranfysvalle02/cinematlas/blob/main/blog.md).

---

## Quickstart

```bash
export MONGODB_URI="mongodb+srv://…"     # MDB_URI also works
export VOYAGE_API_KEY="pa-…"
```

```python
from cinematlas import Cinematlas

engine = Cinematlas()
engine.ensure_indexes()                                  # once; idempotent

engine.ingest("https://www.youtube.com/watch?v=5NhYvbMdbBU")
engine.ingest("lecture.mov")                             # or a URL, bytes, file object, web upload

results = engine.search("how loud is a sonic boom?")     # the scene, down to the second
results.top.link                                         # 'https://…#t=34'
results.top.text                                         # 'Sonic booms are about 110 decibels.'
results.top.explain()                                    # rank per source, relevance, score

engine.search_scene_vector("a baby's hand holding a finger")   # joint vector only: the scene, ~80 ms
```

Results are plain dicts underneath (`json.dumps` works). Cinematlas doesn't pick an LLM for you;
`results.to_context()` gives you numbered, citable excerpts to pass to one.

---

## Examples

Runnable scripts in [`examples/`](https://github.com/ranfysvalle02/cinematlas/tree/main/examples). Put
`MONGODB_URI` and `VOYAGE_API_KEY` in `.env`; no other keys are needed. Examples 1–4 search the benchmark
corpus (six NASA interviews in `cinematlas_bench`); swap in your own collection name to search yours.

| | Example | Shows |
| --- | --- | --- |
| Basic | [`01_search.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/01_search.py) | A question in, the second that answers it out |
| Basic | [`02_fast_scene_search.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/02_fast_scene_search.py) | The joint vector alone (~65 ms) against the full `search()` |
| Advanced | [`03_why_it_ranked.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/03_why_it_ranked.py) | How a question is read as said or shown, and why each hit ranked |
| Advanced | [`04_answer_with_ollama.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/04_answer_with_ollama.py) | A cited answer from a local LLM ([Ollama](https://ollama.com)), each citation a deep link |
| Advanced | [`05_ingest_your_video.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/05_ingest_your_video.py) | Index any URL or file with live progress, then search it |
| Advanced | [`06_fastapi_app.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/06_fastapi_app.py) | A video upload + search API in about 20 lines |

```bash
uv run python examples/01_search.py "how loud is a sonic boom?"
```

---

## How it works

```
 ingest(video)
   ├─ scenes ── PySceneDetect cuts, ≤30 s each
   ├─ said ──── faster-whisper → timestamped sentences, aligned to scenes
   ├─ shown ─── the middle keyframe of each scene
   └─ Voyage ── one joint keyframe+transcript vector per scene (plus keyframe-only and transcript vectors)
                → one MongoDB Atlas document per scene

 search(question)
   ├─ $rankFusion over scene, keyframe, transcript and BM25 retrieval       one query
   ├─ $rerank over candidate sentences                                      picks the second
   └─ routing by reranker confidence                                        said vs shown
```

| Need | Call |
| --- | --- |
| The scene and the exact second (default) | `search(q)` |
| The scene, fastest | `search_scene_vector(q)` |
| Speech only | `search(q, sources=("transcript", "text"))` |
| Your own blend | `search(q, weights={"scene": 2, "transcript": 1, "rerank": 1})` |
| One source | `search_transcript` · `search_text` · `search_visual_vector` · `search_scene_vector` |

All of them accept `video_id=`. Every hit carries `moment` (`{start, end, text}`), `moment_link`
(YouTube `?t=431s`, files `#t=431`), `ranks`, `relevance` and the scene's fields.

`ingest()` accepts a URL (YouTube or any file link, scheme optional), a path, `bytes`, a file object, or a
FastAPI `UploadFile` / Flask `FileStorage`. Remote URLs are treated as untrusted: private addresses are
refused, downloads are capped, and signed-URL credentials are stripped before storage. Re-ingesting a
video replaces it without a gap.

---

## CLI

```bash
cinematlas doctor                        # checks the deployment and prints the exact fix for each problem
cinematlas setup [--update]              # create indexes; --update upgrades them in place
cinematlas ingest <url|path|->           # progress on stderr, JSON on stdout
cinematlas search "<question>" [-k 5] [--by hybrid|transcript|visual|text] [--format table|json|context]
```

Global options: `--uri`, `--db`, `--collection`, `--transcript-mode`, `-v`.

## Atlas features used

`$rankFusion` (8.0+) for one-query hybrid retrieval, `$rerank` (8.3+) for in-database sentence
reranking, Automated Embedding for transcripts, Atlas Search for BM25, scalar quantization and BSON
float32 vectors. Each has an equivalent fallback, and `cinematlas doctor` tells you which path is in use.

## Development

```bash
uv sync
uv run pytest -m "not integration and not media"    # unit, offline (~9 s)
uv run pytest -m media                               # real ffmpeg / Whisper on a committed NASA fixture
uv run pytest -m integration                         # live Atlas + Docker Atlas Local (reads .env)
uv run python bench/ingest.py && uv run python bench/ingest.py --no-captions && uv run python bench/run.py
```

MIT license. Test and benchmark media: NASA, public domain.
