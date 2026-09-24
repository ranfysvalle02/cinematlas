# Cinematlas

**Ask a question. Get the second in the video that answers it.**

```bash
pip install "cinematlas[whisper]"
cinematlas doctor
cinematlas ingest "https://www.youtube.com/watch?v=5NhYvbMdbBU"
cinematlas search "how loud is a sonic boom?"
```

```text
 1. 5NhYvbMdbBU#11 @    0:56  Sonic booms can be about as loud as a balloon popping.
    https://www.youtube.com/watch?v=5NhYvbMdbBU&t=56s
 2. 5NhYvbMdbBU#10 @    0:51  These sonic booms are really loud.
    https://www.youtube.com/watch?v=5NhYvbMdbBU&t=51s
```

---

## The finding: fuse in the embedding, not in the ranking

Video search has two kinds of question. *"How many medals has his beer won?"* is about what was **said**.
*"The one with the girl on hay bales"* is about what was **shown**.

The standard design indexes speech and pictures separately, retrieves from each, and merges the ranked
lists. That fails, because the lists disagree on every question that's about only one of the two, and
merging averages the disagreement away. Cinematlas embeds each scene's keyframe **and** its transcript
into **one** vector, so there's nothing to reconcile.

| Mean Hit@1 | First corpus | Held-out corpus |
| --- | --- | --- |
| merged rankings (rank fusion, tuned weights, reranked) | 0.65 | 0.21 |
| **one joint image+speech vector** | **0.83** | **0.62** |
| questions where exactly one wins (joint vs merged) | 14 vs 3, p = 0.013 | 40 vs 7, p < 0.001 |

**It holds on video we never tuned on.** The held-out corpus is a different domain (a silent station
tour, astronaut Q&A, science demos; 386 scenes, no burned-in captions), with 80 questions written by an
AI agent that saw only the videos, never the code or results.

**It isn't reading subtitles.** On the first corpus, where every frame has burned-in captions, cropping
them made keyframes alone worse on speech (0.53 → 0.40) but left the joint vector intact (0.73 → 0.77).

**So the default ranks with that one vector.** `search()` finds scenes with the joint vector and uses a
sentence reranker only to pick the exact second. The router we built to rescue merged rankings ties it on
both corpora (p = 1.0 and p = 0.69) at about twice the latency, so it's now opt-in. The two do differ: the
default is better on questions about what was shown; routing leans ahead on what was said and lands on
the exact second more often. If your users mostly ask about speech, pass `routing="adaptive"`.

[Full results, both corpora, caption ablation and limits](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md) ·
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

engine.ingest("https://www.youtube.com/watch?v=5NhYvbMdbBU")   # NASA: 60 Second Science, Sonic Booms
engine.ingest("lecture.mov")                             # or a URL, bytes, file object, web upload

results = engine.search("how loud is a sonic boom?")     # the scene, down to the second
results.top.link                                         # 'https://www.youtube.com/watch?v=5NhYvbMdbBU&t=56s'
results.top.text                                         # 'Sonic booms can be about as loud as a balloon popping.'
results.top.explain()                                    # rank per source, relevance, score

engine.search_scene_vector("an airplane in the sky")    # the joint vector alone: scenes only, ~100 ms
```

Results are plain dicts underneath (`json.dumps` works). Cinematlas doesn't pick an LLM for you;
`results.to_context()` gives you numbered, citable excerpts to pass to one.

---

## Beyond video: `cinematlas.core`

The finding isn't about video. Any records whose signals describe the same thing (product photos and
titles, slides and their text, diagrams and captions) search better with one joint vector than with
separate indexes merged afterwards. `cinematlas.core` is that idea as a small library:

```python
from cinematlas.core import Atlas, Text, Image

photos = Atlas().collection("nasa.photos",
    embed=Text("title") + Image("image"),      # parts compose into ONE joint vector
    moment="description",                      # the reranker picks the best sentence
    filters=["center"], key="nasa_id")         # filterable fields; re-adding a key replaces it
photos.setup()                                 # the vector index; idempotent, updates in place
photos.add(records)                            # any iterable of dicts, or a loader
photos.wait_until_searchable()

photos.search("astronaut fixing a telescope in space").top.title   # 'Making Room for Hubble's New Camera'
photos.search(Image("mars.jpg"), where={"center": "JPL"})           # query by picture, filtered
```

That output is real: [`examples/photos.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/photos.py) indexes about 200 NASA photos and runs it.

**Built in.** Parts: `Text` (labels, nested or computed fields, truncation) and `Image` (PIL, bytes,
path or URL, downscaled to Voyage's limits). Loaders: `PDFPages` (each page's image and text; `pip install
'cinematlas[pdf]'`), `ImageFolder` (images with same-named `.txt` captions) and `JSONLines`.
`atlas.collection("decks", like=PDFPages)` borrows a loader's suggested setup.

**Extend it.** A part is anything that turns a record into text or images for the vector; a loader is
anything that yields records:

```python
from cinematlas.core import Part, Loader, Collection

class Price(Part):                                   # numbers become words the model understands
    def inputs(self, record):
        return [f"costs ${record[self.field]:.0f}"] if record.get(self.field) else []

class Tickets(Loader):                               # records from anywhere
    key, moment = "id", "body"
    embed = Text("subject") + Text("body")
    def __iter__(self):
        yield from my_helpdesk_api.tickets()

@Collection.extend                                   # add methods to every collection, jQuery-style
def newest(self, k=5):
    return list(self.mongo.find({}, {"embedding": 0}).sort("_id", -1).limit(k))
```

Ship a plugin as a package with a `cinematlas.plugins` entry point, and `cinematlas.core.plugins()`
lists it next to the built-ins.

---

## Examples

Five runnable scripts in [`examples/`](https://github.com/ranfysvalle02/cinematlas/tree/main/examples).
They need only `MONGODB_URI` and `VOYAGE_API_KEY` in `.env`. The first two search a demo corpus of six
NASA interviews, the next two index your own video, and the last one searches photos.

```bash
uv run python examples/search.py "a little girl standing on hay bales"
uv run python examples/search.py --adaptive "what did he say about his first flight?"
uv run python examples/ask.py "What first got these people interested in aviation?"
uv run python examples/index_and_search.py lecture.mp4 "when is the exam?"
uv run python examples/photos.py "a rover's tracks on red sand" --center JPL
```

| Example | What it does |
| --- | --- |
| [`search.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/search.py) | The scene and the second for any question, with why each hit ranked. `--adaptive` shows routing reading a question as said or shown |
| [`ask.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/ask.py) | A cited answer from a local LLM ([Ollama](https://ollama.com), no API key), each citation a link to the exact second |
| [`index_and_search.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/index_and_search.py) | Index any URL, YouTube link or file with live progress, then search it |
| [`api.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/api.py) | A FastAPI service: `POST /videos` to upload, `GET /search` for deep links |
| [`photos.py`](https://github.com/ranfysvalle02/cinematlas/blob/main/examples/photos.py) | `cinematlas.core` on about 200 NASA photos: search by text or by picture, filter by center |

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
   ├─ joint-vector search over scenes                                        one query, ranks scenes
   └─ $rerank over those scenes' sentences                                   picks the second

 search(question, routing="adaptive")
   ├─ $rankFusion over scene, keyframe, transcript and BM25 retrieval        one query
   └─ weights set per question by the reranker's confidence                  said vs shown
```

| Need | Call |
| --- | --- |
| The scene and the exact second (default) | `search(q)` |
| Mostly questions about speech | `search(q, routing="adaptive")` |
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
cinematlas search "<question>" [-k 5] [--by hybrid|adaptive|transcript|visual|text] [--format table|json|context]
```

Global options: `--uri`, `--db`, `--collection`, `--transcript-mode`, `-v`.

## Atlas features used

Atlas Vector Search for the joint vectors, `$rerank` (8.3+) for in-database sentence reranking,
`$rankFusion` (8.0+) for adaptive routing's one-query fusion, Automated Embedding for transcripts,
Atlas Search for BM25, scalar quantization and BSON float32 vectors. Each has an equivalent fallback, and `cinematlas doctor` tells you which path is in use.

## Development

```bash
uv sync
uv run pytest -m "not integration and not media"    # unit, offline (~9 s)
uv run pytest -m media                               # real ffmpeg / Whisper on a committed NASA fixture
uv run pytest -m integration                         # live Atlas + Docker Atlas Local (reads .env)
uv run python bench/ingest.py                        # the benchmark corpora, once:
uv run python bench/ingest.py --no-captions          #   caption ablation
uv run python bench/ingest.py --station              #   held-out corpus
uv run python bench/run.py                           # both corpora, caption ablation, paired tests
```

MIT license. Test and benchmark media: NASA, public domain.
