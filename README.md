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

A record usually carries several signals about the same thing: a scene's picture and speech, a photo and
its caption. The standard design gives each signal its own index and merges the ranked lists afterwards.
That loses: on any question about only one signal, the lists disagree, and merging averages the right
answer away. Cinematlas embeds a record's signals **together, into one vector**, so there's nothing to
reconcile.

| Hit@1 | Video | Held-out video | Photos |
| --- | --- | --- | --- |
| merged rankings (rank fusion; on video also tuned weights + reranker) | 0.65 | 0.21 | 0.62 |
| best of five merges, incl. learned weights | 0.72 | 0.50 | 0.78 |
| **one joint vector per record** | **0.83** | **0.62** | **0.93** |
| joint vs merged, questions only one got right | 14 vs 3, p = 0.013 | 40 vs 7, p < 0.001 | 25 vs 1, p < 0.001 |

- **On data nobody tuned on.** The held-out video (a different domain, no burned-in captions) and the
  photos (not video at all) have 160 questions written by an AI agent that never saw the code or results.
- **Not by reading subtitles.** Cropping burned-in captions hurt keyframes alone (0.53 → 0.40 on speech),
  not the joint vector (0.73 → 0.77).
- **Even when the parts disagree.** Pair each photo with another photo's text: the joint vector drops to
  0.70, merged rankings to 0.11. We predicted the reverse.
- **Close to the ceiling.** An oracle that sends each question to its best signal, knowing the answer,
  bounds what routing could reach. The joint vector covers 43–71% of the distance from the best merge to it.

**So `search()` ranks with the joint vector** and uses a reranker only to pick the exact second. Adaptive
routing, built to rescue merged rankings, ties it at twice the latency. It leans ahead on questions about
what was said and behind on what was shown; pass `routing="adaptive"` if your users mostly ask about speech.

[Paper: methods, seven predictions, limits](https://github.com/ranfysvalle02/cinematlas/blob/main/paper.md) · [every table](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md) ·
[the story](https://github.com/ranfysvalle02/cinematlas/blob/main/blog.md)

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
path or URL, downscaled to Voyage's limits). Loaders, each with a suggested setup you can borrow with
`atlas.collection("decks", like=Slides)`:

| Loader | One record per | Install |
| --- | --- | --- |
| `Slides("deck.pptx", pdf="deck.pdf")` | slide: title, body, **speaker notes**, and the rendered slide from its PDF export | `cinematlas[slides]` |
| `Screenshots("shots/")` | screenshot, with its on-screen text read by OCR line by line | `cinematlas[ocr]` |
| `PDFPages("paper.pdf")` | page: its image and text | `cinematlas[pdf]` |
| `ImageFolder("photos/")` | image, with a caption from a same-named `.txt` | |
| `JSONLines("rows.jsonl")` | line | |

```python
decks = atlas.collection("decks", like=Slides)
decks.add(Slides("q3-review.pptx", pdf="q3-review.pdf"))
decks.search("the slide where we showed Q3 churn").top.text    # the speaker-note sentence about churn

shots = atlas.collection("shots", like=Screenshots)
shots.add(Screenshots("qa-run-42/"))
shots.search("the screen with the red error banner").top.text  # 'Payment failed: card declined'
```

**Check it on your own data.** Every vector library says its approach wins; this one lets you check.
Create the collection with `late=True` (it also stores one vector per part), label 30–50 questions, and
`evaluate()` runs the joint vector against merged per-part rankings with the same paired test:

```python
photos = atlas.collection("photos", embed=Text("title") + Image("image"), key="id", late=True)
...
print(photos.evaluate([{"q": "astronaut fixing a telescope", "relevant": ["sts082-717-029"]}, ...]))
```
```text
                  Hit@1  Hit@10    MRR
joint vector       0.93    0.99   0.95
merged rankings    0.62    0.90   0.71

The joint vector wins on your data: joint 0.93 vs merged 0.62 Hit@1 on 80 questions (25 vs 1 disputed, p = < 0.001).
```

The default challenger is Reciprocal Rank Fusion (what Atlas `$rankFusion` does); `fusion="sum"` tests
against the strongest merge we found.

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
Atlas Search for BM25, scalar quantization and BSON float32 vectors. Each has an equivalent fallback, and
`cinematlas doctor` tells you which path is in use.

## Development

```bash
uv sync
uv run pytest -m "not integration and not media"    # unit, offline (~9 s)
uv run pytest -m media                               # real ffmpeg / Whisper on a committed NASA fixture
uv run pytest -m integration                         # live Atlas + Docker Atlas Local (reads .env)
uv run python bench/ingest.py                        # benchmark corpora, once: interviews,
uv run python bench/ingest.py --no-captions          #   caption ablation,
uv run python bench/ingest.py --station              #   held-out video,
uv run python bench/photos.py --ingest               #   photos (aligned and misaligned)
uv run python bench/run.py                           # every table and paired test → bench/RESULTS.md
```

MIT license. Test and benchmark media: NASA, public domain.
