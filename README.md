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

**Or try it in the browser:** `pip install "cinematlas[demo,whisper]"` and `cinematlas demo`. Add a video
by URL or upload, ask a question, and the player jumps to the second that answers it.

![The Cinematlas demo: a question, ranked moments, and the player at 0:56](https://raw.githubusercontent.com/ranfysvalle02/cinematlas/main/docs/demo.png)

---

## The finding: fuse within a unit, chunk across units

A record carries several signals about the same thing: a scene's picture and speech, a photo and its
caption. The standard design gives each signal its own index and merges the ranked lists. That loses:
on any question about one signal, the lists disagree, and merging averages the right answer away.

**Fuse.** Embed a record's signals together, into one vector.

| Hit@1 | Video | Held-out video | NASA photos | Met artworks |
| --- | --- | --- | --- | --- |
| merged rankings, as usually built | 0.65 | 0.21 | 0.62 | 0.62 |
| best merge we found (incl. learned weights) | 0.72 | 0.50 | 0.78 | 0.81 |
| **one joint vector per record** | **0.83** | **0.62** | **0.93** | **0.95** |

It wins on data nobody tuned on (240 questions written blind by an AI agent, on space footage, space
photography and museum art), isn't reading burned-in subtitles, wins even when a record's parts describe
different things (0.70 vs 0.11), and holds on the kind of queries people really type: terse and from
memory (0.80 vs 0.70 against the strongest merge) or full of typos (0.84 vs 0.61).

**Chunk.** One vector per record breaks when a part is long. Bury each photo's description among 31
others and the joint vector falls below chunked late fusion; the long text even drowns out the photo.
Embed the photo together with *each chunk* instead:

| Hit@1, answer is 1/32 of the text | |
| --- | --- |
| one joint vector per record | 0.54 |
| chunked late fusion (ideal chunk boundaries) | 0.81 |
| **the photo fused into each chunk** (ideal boundaries) | **0.94** |
Ideal boundaries aren't needed. Strip the paragraph breaks so a chunker has to find the topics itself,
and `Semantic` (cut where adjacent sentences stop being similar) scores **0.90 against 0.93** for ideal
boundaries, statistically indistinguishable; fixed-size chunks score 0.82–0.85 and one vector per record
0.59 (8 descriptions per record).

Both come with paired significance tests and fourteen predictions written down before each run, four of
which failed. [TL;DR](https://github.com/ranfysvalle02/cinematlas/blob/main/TLDR.md) · [Paper](https://github.com/ranfysvalle02/cinematlas/blob/main/paper.md) · [every table](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md) · [the story](https://github.com/ranfysvalle02/cinematlas/blob/main/blog.md) ·
[review](https://github.com/ranfysvalle02/cinematlas/blob/main/REVIEW.md)

**In the library:** video `search()` ranks scenes with the joint vector and uses a reranker only to pick
the exact second (`routing="adaptive"` leans ahead on speech questions, at twice the latency).
`cinematlas.core` applies both halves to any records: `Text("title") + Image("photo")` to fuse,
`Text("body", chunk=…)` to chunk.

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

**Long text? Chunk it, fused.** `Text("body", chunk=800)` splits long text into its paragraphs (up to 800
characters each) and embeds each piece *together with the record's other parts*; search keeps each
record's best piece and returns it as the moment. For text without paragraphs (transcripts, OCR, scraped
pages) use `chunk=Semantic(800)`, which cuts where the topic changes:

```python
manuals = atlas.collection("manuals", embed=Text("title") + Text("body", chunk=800) + Image("cover"),
                           key="id", moment="body")
manuals.search("how do I reset the pressure valve?").top.text   # the passage that answers
```

**Built in.** Parts: `Text` (labels, nested or computed fields, truncation, `chunk=`) and `Image` (PIL,
bytes, path or URL, downscaled to Voyage's limits). Loaders, each with a suggested setup you can borrow with
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

`ingest()` accepts:

| Source | Fetched with |
| --- | --- |
| `s3://bucket/key`, `gs://bucket/object` | the cloud SDK and your ambient credentials (`cinematlas[s3]`, `cinematlas[gcs]`) |
| a direct video link, including presigned S3 / GCS / Azure URLs | a streaming download, every redirect SSRF-checked |
| a page such as YouTube | yt-dlp |
| a path, `bytes`, a file object, a FastAPI `UploadFile` / Flask `FileStorage` | streamed to disk |

Remote URLs are treated as untrusted: private addresses are refused, downloads are capped at
`max_download_mb` (cloud objects are checked before downloading), and signed-URL credentials are stripped
before storage. Re-ingesting a video replaces it without a gap.

---

## CLI

```bash
cinematlas doctor                        # checks the deployment and prints the exact fix for each problem
cinematlas setup [--update]              # create indexes; --update upgrades them in place
cinematlas ingest <url|path|->           # progress on stderr, JSON on stdout
cinematlas search "<question>" [-k 5] [--by hybrid|adaptive|transcript|visual|text] [--format table|json|context]
cinematlas demo [--port 8765]            # local web app: add videos, search, jump to the second
```

Global options: `--uri`, `--db`, `--collection`, `--transcript-mode`, `-v`.

## Atlas features used

Atlas Vector Search for the joint vectors, `$rerank` (8.3+) for in-database sentence reranking,
`$rankFusion` (8.0+) for adaptive routing's one-query fusion, Automated Embedding for transcripts,
Atlas Search for BM25, scalar quantization and BSON float32 vectors. Each has an equivalent fallback, and
`cinematlas doctor` tells you which path is in use.

## Development

```
cinematlas/
  engine.py        Cinematlas: the facade (configuration + public API)
  media.py         download, uploads, audio, scene cuts, keyframes, S3
  urlsafety.py     remote URLs are untrusted input (SSRF guard)
  transcribe.py    speech to timestamped sentences (OpenAI Whisper or faster-whisper)
  embed.py         keyframe, joint, transcript and query vectors
  ingest.py        the pipeline and its gapless replace
  search.py        single sources, fusion, reranking, routing
  capabilities.py  native-stage fallbacks and routing calibration
  doctor.py        what's wrong and how to fix it
  demo/            cinematlas demo: FastAPI server + a one-file web page
  core/            joint-vector search for any records: parts, chunkers, loaders, evaluate()
```

```bash
uv sync
uv run pytest -m "not integration and not media"    # unit, offline (~9 s)
uv run pytest -m media                               # real ffmpeg / Whisper on a committed NASA fixture
uv run pytest -m integration                         # live Atlas + Docker Atlas Local (reads .env)
uv run python bench/ingest.py                        # benchmark corpora, once: interviews,
uv run python bench/ingest.py --no-captions          #   caption ablation,
uv run python bench/ingest.py --station              #   held-out video,
uv run python bench/photos.py --ingest               #   photos (aligned and misaligned)
uv run python bench/met.py --ingest                  #   Met artworks
uv run python bench/boundary.py --ingest             #   boundary tests (see paper.md for the full list)
uv run python bench/run.py                           # every table and paired test → bench/RESULTS.md
```

MIT license. Test and benchmark media: NASA, public domain.
