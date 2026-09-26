# What's new

Every release, newest first. Times are US Eastern (UTC−4). For releases published to PyPI, the time
is the upload time; releases marked *GitHub only* were pushed but not uploaded. Upgrading from 0.13?
See [the migration table](#upgrading-from-013).

---

## 0.16.2 · 2026-09-26 10:45 · *this release*

**The quickstart works on a fresh cluster, and every document matches the code.**

- `cinematlas ingest` creates the search indexes first (idempotent), so the three-command quickstart
  (`ingest`, then `search`) works on a new collection. Verified live: `0:56 "Sonic booms can be about as
  loud as a balloon popping."`
- Docs: `REVIEW.md` rescored for v0.16. The README architecture map covers the new modules, and the
  benchmark's generated advice uses `.adaptive()`. This file is new.
- Everything from 0.15.0–0.16.1 reaches PyPI for the first time in this release.

## 0.16.1 · 2026-09-26 07:58 · *GitHub only*

**Polish pass, from a full review of everything since 0.13.**

- `atlas.videos()` and `Cinematlas(atlas=...)` use the atlas's embedding model and reranker. Before,
  they silently used the defaults. Passing a second set of connection options now raises.
- `Cinematlas(mongo_client=...)` no longer closes a client you injected.
- A typo like `q.limt` or a probe like `hasattr(q, "shape")` fails at once instead of running a search.
- `copy`/`deepcopy` of a query gives the same query on the same engine, with no cached results.
- `.weights()` rejects unknown sources, `with_retries` needs at least one attempt, and duplicate filter
  values collapse.
- Index drift reports metadata filters that are no longer declared.
- CLI: `--filter FIELD`, `ingest --meta KEY=VALUE`, `search --where KEY=VALUE` (use `a,b` for any of them).
- Benchmark: the captions ablation called removed methods; all arms are smoke-tested live.

## 0.16.0 · 2026-09-26 04:16 · *GitHub only*

**One result type, awaitable queries, a progress bar, OpenTelemetry.**

- Results: one `Hit` / `Hits` base. Hits are still plain dicts (every stored field, `json.dumps` works)
  with typed accessors: `rank`, `score`, `moment`, `text`, `explain()`. Video hits (`SearchHit`) add
  `video_id`, `scene_id`, `timestamp`, `link`, `ranks`. In `core`, `RecordHit` / `RecordHits` replace
  `Hit` / `Hits`.
- Async: `await engine.search(q).limit(5)`. It runs the same ranking code in a worker thread; 10
  concurrent awaits returned the same rankings as sequential calls, in 2.2 s instead of 4.8 s.
- `cinematlas[progress]`: a tqdm bar over the five ingest stages.
- `cinematlas[otel]`: an ingest span with a child per stage and a span per search, each with its Voyage
  usage. Without the extra, it does nothing.

## 0.15.0 · 2026-09-26 04:00 · *GitHub only* · **breaking**

**One search API, one connection, a search-only install.**

- `engine.search(q)` returns a lazy, immutable query: `.video()`, `.where()`, `.limit()`, `.only()`,
  `.using()`, `.adaptive()`, `.routing()`, `.weights()`, `.rerank()`, `.candidates()`. The six
  `search_*` methods and `ingest_video` / `ingest_file` are gone.
- Metadata filters: `Cinematlas(filters=("course",))` builds them into every index, and
  `ingest(..., metadata={...})` stores them. Every source filters before ranking, including native
  `$rankFusion` and `$rerank`.
- `cinematlas.core`: `collection.search(q)` uses the same builder, and `.merged("sum")` replaces
  `search_merged`.
- One connection: `Atlas` owns the Mongo client, the Voyage client and the usage meter, and
  `atlas.videos("db.talks")` is a video collection on it.
- Voyage usage metering: `result.usage` per ingest, `engine.usage` running, and `usage.cost(prices)`
  estimates dollars from prices you pass. 429s back off longer, with jitter, on one retry path.
- `pip install cinematlas` is search-only (pymongo, voyageai, pillow), and `cinematlas[video]` adds
  OpenCV, PySceneDetect and yt-dlp.
- `docs/quickstart.ipynb`.
- Verified: a paired run against 0.13 on 30 benchmark questions × 4 configurations, with 120/120
  identical rankings.
- (0.14.0 was planned as a smaller release and folded into this one; it was never published.)

## 0.13.0 · 2026-09-24 16:29

**Reliable sources without yt-dlp.** `s3://` and `gs://` via the cloud SDKs, with sizes checked before
downloading. Direct and presigned video links stream to disk with every redirect SSRF-checked, and
yt-dlp is only used for pages such as YouTube.

## 0.12.0 · 2026-09-24 16:14

**`cinematlas demo`.** A local web app: add a video by URL or upload, ask a question, and the player
jumps to the second. FastAPI and a single HTML file, with no build step.

## 0.11.0 · 2026-09-24 15:02

**The result holds on realistic queries.** Terse simulated searchers: joint 0.80 vs 0.70 against the
strongest merge. Messy typing: 0.84 vs 0.61. Both predictions were committed before the runs.

## 0.10.1 · 2026-09-24 14:04

The PyPI page links `TLDR.md`.

## 0.10.0 · 2026-09-24 13:52

**The finding holds outside NASA.** 400 Met Museum artworks with blind questions: joint 0.95 vs merged
0.62 (and 0.81 against CombSUM, the strongest merge), p < 0.001.

## 0.9.0 · 2026-09-24 12:55

**Real chunkers close the boundary.** `Semantic(800)` finds topic boundaries on its own: 0.90 vs 0.93
for ideal boundaries. `engine.py` was split into focused modules behind the facade, `REVIEW.md` was
added, and a bug that lost chunks across write batches was fixed.

## 0.8.0 · 2026-09-24 10:05

**Where early fusion loses, and the fix.** With long parts, one vector per record falls to 0.54.
Embedding the photo with each chunk (`Text(field, chunk=N)`) restores 0.94.

## 0.7.0 · 2026-09-24 05:23

**`evaluate()` on your own data.** `late=True` also stores one vector per part, and `evaluate()`
runs joint vs merged with an exact paired test. NASA photos: joint 0.93 vs merged 0.62. New `Slides`
and `Screenshots` loaders.

## 0.6.0 · 2026-09-24 04:35

**`cinematlas.core`: joint-vector search for any records.** Parts compose with `+` (`Text`, `Image`);
`PDFPages`, `ImageFolder` and `JSONLines` loaders; plugins.

## 0.5.1 · 2026-09-24 04:11

Polish: the README and blog quote real outputs, and the moment-accuracy tradeoff is stated.

## 0.5.0 · 2026-09-24 03:49

**Scene-first search by default.** A held-out corpus confirms the finding (joint 0.62 vs rank fusion
0.21). `search()` ranks with the joint vector and uses the reranker only to pick the second.

## 0.4.5 · 2026-09-24 02:39

Examples on the PyPI page.

## 0.4.4 · 2026-09-24 02:16

The story (`blog.md`) linked from PyPI.

## 0.4.3 · 2026-09-24 01:58

CI fix for Python 3.10 (`onnxruntime<1.24`), and refreshed PyPI metadata.

## 0.1.0 – 0.4.2 · 2026-09-23 23:00 – 2026-09-24 01:12

Early releases, published before the repository's history begins (its first commit is 0.4.2 at 01:48).
That first commit already carried the founding result: a joint keyframe+transcript vector per scene
beats rank fusion of the same signals, 0.83 vs 0.65 Hit@1 (p = 0.013).

---

## Upgrading from 0.13

| 0.13 | 0.15+ |
| --- | --- |
| `pip install cinematlas` (video included) | `pip install "cinematlas[video]"` to ingest; plain `cinematlas` searches |
| `engine.search(q, top_k=3, video_id="v")` | `engine.search(q).video("v").limit(3)` |
| `engine.search(q, routing="adaptive")` | `engine.search(q).adaptive()` |
| `engine.search(q, weights={...})` | `engine.search(q).weights(scene=2, rerank=1)` |
| `engine.search(q, sources=("transcript", "text"))` | `engine.search(q).using("transcript", "text")` |
| `engine.search_scene_vector(q)` · `search_visual_vector` · `search_transcript` · `search_text` | `engine.search(q).only("scene" \| "visual" \| "transcript" \| "text")` |
| `engine.ingest_video(url)` / `ingest_file(f)` → scene count | `engine.ingest(url_or_file).scenes` |
| `collection.search(q, k=5, where={...}, moment=False)` | `collection.search(q).where(...).limit(5).rerank(False)` |
| `collection.search_merged(q, fusion="sum")` | `collection.search(q).merged("sum")` |
| `cinematlas.core.Hit` / `Hits` | `cinematlas.core.RecordHit` / `RecordHits` (base: `cinematlas.Hit` / `Hits`) |
| `results == []` | `results.run() == []` (queries compare by identity) |
| `Cinematlas(mongo_client=c)` closed `c` on exit | it leaves `c` open; you close it |
