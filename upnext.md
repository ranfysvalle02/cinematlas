# Up next

What to build next, checked against the code as of 0.16.0. A lot of the outside wishlist was already
built, so this file starts by saying what exists. That keeps us from building it twice.

## Already shipped (no work needed)

| Wishlist item | Where it lives |
|---|---|
| Keyframes picked by scene change, not a fixed rate | `media.detect_scene_spans` (PySceneDetect `ContentDetector`), `split_long_spans` |
| Scenes with start/end times, deep links to the second | `_utils.build_deep_link`, `results.SearchHit` |
| Speech lined up with scenes, pluggable Whisper | `transcribe.Transcriber` (faster-whisper / OpenAI), `_utils.assign_segments` |
| Joint image+speech vector | `Embedder.embed_keyframes_batched(transcripts=...)` |
| Batching and retries | `Embedder.multimodal_aligned`, `embed_transcripts` (hand-rolled backoff) |
| Search indexes created automatically, sized to the model | `indexes.ensure_search_indexes`, `definition_drift`, `wait_until_queryable` |
| Hybrid RRF search | native `$rankFusion` with a client-side fallback (`retrieval.rrf_fuse`), `$rerank` |
| Bulk upserts | `core/collection.py` (`bulk_write` + `ReplaceOne`); video uses `insert_many` + a gapless replace |
| Protocols that can be swapped out | `IngestHost`, `SearchHost` |
| Optional extras | `s3`, `gcs`, `openai`, `whisper`, `pdf`, `slides`, `ocr`, `demo`, `all` |
| Progress output | CLI `_progress_printer` |
| Checking your setup | `cinematlas doctor` |

## Phase 1: done in 0.14.0

- Voyage usage metering: `MeteredVoyage` wraps the client, `engine.usage` / `atlas.usage` keep running
  totals, `IngestResult.usage` shows what one ingest used, and `Usage.cost(prices)` estimates dollars
  from prices you pass in (no price table ships with the package, because prices change).
- 429-aware backoff with jitter (`usage.backoff`), shared by the video and `core` embed paths.
- `docs/quickstart.ipynb`.
- Per-stage timings were already there (`IngestResult.stages`).

## Phase 2: rethink the API (no users yet, so no migrations and no compatibility shims)

We have no users, so nothing has to stay backward compatible. Break whatever needs breaking and make
the story as simple as possible: **one video → scenes → the second that answers you.**

1. ~~**Search API.**~~ **Done.** `engine.search(q)` returns a lazy, immutable `Search`
   (`cinematlas/query.py`): `.video()`, `.where()`, `.limit()`, `.only()`, `.using()`, `.adaptive()`,
   `.routing()`, `.weights()`, `.rerank()`, `.candidates()`. The six `search_*` methods and
   `ingest_video`/`ingest_file` are gone. Metadata filters: `Cinematlas(filters=(...))` builds them into
   every index, `ingest(metadata=...)` stores them, and they pre-filter every source (vector
   `filter`, `$search` `equals`/`in`). Verified live, including native `$rankFusion` + `$rerank`.
   Paired check against 0.13: 120/120 identical ranked lists (30 bench questions × 4 configs).
2. ~~**Unify the two front doors.**~~ **Done.** Rather than rebuilding video search on `Collection`,
   which would lose the multi-source fusion the bench results depend on, the two share their seams:
   - **One connection.** `Atlas` owns the Mongo client, the metered Voyage client and `usage`.
     `atlas.videos("db.scenes", ...)` returns a video collection on that connection.
     `Cinematlas(...)` is the one-line shortcut and holds its own `Atlas` (`engine.atlas`).
   - **One retry path.** `usage.with_retries` handles video keyframes, transcripts and `core` records.
   - **One builder.** A `Query` base provides `.limit/.where/.rerank/.candidates` and lazy reading.
     `Search` adds the video sources, and `RecordSearch` adds `.merged(fusion)`, which replaces
     `search_merged`. `core` `where` still takes any MQL value; video `where` takes strings, because
     those fields also filter full-text search.
   Paired check: still 120/120 identical to 0.13.
3. ~~**Lean install.**~~ **Done.** Core is pymongo + voyageai + pillow (pillow ships with voyageai
   anyway). `cinematlas[video]` adds OpenCV, PySceneDetect and yt-dlp. Without it, ingest raises
   `DependencyError` naming the package and the extra. Checked in a clean venv: 92 MB, none of the
   video packages installed.
4. ~~**Typed models.**~~ **Done (0.16.0).** Results stay dicts, on purpose: hits carry arbitrary
   stored fields, and plain JSON is the point. A single `Hit`/`Hits` base (`cinematlas.results`) has
   attribute access and typed accessors (`rank`, `score`, `moment: Moment`, `text`, `explain()`).
   Video `SearchHit` adds `video_id`/`scene_id`/`timestamp`/`link`/`ranks`; `core.RecordHit` and
   `RecordHits` replace `core.Hit`/`Hits`. A record's own fields are never shadowed.
5. ~~**Async search.**~~ **Done (0.16.0).** Every query is awaitable (`await engine.search(q)`,
   `await q.arun()`). It runs the one ranking implementation in a worker thread instead of a second
   async copy, so the paired check stays valid. Queries use identity equality and hashing (so
   `asyncio.gather` works); concurrent reads of one query run it once. Live: 10 concurrent awaits gave
   rankings identical to sequential runs, in 2.2s vs 4.8s.
6. ~~**Optional `tqdm`.**~~ **Done (0.16.0).** `cinematlas[progress]` draws a bar over the five ingest
   stages on a terminal. Otherwise the CLI prints one line per stage.

## Phase 3

- ~~**OpenTelemetry spans.**~~ **Done (0.16.0).** `cinematlas[otel]` (API only; bring your own SDK and
  exporter). `cinematlas.ingest` has a child per stage, and `cinematlas.search` covers each run; both
  carry `cinematlas.voyage.*` usage. Without the extra, nothing changes.
- **Generic `BaseEmbedder` / `BaseVectorStore`: decided against, not deferred.** The library's claim is
  measured on Voyage + Atlas (joint vectors, `$rankFusion`, `$rerank`, autoEmbed), and a
  provider-neutral layer would sit between users and exactly those features. The seams that make it
  testable already exist (`IngestHost`, `SearchHost`, injectable clients). Revisit only if a real user
  needs another provider.

**Nothing is left open.** New work starts as a new entry here, with its expected effect written down
first.

## Rule for every item

Same method as the benchmark work: write down the expected effect before you build. Any change
that could touch retrieval quality gets a paired run on `bench/` before release.
