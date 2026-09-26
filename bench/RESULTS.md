# Retrieval benchmark

Corpus: 6 NASA *The Quiet Crew* interviews (65 scenes; one program, one topic, overlapping vocabulary, burned-in captions). Two labelled question sets:

* **Speech** (30, [queries.json](queries.json)): paraphrased questions answered by what someone *says*. Relevant = a scene of the right person containing the answer phrase.
* **Visual** (30, [queries_visual.json](queries_visual.json)): questions about what is *shown*, written from the keyframes. Relevant = one of the labelled scenes.

| Configuration | Speech Hit@1 | Visual Hit@1 | **Mean Hit@1** | Speech MRR | Visual MRR | Moment@1 | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| visual only (keyframes) | 0.53 | 0.90 | **0.72** | 0.603 | 0.925 | — | 297 ms |
| joint vector only (image+speech) | 0.73 | 0.93 | **0.83** | 0.801 | 0.956 | — | 89 ms |
| full-text only (Atlas Search BM25) | 0.60 | 0.30 | **0.45** | 0.652 | 0.360 | — | 90 ms |
| transcript only · autoEmbed voyage-4 | 0.83 | 0.23 | **0.53** | 0.886 | 0.365 | — | 164 ms |
| transcript only · client voyage-4 | 0.83 | 0.20 | **0.52** | 0.886 | 0.347 | — | 265 ms |
| transcript only · client, voyage-4-lite queries | 0.83 | 0.23 | **0.53** | 0.889 | 0.366 | — | 261 ms |
| transcript + rerank | 0.90 | 0.43 | **0.67** | 0.937 | 0.522 | 0.78 | 443 ms |
| fixed fusion, equal weights, no rerank | 0.57 | 0.57 | **0.57** | 0.727 | 0.672 | 0.57 | 180 ms |
| fixed fusion, equal weights + rerank | 0.70 | 0.57 | **0.63** | 0.799 | 0.661 | 0.71 | 476 ms |
| fixed fusion, tuned weights + rerank | 0.80 | 0.50 | **0.65** | 0.853 | 0.600 | 0.75 | 485 ms |
| adaptive routing · autoEmbed | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 477 ms |
| **scene-first (default)**: joint vector ranks, reranker picks the second | 0.73 | 0.93 | **0.83** | 0.801 | 0.956 | 0.73 | 283 ms |
| adaptive · client-side fusion | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 907 ms |
| adaptive · client transcript mode | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 659 ms |

*Moment@1*: among top-1 speech hits, the returned moment contains the answer or starts within 3 s of it. Latency is the speech-set p50 from a laptop over the internet, including Voyage calls. Hybrid runs as one native `$rankFusion` query unless marked client-side; reranking runs as native `$rerank`. Vector indexes use scalar quantization; vectors are stored as BSON float32.

**Reading this table.** One joint image+speech vector per scene (mean 0.83) beats fusing the same signals after retrieval (best tuned fusion 0.65): it wins 14 questions the fusion misses and loses 3 (exact McNemar p = 0.013). Separate lists disagree on every question that is about only one of the two, and rank fusion averages the disagreement away; a joint vector never produces it. The default, scene-first, ranks with that vector and uses the reranker only to pick the second (Moment@1 0.73), in 283 ms. Adaptive routing, which repairs late fusion by choosing a specialist per question, ties it (4 vs 3, p = 1.000) at 477 ms: it leans ahead on questions about what was said, behind on what was shown.

**Limits.** 60 questions over 6 videos from one program, written by the authors; one question is 3.3 points, so only gaps confirmed by the paired test count. Weights and routing thresholds were tuned on this set, which is why the held-out corpus below exists. Latency is one laptop to one cloud region, comparative only.

## Caption ablation

Every frame in this corpus shows its dialogue as a burned-in caption, so image vectors can read the speech. Most real video has no burned-in captions. This run crops the caption band (bottom 15%) off each keyframe before embedding (collection `scenes_nocaptions`); transcripts, questions and labels are unchanged.

| Configuration | Speech Hit@1 | Visual Hit@1 | Mean Hit@1 |
| --- | --- | --- | --- |
| visual only (keyframes) | 0.53 → 0.40 | 0.90 → 0.97 | 0.72 → **0.68** |
| joint vector only (image+speech) | 0.73 → 0.77 | 0.93 → 0.93 | 0.83 → **0.85** |
| transcript + rerank | 0.90 → 0.90 | 0.43 → 0.43 | 0.67 → **0.67** |
| adaptive routing | 0.83 → 0.87 | 0.80 → 0.83 | 0.82 → **0.85** |

Without captions, keyframes alone lose speech questions (0.53 → 0.40): pixels were reading the subtitles. The joint vector doesn't need them (0.73 → 0.77), because the speech is inside the embedding, not painted on the frame.

**Paired comparison, adaptive routing vs joint vector only** (all 60 questions). Only questions where exactly one system is right carry information; *p* is an exact two-sided McNemar test.

| Corpus | Routing right, joint vector wrong | Joint vector right, routing wrong | p |
| --- | --- | --- | --- |
| captions | 3 | 4 | 1.000 |
| no captions | 3 | 3 | 1.000 |

## Held-out corpus

6 NASA videos from a different domain (food, nbl, pettit, potty, rubins, tour: a silent 15-minute station tour, astronaut Q&A, science demos, food science; 386 scenes, no burned-in captions). 40 speech and 40 visual questions ([speech](queries_station_speech.json), [visual](queries_station_visual.json)) were written by an agent that saw only these videos' keyframes and transcripts, never the code or any results. Routing thresholds and fusion weights were tuned on the first corpus only.

| Configuration | Speech Hit@1 | Visual Hit@1 | Mean Hit@1 | Moment@1 | p50 |
| --- | --- | --- | --- | --- | --- |
| keyframes only | 0.12 | 0.88 | **0.50** | — | 67 ms |
| transcript + rerank | 0.55 | 0.07 | **0.31** | 0.86 | 354 ms |
| rank fusion, tuned weights + rerank | 0.33 | 0.10 | **0.21** | 0.85 | 488 ms |
| joint vector only | 0.35 | 0.90 | **0.62** | — | 69 ms |
| adaptive routing | 0.50 | 0.68 | **0.59** | 0.85 | 484 ms |
| scene-first | 0.35 | 0.90 | **0.62** | 0.71 | 213 ms |

| Paired comparison | A right, B wrong | B right, A wrong | p |
| --- | --- | --- | --- |
| joint vector only vs rank fusion, tuned weights + rerank | 40 | 7 | < 0.001 |
| scene-first vs adaptive routing | 14 | 11 | 0.690 |
| joint vector only vs adaptive routing | 14 | 11 | 0.690 |

**Decision rule, fixed before this corpus was run:** scene-first becomes the default if it is not significantly worse than adaptive routing on either corpus and it is faster. On this corpus it is not significantly worse (14 vs 11, p = 0.690) and faster (213 vs 484 ms).

**Where scene-first and routing differ.** They tie overall, but not per category: scene-first is better on questions about what was shown, routing leans ahead on what was said. Finding the exact second is not a difference: on the questions where both found the right scene, scene-first picked the right second 6/9 times and routing 6/9 here (16/22 and 17/22 on the first corpus). The Moment@1 columns above differ only because each method is scored on its own correct answers. If your users mostly ask about speech, use `.adaptive()` (`engine.search(q).adaptive()`).

| Corpus | Questions | Scene-first | Routing | Scene-first only | Routing only | p |
| --- | --- | --- | --- | --- | --- | --- |
| first | speech | 0.73 | 0.83 | 0 | 3 | 0.250 |
| first | visual | 0.93 | 0.80 | 4 | 0 | 0.125 |
| held-out | speech | 0.35 | 0.50 | 5 | 11 | 0.210 |
| held-out | visual | 0.90 | 0.68 | 9 | 0 | 0.004 |

## Beyond video: photos

394 public-domain NASA photos across 16 topics ([corpus](photos_corpus.json)), each embedded as `Text(title) + Text(description) + Image(photo)`. The joint vector is compared with merged per-part rankings (Reciprocal Rank Fusion over one vector per part) using `Collection.evaluate()`, the same check users can run on their own data. 80 questions ([questions](queries_photos.json)), half about what a photo shows and half about facts in its text, were written by an AI agent that saw only the photos and their text, never the code or results.

The misaligned collection is the boundary test: identical photos and text, but each photo is paired with another photo's title and description, so the parts of a record no longer describe the same thing. Predictions, recorded before these runs: the joint vector wins on aligned records, and its advantage disappears on misaligned ones.

| Collection | Joint Hit@1 | Merged Hit@1 | Joint only | Merged only | p |
| --- | --- | --- | --- | --- | --- |
| aligned: each photo with its own title and description | 0.93 | 0.62 | 25 | 1 | < 0.001 |
| misaligned: each photo with another photo's title and description | 0.70 | 0.11 | 47 | 0 | < 0.001 |

| Collection | Questions | Joint | Merged | Joint only | Merged only | p |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | visual | 0.88 | 0.62 | 10 | 0 | 0.002 |
| aligned | text | 0.97 | 0.62 | 15 | 1 | < 0.001 |
| misaligned | visual | 0.50 | 0.07 | 17 | 0 | < 0.001 |
| misaligned | text | 0.90 | 0.15 | 30 | 0 | < 0.001 |

Going from aligned to misaligned, the joint vector's Hit@1 changes by -0.23 and merged rankings' by -0.51.

**Predictions vs outcome.** First prediction (joint wins on aligned records): held. Second prediction (its advantage disappears on misaligned records): did not hold. Misalignment hurt merged rankings more than the joint vector: Reciprocal Rank Fusion rewards records that rank well in every list, and once a record's parts describe different things its lists stop agreeing. The joint vector did degrade, mostly on questions about the photo, which two unrelated text parts now outweigh. Where merging rankings beats early fusion, if anywhere, is still open; a smarter merge than RRF is the next thing to test.

Reproduce: `uv run python bench/photos.py --ingest`, then `uv run python bench/run.py`.

## Outside NASA: the Met

400 public-domain artworks from the Metropolitan Museum's open-access collection (CC0), 16 subjects from armor to calligraphy ([corpus](met_corpus.json)), each embedded as `Text(title) + Text(details) + Image(photo)`, details being artist, date, medium, culture and department. 80 questions ([questions](queries_met.json)), half about what a work shows and half about its catalogue facts, were written by an AI agent that saw only the images and records. Run with `Collection.evaluate()` against two merges. Code: [met.py](met.py).

| Merged with | Joint Hit@1 | Merged Hit@1 | Visual (joint vs merged) | Text (joint vs merged) | Joint only vs merged only | p |
| --- | --- | --- | --- | --- | --- | --- |
| RRF | 0.95 | 0.62 | 0.95 vs 0.57 | 0.95 vs 0.68 | 28 vs 2 | < 0.001 |
| CombSUM (strongest merge) | 0.95 | 0.81 | 0.95 vs 0.80 | 0.95 vs 0.82 | 11 vs 0 | < 0.001 |

**Prediction vs outcome** (recorded before any Met data existed): the joint vector beats merged rankings significantly against both RRF and CombSUM: held.

## Are the questions too clean?

Benchmark questions written by an AI agent that sees the answer tend to be complete, well spelled and detail-rich, which could flatter a joint vector. Two stress tests on the Met artworks ([predictions](PREDICTIONS.md), committed before the runs): **terse searchers**, an agent simulating a visitor who glimpsed one work and later types 2–5 words from memory, one detail, sometimes misspelled ([queries](queries_met_searchers.json)); and **messy typing**, the 80 questions degraded by a fixed-seed script (filler dropped, at most five words, a typo in about one word in four; `met.noisy`).

| Queries | Joint | CombSUM | RRF | Joint only vs CombSUM only | p |
| --- | --- | --- | --- | --- | --- |
| full questions | **0.95** | 0.81 | 0.62 | 11 vs 0 | < 0.001 |
| messy typing (deterministic) | **0.84** | 0.61 | 0.46 | 20 vs 2 | < 0.001 |
| terse searchers (simulated) | **0.80** | 0.70 | 0.47 | 10 vs 2 | 0.039 |

Both predictions held. The joint vector keeps a significant lead over the strongest merge on terse, single-detail queries, and under messy typing the gap widens: merged rankings degrade faster than the joint vector. Real users remain untested.

## Can a smarter merge win?

Merged rankings above use Reciprocal Rank Fusion. Here every signal's own index is searched once per question (top 50, with scores) and merged five ways, all from the same retrieved lists: RRF; CombSUM (add min-max-normalized scores); CombMNZ (CombSUM times the number of lists that found the record); CombMAX (a record's best single-signal score, which needs no agreement between lists); and CombSUM with per-signal weights learned by 2-fold cross-validation. The oracle picks, for each question, whichever single signal ranks the answer highest, knowing the answer: not a real method, but an upper bound for any router that sends each question to one signal. Merges use only the separate signals (video: keyframe, transcript, full text; photos: title, description, photo), never the joint vector. Cells: Hit@1 (joint only vs method only, exact McNemar p). Code: [fusion.py](fusion.py).

| Method | video (interviews) | held-out video (station) | photos, aligned | photos, misaligned |
| --- | --- | --- | --- | --- |
| joint vector | **0.83** | **0.62** | **0.93** | **0.70** |
| rrf | 0.55 (19 vs 2, p < 0.001) | 0.11 (43 vs 2, p < 0.001) | 0.62 (25 vs 1, p < 0.001) | 0.11 (47 vs 0, p < 0.001) |
| sum | 0.70 (11 vs 3, p 0.057) | 0.24 (39 vs 8, p < 0.001) | 0.78 (13 vs 1, p 0.002) | 0.46 (22 vs 3, p < 0.001) |
| mnz | 0.65 (14 vs 3, p 0.013) | 0.15 (42 vs 4, p < 0.001) | 0.76 (14 vs 1, p < 0.001) | 0.35 (28 vs 0, p < 0.001) |
| max | 0.72 (9 vs 2, p 0.065) | 0.50 (12 vs 2, p 0.013) | 0.56 (31 vs 2, p < 0.001) | 0.38 (27 vs 1, p < 0.001) |
| learned (2-fold CV) | 0.70 (10 vs 2, p 0.039) | 0.50 (12 vs 2, p 0.013) | 0.72 (17 vs 1, p < 0.001) | 0.46 (23 vs 4, p < 0.001) |
| oracle single signal | 0.92 (2 vs 7, p 0.180) | 0.79 (2 vs 15, p 0.002) | 0.99 (0 vs 5, p 0.062) | 0.94 (1 vs 20, p < 0.001) |

**No real merge beats the joint vector on any corpus.** It is ahead in all 20 comparisons, significantly in 18. Of the gap between the best real merge and the oracle, the joint vector closes 58% (video (interviews)), 43% (held-out video (station)), 71% (photos, aligned), 50% (photos, misaligned).

**Predictions vs outcome** (recorded before this run). Score-based merges beat RRF but none beats the joint vector: mostly held; they beat RRF in 15 of 16 cases and none beats the joint vector. CombMAX ties or beats the joint vector on misaligned photos: did not hold (0.38 vs 0.70). The oracle beats the joint vector everywhere: it's ahead on every corpus, significantly on held-out video (station), photos, misaligned. That remaining gap is what perfect per-question routing could still add on top of early fusion.

## Where early fusion loses

Two controlled tests on the photo corpus and its 80 questions ([boundary.py](boundary.py)). Cells: Hit@1 (visual / text questions), and against the joint vector: questions only it got right vs only the method got right, exact McNemar p.

**B1: a single unrelated part.** Each record is `Text(description) + Image(photo)`, one part each, with the photo's own description (aligned) or another photo's (misaligned). This removes the two-texts-vs-one-photo imbalance of the earlier misaligned test.

| Method | single part, aligned | single part, misaligned |
| --- | --- | --- |
| joint vector | **0.93 (0.88 / 0.97)** | **0.70 (0.47 / 0.93)** |
| merged (rrf) | 0.62 (0.68 / 0.57), 26 vs 2, p < 0.001 | 0.11 (0.10 / 0.12), 51 vs 4, p < 0.001 |
| merged (sum) | 0.78 (0.78 / 0.78), 14 vs 2, p 0.004 | 0.39 (0.12 / 0.65), 30 vs 5, p < 0.001 |

**B2: long records.** Each record is `Text(body) + Image(photo)`, the body being the photo's own description buried among 0, 7 or 31 descriptions of NASA photos from outside the corpus ([distractors](distractors.json)), so every answer stays unique. Chunked late fusion gets the ideal chunking: one vector per description, a record scored by its best chunk, merged with the photo's vector. Unchunked late fusion merges one whole-body vector with the photo's.

| Method | long text, 1 description | long text, 8 descriptions | long text, 32 descriptions |
| --- | --- | --- | --- |
| joint vector | **0.93 (0.88 / 0.97)** | **0.59 (0.60 / 0.57)** | **0.54 (0.55 / 0.53)** |
| chunked late (rrf) | 0.64 (0.72 / 0.55), 25 vs 2, p < 0.001 | 0.62 (0.75 / 0.50), 14 vs 17, p 0.720 | 0.61 (0.68 / 0.55), 14 vs 20, p 0.392 |
| chunked late (sum) | 0.75 (0.78 / 0.72), 16 vs 2, p 0.001 | 0.82 (0.75 / 0.90), 3 vs 22, p < 0.001 | 0.81 (0.72 / 0.90), 4 vs 26, p < 0.001 |
| unchunked late (sum) | 0.76 (0.78 / 0.75), 15 vs 2, p 0.002 | 0.55 (0.53 / 0.57), 12 vs 9, p 0.664 | 0.40 (0.42 / 0.38), 16 vs 5, p 0.027 |
| chunk-level joint | — | 0.93 (0.88 / 0.97), 1 vs 28, p < 0.001 | 0.94 (0.90 / 0.97), 0 vs 32, p < 0.001 |

**Predictions vs outcome** (recorded before these runs). The joint vector still wins with one unrelated part but loses to CombSUM on visual questions: did not hold; it wins overall and on visual questions (0.47 vs 0.12). On long records, chunked late fusion beats the joint vector significantly: held, on long text, 8 descriptions, long text, 32 descriptions. This is the boundary: one vector per record stops working when a part is long enough that the relevant passage is a small share of it, and the long text also drowns out the photo (the joint vector's visual accuracy falls with length though the photo never changes). Chunk-level fusion (the photo embedded together with each chunk, a record scoring its best chunk), predicted after seeing B2 to beat chunked late fusion and recover visual accuracy to 0.80 or more: long text, 8 descriptions: 0.93 vs chunked late 0.82 (11 vs 3, p 0.057), visual 0.88; long text, 32 descriptions: 0.94 vs chunked late 0.81 (12 vs 2, p 0.013), visual 0.90.

**Real chunkers.** Chunk-level fusion through `cinematlas.core`, with the chunkers users get. "Unmarked" removes the paragraph breaks, so a chunker has to find topic boundaries itself (with them present, `Paragraphs` gets the ideal boundaries by construction). The semantic chunker's rule and defaults were chosen on documents built only from the distractor pool, which no question targets. Ideal: chunk-level fusion with one description per chunk, at the same length.

| Chunker | Descriptions per record | Chunks stored | Hit@1 (visual / text) | Ideal | vs ideal |
| --- | --- | --- | --- | --- | --- |
| unmarked, fixed 1600 chars | 8 | 1507 | 0.82 (0.80 / 0.85) | 0.93 | 3 vs 11, p 0.057 |
| unmarked, fixed 500 chars | 8 | 5032 | 0.85 (0.80 / 0.90) | 0.93 | 3 vs 9, p 0.146 |
| unmarked, Semantic(800) | 8 | 5734 | 0.90 (0.88 / 0.93) | 0.93 | 3 vs 5, p 0.727 |

**Predictions vs outcome** (recorded before these runs). With paragraph breaks removed, the semantic chunker beats fixed-size chunking and gets within 0.05 of the ideal (>= 0.88): held (0.90; fixed-size 0.82, 0.85). Directly against fixed-size chunking its lead is not significant at this size (10 vs 4, p 0.180; 6 vs 2, p 0.289); every real chunker keeps chunk-level fusion far above one vector per record.

Reproduce: `uv run python bench/ingest.py`, then `--no-captions` and `--station`, then `uv run python bench/run.py` · generated 2026-09-24
