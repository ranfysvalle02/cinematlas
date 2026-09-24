# Retrieval benchmark

Corpus: 6 NASA *The Quiet Crew* interviews (65 scenes; one program, one topic, overlapping vocabulary, burned-in captions). Two labelled question sets:

* **Speech** (30, [queries.json](queries.json)): paraphrased questions answered by what someone *says*. Relevant = a scene of the right person containing the answer phrase.
* **Visual** (30, [queries_visual.json](queries_visual.json)): questions about what is *shown*, written from the keyframes. Relevant = one of the labelled scenes.

| Configuration | Speech Hit@1 | Visual Hit@1 | **Mean Hit@1** | Speech MRR | Visual MRR | Moment@1 | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| visual only (keyframes) | 0.53 | 0.90 | **0.72** | 0.603 | 0.925 | — | 385 ms |
| joint vector only (image+speech) | 0.73 | 0.93 | **0.83** | 0.801 | 0.956 | — | 141 ms |
| full-text only (Atlas Search BM25) | 0.60 | 0.30 | **0.45** | 0.652 | 0.360 | — | 120 ms |
| transcript only · autoEmbed voyage-4 | 0.83 | 0.23 | **0.53** | 0.886 | 0.365 | — | 190 ms |
| transcript only · client voyage-4 | 0.83 | 0.20 | **0.52** | 0.886 | 0.347 | — | 351 ms |
| transcript only · client, voyage-4-lite queries | 0.83 | 0.23 | **0.53** | 0.889 | 0.366 | — | 341 ms |
| transcript + rerank | 0.90 | 0.43 | **0.67** | 0.937 | 0.522 | 0.78 | 589 ms |
| fixed fusion, equal weights, no rerank | 0.57 | 0.60 | **0.58** | 0.727 | 0.688 | 0.57 | 231 ms |
| fixed fusion, equal weights + rerank | 0.70 | 0.57 | **0.63** | 0.799 | 0.661 | 0.71 | 611 ms |
| fixed fusion, tuned weights + rerank | 0.80 | 0.50 | **0.65** | 0.853 | 0.600 | 0.75 | 606 ms |
| adaptive routing · autoEmbed | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 606 ms |
| **scene-first (default)**: joint vector ranks, reranker picks the second | 0.73 | 0.93 | **0.83** | 0.801 | 0.956 | 0.73 | 338 ms |
| adaptive · client-side fusion | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 1246 ms |
| adaptive · client transcript mode | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 817 ms |

*Moment@1*: among top-1 speech hits, the returned moment contains the answer or starts within 3 s of it. Latency is the speech-set p50 from a laptop over the internet, including Voyage calls. Hybrid runs as one native `$rankFusion` query unless marked client-side; reranking runs as native `$rerank`. Vector indexes use scalar quantization; vectors are stored as BSON float32.

**Reading this table.** One joint image+speech vector per scene (mean 0.83) beats fusing the same signals after retrieval (best tuned fusion 0.65): it wins 14 questions the fusion misses and loses 3 (exact McNemar p = 0.013). Separate lists disagree on every question that is about only one of the two, and rank fusion averages the disagreement away; a joint vector never produces it. The default, scene-first, ranks with that vector and uses the reranker only to pick the second (Moment@1 0.73), in 338 ms. Adaptive routing, which repairs late fusion by choosing a specialist per question, ties it (4 vs 3, p = 1.000) at 606 ms: it leans ahead on questions about what was said, behind on what was shown.

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
| keyframes only | 0.12 | 0.88 | **0.50** | — | 136 ms |
| transcript + rerank | 0.55 | 0.07 | **0.31** | 0.86 | 463 ms |
| rank fusion, tuned weights + rerank | 0.33 | 0.10 | **0.21** | 0.85 | 611 ms |
| joint vector only | 0.35 | 0.90 | **0.62** | — | 92 ms |
| adaptive routing | 0.50 | 0.68 | **0.59** | 0.85 | 590 ms |
| scene-first | 0.35 | 0.90 | **0.62** | 0.71 | 267 ms |

| Paired comparison | A right, B wrong | B right, A wrong | p |
| --- | --- | --- | --- |
| joint vector only vs rank fusion, tuned weights + rerank | 40 | 7 | < 0.001 |
| scene-first vs adaptive routing | 14 | 11 | 0.690 |
| joint vector only vs adaptive routing | 14 | 11 | 0.690 |

**Decision rule, fixed before this corpus was run:** scene-first becomes the default if it is not significantly worse than adaptive routing on either corpus and it is faster. On this corpus it is not significantly worse (14 vs 11, p = 0.690) and faster (267 vs 590 ms).

**Where scene-first and routing differ.** They tie overall, but not per category: scene-first is better on questions about what was shown, routing leans ahead on what was said. Finding the exact second is not a difference: on the questions where both found the right scene, scene-first picked the right second 6/9 times and routing 6/9 here (16/22 and 17/22 on the first corpus). The Moment@1 columns above differ only because each method is scored on its own correct answers. If your users mostly ask about speech, pass `routing="adaptive"`.

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

Reproduce: `uv run python bench/ingest.py`, then `--no-captions` and `--station`, then `uv run python bench/run.py` · generated 2026-09-24
