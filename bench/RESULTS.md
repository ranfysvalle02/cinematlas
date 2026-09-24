# Retrieval benchmark

Corpus: 6 NASA *The Quiet Crew* interviews (65 scenes; one program, one topic, overlapping vocabulary, burned-in captions). Two labelled question sets:

* **Speech** (30, [queries.json](queries.json)): paraphrased questions answered by what someone *says*. Relevant = a scene of the right person containing the answer phrase.
* **Visual** (30, [queries_visual.json](queries_visual.json)): questions about what is *shown*, written from the keyframes. Relevant = one of the labelled scenes.

| Configuration | Speech Hit@1 | Visual Hit@1 | **Mean Hit@1** | Speech MRR | Visual MRR | Moment@1 | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| visual only (keyframes) | 0.53 | 0.90 | **0.72** | 0.603 | 0.925 | — | 1739 ms |
| joint vector only (image+speech) | 0.73 | 0.93 | **0.83** | 0.801 | 0.956 | — | 136 ms |
| full-text only (Atlas Search BM25) | 0.60 | 0.30 | **0.45** | 0.652 | 0.360 | — | 118 ms |
| transcript only · autoEmbed voyage-4 | 0.83 | 0.23 | **0.53** | 0.886 | 0.365 | — | 190 ms |
| transcript only · client voyage-4 | 0.83 | 0.20 | **0.52** | 0.886 | 0.347 | — | 348 ms |
| transcript only · client, voyage-4-lite queries | 0.83 | 0.23 | **0.53** | 0.889 | 0.366 | — | 335 ms |
| transcript + rerank | 0.90 | 0.43 | **0.67** | 0.937 | 0.522 | 0.78 | 579 ms |
| fixed fusion, equal weights, no rerank | 0.57 | 0.57 | **0.57** | 0.727 | 0.672 | 0.57 | 237 ms |
| fixed fusion, equal weights + rerank | 0.70 | 0.57 | **0.63** | 0.799 | 0.661 | 0.71 | 620 ms |
| fixed fusion, tuned weights + rerank | 0.80 | 0.50 | **0.65** | 0.853 | 0.600 | 0.75 | 611 ms |
| adaptive routing · autoEmbed | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 607 ms |
| **scene-first (default)**: joint vector ranks, reranker picks the second | 0.73 | 0.93 | **0.83** | 0.801 | 0.956 | 0.73 | 359 ms |
| adaptive · client-side fusion | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 1245 ms |
| adaptive · client transcript mode | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 2180 ms |

*Moment@1*: among top-1 speech hits, the returned moment contains the answer or starts within 3 s of it. Latency is the speech-set p50 from a laptop over the internet, including Voyage calls. Hybrid runs as one native `$rankFusion` query unless marked client-side; reranking runs as native `$rerank`. Vector indexes use scalar quantization; vectors are stored as BSON float32.

**Reading this table.** One joint image+speech vector per scene (mean 0.83) beats fusing the same signals after retrieval (best tuned fusion 0.65): it wins 14 questions the fusion misses and loses 3 (exact McNemar p = 0.013). Separate lists disagree on every question that is about only one of the two, and rank fusion averages the disagreement away; a joint vector never produces it. The default, scene-first, ranks with that vector and uses the reranker only to pick the second (Moment@1 0.73), in 359 ms. Adaptive routing, which repairs late fusion by choosing a specialist per question, ties it (4 vs 3, p = 1.000) at 607 ms: it leans ahead on questions about what was said, behind on what was shown.

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
| keyframes only | 0.12 | 0.88 | **0.50** | — | 103 ms |
| transcript + rerank | 0.55 | 0.07 | **0.31** | 0.86 | 468 ms |
| rank fusion, tuned weights + rerank | 0.33 | 0.10 | **0.21** | 0.85 | 622 ms |
| joint vector only | 0.35 | 0.90 | **0.62** | — | 142 ms |
| adaptive routing | 0.50 | 0.68 | **0.59** | 0.85 | 646 ms |
| scene-first | 0.35 | 0.90 | **0.62** | 0.71 | 325 ms |

| Paired comparison | A right, B wrong | B right, A wrong | p |
| --- | --- | --- | --- |
| joint vector only vs rank fusion, tuned weights + rerank | 40 | 7 | < 0.001 |
| scene-first vs adaptive routing | 14 | 11 | 0.690 |
| joint vector only vs adaptive routing | 14 | 11 | 0.690 |

**Decision rule, fixed before this corpus was run:** scene-first becomes the default if it is not significantly worse than adaptive routing on either corpus and it is faster. On this corpus it is not significantly worse (14 vs 11, p = 0.690) and faster (325 vs 646 ms).

**Where scene-first and routing differ.** They tie overall, but not per category: scene-first is better on questions about what was shown, routing leans ahead on what was said. Routing also lands on the exact second more often when it finds the right scene (Moment@1 0.85 vs 0.71 here, 0.76 vs 0.73 on the first corpus). If your users mostly ask about speech, pass `routing="adaptive"`.

| Corpus | Questions | Scene-first | Routing | Scene-first only | Routing only | p |
| --- | --- | --- | --- | --- | --- | --- |
| first | speech | 0.73 | 0.83 | 0 | 3 | 0.250 |
| first | visual | 0.93 | 0.80 | 4 | 0 | 0.125 |
| held-out | speech | 0.35 | 0.50 | 5 | 11 | 0.210 |
| held-out | visual | 0.90 | 0.68 | 9 | 0 | 0.004 |

Reproduce: `uv run python bench/ingest.py`, then `--no-captions` and `--station`, then `uv run python bench/run.py` · generated 2026-09-24
