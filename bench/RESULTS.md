# Retrieval benchmark

Corpus: 6 NASA *The Quiet Crew* interviews (65 scenes; one program, one topic, overlapping vocabulary, burned-in captions). Two labelled question sets:

* **Speech** (30, [queries.json](queries.json)): paraphrased questions answered by what someone *says*. Relevant = a scene of the right person containing the answer phrase.
* **Visual** (30, [queries_visual.json](queries_visual.json)): questions about what is *shown*, written from the keyframes. Relevant = one of the labelled scenes.

| Configuration | Speech Hit@1 | Visual Hit@1 | **Mean Hit@1** | Speech MRR | Visual MRR | Moment@1 | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| visual only (keyframes) | 0.53 | 0.90 | **0.72** | 0.603 | 0.925 | — | 302 ms |
| **scene only (joint image+speech)** · fast mode | 0.73 | 0.93 | **0.83** | 0.801 | 0.956 | — | 79 ms |
| full-text only (Atlas Search BM25) | 0.60 | 0.30 | **0.45** | 0.652 | 0.360 | — | 68 ms |
| transcript only · autoEmbed voyage-4 | 0.83 | 0.23 | **0.53** | 0.886 | 0.365 | — | 134 ms |
| transcript only · client voyage-4 | 0.83 | 0.20 | **0.52** | 0.886 | 0.347 | — | 269 ms |
| transcript only · client, voyage-4-lite queries | 0.83 | 0.23 | **0.53** | 0.889 | 0.366 | — | 261 ms |
| transcript + rerank | 0.90 | 0.43 | **0.67** | 0.937 | 0.522 | 0.78 | 436 ms |
| fixed fusion, equal weights, no rerank | 0.57 | 0.57 | **0.57** | 0.727 | 0.672 | 0.57 | 174 ms |
| fixed fusion, equal weights + rerank | 0.70 | 0.57 | **0.63** | 0.799 | 0.661 | 0.71 | 479 ms |
| fixed fusion, tuned weights + rerank | 0.80 | 0.50 | **0.65** | 0.853 | 0.600 | 0.75 | 471 ms |
| **adaptive routing (default)** · autoEmbed | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 475 ms |
| adaptive · client-side fusion | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 930 ms |
| adaptive · client transcript mode | 0.83 | 0.80 | **0.82** | 0.888 | 0.850 | 0.76 | 659 ms |

*Moment@1*: among top-1 speech hits, the returned moment contains the answer or starts within 3 s of it. Latency is the speech-set p50 from a laptop over the internet, including Voyage calls. Hybrid runs as one native `$rankFusion` query unless marked client-side; reranking runs as native `$rerank`. Vector indexes use scalar quantization; vectors are stored as BSON float32.

**Reading this table.** One joint image+speech vector per scene (mean 0.83) beats fusing the same signals after retrieval (best tuned fusion 0.65): it wins 14 questions the fusion misses and loses 3 (exact McNemar p = 0.013). Separate lists disagree on every question that is about only one of the two, and rank fusion averages the disagreement away; a joint vector never produces it. Adaptive routing repairs late fusion by choosing a specialist per question and reaches 0.82, statistically tied with scene-only (3 vs 4, p = 1.00). What routing adds is the exact second (Moment@1 0.76); scene-only returns the scene, at 79 ms.

**Limits.** 60 questions over 6 videos from one program, written by the authors; one question is 3.3 points, so only gaps confirmed by the paired test count. Weights and routing thresholds were tuned on this set. Latency is one laptop to one cloud region, comparative only.

## Caption ablation

Every frame in this corpus shows its dialogue as a burned-in caption, so image vectors can read the speech. Most real video has no burned-in captions. This run crops the caption band (bottom 15%) off each keyframe before embedding (collection `scenes_nocaptions`); transcripts, questions and labels are unchanged.

| Configuration | Speech Hit@1 | Visual Hit@1 | Mean Hit@1 |
| --- | --- | --- | --- |
| visual only (keyframes) | 0.53 → 0.40 | 0.90 → 0.97 | 0.72 → **0.68** |
| scene only (joint image+speech) | 0.73 → 0.77 | 0.93 → 0.93 | 0.83 → **0.85** |
| transcript + rerank | 0.90 → 0.90 | 0.43 → 0.43 | 0.67 → **0.67** |
| adaptive routing (default) | 0.83 → 0.87 | 0.80 → 0.83 | 0.82 → **0.85** |

Without captions, keyframes alone lose speech questions (0.53 → 0.40): pixels were reading the subtitles. The joint vector doesn't need them (0.73 → 0.77), because the speech is inside the embedding, not painted on the frame.

**Paired comparison, adaptive routing vs scene only** (all 60 questions). Only questions where exactly one system is right carry information; *p* is an exact two-sided McNemar test.

| Corpus | Routing right, scene-only wrong | Scene-only right, routing wrong | p |
| --- | --- | --- | --- |
| captions | 3 | 4 | 1.000 |
| no captions | 3 | 3 | 1.000 |

Reproduce: `uv run python bench/ingest.py && uv run python bench/ingest.py --no-captions && uv run python bench/run.py` · generated 2026-09-24
