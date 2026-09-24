# Fuse in the embedding, not in the ranking

**Early vs late fusion for multimodal retrieval: four corpora, twelve pre-stated predictions, one boundary**

*Cinematlas project · September 2026 · all code, corpora, questions and results:
[github.com/ranfysvalle02/cinematlas](https://github.com/ranfysvalle02/cinematlas)*

---

## Abstract

Multimodal retrieval systems usually index each signal separately (transcript, keyframe, title, photo)
and merge the ranked lists, typically with Reciprocal Rank Fusion (late fusion). We compare this with
embedding each record's signals together into one vector with a multimodal model (early fusion), on
four corpora: talking-head video, held-out video from a different domain, NASA photos with text, and
museum artworks. Early fusion wins every comparison: Hit@1 0.83 vs 0.65, 0.62 vs 0.21, 0.93 vs 0.62 and
0.95 vs 0.62, each significant under an exact paired test. No smarter merge closes the gap: CombSUM,
CombMNZ, CombMAX and cross-validated weights all lose to the joint vector, which instead recovers 43–71%
of the distance between the best merge and an oracle that routes each question to its best signal. Two
predictions we expected to limit the finding did not hold. Removing burned-in
captions did not hurt the joint vector, and pairing each photo with unrelated text hurt late fusion more
than early fusion (−0.51 vs −0.23 Hit@1), because rank fusion depends on the separate lists agreeing.
We then found the boundary: when one signal is long, one vector per record loses to chunked late fusion
(0.54 vs 0.81 Hit@1 with the answer 1/32 of the text), and the long text also drowns out the image. Fusing
at chunk granularity, embedding the image together with each chunk, restores it (0.94). The rule that
survives: fuse within a unit, chunk across units. A previously reported gap in exact-moment accuracy turned
out to be a measurement artifact.

## 1. Question

A record in a multimodal collection carries several signals that describe the same thing: a video scene's
picture and speech, a photo and its caption, a slide and its speaker notes. Two ways to search them:

- **Late fusion.** Embed each signal on its own, retrieve from each index, merge the ranked lists.
  Modular and easy to explain; the default in most multimodal retrieval pipelines.
- **Early fusion.** Embed all of a record's signals together as one interleaved input to a multimodal
  model, producing one vector; retrieve once.

Which retrieves better, and when does the answer change?

## 2. Method

**Corpora.** All media is public domain (NASA).

| Corpus | Records | Signals | Questions | Written by |
| --- | --- | --- | --- | --- |
| Interviews: 6 videos, one program, burned-in captions | 65 scenes | keyframe, transcript | 30 speech + 30 visual | the authors |
| Station: 6 videos, different domain, no captions | 386 scenes | keyframe, transcript | 40 speech + 40 visual | an AI agent, blind |
| Photos: 16 topics | 394 photos | title, description, photo | 40 text + 40 visual | an AI agent, blind |
| Met artworks: 16 subjects (CC0) | 400 works | title, catalogue details, image | 40 text + 40 visual | an AI agent, blind |

"Blind" means the question writer saw only the media (keyframes, transcripts, photos, captions), never
the code, the systems or any results. Every label was checked mechanically (answer phrases occur in
exactly one scene; every referenced record exists) and left unedited.

**Systems.** Embeddings come from `voyage-multimodal-3.5` for images and joint inputs, and `voyage-4` for
transcript text; reranking uses `rerank-2.5`; storage and search use MongoDB Atlas Vector Search.

- *Joint vector (early fusion):* one vector per record from all its signals together.
- *Merged rankings (late fusion):* one vector per signal, each searched separately, merged with
  Reciprocal Rank Fusion (k = 60). On video, this includes full-text search and a sentence reranker with
  weights tuned on the first corpus.
- *Adaptive routing:* late fusion re-weighted per question by the reranker's confidence that the question
  is about speech. Built to rescue late fusion; tuned on the first corpus.
- *Scene-first:* the joint vector ranks; the reranker only picks each result's best sentence.

**Measurement.** Hit@1: the top result is a labelled record. Because small benchmarks make averages
unreliable (one question is 1.25–3.3 points here), comparisons use an exact two-sided McNemar test on the
questions where exactly one system is right. Moment accuracy: the returned sentence contains the answer
or starts within 3 s of it.

**Pre-stated predictions and decisions.** Each was written down before the run that tested it.

1. On the first corpus, the joint vector's speech accuracy comes from reading burned-in captions, and
   falls without them.
2. *Decision rule:* scene-first becomes the library default if it is not significantly worse than
   adaptive routing on either video corpus, and it is faster.
3. On aligned photos, the joint vector beats merged rankings.
4. On misaligned photos (each photo paired with another photo's title and description), the joint
   vector's advantage disappears.
5. Score-based merges (CombSUM, CombMNZ, CombMAX, learned weights) beat RRF, but none beats the joint
   vector significantly on any corpus.
6. On misaligned photos, CombMAX, which needs no agreement between lists, ties or beats the joint vector.
7. An oracle that picks each question's best single signal (knowing the answer) beats the joint vector
   on every corpus.
8. With a single unrelated part (one text, one image), the joint vector still wins overall but loses to
   CombSUM on visual questions.
9. As records grow long (the answer 1/8, then 1/32 of the text), chunked late fusion overtakes the joint
   vector significantly.
10. *Recorded after 8–9:* early fusion at chunk granularity beats chunked late fusion on long records and
    recovers visual accuracy to ≥ 0.80.
11. With paragraph breaks removed, a semantic chunker beats fixed-size chunking and comes within 0.05 of
    ideal boundaries.
12. On a non-NASA domain (Metropolitan Museum artworks), the joint vector beats merged rankings
    significantly, against both RRF and CombSUM.

## 3. Results

### 3.1 Early fusion beats late fusion on every corpus

| Corpus | Joint vector | Merged rankings | Joint only | Merged only | p |
| --- | --- | --- | --- | --- | --- |
| Interviews | **0.83** | 0.65 | 14 | 3 | 0.013 |
| Station (held out) | **0.62** | 0.21 | 40 | 7 | < 0.001 |
| Photos | **0.93** | 0.62 | 25 | 1 | < 0.001 |
| Met artworks (not NASA) | **0.95** | 0.62 | 28 | 2 | < 0.001 |
| Met artworks, vs CombSUM | **0.95** | 0.81 | 11 | 0 | < 0.001 |

The gap widened on data nobody tuned on: late fusion's weights, set on the interviews, fell to 0.21 on
the station videos, while the joint vector held at 0.62.

### 3.2 It isn't reading subtitles (prediction 1: did not hold)

Cropping the caption band off every keyframe of the interview corpus and re-embedding:

| Speech questions | With captions | Captions cropped |
| --- | --- | --- |
| Keyframe vector alone | 0.53 | 0.40 |
| Joint vector | 0.73 | 0.77 |

The keyframe-only vector *was* reading the captions. The joint vector wasn't: its speech signal comes
from the transcript inside the embedding.

### 3.3 The router was repairing late fusion (decision rule: scene-first adopted)

| Corpus | Scene-first vs adaptive routing | p | Latency |
| --- | --- | --- | --- |
| Interviews | 4 vs 3 | 1.0 | about half |
| Station | 14 vs 11 | 0.69 | about half |

Adaptive routing lifts late fusion to parity with the joint vector but no further. They differ by question
type: scene-first is better on questions about what is shown (9 vs 0 on the station corpus, p = 0.004),
while routing leans ahead on questions about what is said (in both corpora, neither gap significant on its
own). Scene-first became the default; routing is one argument away.

### 3.4 Misalignment hurts late fusion more (prediction 4: did not hold)

| Photos | Joint vector | Merged rankings |
| --- | --- | --- |
| Aligned | 0.93 | 0.62 |
| Misaligned | 0.70 | 0.11 |
| Change | −0.23 | **−0.51** |

We expected early fusion to suffer when a record's parts describe different things, and merged rankings
to hold up because each list stays clean. The opposite happened. Reciprocal Rank Fusion rewards records
that rank well in *every* list. When the parts are aligned, the right record ranks well in the title, the
description and the photo lists at once; when they aren't, its lists stop agreeing and it loses to records
that rank moderately everywhere. The joint vector did degrade: questions about the photo fell from 0.88 to
0.50, since two unrelated text parts now outweigh one image. But merging made it worse, not better.

### 3.5 The moment-accuracy gap was a measurement artifact

Moment@1 was first reported as 0.71 for scene-first vs 0.85 for routing (station corpus). Each figure was
computed over that system's own correct answers, which are different questions. Restricted to questions
where both found the right scene, scene-first picked the right sentence on 16 of 22 (interviews) and 6 of
9 (station); routing on 17 of 22 and 6 of 9. No difference.

### 3.6 No smarter merge wins (prediction 5: mostly held; 6: did not hold; 7: held in direction)

Every signal's own index was searched once per question, and the same lists were merged five ways.
Hit@1, with the joint vector's disputed-question record against each:

| Method | Interviews | Station | Photos | Misaligned photos |
| --- | --- | --- | --- | --- |
| **Joint vector** | **0.83** | **0.62** | **0.93** | **0.70** |
| RRF | 0.55 | 0.11 | 0.62 | 0.11 |
| CombSUM | 0.70 | 0.24 | 0.78 | 0.46 |
| CombMNZ | 0.65 | 0.15 | 0.76 | 0.35 |
| CombMAX | 0.72 | 0.50 | 0.56 | 0.38 |
| CombSUM, weights by 2-fold CV | 0.70 | 0.50 | 0.72 | 0.46 |
| *Oracle: best single signal per question* | *0.92* | *0.79* | *0.99* | *0.94* |

The joint vector leads all 20 real comparisons, significantly in 18 (the other two at p = 0.057 and
0.065, both on the smallest corpus). Score-based merges do beat RRF in 15 of 16 cases, so RRF is a weak
baseline, but the strongest real merge still trails by 11–24 points. CombMAX, which should thrive when
lists disagree, doesn't. Learned weights generalize no better than untuned CombSUM.

The oracle is not a method: it chooses each question's best signal after seeing the answer. It bounds
what any router over single signals could reach. Measured on that scale, starting from the best real
merge, the joint vector covers 58% (interviews), 43% (station), 71% (photos) and 50% (misaligned photos)
of the distance to the oracle. What remains is headroom for a router that could tell, per question, which
signal to trust, which is what adaptive routing attempts, and a direction for future work.

### 3.7 The boundary is record length (prediction 8: did not hold; 9: held)

With a single unrelated part the joint vector still wins, overall (0.70 vs 0.39 for CombSUM) and on visual
questions (0.47 vs 0.12): misalignment is not the boundary.

Length is. Each photo's own description was buried among 0, 7 or 31 descriptions of NASA photos from
outside the corpus, so answers stay unique while the relevant passage shrinks to 1/8 and 1/32 of the text.
Late fusion was given ideal chunking (one vector per description, a record scored by its best chunk).

| Hit@1 (visual / text) | 1 description | 8 descriptions | 32 descriptions |
| --- | --- | --- | --- |
| Joint vector, whole record | **0.93** (0.88 / 0.97) | 0.59 (0.60 / 0.57) | 0.54 (0.55 / 0.53) |
| Chunked late fusion (CombSUM) | 0.75 | **0.82** (0.75 / 0.90) | **0.81** (0.72 / 0.90) |
| Unchunked late fusion (CombSUM) | 0.76 | 0.55 | 0.40 |

Chunked late fusion overtakes the joint vector significantly at 8 (22 vs 3 disputed, p < 0.001) and 32
(26 vs 4, p < 0.001). Two effects compound. The relevant passage becomes a small share of what one vector
summarizes, and the long text drowns out the other signal: the joint vector's accuracy on *visual*
questions falls from 0.88 to 0.55 though the photo never changes. Unchunked late fusion degrades too,
faster, so chunking, not merging, is what rescues late fusion here.

### 3.8 Fusing per chunk restores it (prediction 10: held)

Early fusion at chunk granularity embeds the photo together with each chunk; a record scores its best
chunk.

| Hit@1 (visual / text) | 8 descriptions | 32 descriptions |
| --- | --- | --- |
| Chunked late fusion | 0.82 | 0.81 |
| **Chunk-level early fusion** | **0.93** (0.88 / 0.97) | **0.94** (0.90 / 0.97) |

It matches the unpadded joint vector at both lengths, beats chunked late fusion (11 vs 3, p = 0.057 at 8;
12 vs 2, p = 0.013 at 32) and restores visual accuracy to 0.88–0.90. The padding costs it nothing: the
right record's best chunk is its own description with its photo, the same input as the unpadded record.
### 3.9 Real chunkers find the boundaries (prediction 11: held)

Both chunked methods above used ideal boundaries. To test real chunkers without handing them the answer,
the paragraph breaks were removed (8 descriptions per record), so a chunker must find topic boundaries on
its own. `Semantic` embeds each sentence and cuts wherever the similarity between adjacent two-sentence
windows falls below the text's mean; its rule and defaults were chosen on documents built only from the
distractor pool, which no question targets.

| Chunker, no paragraph breaks | Hit@1 (visual / text) | vs ideal boundaries (0.93) |
| --- | --- | --- |
| Fixed-size, 1,600 characters | 0.82 (0.80 / 0.85) | 3 vs 11, p = 0.057 |
| Fixed-size, 500 characters | 0.85 (0.80 / 0.90) | 3 vs 9, p = 0.15 |
| **Semantic, 800 characters** | **0.90** (0.88 / 0.93) | 3 vs 5, p = 0.73 |

Every real chunker keeps chunk-level fusion far above one vector per record (0.59 at this length).
The semantic chunker is statistically indistinguishable from ideal boundaries; its lead over fixed-size
chunking (10 vs 4 and 6 vs 2) is not significant at 80 questions.

## 4. Discussion

**Why late fusion loses.** Every question about only one signal (most questions) produces lists that
disagree: the transcript index knows the answer, the keyframe index returns noise, and rank fusion averages
them. Weights can't fix this, because the right weights depend on the question and are fixed before it
arrives; routing fixes it per question and reaches parity, not better. A joint vector never produces the
disagreement in the first place.

**The boundary.** Misalignment is not it: early fusion degrades when a record's parts describe different
things, but merged rankings degrade more, whether two parts are unrelated (§3.4) or one (§3.7). Smarter
merges are not it either (§3.6). Length is (§3.7): once the relevant passage is a small share of a long
part, one vector per record loses to chunked late fusion, and fusing per chunk wins it back (§3.8). What
remains open: learned rerankers over the union of candidates, which are closer to routing than merging;
very long *non-text* parts (hours of video, large image sets); and whether the per-chunk gain holds with
chunk boundaries that cut through the relevant passage.

**Practical rule.** Fuse within a unit, chunk across units: embed a record's signals together, and when
one of them is long, split it and embed each piece together with the others. Keep separate vectors only
to *check* this on your own data: `Collection.evaluate()` in `cinematlas.core` runs exactly
the comparison in §3.1 on your labelled questions and reports the paired test.

## 5. Limitations

- Three of the four corpora are NASA media; the fourth is museum art. Lectures, meetings, sports, e-commerce and document collections may
  behave differently.
- 60–80 questions per corpus. The paired test separates real gaps from noise; it doesn't make small
  corpora representative.
- Questions for the second and third corpora were written by an AI agent. It was blind to the systems,
  but not a human searcher, and it may phrase queries in ways embedding models find easy.
- Late fusion was tested as five merges (RRF, CombSUM, CombMNZ, CombMAX, cross-validated weights) and an
  oracle. Learned-to-rank models over merged candidates were not tested.
- The interview corpus tuned the late-fusion weights and routing thresholds; its numbers favour those
  systems, which is why the other two corpora exist.
- The long records in §3.7–3.8 are constructed: real descriptions padded with other real descriptions.
  Real long documents have internal structure and chunk boundaries that can split the relevant passage.
- One embedding provider (Voyage AI). Early fusion needs a model that embeds interleaved text and images
  well; results will track that model's quality.

## 6. Reproduce

```bash
uv run python bench/ingest.py                                  # interviews
uv run python bench/ingest.py --no-captions                    # caption ablation
uv run python bench/ingest.py --station                        # station corpus
uv run python bench/photos.py --ingest                         # photos, aligned and misaligned
uv run python bench/met.py --ingest                            # Met artworks
uv run python bench/boundary.py --ingest                       # single-part and long-record tests
uv run python bench/boundary.py --ingest-chunk-fusion 8 32     # chunk-level fusion, ideal boundaries
uv run python bench/boundary.py --ingest-chunkers              # real chunkers, no paragraph breaks
uv run python bench/run.py                                     # every table here → bench/RESULTS.md
```

Full tables: [bench/RESULTS.md](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md).
The narrative version: [blog.md](https://github.com/ranfysvalle02/cinematlas/blob/main/blog.md).
