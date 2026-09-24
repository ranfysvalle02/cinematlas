# Fuse in the embedding, not in the ranking

**Early vs late fusion for multimodal retrieval: three corpora, four pre-stated predictions**

*Cinematlas project · September 2026 · all code, corpora, questions and results:
[github.com/ranfysvalle02/cinematlas](https://github.com/ranfysvalle02/cinematlas)*

---

## Abstract

Multimodal retrieval systems usually index each signal separately (transcript, keyframe, title, photo)
and merge the ranked lists, typically with Reciprocal Rank Fusion (late fusion). We compare this with
embedding each record's signals together into one vector with a multimodal model (early fusion), on
three corpora: talking-head video, held-out video from a different domain, and photos with text. Early
fusion wins every comparison: Hit@1 0.83 vs 0.65, 0.62 vs 0.21 and 0.93 vs 0.62, each significant under
an exact paired test. Two predictions we expected to limit the finding did not hold. Removing burned-in
captions did not hurt the joint vector, and pairing each photo with unrelated text hurt late fusion more
than early fusion (−0.51 vs −0.23 Hit@1), because rank fusion depends on the separate lists agreeing.
A previously reported gap in exact-moment accuracy between two systems turned out to be a measurement
artifact.

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

## 3. Results

### 3.1 Early fusion beats late fusion on every corpus

| Corpus | Joint vector | Merged rankings | Joint only | Merged only | p |
| --- | --- | --- | --- | --- | --- |
| Interviews | **0.83** | 0.65 | 14 | 3 | 0.013 |
| Station (held out) | **0.62** | 0.21 | 40 | 7 | < 0.001 |
| Photos | **0.93** | 0.62 | 25 | 1 | < 0.001 |

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

## 4. Discussion

**Why late fusion loses.** Every question about only one signal (most questions) produces lists that
disagree: the transcript index knows the answer, the keyframe index returns noise, and rank fusion averages
them. Weights can't fix this, because the right weights depend on the question and are fixed before it
arrives; routing fixes it per question and reaches parity, not better. A joint vector never produces the
disagreement in the first place.

**The boundary.** We set out to map where early fusion stops winning, and the obvious candidate (parts
that don't describe the same thing) did not produce it. Early fusion degrades with misalignment; late
fusion with RRF degrades more. Open questions: a smarter merge than RRF (score-based, or learned), a single
unrelated part rather than two, and signals too large for one multimodal input (long documents), where
early fusion must compress.

**Practical rule.** If a record's signals describe the same thing, embed them together. Keep separate
vectors only to *check* this on your own data: `Collection.evaluate()` in `cinematlas.core` runs exactly
the comparison in §3.1 on your labelled questions and reports the paired test.

## 5. Limitations

- All three corpora are NASA media. Lectures, meetings, sports, e-commerce and document collections may
  behave differently.
- 60–80 questions per corpus. The paired test separates real gaps from noise; it doesn't make small
  corpora representative.
- Questions for the second and third corpora were written by an AI agent. It was blind to the systems,
  but not a human searcher, and it may phrase queries in ways embedding models find easy.
- Late fusion here means Reciprocal Rank Fusion, what Atlas `$rankFusion` implements and the common
  default. Other merges were not tested.
- The interview corpus tuned the late-fusion weights and routing thresholds; its numbers favour those
  systems, which is why the other two corpora exist.
- One embedding provider (Voyage AI). Early fusion needs a model that embeds interleaved text and images
  well; results will track that model's quality.

## 6. Reproduce

```bash
uv run python bench/ingest.py                    # interviews
uv run python bench/ingest.py --no-captions      # caption ablation
uv run python bench/ingest.py --station          # station corpus
uv run python bench/photos.py --ingest           # photos, aligned and misaligned
uv run python bench/run.py                       # every table here → bench/RESULTS.md
```

Full tables: [bench/RESULTS.md](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md).
The narrative version: [blog.md](https://github.com/ranfysvalle02/cinematlas/blob/main/blog.md).
