# Fuse in the embedding, not in the ranking

*What building video search taught us about where fusion belongs.*

---

Every question you ask a video is one of two kinds.

*"How many medals has his beer won?"* is about what someone **said**. The answer is in the audio. The
picture on screen could be anything: a brewery, a face, a logo.

*"The one with the girl on hay bales"* is about what was **shown**. Nobody says "hay bales." The answer
exists only in pixels.

A search system that handles one kind well will usually fail the other. What we learned building
Cinematlas is that how you handle both depends on where you combine the two signals. Get that one
decision right and a lot of machinery you thought you needed goes away.

## The design everyone builds first

The obvious architecture is the one we built first, and the one most multimodal RAG tutorials still
teach:

1. Transcribe the speech. Embed the sentences. Index them.
2. Extract keyframes. Embed the images. Index them.
3. At query time, search both indexes, then merge the two ranked lists with Reciprocal Rank Fusion.
4. Rerank the merged candidates.

It's modular, each part is testable, each index is simple, and it feels like good engineering.

It scored **0.65**.

| Retrieval | Said | Shown | Mean Hit@1 |
| --- | --- | --- | --- |
| transcript + reranker | **0.90** | 0.43 | 0.67 |
| keyframes | 0.53 | **0.90** | 0.72 |
| both, merged with tuned rank fusion + reranker | 0.80 | 0.50 | 0.65 |

Read the last row carefully. We took the best speech retriever and the best visual retriever, combined
them, and got something **worse than either one on its own**, on average. It lost the visual questions
the keyframes were winning (0.90 → 0.50), and it didn't even keep the speech questions (0.90 → 0.80).

## Why merging ranked lists fails

Take a speech question. The transcript index puts the right scene first. The keyframe index has no idea;
to it, the question is noise, so it returns some confident-looking but arbitrary ordering of talking
heads. Rank fusion treats both lists as evidence and averages them. The right answer gets pulled down by
a list that never had a vote worth counting.

Now flip it for a visual question. Same thing in reverse.

**Every question that is about only one of the two modalities produces two lists that disagree.** That's
most questions, and rank fusion's whole mechanism is averaging, so it averages the disagreement away.
Tuning the weights doesn't help, because the right weights depend on the question, and the weights are
fixed before the question arrives.

## The fix we built, and why it wasn't the real fix

If the right weights depend on the question, decide per question. That's what we did next.

A sentence reranker scores how well any transcript sentence answers the query. When the best score is
high, the question is almost certainly about speech. When it's low, the answer probably isn't in the
audio. So we turned that score into a confidence and **routed**: confident speech questions go to the
transcript specialist, confident visual ones to the visual specialist, and a soft blend in between.

It worked: **0.82**. We calibrated thresholds per reranker, wrote documentation about "the router," and
made it the default.

Then we noticed a row we'd been treating as a sideshow.

## The row that was already winning

Alongside the separate indexes, Cinematlas stores one more vector per scene: the keyframe **and** the
transcript embedded together, as a single input, by a multimodal model (Voyage `voyage-multimodal-3.5`).
One scene, one vector, picture and speech in the same space.

We'd labeled it "fast mode." It scored **0.83**.

No router, no reranker, no calibration, no fusion weights. One vector search, in about **80 ms** against
**475 ms** for the routed pipeline.

A tie on a 60-question benchmark proves nothing by itself: one question is 3.3 points. So we stopped
comparing averages and compared **questions**. Only questions where exactly one system is right carry
information, and an exact McNemar test tells you whether the split is luck.

| Comparison | A right, B wrong | B right, A wrong | p |
| --- | --- | --- | --- |
| joint vector vs. tuned rank fusion | **14** | 3 | **0.013** |
| joint vector vs. equal-weight rank fusion | **17** | 1 | **0.0001** |
| joint vector vs. the router | 4 | 3 | 1.0 |

The joint vector beats merged lists decisively. It ties the router exactly. **The router was never
solving the modality problem. It was repairing the damage rank fusion did.** Fuse the signals before
retrieval, inside the embedding, and there's no disagreement to repair.

## Trying to prove ourselves wrong

There was an obvious objection, and we raised it ourselves.

Every frame in our benchmark corpus, six NASA interviews, has the dialogue **burned in as captions**.
A multimodal model looking at those frames can read the words. Maybe the joint vector wasn't
understanding picture and speech together at all. Maybe it was just reading subtitles, and it would
collapse on normal video.

We made a prediction we could lose: crop the caption band off every keyframe, re-embed the whole corpus,
and the joint vector's speech score should fall toward the keyframe-only score.

It didn't.

| Retrieval | Speech, with captions | Speech, captions cropped |
| --- | --- | --- |
| keyframes alone | 0.53 | **0.40** |
| joint image+speech vector | 0.73 | **0.77** |

The keyframes **did** lose: 0.53 → 0.40. The pixels really had been reading the subtitles. The joint
vector didn't lose anything. It went slightly *up*, and its visual score held at 0.93. The speech was never
coming from the painted-on words. It was coming from the transcript, inside the embedding.

The prediction failed, and the failure is the strongest evidence in this post. We gave the finding a
real chance to break, and it held.

## Testing it on video we never saw

Everything so far came from one benchmark: 60 questions we wrote ourselves, on six interviews about one
aircraft, with the router's thresholds tuned on those same questions. That's enough to say which gaps are
real on this set. It isn't enough to say the finding holds anywhere else.

So we built a second benchmark designed to break it:

- **A different domain:** six NASA videos about life on the space station, including a silent 15-minute
  walkthrough, an astronaut answering questions about space toilets, science demos and a food lab. 386
  scenes, six times the first corpus.
- **No burned-in captions.**
- **Questions we didn't write.** An agent saw only the keyframes and transcripts, never the code or any
  results, and wrote 40 speech and 40 visual questions.
- **Nothing tuned on it.** Weights and thresholds stayed exactly as the first corpus set them.

Before running it, we wrote down the decision it would settle: *if ranking by the joint vector alone is
not significantly worse than the router on either corpus, and it's faster, it becomes the default.*

| Mean Hit@1 | First corpus | Held-out corpus |
| --- | --- | --- |
| merged rankings, tuned + reranked | 0.65 | 0.21 |
| **one joint vector** | **0.83** | **0.62** |
| joint vector vs merged: disputed questions | 14 vs 3, p = 0.013 | **40 vs 7, p < 0.001** |

The finding didn't just survive; the gap got wider. On unfamiliar video, merged rankings fell apart
(0.21), and the joint vector won 40 of the 47 questions where the two disagreed.

And the rule settled the default. Ranking by the joint vector tied the router again (14 vs 11, p = 0.69)
at under half the latency.

## What each part is for now

- **The joint vector finds the scene.** That's where the two modalities are fused, and it's the default
  ranking.
- **The reranker finds the second.** A scene can run 30 seconds, and the promise is *the second*. The
  reranker scores only the sentences inside the scenes the vector found, and picks the one that answers.
- **The router is opt-in.** Here's the honest part: it ties overall, but not everywhere. Ranking by the
  joint vector is better on questions about what was *shown* (9 vs 0 on the held-out set, p = 0.004). The
  router leans ahead on what was *said*, in both corpora, though neither gap is significant on its own.
  If your users mostly ask about speech, `routing="adaptive"` is one argument away.

We could have tuned the default until the speech gap closed. But the held-out set only means something if
nobody tunes on it, so the default is what the rule we wrote down said it should be.

## The lesson that transfers

This isn't specific to video. Any time you retrieve over more than one signal (text and images, code and
docs, product titles and photos), you face the same fork:

- **Late fusion:** retrieve separately, merge rankings. Each retriever only ever sees half the
  evidence, and the merge step has to guess how much to trust each half, per query, without the query's
  help.
- **Early fusion:** embed the signals together, retrieve once. The model sees both halves at once and
  decides, for this item, what matters.

Late fusion is easier to build and easier to explain on a whiteboard. It's also where accuracy goes to
die, and where teams end up bolting on routers, classifiers and per-query weight tuning to win it back.

**If your signals describe the same thing, fuse them in the embedding, not in the ranking.**

## What we don't know yet

Two corpora, both NASA, 140 questions in total. The paired test tells us which gaps are real; two domains
are better than one, but lectures, sports, surveillance and meetings could still behave differently. The
router's per-category lean toward speech is consistent but not yet significant, and more speech-heavy
data would settle it. Adaptive routing also isn't perfectly repeatable: between two runs it changed its
answer on one held-out question. Everything here reproduces from the repo; both corpora, the caption ablation and
the limits are in
[bench/RESULTS.md](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md). If you run
it on your own video and it breaks, we want to know.

---

```bash
pip install "cinematlas[whisper]"
cinematlas ingest "https://www.youtube.com/watch?v=5NhYvbMdbBU"
cinematlas search "how loud is a sonic boom?"
```

[github.com/ranfysvalle02/cinematlas](https://github.com/ranfysvalle02/cinematlas) · MIT ·
benchmark media: NASA, public domain
