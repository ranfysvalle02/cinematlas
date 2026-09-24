# Fuse in the embedding, not in the ranking

*What building video search taught us about where fusion belongs, and where it breaks.*

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

1. Transcribe the speech. Embed the sentences. Index them, plus full-text search.
2. Extract keyframes. Embed the images. Index them.
3. At query time, search every index, then merge the ranked lists with Reciprocal Rank Fusion.
4. Rerank the merged candidates.

It's modular, each part is testable, each index is simple, and it feels like good engineering.

It scored **0.65**.

| Retrieval | Said | Shown | Mean Hit@1 |
| --- | --- | --- | --- |
| transcript + reranker | **0.90** | 0.43 | 0.67 |
| keyframes | 0.53 | **0.90** | 0.72 |
| every index, merged with tuned rank fusion + reranker | 0.80 | 0.50 | 0.65 |

Read the last row carefully. We took the best speech retriever and the best visual retriever, merged them
with everything else we had, and got something **worse than either one on its own**, on average. It lost
the visual questions the keyframes were winning (0.90 → 0.50) and didn't even keep the speech questions
(0.90 → 0.80).

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

It had been one of the lists in the merge all along, where the averaging drowned it out. On its own: no
router, no reranker, no calibration, no fusion weights. One vector search, at about a fifth of the
routed pipeline's latency.

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

All of that came from 60 questions we wrote ourselves, with the router tuned on them. So we built a second
benchmark designed to break it: six NASA videos from a different domain (a silent station walkthrough,
astronaut Q&A, science demos), 386 scenes, no captions, 80 questions written by an AI agent that saw only
the videos, and nothing tuned on it. We also wrote down, beforehand, the decision it would settle: if the
joint vector alone isn't significantly worse than the router, it becomes the default.

The gap got wider. Merged rankings fell to **0.21**; the joint vector held at **0.62**, winning 40 of the
47 questions where they disagreed. It tied the router again at under half the latency, so it became the
default.

## Beyond video

The argument never depended on video, so we tested it where it should apply just as well: 394 NASA
photos, each with a title and a description, and 80 questions written by an AI agent that saw only the
photos and their text. Half ask what a photo shows, half ask about facts in its text. We ran the
comparison with `evaluate()`, the same check anyone can run on their own collection.

The joint vector scored **0.93**; merged rankings **0.62**. It won 25 of the 26 questions where they
disagreed.

## Wrong again, twice

Then we tried to find where it stops. The obvious boundary: when a record's parts *don't* describe the
same thing, one vector should blur them and separate indexes should hold up. We paired every photo with
another photo's title and description and predicted, in writing, that the joint vector's advantage would
disappear.

It didn't. The joint vector fell to 0.70; merged rankings fell to **0.11**. Rank fusion rewards records
that rank well in *every* list, so it only works when the lists agree, and misaligned parts are exactly
when they don't.

Maybe rank fusion was just a weak merge. We tried every alternative we could think of: add the similarity
scores, reward records that many lists agree on, take each record's single best score (which needs no
agreement at all), and learn per-signal weights on half the questions to test on the other half. On all
four collections, every one of them lost to the joint vector, by 11 to 24 points for the best of them.

To see how much room is left, we added an oracle: for each question, whichever single signal ranks the
answer highest, chosen *knowing the answer*. No real system can do that. The joint vector gets 43–71% of
the way from the best merge to the oracle without knowing anything about the question. The rest is what a
perfect router could still add.

## Where it finally breaks

If misaligned parts and smarter merges couldn't beat one vector, what can? A vector has a fixed size.
Put enough text into it and the part that matters becomes a rounding error.

So we buried each photo's description among 7, then 31, descriptions of other NASA photos, and predicted
that chunked late fusion (one vector per description, a record scored by its best chunk) would finally
win. It did. With the answer 1/32 of the text, one vector per record fell from 0.93 to 0.54; chunked late
fusion held at 0.81. Worse, the long text drowned out the photo: the joint vector's accuracy on questions
about the *picture* fell from 0.88 to 0.55, though the picture never changed.

That's the boundary. And it suggests its own fix: don't choose between fusing and chunking. Chunk the
long part, and embed the photo *together with each chunk*. At 32 descriptions, that scored **0.94**, as
good as with no padding at all, and the picture questions came back to 0.90. The rule that survives:

**Fuse within a unit, chunk across units.**

Those 0.94 used ideal boundaries, one description per chunk. Real text doesn't come pre-cut, so we
removed the paragraph breaks and let chunkers find the topics themselves. A semantic chunker (cut where
adjacent sentences stop being similar) scored 0.90, statistically indistinguishable from ideal. In the
library it's one argument: `Text("body", chunk=800)` for text with paragraphs, `chunk=Semantic(800)` for
text without.

## What each part is for

- **The joint vector finds the record.** It's the default ranking in the video search and in
  `cinematlas.core`, per chunk when a part is long.
- **The reranker finds the second.** It scores only the sentences inside what the vector found. (We once
  reported the router finds the second more often, 0.85 vs 0.71; those figures came from different
  questions. Matched, they're equal.)
- **The router is opt-in.** It leans ahead on questions about what was *said*, behind on what was *shown*
  (9 vs 0 on the held-out set). If your users mostly ask about speech, it's one argument away.

## The lesson that transfers

Any time you retrieve over more than one signal (text and images, slides and speaker notes, screenshots
and their text), you face the same fork:

- **Late fusion:** retrieve separately, merge rankings. Each retriever sees half the evidence, and the
  merge has to guess how much to trust each half, per query, without the query's help.
- **Early fusion:** embed the signals together, retrieve once. The model sees both halves at once and
  decides, for this item, what matters.

Late fusion is easier to build and easier to explain on a whiteboard. It's also where accuracy goes to
die, and where teams end up bolting on routers, classifiers and per-query weights to win it back.

**If your signals describe the same thing, fuse them in the embedding, not in the ranking. If one of
them is long, chunk it, and fuse each chunk.**

## What we don't know yet

Three corpora, all NASA, 220 questions. The paired test tells us which gaps are real; it doesn't make
three corpora representative of lectures, meetings, e-commerce or documents. The router's lean toward
speech questions is consistent but not yet significant. Adaptive routing isn't perfectly repeatable:
between two runs it changed its answer on one held-out question. The long records we tested were built by
padding real descriptions, not real long documents with their own structure.

The write-up, with eleven predictions made in advance and how each came out, is
[paper.md](https://github.com/ranfysvalle02/cinematlas/blob/main/paper.md); every table is in
[bench/RESULTS.md](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md). If it breaks
on your data, `evaluate()` will tell you, and we want to know.

---

```bash
pip install "cinematlas[whisper]"
cinematlas ingest "https://www.youtube.com/watch?v=5NhYvbMdbBU"
cinematlas search "how loud is a sonic boom?"   # → 0:56 "about as loud as a balloon popping"
```

[github.com/ranfysvalle02/cinematlas](https://github.com/ranfysvalle02/cinematlas) · MIT ·
benchmark media: NASA, public domain
