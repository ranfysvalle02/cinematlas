# TL;DR

**If a record's signals describe the same thing, embed them together. If one of them is long, chunk it
and embed each chunk together with the others.** *Fuse within a unit, chunk across units.*

Most multimodal search gives each signal (speech, frames, captions, photos) its own index and merges the
ranked lists. On four corpora in three domains, one joint vector per record beat that design every time,
including against every smarter merge we could build.

---

## The key insights

| Hit@1 | Video | Held-out video | NASA photos | Met artworks |
| --- | --- | --- | --- | --- |
| merged rankings, as usually built | 0.65 | 0.21 | 0.62 | 0.62 |
| best merge we found (incl. learned weights) | 0.72 | 0.50 | 0.78 | 0.81 |
| **one joint vector per record** | **0.83** | **0.62** | **0.93** | **0.95** |

1. **Merging rankings loses because the lists disagree.** A question about what was *said* gets the right
   answer from the transcript index and noise from the image index; rank fusion averages them. It only
   works when every list agrees, which is exactly when you didn't need it.
2. **A joint vector never produces the disagreement.** The model sees all of a record's signals at once.
   It wins even when the parts describe different things (0.70 vs 0.11), and on messy real-world queries:
   terse and from memory (0.80 vs 0.70 against the strongest merge), or full of typos (0.84 vs 0.61),
   where merged rankings degrade faster.
3. **It breaks on long parts, and chunking fixes it.** With the answer 1/32 of a record's text, one vector
   per record falls to 0.54. Fuse the image into each chunk and it's back to 0.94; a semantic chunker finds
   the topic boundaries on its own (0.90 vs 0.93 ideal).
4. **The router you'd build to rescue merged rankings is optional.** It ties the joint vector at twice the
   latency, so video search now ranks with the joint vector by default.

## The lessons

- **Compare questions, not averages.** On 60–80 questions an average can't separate a real gap from luck.
  An exact paired test on the questions where exactly one system is right can.
- **Write predictions down before you run.** Fourteen here, four wrong, the newest ones committed to
  git before the run so the timestamp proves it. The wrong ones were the most
  informative: captions weren't the reason it worked, misalignment wasn't its boundary.
- **Hold data back.** The first corpus tuned everything; two of the three later ones were never looked at
  until the final run, with questions written blind.
- **Check the counts.** A chunk-loss bug surfaced because stored chunks didn't match what was added. Its
  published number was withdrawn, the data repaired, and the result re-measured.
- **Try to break it.** Every section of the paper is an attempt to make the finding fail; one did, and
  became the second half of the rule.

## Quickstart

```bash
pip install "cinematlas[whisper]"
export MONGODB_URI="mongodb+srv://…"  VOYAGE_API_KEY="pa-…"
cinematlas ingest "https://www.youtube.com/watch?v=5NhYvbMdbBU"
cinematlas search "how loud is a sonic boom?"     # → 0:56 "about as loud as a balloon popping"
cinematlas demo                                   # the same in a browser, with a player that jumps there
```

Any records, not just video:

```python
from cinematlas.core import Atlas, Image, Semantic, Text

docs = Atlas().collection("manuals",
    embed=Text("title") + Text("body", chunk=Semantic(800)) + Image("cover"),   # fuse, and chunk
    key="id", moment="body", late=True)                                         # late=True enables evaluate()
docs.setup(); docs.add(records); docs.wait_until_searchable()
docs.search("how do I reset the pressure valve?").top.text      # the passage that answers
print(docs.evaluate(my_labelled_questions, fusion="sum"))       # does it hold on *your* data?
```

## Why this matters

Late fusion is the default in most multimodal retrieval pipelines, because it's modular and easy to
explain. It's also where accuracy goes, and where teams end up bolting on routers, classifiers and
per-query weights to win it back. The rule here is simpler to build *and* more accurate, it says where it
stops working and what to do then, and `evaluate()` lets anyone check it on their own data in an
afternoon instead of taking it on trust.

---

[README](https://github.com/ranfysvalle02/cinematlas/blob/main/README.md) (use it) ·
[blog](https://github.com/ranfysvalle02/cinematlas/blob/main/blog.md) (the story) ·
[paper](https://github.com/ranfysvalle02/cinematlas/blob/main/paper.md) (methods and limits) ·
[RESULTS](https://github.com/ranfysvalle02/cinematlas/blob/main/bench/RESULTS.md) (every table) ·
[REVIEW](https://github.com/ranfysvalle02/cinematlas/blob/main/REVIEW.md) (an honest score)
