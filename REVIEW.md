# Cinematlas: project review

*A candid assessment of the project as of v0.11: the research, the software, and what's left.*

---

## TL;DR

**9 / 10.** A small library with an unusually well-tested finding: embed a record's signals together
(early fusion) instead of indexing them separately and merging rankings (late fusion), and chunk long
parts, fusing each chunk. It holds across four corpora in three domains (video, space photography,
museum art), three of which nobody tuned on, and beats every merging method we could find. The boundary where it stops working was found and fixed rather than hidden.
What's left is real users: the questions were written by us or an AI agent, and while the result
survives terse, from-memory and misspelled queries, simulated searchers aren't real ones.

## Executive summary

Cinematlas started as video search on MongoDB Atlas and Voyage AI ("ask a question, get the second in the
video that answers it") and turned into a study of where multimodal fusion belongs.

**The finding.** Merging separate rankings loses because the lists disagree on any question that's about
one signal, and merging averages the right answer away. One joint vector per record wins on talking-head
video (0.83 vs 0.65 Hit@1), held-out video (0.62 vs 0.21), NASA photos (0.93 vs 0.62) and Met artworks
(0.95 vs 0.62), each significant under an exact paired test, and against the strongest of five merges,
including learned weights.

**The boundary.** One vector per record breaks when a part is long: with the answer 1/32 of a record's
text, it falls to 0.54. Fusing the image into each chunk restores 0.94. With the paragraph breaks
removed, a semantic chunker finds the boundaries on its own (0.90, statistically indistinguishable from
ideal). The rule that survives: **fuse within a unit, chunk across units.**

**The product.** `pip install cinematlas`: video search whose default ranks with the joint vector, plus
`cinematlas.core`, which applies both halves of the rule to any records (`Text + Image`, `chunk=`,
loaders for slides, screenshots and PDFs) and ships `evaluate()`, so users can check the finding on
their own data instead of taking it on trust.

---

## Scores

| Dimension | Score | Why |
| --- | --- | --- |
| **Research rigor** | **9.5** | Paired significance tests, not averages. Held-out data with questions written blind. Fourteen predictions recorded before the runs that tested them, four of which failed and are reported; the newest are committed to git before their runs. A tuning set kept separate from the test set. An oracle ceiling. |
| **Insight** | **8.5** | Early vs late fusion is a known axis in multimodal learning; the contribution is a practical, falsifiable rule for retrieval, a measured boundary (length, not misalignment), and a working fix (chunk-level fusion). The explanation of *why* rank fusion fails (it needs lists to agree) is clean and testable. |
| **Software engineering** | **8.5** | 310 unit tests, media tests on real ffmpeg/Whisper, live integration tests on Atlas cloud and Atlas Local. The engine was split into focused modules with no API change. Graceful fallbacks everywhere. Deductions: integration tests don't run in CI, and the facade is 540 lines (mostly docstrings for the public API). |
| **Developer experience** | **9** | Four commands from install to a deep link. `doctor` prints the exact fix. Every README output is real. Five examples, each run live. `evaluate()` turns the claim into something users can verify. |
| **Documentation & storytelling** | **9** | Three documents with distinct jobs: README (use it), blog (the story, including where we were wrong), paper (methods, predictions, limits). Every number traces to a generated `RESULTS.md`. |
| **Reproducibility** | **8.5** | Every table regenerates from `bench/`; corpora, questions and distractors are committed. It still needs an Atlas cluster, a Voyage key and hours of embedding. |
| **External validity** | **8** | Four corpora in three domains: video, space photography and museum art (0.95 vs 0.62). The result survives terse from-memory queries (0.80 vs 0.70 vs the strongest merge) and messy typing (0.84 vs 0.61). Still no real users, and the long-record tests pad real descriptions rather than using real long documents. |
| **Production readiness** | **7.5** | Safe URL handling, idempotent setup, gapless re-ingest, per-record failure isolation. Deductions: one embedding provider, a young `core` API, and chunk-level fusion re-embeds the image with every chunk, which multiplies embedding cost. |
| **Overall** | **9** | The core claim now survives every objection we could test without real users. |

---

## What's genuinely strong

1. **It tried to break itself, and says so.** Burned-in captions, misaligned parts, smarter merges and
   long records were each set up as a way for the finding to fail. Two of those predictions were wrong
   in the finding's favour, and the long-record test is where it did fail, which is reported and fixed.
2. **Corrections are part of the record.** A moment-accuracy gap (0.71 vs 0.85) turned out to be a
   measurement artifact. A chunk-loss bug in `chunk=` (chunks of one record deleted when a write batch
   straddled two records) invalidated a published number; it was caught by checking stored counts, fixed
   with a regression test, and the data repaired before re-measuring.
3. **The rule is actionable.** It tells an engineer what to build (joint vectors), when that stops working
   (long parts), and what to do then (chunk and fuse), with one argument for each in the library.
4. **Users can check it.** `collection.evaluate(questions, fusion="sum")` runs the same paired comparison
   on their own data against the strongest merge.

## What holds it back

1. **No real users.** The result survives simulated terse searchers and messy typing, but every query
   was written by us, an AI agent or a script. Real users on a real product are the test left.
2. **Sample sizes.** 60–80 questions per corpus separates large effects from noise but leaves smaller
   ones unresolved: the router's lead on speech questions, and the semantic chunker's lead over
   fixed-size chunks, are consistent but not significant.
3. **Cost of chunk-level fusion.** Embedding the image with every chunk is simple and accurate but
   expensive for long documents with many chunks; smarter sharing (e.g. one image embedding reused
   across chunks via a model that supports it) is unexplored.
4. **CI coverage** (minor). Unit and media tests run in CI; the Atlas integration tier runs locally.

## What would move it to 10

- Run `evaluate()` on a real dataset with questions written by people.
- Test chunk-level fusion on real long documents (manuals, papers) with their own structure.

---

## By the numbers

| | |
| --- | --- |
| Corpora | 4 (interviews, held-out video, NASA photos, Met artworks) + constructed boundary sets |
| Benchmark questions | 380 (plus 80 noise-degraded), 320 written blind by AI agents |
| Pre-stated predictions | 14 (10 held, 4 failed) |
| Late-fusion methods beaten | 5 (RRF, CombSUM, CombMNZ, CombMAX, cross-validated weights) |
| Tests | 310 unit + media + 18 live integration (Atlas cloud and Atlas Local) |
| Source modules | 9 engine modules + `cinematlas.core` (8 modules) |
| Public docs | README, blog, paper, `bench/RESULTS.md`, this review |
