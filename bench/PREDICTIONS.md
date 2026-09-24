# Predictions

Written down, committed and pushed **before** the run that tests them, so the git history timestamps
them. Outcomes are filled in afterwards and never edited retroactively. Predictions 1–12 predate this
file; they are listed, with outcomes, in [paper.md](../paper.md) §2.

## 13. Terse, single-detail queries (simulated searchers)

On the Met artworks, an agent plays a searcher who glimpsed one work and later searches for it: 2–6
words, one remembered detail, sometimes the wrong word or a typo. Each query's answer is the work it
was written for (known-item search). **Prediction:** the joint vector still beats merged rankings
significantly against CombSUM, the strongest merge, but by less than on the full questions (0.95 vs 0.81).

**Outcome:** *pending*

## 14. Messy typing (deterministic noise)

The 80 existing Met questions, degraded by a fixed-seed script: typos in about one word in four, filler
words dropped, and cut to at most five words. **Prediction:** both methods lose accuracy; the joint
vector still beats CombSUM significantly.

**Outcome:** *pending*
