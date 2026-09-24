# Predictions

Written down, committed and pushed **before** the run that tests them, so the git history timestamps
them. Outcomes are filled in afterwards and never edited retroactively. Predictions 1–12 predate this
file; they are listed, with outcomes, in [paper.md](../paper.md) §2.

## 13. Terse, single-detail queries (simulated searchers)

On the Met artworks, an agent plays a searcher who glimpsed one work and later searches for it: 2–6
words, one remembered detail, sometimes the wrong word or a typo. Each query's answer is the work it
was written for (known-item search). **Prediction:** the joint vector still beats merged rankings
significantly against CombSUM, the strongest merge, but by less than on the full questions (0.95 vs 0.81).

**Outcome: held.** Joint 0.80 vs CombSUM 0.70 (10 vs 2 disputed, p = 0.039) and vs RRF 0.47. The margin
shrank from 14 points to 10, as predicted. Queries: [queries_met_searchers.json](queries_met_searchers.json),
2–5 words, 16 of 80 with deliberate typos.

## 14. Messy typing (deterministic noise)

The 80 existing Met questions, degraded by a fixed-seed script: typos in about one word in four, filler
words dropped, and cut to at most five words. **Prediction:** both methods lose accuracy; the joint
vector still beats CombSUM significantly.

**Outcome: held, and the gap widened.** Joint 0.95 → 0.84, CombSUM 0.81 → 0.61, RRF 0.62 → 0.46; joint vs
CombSUM 20 vs 2 disputed, p < 0.001. Merged rankings degrade faster under noise than the joint vector.
