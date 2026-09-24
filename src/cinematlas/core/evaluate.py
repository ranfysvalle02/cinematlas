"""Check the finding on your own data: joint vector vs merged rankings, with a paired test.

    report = collection.evaluate([
        {"q": "astronaut fixing a telescope", "relevant": ["sts082-717-029"]},
        ...                                   # 30–50 questions is enough to see real gaps
    ])
    print(report)

Needs a collection created with ``late=True``, which also stores one vector per part so the
merged-rankings baseline can run on exactly the same data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test: (questions only A got, questions only B got, p)."""
    a_only = sum(x and not y for x, y in zip(a, b, strict=True))
    b_only = sum(y and not x for x, y in zip(a, b, strict=True))
    n = a_only + b_only
    if not n:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return a_only, b_only, min(1.0, 2 * tail)


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


@dataclass
class Scores:
    """Hit@1, Hit@k and MRR for one method, plus which questions it got right at rank 1."""

    hit1: float
    hitk: float
    mrr: float
    correct: list[bool] = field(repr=False)


def score(ranked: Sequence[Sequence[Any]], relevant: Sequence[set[Any]], k: int) -> Scores:
    firsts = []
    for ids, rel in zip(ranked, relevant, strict=True):
        firsts.append(next((i for i, x in enumerate(ids[:k], 1) if x in rel), None))
    n = len(firsts) or 1
    return Scores(hit1=sum(f == 1 for f in firsts) / n, hitk=sum(f is not None for f in firsts) / n,
                  mrr=sum(1 / f for f in firsts if f) / n, correct=[f == 1 for f in firsts])


@dataclass
class EvalReport:
    """Joint vector vs merged rankings on your questions."""

    n: int
    k: int
    joint: Scores
    merged: Scores
    joint_only: int
    merged_only: int
    p: float
    rows: list[dict[str, Any]] = field(repr=False, default_factory=list)

    @property
    def winner(self) -> str | None:
        """``"joint"`` or ``"merged"`` if the gap is significant (p < 0.05), else ``None``."""
        if self.p >= 0.05:
            return None
        return "joint" if self.joint_only > self.merged_only else "merged"

    def verdict(self) -> str:
        head = f"joint {self.joint.hit1:.2f} vs merged {self.merged.hit1:.2f} Hit@1 on {self.n} questions"
        split = f"{self.joint_only} vs {self.merged_only} disputed, p = {fmt_p(self.p)}"
        if self.winner == "joint":
            return f"The joint vector wins on your data: {head} ({split})."
        if self.winner == "merged":
            return f"Merged rankings win on your data: {head} ({split}). Your parts may not describe the same thing."
        more = "" if self.n >= 30 else " Label more questions (30+) to see smaller gaps."
        return f"No significant difference yet: {head} ({split}).{more}"

    def __str__(self) -> str:
        return "\n".join([
            f"{'':16}{'Hit@1':>7}{f'Hit@{self.k}':>8}{'MRR':>7}",
            f"{'joint vector':16}{self.joint.hit1:7.2f}{self.joint.hitk:8.2f}{self.joint.mrr:7.2f}",
            f"{'merged rankings':16}{self.merged.hit1:7.2f}{self.merged.hitk:8.2f}{self.merged.mrr:7.2f}",
            "",
            self.verdict(),
        ])


def normalize_questions(questions: Sequence[Mapping[str, Any]]) -> list[tuple[Any, set[Any]]]:
    out = []
    for i, q in enumerate(questions):
        query = q.get("q", q.get("query"))
        relevant = q.get("relevant", q.get("answer"))
        if query is None or relevant is None:
            raise ValueError(f"Question {i} needs 'q' and 'relevant' (a key or list of keys): {dict(q)}")
        out.append((query, set(relevant) if isinstance(relevant, (list, tuple, set)) else {relevant}))
    return out
