"""Ways to merge separate rankings (late fusion), for comparison with the joint vector.

Each takes ``{list name: [(record id, score), ...best first]}`` and returns record ids, best first.
On every corpus in bench/, all of them lose to one joint vector; CombSUM is the strongest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

Ranked = Sequence[tuple[Any, float]]
RRF_K = 60


def rrf(lists: Mapping[str, Ranked], weights: Mapping[str, float] | None = None) -> list:
    """Reciprocal Rank Fusion: sum of w / (60 + rank). What Atlas ``$rankFusion`` does."""
    score: dict = {}
    for name, ranked in lists.items():
        for rank, (doc, _) in enumerate(ranked, 1):
            score[doc] = score.get(doc, 0.0) + (weights or {}).get(name, 1.0) / (RRF_K + rank)
    return sorted(score, key=lambda d: -score[d])


def normalized(ranked: Ranked) -> dict:
    """Min-max scale one list's scores to [0, 1]."""
    if not ranked:
        return {}
    hi, lo = ranked[0][1], ranked[-1][1]
    return {doc: (s - lo) / (hi - lo) if hi > lo else 1.0 for doc, s in ranked}


def comb_sum(lists: Mapping[str, Ranked], weights: Mapping[str, float] | None = None, *, mnz: bool = False) -> list:
    """CombSUM: add normalized scores. ``mnz=True`` (CombMNZ) multiplies by how many lists found the record."""
    score: dict = {}
    hits: dict = {}
    for name, ranked in lists.items():
        w = (weights or {}).get(name, 1.0)
        for doc, s in normalized(ranked).items():
            score[doc] = score.get(doc, 0.0) + w * s
            hits[doc] = hits.get(doc, 0) + 1
    if mnz:
        score = {d: s * hits[d] for d, s in score.items()}
    return sorted(score, key=lambda d: -score[d])


def comb_mnz(lists: Mapping[str, Ranked], weights: Mapping[str, float] | None = None) -> list:
    return comb_sum(lists, weights, mnz=True)


def comb_max(lists: Mapping[str, Ranked], weights: Mapping[str, float] | None = None) -> list:
    """CombMAX: each record's best normalized score in any list. Needs no agreement between lists."""
    score: dict = {}
    for ranked in lists.values():
        for doc, s in normalized(ranked).items():
            score[doc] = max(score.get(doc, 0.0), s)
    return sorted(score, key=lambda d: -score[d])


MERGES: dict[str, Callable[..., list]] = {"rrf": rrf, "sum": comb_sum, "mnz": comb_mnz, "max": comb_max}
