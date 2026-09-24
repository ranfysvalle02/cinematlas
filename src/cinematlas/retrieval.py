"""Ranking primitives for hybrid search (pure: no I/O).

Hybrid search fuses several *ranked lists* of scenes (keyframe vectors, joint
image+transcript vectors, transcript vectors, and a sentence-level reranker)
with Reciprocal Rank Fusion. RRF uses only ranks, so it combines retrievers
whose scores live on incompatible scales without any normalization, and the
reranker slots in as just another list.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

RRF_K = 60  # standard constant from Cormack et al. (2009); damps the head of each list

SceneKey = tuple[str, int]


def scene_key(doc: Mapping[str, Any]) -> SceneKey:
    return (doc["video_id"], doc["scene_id"])


def rrf_fuse(
    ranked: Mapping[str, Sequence[SceneKey]],
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
) -> list[tuple[SceneKey, float]]:
    """Fuse ranked lists: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1.

    Ties are broken by best single rank, then key, so results are deterministic.
    """
    scores: dict[SceneKey, float] = {}
    best_rank: dict[SceneKey, int] = {}
    for name, keys in ranked.items():
        w = (weights or {}).get(name, 1.0)
        if w <= 0:
            continue
        for rank, key in enumerate(dict.fromkeys(keys), start=1):  # dedupe, keep first rank
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by do does for from how i in is it its of on or so that the their them "
    "they this to was we what when where which who why will with you".split()
)


def _terms(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP}


def lexical_moment(query: str, segments: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Best-matching segment by content-word overlap. Fallback when no reranker is used."""
    q = _terms(query)
    if not segments:
        return None
    scored = [(len(q & _terms(s["text"])), -i, s) for i, s in enumerate(segments)]
    best = max(scored, key=lambda t: (t[0], t[1]))
    return best[2] if best[0] > 0 else None


def rerank_segments(
    rerank_fn: Any,
    query: str,
    docs: Sequence[Mapping[str, Any]],
    max_segments: int = 400,
) -> tuple[list[SceneKey], dict[SceneKey, tuple[float, Mapping[str, Any]]]]:
    """Rerank every sentence of the candidate scenes; a scene scores as its best sentence.

    ``rerank_fn(query, texts) -> [(index, relevance)]``. Returns the scene ranking and,
    per scene, ``(relevance, best_segment)``, which is the scene's *moment*. Scenes
    without speech are absent from this list (the visual lists still rank them).
    """
    flat: list[tuple[SceneKey, Mapping[str, Any]]] = []
    for doc in docs:
        segments = doc.get("segments") or (
            [{"start": doc["timestamp_start"], "end": doc["timestamp_end"], "text": doc["transcript"]}]
            if doc.get("transcript") else []
        )
        flat.extend((scene_key(doc), s) for s in segments if s.get("text"))
    flat = flat[:max_segments]
    if not flat:
        return [], {}

    best: dict[SceneKey, tuple[float, Mapping[str, Any]]] = {}
    for idx, relevance in rerank_fn(query, [s["text"] for _, s in flat]):
        key, seg = flat[idx]
        if key not in best or relevance > best[key][0]:
            best[key] = (relevance, seg)
    order = sorted(best, key=lambda key: -best[key][0])
    return order, best


def _clock(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


def to_context(results: Sequence[Mapping[str, Any]], *, full_scene: bool = False) -> str:
    """Render :meth:`Cinematlas.search` results as numbered, citable context for any LLM.

    Each entry is ``[n] <video> @ m:ss <link>`` followed by the matching sentence (or the
    whole scene transcript with ``full_scene=True``). Ask the model to cite ``[n]`` and map
    citations back with ``results[n - 1]["moment_link"]``. Scenes without speech are listed
    so visual matches stay visible to the model.
    """
    blocks = []
    for n, r in enumerate(results, start=1):
        moment = r.get("moment") or {}
        start = moment.get("start", r.get("timestamp_start", 0.0))
        where = r.get("moment_link") or r.get("filename") or ""
        text = (r.get("transcript") if full_scene or not moment else moment.get("text")) or "(no speech in this scene)"
        blocks.append(f"[{n}] {r.get('video_id')} @ {_clock(start)} {where}".rstrip() + f"\n{text.strip()}")
    return "\n\n".join(blocks)
