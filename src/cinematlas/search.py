"""Search: single-source queries, native or client-side fusion, sentence reranking, routing.

Functions take a :class:`SearchHost` (the engine facade satisfies it): what the engine *is* stays in
the engine, and this module holds only how a question becomes ranked moments.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol

from pymongo.errors import PyMongoError

from ._utils import (
    SEARCH_PROJECTION,
    build_deep_link,
    build_native_rerank_pipeline,
    build_rank_fusion_pipeline,
    build_text_search_stage,
    build_vector_search_pipeline,
)
from .capabilities import say_once, warn_once
from .exceptions import SearchError
from .indexes import (
    DEFAULT_AUTO_INDEX,
    DEFAULT_SCENE_INDEX,
    DEFAULT_TEXT_INDEX,
    DEFAULT_TRANSCRIPT_INDEX,
    DEFAULT_VISUAL_INDEX,
)
from .results import SearchHit, SearchResults
from .retrieval import lexical_moment, rerank_segments, rrf_fuse, scene_key

logger = logging.getLogger("cinematlas")

SOURCES = ("visual", "scene", "transcript", "text")
# Fusion weights, tuned on bench/'s first corpus: the keyframe list is noisy for questions about speech,
# so it acts as a tie-breaker; the sentence-level reranker carries the most signal.
DEFAULT_WEIGHTS = {"visual": 0.25, "scene": 1.0, "transcript": 1.0, "text": 1.0, "rerank": 2.0}
# Default search is scene-first: one joint keyframe+speech vector per scene ranks the scenes and the
# reranker only picks each scene's moment. On both bench/ video corpora it ties adaptive routing (it wins
# questions about what's shown, leans behind on what's said) at about half the latency.
#
# Adaptive routing (routing="adaptive"): the sentence reranker's top relevance says whether a question
# is about what was said or shown: speech confidence f = clamp((relevance - lo) / (hi - lo), 0, 1)
# scales the speech group by f and the visual group by 1 - f.
SPEECH_WEIGHTS = {"transcript": 1.0, "text": 0.25, "rerank": 2.0}
VISUAL_WEIGHTS = {"scene": 1.0, "visual": 0.25}

SceneKey = tuple[str, int]


class SearchHost(Protocol):
    collection: Any
    rerank_model: str | None
    scene_embeddings: bool
    native_fusion: bool | None
    native_rerank: bool | None

    @property
    def transcript_mode(self) -> str: ...
    @property
    def server_version(self) -> str | None: ...
    @property
    def routing_calibration(self) -> tuple[float, float] | None: ...
    def _embed_multimodal_query(self, query_text: str) -> list[float]: ...
    def _embed_text_query(self, query_text: str) -> list[float]: ...
    def _rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]: ...
    def search(self, query_text: str, top_k: int = 5, video_id: str | None = None, **kw: Any) -> SearchResults: ...


# ---------------------------------------------------------------------- single sources
def run(collection: Any, pipeline: list[dict[str, Any]], what: str) -> SearchResults:
    try:
        return SearchResults(collection.aggregate(pipeline))
    except PyMongoError as e:
        raise SearchError(f"{what} failed: {e}") from e


def vector_search(host: SearchHost, query_vec: list[float], *, index_name: str, path: str, top_k: int,
                  video_id: str | None) -> SearchResults:
    pipeline = build_vector_search_pipeline(index_name=index_name, path=path, top_k=top_k, query_vector=query_vec,
                                            video_id=video_id)
    return run(host.collection, pipeline, f"MongoDB vector search on {index_name!r}")


def search_client_transcript(host: SearchHost, query_text: str, top_k: int, index_name: str,
                             video_id: str | None) -> SearchResults:
    if not query_text:
        return SearchResults()
    pipeline = build_vector_search_pipeline(index_name=index_name, path="transcript_embedding", top_k=top_k,
                                            query_vector=host._embed_text_query(query_text), video_id=video_id)
    return run(host.collection, pipeline, "Atlas Vector Search")


def search_auto_transcript(host: SearchHost, query_text: str, top_k: int, index_name: str,
                           video_id: str | None) -> SearchResults:
    if not query_text:
        return SearchResults()
    pipeline = build_vector_search_pipeline(index_name=index_name, path="transcript", top_k=top_k,
                                            query_text=query_text, video_id=video_id)
    return run(host.collection, pipeline, "Atlas Vector Search")


def search_text(host: SearchHost, query_text: str, top_k: int, video_id: str | None) -> SearchResults:
    if not query_text:
        return SearchResults()
    pipeline = [build_text_search_stage(DEFAULT_TEXT_INDEX, query_text, video_id), {"$limit": top_k},
                {"$project": {**{k: v for k, v in SEARCH_PROJECTION.items() if k != "score"},
                              "score": {"$meta": "searchScore"}}}]
    return run(host.collection, pipeline, "Atlas full-text search")


# ---------------------------------------------------------------------- fusion
def source_pipeline(host: SearchHost, name: str, query_text: str, n: int, video_id: str | None,
                    mm_vec: list[float] | None) -> list[dict[str, Any]]:
    """The retrieval pipeline for one source (usable standalone or inside ``$rankFusion``)."""
    if name == "text":
        return [build_text_search_stage(DEFAULT_TEXT_INDEX, query_text, video_id), {"$limit": n}]
    if name == "transcript":
        if host.transcript_mode == "autoembed":
            stage = build_vector_search_pipeline(index_name=DEFAULT_AUTO_INDEX, path="transcript", top_k=n,
                                                 query_text=query_text, video_id=video_id)[0]
        else:
            stage = build_vector_search_pipeline(index_name=DEFAULT_TRANSCRIPT_INDEX, path="transcript_embedding",
                                                 top_k=n, query_vector=host._embed_text_query(query_text),
                                                 video_id=video_id)[0]
        return [stage]
    index, path = ((DEFAULT_VISUAL_INDEX, "visual_embedding") if name == "visual"
                   else (DEFAULT_SCENE_INDEX, "scene_embedding"))
    return [build_vector_search_pipeline(index_name=index, path=path, top_k=n, query_vector=mm_vec,
                                         video_id=video_id)[0]]


def retrieve_native(host: SearchHost, pipelines: dict[str, list[dict[str, Any]]], weights: dict[str, float],
                    n: int) -> tuple[dict[str, list[SceneKey]], dict[SceneKey, dict[str, Any]]]:
    """One round trip: Atlas ``$rankFusion`` returns the union plus each document's per-source rank."""
    rows = list(host.collection.aggregate(build_rank_fusion_pipeline(pipelines, weights, n * len(pipelines))))
    per_source: dict[str, list[tuple[int, SceneKey]]] = {name: [] for name in pipelines}
    docs: dict[SceneKey, dict[str, Any]] = {}
    for row in rows:
        key = scene_key(row)
        docs[key] = {k: v for k, v in row.items() if k not in ("score", "fusion")}
        for detail in (row.get("fusion") or {}).get("details", []):
            rank = detail.get("rank")
            if isinstance(rank, int):
                per_source.setdefault(detail["inputPipelineName"], []).append((rank, key))
    return {name: [k for _, k in sorted(v)] for name, v in per_source.items()}, docs


def rerank_scenes(host: SearchHost, query_text: str, docs: dict[SceneKey, dict[str, Any]]
                  ) -> tuple[list[SceneKey], dict[SceneKey, tuple[float, dict[str, Any]]]]:
    """Sentence-level rerank: Atlas-native ``$rerank`` when enabled, Voyage API otherwise."""
    if host.native_rerank is not False:
        try:
            rows = list(host.collection.aggregate(build_native_rerank_pipeline(list(docs), query_text,
                                                                               host.rerank_model)))
            best: dict[SceneKey, tuple[float, dict[str, Any]]] = {}
            for row in rows:
                key, seg = scene_key(row), row.get("segment") or {
                    "start": row.get("timestamp_start", 0.0), "end": row.get("timestamp_end", 0.0),
                    "text": row.get("transcript", "")}
                if key not in best or row["relevance"] > best[key][0]:
                    best[key] = (row["relevance"], seg)
            host.native_rerank = True
            return sorted(best, key=lambda k: -best[k][0]), best
        except PyMongoError as e:
            if host.native_rerank:
                raise
            warn_once("$rerank", e, host.server_version)
            host.native_rerank = False
    return rerank_segments(host._rerank, query_text, list(docs.values()))


def search(host: SearchHost, query_text: str, top_k: int = 5, video_id: str | None = None, *,
           sources: Sequence[str] = SOURCES, rerank: bool = True, candidates: int | None = None,
           weights: dict[str, float] | None = None, routing: str | None = None) -> SearchResults:
    """See :meth:`cinematlas.Cinematlas.search`."""
    if not query_text:
        return SearchResults()
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    if routing is None:
        if weights is None and tuple(sources) == SOURCES and host.scene_embeddings:
            try:
                results = host.search(query_text, top_k, video_id, rerank=rerank, candidates=candidates,
                                      routing="scene")
                if results:  # empty: scenes ingested before scene vectors existed; fuse instead
                    return results
            except SearchError as e:  # no scene index here (older deployment): fuse everything instead
                say_once(("scene-first", type(e).__name__),
                         f"Scene-first search unavailable ({e}); using adaptive fusion. "
                         "Fix: engine.ensure_indexes() and re-ingest with scene_embeddings=True.")
        routing = "adaptive"
    if routing not in ("adaptive", "fixed", "scene"):
        raise ValueError(f"routing must be 'adaptive', 'fixed' or 'scene', got {routing!r}")
    n = candidates or max(top_k * 4, 20)
    user_weights = weights
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if routing == "scene":
        # The joint image+speech vector ranks scenes; the reranker only picks each scene's moment.
        sources, n, weights = ("scene",), candidates or top_k, {"scene": 1.0, "rerank": 0.0}

    mm_vec = host._embed_multimodal_query(query_text) if {"visual", "scene"} & set(sources) else None
    pipelines: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, Exception] = {}
    for name in sources:
        try:
            pipelines[name] = source_pipeline(host, name, query_text, n, video_id, mm_vec)
        except SearchError as e:
            errors[name] = e

    ranked: dict[str, list[SceneKey]] = {}
    docs: dict[SceneKey, dict[str, Any]] = {}
    if pipelines and host.native_fusion is not False:
        try:
            ranked, docs = retrieve_native(host, pipelines, weights, n)
            host.native_fusion = True
        except PyMongoError as e:
            if host.native_fusion:
                raise SearchError(f"$rankFusion failed: {e}") from e
            warn_once("$rankFusion", e, host.server_version)
            host.native_fusion = False
    if host.native_fusion is False:
        for name, pipeline in pipelines.items():
            try:
                hits = list(host.collection.aggregate(
                    [*pipeline, {"$project": {k: v for k, v in SEARCH_PROJECTION.items() if k != "score"}}]))
            except PyMongoError as e:
                errors[name] = e
                logger.warning(f"search source {name!r} failed, continuing without it: {e}")
                continue
            ranked[name] = [scene_key(h) for h in hits]
            for h in hits:
                docs.setdefault(scene_key(h), h)
    if errors and len(errors) == len(sources):
        raise SearchError(f"All search sources failed: {errors}")

    moments: dict[SceneKey, tuple[float, dict[str, Any]]] = {}
    if rerank and host.rerank_model and docs:
        try:
            ranked["rerank"], moments = rerank_scenes(host, query_text, docs)
        except Exception as e:
            logger.warning(f"Rerank failed, using retrieval ranks only: {e}")

    speech_confidence = None
    thresholds = host.routing_calibration if routing == "adaptive" and user_weights is None else None
    if thresholds and moments:
        lo, hi = thresholds
        top_relevance = max(rel for rel, _ in moments.values())
        speech_confidence = min(1.0, max(0.0, (top_relevance - lo) / (hi - lo)))
        weights = {**{k: w * speech_confidence for k, w in SPEECH_WEIGHTS.items()},
                   **{k: w * (1.0 - speech_confidence) for k, w in VISUAL_WEIGHTS.items()}}

    results = SearchResults()
    results.speech_confidence = speech_confidence
    results.weights = {k: round(v, 3) for k, v in weights.items() if k in ranked}
    for key, score in rrf_fuse(ranked, weights)[:top_k]:
        doc = dict(docs[key])
        if key in moments:
            relevance, moment = moments[key]
        else:
            relevance, moment = None, lexical_moment(query_text, doc.get("segments") or [])
        start = moment["start"] if moment else doc.get("timestamp_start", 0.0)
        url = doc.get("video_url")
        results.append(SearchHit({
            **doc,
            "score": score,
            "ranks": {name: keys.index(key) + 1 for name, keys in ranked.items() if key in keys},
            "relevance": relevance,
            "moment": dict(moment) if moment else None,
            "moment_link": build_deep_link(url, start) if url else None,
        }))
    return results
