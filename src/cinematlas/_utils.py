"""Pure helpers (no I/O) used by the ingestion and search pipelines."""

import hashlib
import math
import re
import time
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_YOUTUBE_HOSTS = {"www.youtube.com", "youtube.com", "m.youtube.com", "music.youtube.com"}
_YOUTU_BE_HOSTS = {"youtu.be", "www.youtu.be"}
# /shorts/<id>, /embed/<id>, /live/<id>, /v/<id>
_YOUTUBE_PATH_ID = re.compile(r"^/(?:shorts|embed|live|v)/([0-9A-Za-z_-]{11})(?:[/?#]|$)")

SEARCH_PROJECTION = {
    "_id": 0,
    "video_id": 1,
    "scene_id": 1,
    "timestamp_start": 1,
    "timestamp_end": 1,
    "transcript": 1,
    "segments": 1,
    "thumbnail_url": 1,
    "deep_link": 1,
    "video_url": 1,
    "source_type": 1,
    "filename": 1,
    "score": {"$meta": "vectorSearchScore"},
}

# Atlas caps numCandidates at 10,000.
MAX_NUM_CANDIDATES = 10_000


def extract_video_id(video_url: str | None) -> str:
    """Derive a stable video ID from a URL.

    YouTube URLs yield their canonical 11-char ID. Anything else yields a
    deterministic hash of the full URL, so two distinct URLs never collide
    (ingestion replaces all documents sharing a video ID).
    """
    if not video_url:
        return f"vid_{int(time.time())}"

    try:
        parsed = urllib.parse.urlparse(video_url)
        host = (parsed.hostname or "").lower()
        if host in _YOUTUBE_HOSTS:
            qs = urllib.parse.parse_qs(parsed.query)
            if qs.get("v") and qs["v"][0]:
                return qs["v"][0]
            match = _YOUTUBE_PATH_ID.match(parsed.path)
            if match:
                return match.group(1)
        elif host in _YOUTU_BE_HOSTS:
            path = parsed.path.strip("/")
            if path:
                return path.split("/")[0]
    except ValueError:
        pass

    return f"vid_{hashlib.md5(video_url.encode('utf-8')).hexdigest()[:10]}"


def is_youtube_url(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in _YOUTUBE_HOSTS or host in _YOUTU_BE_HOSTS


def build_deep_link(video_url: str, start_sec: float) -> str:
    """Link that opens ``video_url`` at ``start_sec``.

    YouTube understands ``?t=<N>s``. Everything else gets a W3C Media Fragment
    (``#t=<N>``), which browsers seek natively for direct video files and which
    never touches the query string, so presigned/signed URLs stay valid.
    """
    parsed = urllib.parse.urlparse(video_url)
    if is_youtube_url(video_url):
        query = [(k, v) for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True) if k != "t"]
        query.append(("t", f"{int(start_sec)}s"))
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query), fragment=""))
    return urllib.parse.urlunparse(parsed._replace(fragment=f"t={int(start_sec)}"))


# A bare "host.tld/..." (no scheme) is treated as https. Local paths are handled before this.
_SCHEMELESS_HOST = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?(?:[/?#]|$)", re.IGNORECASE)


def normalize_source_url(source: str) -> str:
    """``www.b.com/v.mp4`` -> ``https://www.b.com/v.mp4``; URLs with a scheme are returned as-is."""
    source = source.strip()
    if "://" not in source and _SCHEMELESS_HOST.match(source):
        return f"https://{source}"
    return source


# Query parameters that carry credentials in signed URLs: AWS SigV4, GCS, Azure SAS,
# CloudFront, and common token conventions. Matched case-insensitively.
_SENSITIVE_PARAM = re.compile(
    r"^(x-amz-.*|x-goog-.*|signature|sig|se|sp|sv|sr|st|spr|skoid|sktid|skt|ske|sks|skv|"
    r"expires|policy|key-pair-id|token|access_token|id_token|auth|authorization|api_key|apikey|key)$",
    re.IGNORECASE,
)


def redact_url(url: str) -> str:
    """Strip userinfo and credential-bearing query params so URLs are safe to persist."""
    parsed = urllib.parse.urlparse(url)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    query = [
        (k, v) for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not _SENSITIVE_PARAM.match(k)
    ]
    return urllib.parse.urlunparse(parsed._replace(netloc=netloc, query=urllib.parse.urlencode(query)))


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def normalize_segments(raw_segments: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Coerce Whisper segments (dicts or SDK objects) to ``{start, end, text[, words]}`` dicts.

    ``words`` (``[{start, end, word}]``) is kept when the backend produced word timestamps.
    """
    segments = []
    for seg in raw_segments or []:
        out: dict[str, Any] = {
            "start": float(_field(seg, "start") or 0.0),
            "end": float(_field(seg, "end") or 0.0),
            "text": (_field(seg, "text") or "").strip(),
        }
        words = _field(seg, "words")
        if words:
            out["words"] = [
                {"start": float(_field(w, "start") or 0.0), "end": float(_field(w, "end") or 0.0),
                 "word": _field(w, "word") or ""}
                for w in words
            ]
        segments.append(out)
    return segments


def split_long_spans(spans: Sequence[tuple[float, float]], max_len: float | None) -> list[tuple[float, float]]:
    """Split spans longer than ``max_len`` into equal contiguous parts no longer than ``max_len``.

    Keeps retrieval precise and deep links accurate for footage without hard cuts
    (interviews with dissolves, lectures, screen recordings).
    """
    if not max_len:
        return list(spans)
    out: list[tuple[float, float]] = []
    for start, end in spans:
        parts = max(1, math.ceil((end - start) / max_len - 1e-9))
        step = (end - start) / parts
        out.extend((start + i * step, end if i == parts - 1 else start + (i + 1) * step) for i in range(parts))
    return out


# Whisper segment boundaries jitter by a few hundred ms; an overlap shorter than this is noise.
MIN_SCENE_OVERLAP_SEC = 0.5


def assign_segments(
    scene_spans: Sequence[tuple],
    segments: Sequence[dict[str, Any]],
    min_overlap: float = MIN_SCENE_OVERLAP_SEC,
) -> list[list[dict[str, Any]]]:
    """Distribute speech segments to the scene(s) they were spoken in.

    * A segment grazing a cut (overlap < ``min_overlap`` on one side, i.e. timestamp
      jitter) goes whole to the scene it overlaps most. No speech is ever dropped.
    * A segment clearly straddling cuts (>= ``min_overlap`` on several sides) is split
      word-by-word at the cuts when word timestamps exist; otherwise it is kept in
      each of those scenes.

    Returns, per scene, ``[{start, end, text}]`` in time order.
    """
    buckets: list[list[dict[str, Any]]] = [[] for _ in scene_spans]
    for seg in segments:
        if not seg["text"]:
            continue
        overlaps = [
            max(0.0, min(seg["end"], s_end) - max(seg["start"], s_start)) for s_start, s_end in scene_spans
        ]
        if not overlaps or max(overlaps) <= 0:
            continue
        best = max(range(len(overlaps)), key=overlaps.__getitem__)
        targets = sorted({best} | {i for i, ov in enumerate(overlaps) if ov >= min_overlap})
        words = seg.get("words")

        if len(targets) == 1 or not words:
            for i in targets:
                buckets[i].append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            continue

        per_scene: dict[int, list[dict[str, Any]]] = {}
        for w in words:
            mid = (w["start"] + w["end"]) / 2
            owner = min(targets, key=lambda i: 0 if scene_spans[i][0] <= mid < scene_spans[i][1]
                        else min(abs(mid - scene_spans[i][0]), abs(mid - scene_spans[i][1])))
            per_scene.setdefault(owner, []).append(w)
        for i, ws in per_scene.items():
            text = "".join(w["word"] for w in ws).strip()
            if text:
                buckets[i].append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": text})

    for b in buckets:
        b.sort(key=lambda x: x["start"])
    return buckets


def assign_transcripts(
    scene_spans: Sequence[tuple],
    segments: Sequence[dict[str, Any]],
    min_overlap: float = MIN_SCENE_OVERLAP_SEC,
) -> list[str]:
    """Per-scene transcript text; see :func:`assign_segments` for the attribution rules."""
    return [" ".join(s["text"] for s in segs) for segs in assign_segments(scene_spans, segments, min_overlap)]


def build_vector_search_pipeline(
    *,
    index_name: str,
    path: str,
    top_k: int,
    query_text: str | None = None,
    query_vector: list[float] | None = None,
    match: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a ``$vectorSearch`` + ``$project`` pipeline (text for autoEmbed, vector otherwise).

    ``match`` is a pre-filter on indexed filter fields, e.g. ``{"video_id": "a", "metadata.genre": "x"}``
    (see :func:`build_match`).
    """
    if (query_text is None) == (query_vector is None):
        raise ValueError("Provide exactly one of query_text or query_vector.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

    stage: dict[str, Any] = {
        "index": index_name,
        "path": path,
        "numCandidates": min(top_k * 10, MAX_NUM_CANDIDATES),
        "limit": top_k,
    }
    if query_text is not None:
        stage["query"] = query_text
    else:
        stage["queryVector"] = query_vector
    if match:
        stage["filter"] = dict(match)

    return [{"$vectorSearch": stage}, {"$project": dict(SEARCH_PROJECTION)}]


def build_match(video_ids: Sequence[str] = (), where: Mapping[str, str | Sequence[str]] | None = None
                ) -> dict[str, Any]:
    """A pre-filter for every source: videos by ID, plus metadata fields (``metadata.<name>``).

    A string matches that value; a list matches any of its values.
    """
    match: dict[str, Any] = {}
    if video_ids:
        match["video_id"] = video_ids[0] if len(video_ids) == 1 else {"$in": list(video_ids)}
    for name, value in (where or {}).items():
        values = [value] if isinstance(value, str) else list(dict.fromkeys(value))
        match[f"metadata.{name}"] = values[0] if len(values) == 1 else {"$in": values}
    return match


def build_text_search_stage(index_name: str, query_text: str, match: Mapping[str, Any] | None = None
                            ) -> dict[str, Any]:
    """Atlas Search (BM25) over transcripts: catches exact names/numbers vectors can blur."""
    text = {"text": {"query": query_text, "path": "transcript"}}
    if not match:
        return {"$search": {"index": index_name, **text}}
    filters = [{"in": {"path": path, "value": cond["$in"]}} if isinstance(cond, dict)
               else {"equals": {"path": path, "value": cond}} for path, cond in match.items()]
    return {"$search": {"index": index_name, "compound": {"must": [text], "filter": filters}}}


def build_rank_fusion_pipeline(
    pipelines: dict[str, list[dict[str, Any]]], weights: dict[str, float], limit: int
) -> list[dict[str, Any]]:
    """Native ``$rankFusion`` (MongoDB 8.0+) with per-pipeline ranks exposed via scoreDetails."""
    return [
        {"$rankFusion": {
            "input": {"pipelines": pipelines},
            "combination": {"weights": {name: float(weights.get(name, 1.0)) for name in pipelines}},
            "scoreDetails": True,
        }},
        {"$limit": limit},
        {"$project": {**{k: v for k, v in SEARCH_PROJECTION.items() if k != "score"},
                      "score": {"$meta": "score"}, "fusion": {"$meta": "scoreDetails"}}},
    ]


def build_native_rerank_pipeline(
    keys: Sequence[tuple[str, int]], query_text: str, model: str, max_docs: int = 1000
) -> list[dict[str, Any]]:
    """Sentence-level ``$rerank`` (MongoDB 8.3+, Atlas): unwind each scene's segments, rerank them."""
    return [
        {"$match": {"$or": [{"video_id": v, "scene_id": s} for v, s in keys]}},
        {"$unwind": {"path": "$segments", "preserveNullAndEmptyArrays": True}},
        {"$set": {"_rerank_text": {"$ifNull": ["$segments.text", {"$ifNull": ["$transcript", ""]}]}}},
        {"$match": {"_rerank_text": {"$ne": ""}}},
        {"$rerank": {"query": {"text": query_text}, "path": "_rerank_text",
                     "numDocsToRerank": max_docs, "model": model}},
        {"$project": {"_id": 0, "video_id": 1, "scene_id": 1, "segment": "$segments",
                      "transcript": 1, "timestamp_start": 1, "timestamp_end": 1,
                      "relevance": {"$meta": "score"}}},
    ]
