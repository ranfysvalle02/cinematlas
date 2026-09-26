"""Atlas Vector Search index definitions and an idempotent helper to create them.

Transcript search has two interchangeable backends:

* ``autoembed`` — Atlas Automated Embedding: Atlas embeds ``transcript`` (and the
  query text) itself with a Voyage text model. Requires a cluster with autoEmbed.
* ``client`` — Cinematlas embeds transcripts with the same Voyage text model and
  stores them in ``transcript_embedding``; a plain vector index serves queries.
  Works on any Atlas / Atlas Local deployment with Vector Search.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

logger = logging.getLogger("cinematlas")

TranscriptMode = Literal["autoembed", "client"]

DEFAULT_VISUAL_INDEX = "cinematlas_vector_index"
DEFAULT_AUTO_INDEX = "cinematlas_auto_index"
DEFAULT_TRANSCRIPT_INDEX = "cinematlas_transcript_index"
DEFAULT_SCENE_INDEX = "cinematlas_scene_index"
DEFAULT_TEXT_INDEX = "cinematlas_text_index"
# Scalar quantization: ~75% less vector memory in mongot, negligible recall cost at this scale.
DEFAULT_QUANTIZATION = "scalar"
DEFAULT_TEXT_MODEL = "voyage-4"
DEFAULT_AUTO_EMBED_MODEL = DEFAULT_TEXT_MODEL  # back-compat alias
# voyage-multimodal-3.5 and the voyage-4 family both default to 1024 dimensions.
DEFAULT_VISUAL_DIMENSIONS = 1024
DEFAULT_TEXT_DIMENSIONS = 1024


def _filter_fields(filters: Sequence[str]) -> list[dict[str, Any]]:
    """``video_id`` plus each declared metadata filter (stored under ``metadata.<name>``)."""
    return [{"type": "filter", "path": "video_id"}, *({"type": "filter", "path": f"metadata.{f}"} for f in filters)]


def _vector_field(path: str, num_dimensions: int, quantization: str | None) -> dict[str, Any]:
    field: dict[str, Any] = {"type": "vector", "path": path, "numDimensions": num_dimensions, "similarity": "cosine"}
    if quantization:
        field["quantization"] = quantization
    return field


def visual_index_definition(
    num_dimensions: int = DEFAULT_VISUAL_DIMENSIONS, quantization: str | None = DEFAULT_QUANTIZATION,
    filters: Sequence[str] = (),
) -> dict[str, Any]:
    return {"fields": [_vector_field("visual_embedding", num_dimensions, quantization), *_filter_fields(filters)]}


def scene_index_definition(
    num_dimensions: int = DEFAULT_VISUAL_DIMENSIONS, quantization: str | None = DEFAULT_QUANTIZATION,
    filters: Sequence[str] = (),
) -> dict[str, Any]:
    """Joint keyframe+transcript vectors (voyage-multimodal-3.5 interleaved input)."""
    return {"fields": [_vector_field("scene_embedding", num_dimensions, quantization), *_filter_fields(filters)]}


def text_index_definition(filters: Sequence[str] = ()) -> dict[str, Any]:
    """Atlas Search (full-text, BM25) index over transcripts, filterable by video and metadata."""
    fields: dict[str, Any] = {
        "transcript": {"type": "string", "analyzer": "lucene.english"},
        "video_id": {"type": "token"},
    }
    if filters:
        fields["metadata"] = {"type": "document", "dynamic": False,
                              "fields": {f: {"type": "token"} for f in filters}}
    return {"mappings": {"dynamic": False, "fields": fields}}


def auto_embed_index_definition(model: str = DEFAULT_TEXT_MODEL, filters: Sequence[str] = ()) -> dict[str, Any]:
    return {"fields": [{"type": "autoEmbed", "modality": "text", "path": "transcript", "model": model},
                       *_filter_fields(filters)]}


def transcript_vector_index_definition(
    num_dimensions: int = DEFAULT_TEXT_DIMENSIONS, quantization: str | None = DEFAULT_QUANTIZATION,
    filters: Sequence[str] = (),
) -> dict[str, Any]:
    return {"fields": [_vector_field("transcript_embedding", num_dimensions, quantization),
                       *_filter_fields(filters)]}


def detect_transcript_mode(
    collection: Collection,
    *,
    auto_index_name: str = DEFAULT_AUTO_INDEX,
    transcript_index_name: str = DEFAULT_TRANSCRIPT_INDEX,
) -> TranscriptMode | None:
    """Infer the transcript backend from the search indexes that exist on ``collection``."""
    names = {ix["name"] for ix in collection.list_search_indexes()}
    if auto_index_name in names:
        return "autoembed"
    if transcript_index_name in names:
        return "client"
    return None


@dataclass(frozen=True)
class IndexStatus:
    """Health of one search index: what exists vs. what Cinematlas wants."""

    name: str
    kind: str  # "vectorSearch" | "search"
    state: str  # "ready" | "building" | "missing" | "failed"
    drift: tuple[str, ...] = ()  # human-readable differences from the desired definition

    @property
    def ok(self) -> bool:
        return self.state == "ready" and not self.drift


def desired_indexes(
    mode: TranscriptMode,
    *,
    text_model: str = DEFAULT_TEXT_MODEL,
    quantization: str | None = DEFAULT_QUANTIZATION,
    num_dimensions: int = DEFAULT_VISUAL_DIMENSIONS,
    text_dimensions: int = DEFAULT_TEXT_DIMENSIONS,
    scene: bool = True,
    text: bool = True,
    names: dict[str, str] | None = None,
    filters: Sequence[str] = (),
) -> dict[str, tuple[str, dict[str, Any]]]:
    """The single source of truth: ``{index_name: (kind, definition)}`` for a transcript mode."""
    n = {"visual": DEFAULT_VISUAL_INDEX, "scene": DEFAULT_SCENE_INDEX, "text": DEFAULT_TEXT_INDEX,
         "auto": DEFAULT_AUTO_INDEX, "transcript": DEFAULT_TRANSCRIPT_INDEX, **(names or {})}
    wanted: dict[str, tuple[str, dict[str, Any]]] = {
        n["visual"]: ("vectorSearch", visual_index_definition(num_dimensions, quantization, filters)),
    }
    if scene:
        wanted[n["scene"]] = ("vectorSearch", scene_index_definition(num_dimensions, quantization, filters))
    if text:
        wanted[n["text"]] = ("search", text_index_definition(filters))
    if mode == "autoembed":
        wanted[n["auto"]] = ("vectorSearch", auto_embed_index_definition(text_model, filters))
    else:
        wanted[n["transcript"]] = ("vectorSearch",
                                   transcript_vector_index_definition(text_dimensions, quantization, filters))
    return wanted


def definition_drift(current: dict[str, Any] | None, desired: dict[str, Any]) -> tuple[str, ...]:
    """Differences that matter: every key Cinematlas sets must match. Server-added defaults are ignored."""
    if not current:
        return ("definition unavailable",)
    diffs: list[str] = []
    if "fields" in desired:  # vectorSearch: match fields by path
        by_path = {f.get("path"): f for f in current.get("fields", [])}
        for want in desired["fields"]:
            have = by_path.get(want["path"])
            if have is None:
                diffs.append(f"{want['path']}: missing {want['type']} field")
                continue
            for key, value in want.items():
                if have.get(key) != value:
                    diffs.append(f"{want['path']}.{key}: {have.get(key)!r} -> {value!r}")
    else:  # Atlas Search: mappings
        have_m, want_m = current.get("mappings", {}), desired["mappings"]
        if have_m.get("dynamic") != want_m.get("dynamic"):
            diffs.append(f"mappings.dynamic: {have_m.get('dynamic')!r} -> {want_m.get('dynamic')!r}")
        for field, spec in want_m.get("fields", {}).items():
            have = have_m.get("fields", {}).get(field)
            if have is None:
                diffs.append(f"mappings.fields.{field}: missing")
                continue
            diffs.extend(_mapping_drift(f"mappings.fields.{field}", have, spec))
    return tuple(diffs)


def _mapping_drift(where: str, have: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """Keys Cinematlas sets must match, recursively; server-added options (norms, store, …) are ignored."""
    diffs = []
    for key, value in want.items():
        got = have.get(key)
        if isinstance(value, dict) and isinstance(got, dict):
            diffs.extend(_mapping_drift(f"{where}.{key}", got, value))
        elif got != value:
            diffs.append(f"{where}.{key}: {got!r} -> {value!r}")
    return diffs


def inspect_indexes(
    collection: Collection, wanted: dict[str, tuple[str, dict[str, Any]]]
) -> list[IndexStatus]:
    """Compare the collection's search indexes against ``wanted`` (see :func:`desired_indexes`)."""
    existing = {ix["name"]: ix for ix in collection.list_search_indexes()}
    report = []
    for name, (kind, definition) in wanted.items():
        ix = existing.get(name)
        if ix is None:
            report.append(IndexStatus(name, kind, "missing"))
            continue
        status = str(ix.get("status", "")).upper()
        state = ("failed" if status == "FAILED" else "ready" if status == "READY" and ix.get("queryable")
                 else "building")
        report.append(IndexStatus(name, kind, state, definition_drift(ix.get("latestDefinition"), definition)))
    return report


def ensure_search_indexes(
    collection: Collection,
    *,
    transcript_mode: Literal["auto", "autoembed", "client"] = "auto",
    visual_index_name: str = DEFAULT_VISUAL_INDEX,
    auto_index_name: str = DEFAULT_AUTO_INDEX,
    transcript_index_name: str = DEFAULT_TRANSCRIPT_INDEX,
    scene_index_name: str | None = DEFAULT_SCENE_INDEX,
    text_index_name: str | None = DEFAULT_TEXT_INDEX,
    quantization: str | None = DEFAULT_QUANTIZATION,
    num_dimensions: int = DEFAULT_VISUAL_DIMENSIONS,
    text_model: str | None = None,
    auto_embed_model: str | None = None,
    text_dimensions: int = DEFAULT_TEXT_DIMENSIONS,
    filters: Sequence[str] = (),
    update: bool = False,
    wait: bool = True,
    timeout_s: float = 600,
    poll_interval_s: float = 5,
) -> TranscriptMode:
    """Create every index Cinematlas needs and return the transcript mode in use.

    ``transcript_mode="auto"`` reuses whichever transcript index already exists; otherwise it
    tries autoEmbed and falls back to a client-side vector index if the cluster rejects it.

    Existing indexes that differ from the desired definition (e.g. created before scalar
    quantization) are reported in the log and, with ``update=True``, updated **in place**
    via ``updateSearchIndex``. The old version keeps serving queries until the new one is built.
    """
    model = text_model or auto_embed_model or DEFAULT_TEXT_MODEL
    if collection.name not in collection.database.list_collection_names():
        collection.database.create_collection(collection.name)
    names = {"visual": visual_index_name, "auto": auto_index_name, "transcript": transcript_index_name,
             **({"scene": scene_index_name} if scene_index_name else {}),
             **({"text": text_index_name} if text_index_name else {})}

    if transcript_mode == "auto":
        mode = detect_transcript_mode(collection, auto_index_name=auto_index_name,
                                      transcript_index_name=transcript_index_name) or "autoembed"
    elif transcript_mode in ("autoembed", "client"):
        mode = transcript_mode  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown transcript_mode {transcript_mode!r}")

    def wanted_for(m: TranscriptMode) -> dict[str, tuple[str, dict[str, Any]]]:
        return desired_indexes(m, text_model=model, quantization=quantization, num_dimensions=num_dimensions,
                               text_dimensions=text_dimensions, scene=bool(scene_index_name),
                               text=bool(text_index_name), names=names, filters=filters)

    wanted = wanted_for(mode)
    existing = {ix["name"] for ix in collection.list_search_indexes()}
    for name, (kind, definition) in wanted.items():
        if name in existing:
            continue
        try:
            collection.create_search_index(SearchIndexModel(name=name, type=kind, definition=definition))
        except OperationFailure as e:
            if not (transcript_mode == "auto" and name == auto_index_name):
                raise
            logger.info(f"autoEmbed unavailable ({e.code}: {str(e).split(', full error')[0]}); "
                        "falling back to client-side transcript embeddings.")
            mode = "client"
            wanted = wanted_for(mode)
            client_name = transcript_index_name
            if client_name not in existing:
                kind_c, def_c = wanted[client_name]
                collection.create_search_index(SearchIndexModel(name=client_name, type=kind_c, definition=def_c))

    updated = []
    for status in inspect_indexes(collection, wanted):
        if not status.drift or status.state == "missing":
            continue
        if update:
            logger.info(f"Updating search index {status.name!r} in place: {'; '.join(status.drift)}")
            collection.update_search_index(status.name, wanted[status.name][1])
            updated.append(status.name)
        else:
            logger.warning(f"Search index {status.name!r} differs from the recommended definition "
                           f"({'; '.join(status.drift)}). Run ensure_indexes(update=True) or "
                           "`cinematlas setup --update` to update it in place.")

    if wait:
        wait_until_queryable(collection, wanted.keys(), timeout_s=timeout_s, poll_interval_s=poll_interval_s,
                             require_ready=updated)
    return mode


def wait_until_queryable(
    collection: Collection,
    names: Iterable[str],
    *,
    timeout_s: float = 600,
    poll_interval_s: float = 5,
    require_ready: Iterable[str] = (),
) -> None:
    """Block until every index is queryable; ``require_ready`` ones must also finish rebuilding."""
    pending = set(names)
    strict = set(require_ready)
    deadline = time.monotonic() + timeout_s
    while pending:
        for ix in collection.list_search_indexes():
            if ix["name"] not in pending:
                continue
            if ix.get("status") == "FAILED":
                raise RuntimeError(f"Search index {ix['name']!r} failed to build: {ix}")
            ready = str(ix.get("status", "")).upper() == "READY"
            if ix.get("queryable") and (ready or ix["name"] not in strict):
                pending.discard(ix["name"])
        if not pending:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Search indexes not queryable after {timeout_s}s: {sorted(pending)}")
        time.sleep(poll_interval_s)
