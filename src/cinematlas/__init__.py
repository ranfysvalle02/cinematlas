"""Cinematlas: multimodal video search on MongoDB Atlas + Voyage AI."""

from importlib.metadata import PackageNotFoundError, version

from ._utils import build_deep_link, extract_video_id
from .doctor import Check, Diagnosis
from .engine import Cinematlas
from .exceptions import CinematlasError, DependencyError, IngestionError, IngestionStatus, SearchError
from .indexes import (
    IndexStatus,
    auto_embed_index_definition,
    desired_indexes,
    detect_transcript_mode,
    ensure_search_indexes,
    inspect_indexes,
    transcript_vector_index_definition,
    visual_index_definition,
)
from .results import IngestResult, SearchHit, SearchResults
from .retrieval import to_context

try:
    __version__ = version("cinematlas")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "Check",
    "Cinematlas",
    "CinematlasError",
    "DependencyError",
    "Diagnosis",
    "IndexStatus",
    "IngestResult",
    "IngestionError",
    "IngestionStatus",
    "SearchError",
    "SearchHit",
    "SearchResults",
    "auto_embed_index_definition",
    "build_deep_link",
    "desired_indexes",
    "detect_transcript_mode",
    "ensure_search_indexes",
    "extract_video_id",
    "inspect_indexes",
    "to_context",
    "transcript_vector_index_definition",
    "visual_index_definition",
    "__version__",
]
