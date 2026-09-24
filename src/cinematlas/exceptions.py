"""Exception hierarchy for Cinematlas."""

from enum import Enum


class CinematlasError(Exception):
    """Base exception class for Cinematlas errors."""


class DependencyError(CinematlasError):
    """Raised when a required third-party library is missing."""


class IngestionError(CinematlasError):
    """Raised when video ingestion pipeline encounters an unrecoverable failure."""


class SearchError(CinematlasError):
    """Raised when search execution against MongoDB Atlas fails."""


class IngestionStatus(str, Enum):
    """``status`` of a stored document: a searchable scene, or a tombstone for a failed ingest."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
