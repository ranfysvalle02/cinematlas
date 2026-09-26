"""OpenTelemetry spans, if ``opentelemetry-api`` is installed (``pip install "cinematlas[otel]"``); no-ops otherwise.

Ingest emits ``cinematlas.ingest`` with one child span per stage (``fetched``, ``scenes``,
``transcribed``, ``embedded``, ``stored``: the same timings as ``IngestResult.stages``). Every search
emits ``cinematlas.search``. Voyage usage for the span is attached as ``cinematlas.voyage.*`` attributes.
Configure an exporter as usual (``opentelemetry-sdk``); cinematlas only uses the API.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

try:
    from opentelemetry import trace as _trace
except ImportError:  # optional
    _trace = None


def enabled() -> bool:
    return _trace is not None


def _tracer() -> Any:
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("cinematlas")
    except PackageNotFoundError:  # pragma: no cover
        v = None
    return _trace.get_tracer("cinematlas", v)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A current span (``None`` without OpenTelemetry); exceptions are recorded on it."""
    if _trace is None:
        yield None
        return
    with _tracer().start_as_current_span(name, attributes=_clean(attributes)) as s:
        yield s


def stage_span(name: str, start_ns: int, end_ns: int, **attributes: Any) -> None:
    """A finished child span for a stage that already ran (ingest times stages as it goes)."""
    if _trace is None:
        return
    s = _tracer().start_span(name, start_time=start_ns, attributes=_clean(attributes))
    s.end(end_time=end_ns)


def set_usage(s: Any, usage: Mapping[str, Mapping[str, int]]) -> None:
    """Attach per-model Voyage usage (from :meth:`Usage.since`) to span ``s``."""
    if s is None or not usage:
        return
    for key in ("calls", "inputs", "total_tokens", "image_pixels"):
        s.set_attribute(f"cinematlas.voyage.{key}", sum(m.get(key, 0) for m in usage.values()))


def _clean(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """OpenTelemetry attributes must be str/bool/int/float (or lists of them); drop the rest."""
    return {f"cinematlas.{k}": v for k, v in attributes.items() if isinstance(v, (str, bool, int, float))}
