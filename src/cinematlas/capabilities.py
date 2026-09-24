"""What this deployment can do, and saying so once: native-stage fallbacks and routing calibration."""

from __future__ import annotations

import logging

from .doctor import explain_native_failure

logger = logging.getLogger("cinematlas")

# Thresholds live on the reranker's score scale, so they're only valid for the model they were
# calibrated on (bench/: speech questions median 0.66, visual 0.42 with rerank-2.5). Uncalibrated
# rerankers step down to fixed fusion rather than guess; pass routing_thresholds=(lo, hi) to calibrate.
ROUTING_CALIBRATION: dict[str, tuple[float, float]] = {"rerank-2.5": (0.45, 0.55)}

_WARNED: set[tuple[str, str]] = set()  # fallbacks already announced in this process


def warn_once(stage: str, error: Exception, server_version: str | None = None) -> None:
    """Explain a native-stage fallback ($rankFusion, $rerank) once per process."""
    reason, fix = explain_native_failure(stage, error, server_version)
    if (stage, reason) not in _WARNED:
        _WARNED.add((stage, reason))
        fallback = "Voyage rerank API (same model)" if stage == "$rerank" else "client-side fusion (same ranking)"
        logger.warning(f"{reason}; using {fallback}. Fix: {fix}")


def say_once(key: tuple[str, str], message: str) -> None:
    """Log ``message`` as a warning the first time ``key`` is seen in this process."""
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


def routing_calibration(rerank_model: str | None, thresholds: tuple[float, float] | None) -> tuple[float, float] | None:
    """``(lo, hi)`` routing thresholds for a reranker, or ``None`` if it has no calibration."""
    if thresholds is not None:
        return thresholds
    if not rerank_model:
        return None
    calibrated = ROUTING_CALIBRATION.get(rerank_model)
    if calibrated is None:
        say_once(("routing", rerank_model),
                 f"No routing calibration for reranker {rerank_model!r}; search() uses fixed fusion. "
                 "Calibrate with bench/ and pass routing_thresholds=(lo, hi), or use rerank-2.5.")
    return calibrated
