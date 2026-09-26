"""Voyage usage metering and retry backoff.

Every Voyage response reports what it consumed (``total_tokens``; multimodal responses add
``text_tokens`` and ``image_pixels``). :class:`MeteredVoyage` wraps the client and adds those up in a
:class:`Usage`, so an ingest or a session can report what it sent and, given your prices, what it cost.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cinematlas")

_COUNTERS = ("text_tokens", "image_pixels", "total_tokens")


@dataclass
class ModelUsage:
    calls: int = 0
    inputs: int = 0
    text_tokens: int = 0
    image_pixels: int = 0
    total_tokens: int = 0


@dataclass
class Usage:
    """Running Voyage totals, per model. Thread-safe; ``snapshot()`` + ``since()`` give per-job deltas."""

    models: dict[str, ModelUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, inputs: int, response: Any) -> None:
        with self._lock:
            m = self.models.setdefault(model, ModelUsage())
            m.calls += 1
            m.inputs += inputs
            for name in _COUNTERS:
                value = getattr(response, name, None)
                if isinstance(value, int):
                    setattr(m, name, getattr(m, name) + value)

    def _models(self) -> list[tuple[str, ModelUsage]]:
        """A consistent copy, safe to iterate while other threads record."""
        with self._lock:
            return [(k, ModelUsage(**vars(v))) for k, v in self.models.items()]

    @property
    def calls(self) -> int:
        return sum(m.calls for _, m in self._models())

    @property
    def total_tokens(self) -> int:
        return sum(m.total_tokens for _, m in self._models())

    @property
    def image_pixels(self) -> int:
        return sum(m.image_pixels for _, m in self._models())

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {k: dict(vars(v)) for k, v in self.models.items()}

    def since(self, before: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
        """Per-model counts added after ``before`` (a :meth:`snapshot`); models with no new calls are omitted."""
        out = {}
        for model, now in self.snapshot().items():
            prev = before.get(model, {})
            delta = {k: v - prev.get(k, 0) for k, v in now.items()}
            if delta["calls"]:
                out[model] = delta
        return out

    def cost(self, prices: Mapping[str, Mapping[str, float]]) -> float:
        """Estimated USD. ``prices`` maps model -> ``{"per_m_tokens": ..., "per_b_pixels": ...}``.

        cinematlas ships no price table (prices change; see https://docs.voyageai.com/docs/pricing).
        For multimodal models, pass ``per_m_tokens`` for text tokens and ``per_b_pixels`` for images;
        for text models ``per_m_tokens`` applies to ``total_tokens``.
        """
        total = 0.0
        for model, m in self._models():
            p = prices.get(model)
            if not p:
                continue
            tokens = m.text_tokens if m.image_pixels or m.text_tokens else m.total_tokens
            total += tokens / 1e6 * p.get("per_m_tokens", 0.0) + m.image_pixels / 1e9 * p.get("per_b_pixels", 0.0)
        return total

    def __str__(self) -> str:
        models = self._models()
        if not models:
            return "Voyage usage: no calls"
        parts = [f"{name}: {m.calls} calls, {m.inputs} inputs, {m.total_tokens:,} tokens"
                 + (f", {m.image_pixels / 1e6:,.1f}M pixels" if m.image_pixels else "")
                 for name, m in models]
        return "Voyage usage: " + "; ".join(parts)


class MeteredVoyage:
    """A Voyage client wrapper that records usage; everything else passes through to the wrapped client."""

    def __init__(self, client: Any, usage: Usage | None = None):
        self.client = client
        self.usage = usage if usage is not None else Usage()

    def embed(self, texts: Any, *args: Any, model: str, **kwargs: Any) -> Any:
        response = self.client.embed(texts, *args, model=model, **kwargs)
        self.usage.record(model, len(texts), response)
        return response

    def multimodal_embed(self, inputs: Any, *args: Any, model: str, **kwargs: Any) -> Any:
        response = self.client.multimodal_embed(inputs, *args, model=model, **kwargs)
        self.usage.record(model, len(inputs), response)
        return response

    def rerank(self, query: str, documents: Any, *args: Any, model: str, **kwargs: Any) -> Any:
        response = self.client.rerank(query, documents, *args, model=model, **kwargs)
        self.usage.record(model, len(documents), response)
        return response

    def __getattr__(self, name: str) -> Any:
        if name in ("client", "usage"):  # not set yet (copy, unpickling): don't recurse
            raise AttributeError(name)
        return getattr(self.client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("client", "usage"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.client, name, value)


def metered(client: Any, usage: Usage | None = None) -> MeteredVoyage:
    """Wrap ``client`` once (idempotent: an already-metered client is returned as is)."""
    return client if isinstance(client, MeteredVoyage) else MeteredVoyage(client, usage)


def is_rate_limit(error: BaseException) -> bool:
    try:
        from voyageai.error import RateLimitError
        if isinstance(error, RateLimitError):
            return True
    except ImportError:  # pragma: no cover - core dependency
        pass
    return "429" in str(error) or "rate limit" in str(error).lower()


def backoff(attempt: int, error: BaseException | None = None, *, cap: float = 30.0) -> float:
    """Seconds to wait before retry ``attempt + 1``, with jitter so parallel workers don't retry in lockstep.

    Rate limits wait longer (2^attempt + 1s floor) than transient errors (1.5^attempt).
    """
    base = 2.0**attempt + 1.0 if error is not None and is_rate_limit(error) else 1.5**attempt
    return min(cap, base * random.uniform(0.75, 1.25))


def with_retries(call: Callable[[], Any], *, expect: int, retries: int = 3,
                 what: str = "Voyage embedding") -> tuple[list[Any] | None, str | None]:
    """The one retry path for every Voyage embedding call.

    Runs ``call()`` up to ``retries`` times, checking it returned ``expect`` embeddings (a misaligned
    batch would put vectors on the wrong records). Returns ``(embeddings, None)``, or ``(None, error)``
    once every attempt failed. Waits between attempts come from :func:`backoff`.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1 (it counts attempts), got {retries!r}")
    error = None
    for attempt in range(1, retries + 1):
        try:
            response = call()
            if len(response.embeddings) != expect:
                raise ValueError(f"Voyage returned {len(response.embeddings)} embeddings for {expect} inputs")
            return list(response.embeddings), None
        except Exception as e:  # rate limits, transient network errors, misaligned batches
            error = f"{type(e).__name__}: {e}"
            logger.warning(f"{what} failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff(attempt, e))
    return None, error
