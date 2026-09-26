"""Result types: plain dicts and lists underneath, pleasant to use on top.

``SearchHit`` is a ``dict`` (JSON-serializable, backwards compatible) with attribute
shortcuts; ``SearchResults`` is a ``list`` that prints as a readable table, renders as
markdown in notebooks, and turns into LLM-ready context in one call.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .retrieval import _clock, to_context


class SearchHit(dict):
    """One scene returned by a search. Every key is also reachable as an attribute."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def start(self) -> float:
        """Second the best-matching sentence starts (scene start when there's no moment)."""
        moment = self.get("moment") or {}
        return float(moment.get("start", self.get("timestamp_start", 0.0)) or 0.0)

    @property
    def timestamp(self) -> str:
        return _clock(self.start)

    @property
    def text(self) -> str:
        """The matching sentence, or the scene transcript."""
        moment = self.get("moment") or {}
        return (moment.get("text") or self.get("transcript") or "").strip()

    @property
    def link(self) -> str | None:
        """Deep link to :attr:`start` (``None`` for uploads)."""
        return self.get("moment_link") or self.get("deep_link")

    def explain(self) -> str:
        """Why this ranked: rank in each source, reranker relevance, fused score."""
        ranks = self.get("ranks") or {}
        parts = [f"{src} #{rank}" for src, rank in sorted(ranks.items(), key=lambda kv: kv[1])]
        if self.get("relevance") is not None:
            parts.append(f"relevance {self['relevance']:.2f}")
        if self.get("score") is not None:
            parts.append(f"score {self['score']:.4f}")
        return ", ".join(parts) or "single-source result"

    def __repr__(self) -> str:
        text = self.text
        snippet = (text[:60] + "…") if len(text) > 60 else text
        return f"<SearchHit {self.get('video_id')}#{self.get('scene_id')} @ {self.timestamp} {snippet!r}>"


class SearchResults(list):
    """Ranked :class:`SearchHit` objects."""

    def __init__(self, hits: Iterable[dict] = ()):
        super().__init__(h if isinstance(h, SearchHit) else SearchHit(h) for h in hits)
        # Set by adaptive search: 0 = "about what's shown", 1 = "about what's said" (None if not routed).
        self.speech_confidence: float | None = None
        self.weights: dict[str, float] = {}  # effective fusion weights used for this query

    @property
    def top(self) -> SearchHit | None:
        return self[0] if self else None

    @property
    def links(self) -> list[str | None]:
        return [h.link for h in self]

    def to_context(self, *, full_scene: bool = False) -> str:
        """Numbered, citable context for any LLM prompt (see :func:`cinematlas.to_context`)."""
        return to_context(self, full_scene=full_scene)

    def __str__(self) -> str:
        if not self:
            return "(no results)"
        lines = []
        for i, h in enumerate(self, 1):
            text = h.text if len(h.text) <= 70 else h.text[:69] + "…"
            lines.append(f"{i:>2}. {h.get('video_id')}#{h.get('scene_id')} @ {h.timestamp:>7}  {text}")
            if h.link:
                lines.append(f"    {h.link}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:  # Jupyter
        rows = ["| # | video | at | moment | link |", "|---|---|---|---|---|"]
        for i, h in enumerate(self, 1):
            text = h.text.replace("|", "\\|")
            link = f"[open]({h.link})" if h.link else ""
            rows.append(f"| {i} | {h.get('video_id')}#{h.get('scene_id')} | {h.timestamp} | {text} | {link} |")
        return "\n".join(rows)


@dataclass(frozen=True)
class IngestResult:
    """What :meth:`Cinematlas.ingest` produced."""

    video_id: str
    scenes: int
    source_type: str  # "url" | "file"
    transcript_mode: str
    seconds: float
    spoken_scenes: int = 0
    stages: dict[str, float] = field(default_factory=dict)  # seconds per pipeline stage
    usage: dict[str, dict[str, int]] = field(default_factory=dict)  # Voyage calls/tokens/pixels, per model

    def __str__(self) -> str:
        return (f"Indexed {self.scenes} scenes ({self.spoken_scenes} with speech) as {self.video_id!r} "
                f"in {self.seconds:.1f}s [{self.transcript_mode}]" + self._usage_note())

    def _usage_note(self) -> str:
        if not self.usage:
            return ""
        calls = sum(m["calls"] for m in self.usage.values())
        tokens = sum(m["total_tokens"] for m in self.usage.values())
        return f", {calls} Voyage calls, {tokens:,} tokens"
