"""Result types: plain dicts and lists underneath, pleasant to use on top.

Every search returns a :class:`Hits` of :class:`Hit`. A hit is a ``dict`` of the stored document's
fields (so ``json.dumps`` works and any field you stored comes back), plus attribute access and typed
accessors for what every search adds: ``rank``, ``score``, ``moment``, ``text``, ``explain()``.
Video scenes are :class:`SearchHit` (``timestamp``, ``link``, per-source ``ranks``); records from
:mod:`cinematlas.core` are :class:`cinematlas.core.RecordHit`.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from .retrieval import _clock, to_context


class Moment(TypedDict, total=False):
    """The best-matching sentence inside a hit, picked by the reranker."""

    text: str
    start: float  # video: seconds
    end: float
    relevance: float


class Hit(dict):
    """One search result: the stored document's fields, plus what the search added. Keys are attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def moment(self) -> Moment | None:
        """The best-matching sentence (``None`` without a reranker or a moment field)."""
        return self.get("moment")

    @property
    def score(self) -> float | None:
        return self.get("score")

    @property
    def rank(self) -> int | None:
        return self.get("rank")

    @property
    def text(self) -> str:
        """The best-matching sentence, or ``""``."""
        return ((self.get("moment") or {}).get("text") or "").strip()

    def explain(self) -> str:
        """Why this ranked."""
        return f"score {self['score']:.4f}" if self.get("score") is not None else "unscored"


class Hits(list):
    """Ranked :class:`Hit` objects, best first."""

    hit_type: type[Hit] = Hit

    def __init__(self, hits: Iterable[dict] = ()):
        super().__init__(h if isinstance(h, self.hit_type) else self.hit_type(h) for h in hits)

    @property
    def top(self) -> Any:
        """The best hit, or ``None``."""
        return self[0] if self else None

    def to_context(self) -> str:
        """Numbered, citable excerpts for any LLM prompt."""
        return "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(self, 1))


class SearchHit(Hit):
    """One video scene: ``video_id``, ``scene_id``, the scene's transcript, and the moment that answers."""

    @property
    def video_id(self) -> str:
        return self.get("video_id", "")

    @property
    def scene_id(self) -> int:
        return self.get("scene_id", 0)

    @property
    def ranks(self) -> dict[str, int]:
        """This scene's position in each source's list (``scene``, ``visual``, ``transcript``, ``text``, ``rerank``)."""
        return self.get("ranks") or {}

    @property
    def relevance(self) -> float | None:
        """The reranker's relevance for the moment."""
        return self.get("relevance")

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


class SearchResults(Hits):
    """Ranked :class:`SearchHit` objects."""

    hit_type = SearchHit

    def __init__(self, hits: Iterable[dict] = ()):
        super().__init__(hits)
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
