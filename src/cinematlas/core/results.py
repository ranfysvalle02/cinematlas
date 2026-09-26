"""Record search results: the shared :class:`cinematlas.results.Hit` / ``Hits``, for any record."""

from __future__ import annotations

from typing import Any

from ..results import Hit, Hits


class RecordHit(Hit):
    """One record returned by a search: its stored fields plus ``rank``, ``score``, ``moment``."""

    def _ident(self) -> Any:  # not a property: a record may have its own "key" field
        return self.get("_key", self.get("_id"))

    @property
    def ranks(self) -> dict[str, int]:
        """Merged searches only: this record's position in each part's list."""
        return self.get("ranks") or {}

    def explain(self) -> str:
        if self.get("ranks"):  # a merged result: its position in each part's list
            parts = [f"merged #{self['rank']}", *(f"{name} #{r}" for name, r in self["ranks"].items())]
        else:
            parts = [f"vector #{self['rank']} (score {self.get('score', 0):.3f})"]
        if self.get("moment"):
            parts.append(f"best sentence relevance {self['moment']['relevance']:.2f}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        snippet = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return f"<RecordHit {self._ident()} rank={self.get('rank')} {snippet!r}>"


class RecordHits(Hits):
    """Ranked :class:`RecordHit` objects, best first."""

    hit_type = RecordHit

    def __init__(self, hits: Any = (), *, display: tuple[str, ...] = ()):
        super().__init__(hits)
        self.display = display  # fields shown when printed

    def _label(self, hit: RecordHit) -> str:
        for name in self.display:
            if hit.get(name):
                return str(hit[name])
        return str(hit._ident())

    def to_context(self) -> str:
        """Numbered, citable excerpts for any LLM prompt."""
        blocks = []
        for i, h in enumerate(self, 1):
            body = h.text or " · ".join(str(h[f]) for f in self.display if h.get(f))
            blocks.append(f"[{i}] {self._label(h)}\n{body}")
        return "\n\n".join(blocks)

    def __str__(self) -> str:
        if not self:
            return "(no results)"
        lines = []
        for i, h in enumerate(self, 1):
            score = f"  ({h['score']:.3f})" if isinstance(h.get("score"), float) else ""
            lines.append(f"{i:>2}. {self._label(h)[:70]}{score}")
            if h.text:
                lines.append(f"    “{h.text[:100]}”")
        return "\n".join(lines)
