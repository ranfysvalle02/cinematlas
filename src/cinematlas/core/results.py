"""Search results: plain dicts and lists underneath, pleasant on top."""

from __future__ import annotations

from typing import Any


class Hit(dict):
    """One record returned by a search. Every key is also an attribute."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def text(self) -> str:
        """The best-matching sentence if the collection has a moment field, else ``""``."""
        return ((self.get("moment") or {}).get("text") or "").strip()

    def explain(self) -> str:
        parts = [f"vector #{self['rank']} (score {self['score']:.3f})"]
        if self.get("moment"):
            parts.append(f"best sentence relevance {self['moment']['relevance']:.2f}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        label = self.get("_key") or self.get("_id")
        snippet = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return f"<Hit {label} score={self.get('score', 0):.3f} {snippet!r}>"


class Hits(list):
    """Ranked :class:`Hit` objects, best first."""

    def __init__(self, hits: Any = (), *, display: tuple[str, ...] = ()):
        super().__init__(h if isinstance(h, Hit) else Hit(h) for h in hits)
        self.display = display  # fields shown when printed

    @property
    def top(self) -> Hit | None:
        return self[0] if self else None

    def _label(self, hit: Hit) -> str:
        for name in self.display:
            if hit.get(name):
                return str(hit[name])
        return str(hit.get("_key") or hit.get("_id"))

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
            lines.append(f"{i:>2}. {self._label(h)[:70]}  ({h.get('score', 0):.3f})")
            if h.text:
                lines.append(f"    “{h.text[:100]}”")
        return "\n".join(lines)
