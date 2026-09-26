"""The one way to search, for video and for any records: ``.search(query)`` returns a lazy, immutable query.

    engine.search("dog catching frisbee").video("abc").where(genre="action").limit(5).top.link
    photos.search("astronaut repairing a telescope").where(center="GSFC").limit(3).top

Each builder method returns a new query, so a base query can be shared and refined safely. A query
runs the first time it is read and reads like the list it produces (iterate, index, ``len``, ``.top``,
``.to_context()``); ``.run()`` returns that list itself.

:class:`Query` holds what every search shares (``limit``, ``where``, ``rerank``, ``candidates``, and
reading); :class:`Search` adds the video sources, :class:`RecordSearch` the merged-rankings baseline.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from . import search as _search
from ._utils import build_match
from .indexes import DEFAULT_AUTO_INDEX, DEFAULT_SCENE_INDEX, DEFAULT_TRANSCRIPT_INDEX, DEFAULT_VISUAL_INDEX
from .results import SearchResults

if TYPE_CHECKING:
    from .core.collection import Collection
    from .engine import Cinematlas

ROUTINGS = ("scene", "adaptive", "fixed")


class Query:
    """Shared behaviour of every search. Subclasses are frozen dataclasses with the fields
    ``k``, ``filters``, ``use_rerank``, ``n_candidates`` and ``_cache``, and implement ``_execute``,
    ``_declared_filters`` and ``_normalize``."""

    k: int
    filters: tuple[tuple[str, Any], ...]
    _cache: list[Any]

    # ------------------------------------------------------------------ builders
    def _with(self, **changes: Any) -> Any:
        return dataclasses.replace(self, _cache=[], **changes)  # type: ignore[type-var]

    def limit(self, k: int) -> Any:
        """How many results to return (default 5)."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"limit must be a positive integer, got {k!r}")
        return self._with(k=k)

    def where(self, **fields: Any) -> Any:
        """Only results whose declared filter fields match. Calling it again adds or overrides fields."""
        declared = self._declared_filters()
        normalized = {}
        for name, value in fields.items():
            if name not in declared:
                raise ValueError(f"{name!r} is not a filter field (declared: {list(declared)}). "
                                 f"{self._declare_hint(name)}")
            normalized[name] = self._normalize(name, value)
        return self._with(filters=tuple({**dict(self.filters), **normalized}.items()))

    def rerank(self, enabled: bool = True) -> Any:
        """Sentence-level reranking (on by default); it picks each result's exact moment."""
        return self._with(use_rerank=enabled)

    def candidates(self, n: int) -> Any:
        """How many candidates to retrieve before the final ranking."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"candidates must be a positive integer, got {n!r}")
        return self._with(n_candidates=n)

    # ------------------------------------------------------------------ hooks
    def _declared_filters(self) -> Sequence[str]:
        raise NotImplementedError

    def _declare_hint(self, name: str) -> str:
        return ""

    def _normalize(self, name: str, value: Any) -> Any:
        return value

    def _execute(self) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------ execution and reading
    def run(self) -> Any:
        """Execute (once; later calls return the same results)."""
        if not self._cache:
            self._cache.append(self._execute())
        return self._cache[0]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.run())

    def __len__(self) -> int:
        return len(self.run())

    def __getitem__(self, i: Any) -> Any:
        return self.run()[i]

    def __bool__(self) -> bool:
        return bool(self.run())

    def __eq__(self, other: object) -> bool:
        return self.run() == (other.run() if isinstance(other, Query) else other)

    __hash__ = None  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:  # .top, .links, .to_context(), .speech_confidence, …
        if name.startswith("_"):  # never run a query for a private or dunder probe (copy, pickle, IPython)
            raise AttributeError(name)
        return getattr(self.run(), name)

    def __str__(self) -> str:
        return str(self.run())

    def _repr_markdown_(self) -> str | None:
        render = getattr(self.run(), "_repr_markdown_", None)
        return render() if render else None

    def _describe(self) -> list[str]:
        return []

    def __repr__(self) -> str:  # side-effect free: never runs the query
        parts = [*self._describe()]
        if self.filters:
            parts.append(f"where={dict(self.filters)}")
        parts.append(f"limit={self.k}")
        return f"<{type(self).__name__} {' '.join(parts)}>"


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class Search(Query):
    """A question plus how to answer it. Build it with the chainable methods; it runs when read."""

    engine: Cinematlas
    text: str
    k: int = 5
    videos: tuple[str, ...] = ()
    filters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    source: str | None = None  # a single source, no fusion
    sources: tuple[str, ...] = _search.SOURCES
    routing_mode: str | None = None
    fusion_weights: tuple[tuple[str, float], ...] | None = None
    use_rerank: bool = True
    n_candidates: int | None = None
    _cache: list[SearchResults] = dataclasses.field(default_factory=list, repr=False)

    def video(self, *video_ids: str) -> Search:
        """Only these videos. Calling it again adds to the set."""
        return self._with(videos=tuple(dict.fromkeys((*self.videos, *(v for v in video_ids if v)))))

    def only(self, source: str) -> Search:
        """One source alone, no fusion or reranking: ``"scene"``, ``"visual"``, ``"transcript"`` or ``"text"``."""
        if source not in _search.SOURCES:
            raise ValueError(f"source must be one of {_search.SOURCES}, got {source!r}")
        return self._with(source=source)

    def using(self, *sources: str) -> Search:
        """Fuse only these sources (implies fusion rather than scene-first ranking)."""
        unknown = set(sources) - set(_search.SOURCES)
        if not sources or unknown:
            raise ValueError(f"sources must be among {_search.SOURCES}, got {sources!r}")
        return self._with(sources=tuple(sources))

    def routing(self, mode: str) -> Search:
        """``"scene"`` (the default when scene vectors exist), ``"adaptive"`` or ``"fixed"`` fusion."""
        if mode not in ROUTINGS:
            raise ValueError(f"routing must be one of {ROUTINGS}, got {mode!r}")
        return self._with(routing_mode=mode)

    def adaptive(self) -> Search:
        """Fuse every source, weighted per question by how much it's about speech vs. picture."""
        return self.routing("adaptive")

    def weights(self, **weights: float) -> Search:
        """Fixed-weight fusion; unspecified sources keep :data:`~cinematlas.search.DEFAULT_WEIGHTS`."""
        return self._with(fusion_weights=tuple(weights.items()))

    # ------------------------------------------------------------------ execution
    @property
    def match(self) -> dict[str, Any]:
        """The pre-filter every source applies."""
        return build_match(self.videos, {k: list(v) for k, v in self.filters})

    def _execute(self) -> SearchResults:
        eng, text, match = self.engine, self.text, self.match or None
        if not text:
            return SearchResults()
        if self.source == "text":
            return _search.search_text(eng, text, self.k, match)
        if self.source == "transcript":
            if eng.transcript_mode == "autoembed":
                return _search.search_auto_transcript(eng, text, self.k, DEFAULT_AUTO_INDEX, match)
            return _search.search_client_transcript(eng, text, self.k, DEFAULT_TRANSCRIPT_INDEX, match)
        if self.source in ("visual", "scene"):
            index, path = ((DEFAULT_VISUAL_INDEX, "visual_embedding") if self.source == "visual"
                           else (DEFAULT_SCENE_INDEX, "scene_embedding"))
            return eng._vector_search(eng._embed_multimodal_query(text), index_name=index, path=path,
                                      top_k=self.k, match=match)
        return _search.search(eng, text, self.k, match, sources=self.sources, rerank=self.use_rerank,
                              candidates=self.n_candidates,
                              weights=dict(self.fusion_weights) if self.fusion_weights is not None else None,
                              routing=self.routing_mode)

    # ------------------------------------------------------------------ Query hooks
    def _declared_filters(self) -> Sequence[str]:
        return self.engine.filters

    def _declare_hint(self, name: str) -> str:
        return f"Declare it: Cinematlas(filters=({name!r}, ...)), then engine.ensure_indexes(update=True)."

    def _normalize(self, name: str, value: Any) -> tuple[str, ...]:
        """Scene metadata filters are strings (they also filter full-text search, which indexes them as tokens)."""
        values = (value,) if isinstance(value, str) else tuple(value) if isinstance(value, (list, tuple)) else ()
        if not values or not all(isinstance(v, str) for v in values):
            raise ValueError(f"where({name}=...) takes a string or a non-empty list of strings, got {value!r}")
        return values

    def _describe(self) -> list[str]:
        parts = [repr(self.text)]
        if self.videos:
            parts.append(f"video={list(self.videos)}")
        if self.source:
            parts.append(f"only={self.source!r}")
        return parts


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class RecordSearch(Query):
    """A search over a :class:`cinematlas.core.Collection`, ranked by each record's joint vector."""

    collection: Collection
    query: Any  # text, Image(...), a PIL image, or a list mixing them
    k: int = 5
    filters: tuple[tuple[str, Any], ...] = ()
    use_rerank: bool = True
    n_candidates: int | None = None
    fusion: str | None = None  # set by .merged(): rank each part separately, then merge
    depth: int = 50
    _cache: list[Any] = dataclasses.field(default_factory=list, repr=False)

    def merged(self, fusion: str = "rrf", *, depth: int = 50) -> RecordSearch:
        """The usual design, for comparison: search each part's own vector, then merge the rankings.

        ``fusion`` is ``"rrf"``, ``"sum"``, ``"mnz"`` or ``"max"`` (see :mod:`cinematlas.core.fusion`); needs
        ``late=True`` on the collection. Each hit's ``ranks`` shows its position in every part's list.
        """
        from .core.fusion import MERGES

        if fusion not in MERGES:
            raise ValueError(f"fusion must be one of {sorted(MERGES)}, got {fusion!r}")
        return self._with(fusion=fusion, depth=depth)

    def _declared_filters(self) -> Sequence[str]:
        return self.collection.filters

    def _declare_hint(self, name: str) -> str:
        return f"Declare it: atlas.collection(..., filters=[{name!r}, ...]), then .setup()."

    def _execute(self) -> Any:
        where = dict(self.filters) or None
        if self.fusion:
            return self.collection._search_merged(self.query, self.k, where=where, depth=self.depth,
                                                  fusion=self.fusion)
        return self.collection._search(self.query, self.k, where=where, moment=self.use_rerank,
                                       candidates=self.n_candidates)

    def _describe(self) -> list[str]:
        return [repr(self.query), *([f"merged={self.fusion!r}"] if self.fusion else [])]
