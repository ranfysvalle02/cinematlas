"""Chunkers: where to cut a long text so each piece stays about one thing.

A chunk that holds two topics dilutes both, and chunk-level fusion only works when chunks are clean
(bench/: 0.94 Hit@1 with one description per chunk, 0.74 when a 1,600-character packer merged two).

* ``Paragraphs(max_chars)``  default. Paragraphs are the author's own topic boundaries: never merge two
                             unless one is tiny; split an over-long one at sentence boundaries.
* ``Semantic(max_chars)``    for text with no paragraph structure (transcripts, OCR, scraped pages):
                             embed every sentence and cut where adjacent sentences stop being similar.

``Text("body", chunk=800)`` means ``Paragraphs(800)``; pass a chunker for anything else:
``Text("body", chunk=Semantic(800))``. A chunker is any callable ``text -> list[str]``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

Chunker = Callable[[str], list[str]]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def _pack(units: Sequence[str], size: int, sep: str = " ") -> list[str]:
    """Greedily join units into pieces of at most ``size`` characters (a unit longer than size is cut)."""
    out: list[str] = []
    for unit in units:
        for piece in (_hard_split(unit, size) if len(unit) > size else [unit]):
            if out and len(out[-1]) + len(sep) + len(piece) <= size:
                out[-1] += sep + piece
            else:
                out.append(piece)
    return out


class Paragraphs:
    """One chunk per paragraph; one-liners (under ``min_chars``) join a neighbour, long ones split at sentences."""

    def __init__(self, max_chars: int = 800, *, min_chars: int = 100):
        if max_chars < 100:
            raise ValueError(f"max_chars must be at least 100, got {max_chars}")
        self.max_chars = max_chars
        self.min_chars = min_chars  # a heading or one-liner, not a short paragraph

    def __call__(self, text: str) -> list[str]:
        chunks: list[str] = []
        for para in (p.strip() for p in _PARAGRAPH.split(text or "")):
            if not para:
                continue
            pieces = [para] if len(para) <= self.max_chars else _pack(sentences(para), self.max_chars)
            for piece in pieces:
                tiny = len(piece) < self.min_chars or (chunks and len(chunks[-1]) < self.min_chars)
                if chunks and tiny and len(chunks[-1]) + 2 + len(piece) <= self.max_chars:
                    chunks[-1] += "\n\n" + piece
                else:
                    chunks.append(piece)
        return chunks

    def __repr__(self) -> str:
        return f"Paragraphs({self.max_chars})"


class Semantic:
    """Cut where the topic changes, found from sentence embeddings, for text without paragraphs.

    Each gap between sentences gets a similarity: cosine between the mean embeddings of the ``window``
    sentences on either side. Every gap below the text's mean similarity is a cut; chunks over
    ``max_chars`` are cut again at their weakest gap, and chunks under ``min_chars`` join the neighbour
    they're most similar to. (Rule and defaults were chosen on documents built from bench/distractors.json,
    which no benchmark question targets: 22% of chunks mixed two topics, vs 38% for fixed 500-character
    chunks at similar granularity. Requiring local minima missed boundaries: 28%.) Embeddings come from
    Voyage (``voyage-4``); inside a collection the chunker borrows the collection's client.
    """

    def __init__(self, max_chars: int = 800, *, min_chars: int = 100, window: int = 2,
                 model: str = "voyage-4", client: Any = None):
        if max_chars < 100:
            raise ValueError(f"max_chars must be at least 100, got {max_chars}")
        self.max_chars, self.min_chars, self.window, self.model = max_chars, min_chars, window, model
        self.client = client

    def __repr__(self) -> str:
        return f"Semantic({self.max_chars})"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.client is None:
            raise ValueError("Semantic chunking needs a Voyage client (a collection provides one automatically)")
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            out += self.client.embed(texts[i:i + 128], model=self.model, input_type="document").embeddings
        return out

    def __call__(self, text: str) -> list[str]:
        units = [piece for s in sentences(text) for piece in _hard_split(s, self.max_chars)]
        if len("".join(units)) <= self.max_chars or len(units) < 3:
            return [" ".join(units)] if units else []
        vectors = self._embed(units)
        w = self.window
        gaps = [_cosine(_mean(vectors[max(0, i + 1 - w):i + 1]), _mean(vectors[i + 1:i + 1 + w]))
                for i in range(len(units) - 1)]  # gaps[i] sits between units[i] and units[i + 1]
        mean = sum(gaps) / len(gaps)
        spans = self._spans(len(units), [i for i, g in enumerate(gaps) if g < mean])
        spans = [s for span in spans for s in self._split_long(span, units, gaps)]
        spans = self._merge_short(spans, units, gaps)
        return [" ".join(units[a:b]) for a, b in spans]

    @staticmethod
    def _spans(n: int, cuts: list[int]) -> list[tuple[int, int]]:
        bounds = [0, *(c + 1 for c in cuts), n]
        return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]

    def _length(self, span: tuple[int, int], units: list[str]) -> int:
        return sum(len(u) + 1 for u in units[span[0]:span[1]])

    def _split_long(self, span: tuple[int, int], units: list[str], gaps: list[float]) -> list[tuple[int, int]]:
        a, b = span
        if self._length(span, units) <= self.max_chars or b - a < 2:
            return [span]
        weakest = min(range(a, b - 1), key=lambda i: gaps[i])
        return self._split_long((a, weakest + 1), units, gaps) + self._split_long((weakest + 1, b), units, gaps)

    def _merge_short(self, spans: list[tuple[int, int]], units: list[str], gaps: list[float]) -> list[tuple[int, int]]:
        spans = list(spans)
        changed = True
        while changed and len(spans) > 1:
            changed = False
            for i, span in enumerate(spans):
                if self._length(span, units) >= self.min_chars:
                    continue
                left = gaps[span[0] - 1] if i > 0 else -math.inf
                right = gaps[span[1] - 1] if i + 1 < len(spans) else -math.inf
                j = i - 1 if left >= right else i + 1
                merged = (min(spans[i][0], spans[j][0]), max(spans[i][1], spans[j][1]))
                if self._length(merged, units) > self.max_chars:
                    continue
                spans[min(i, j)] = merged
                del spans[max(i, j)]
                changed = True
                break
        return spans


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    return [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def as_chunker(chunk: int | Chunker | None) -> Chunker | None:
    if chunk is None or callable(chunk):
        return chunk
    return Paragraphs(int(chunk))
