"""Parts: what goes into a record's joint vector.

A part turns one record into the inputs Voyage's multimodal model embeds (strings and PIL images).
Parts compose with ``+`` into a :class:`Joint`, which embeds every part of a record *together*,
as one interleaved input, so a single vector carries all of them. That's the whole point: fuse in
the embedding, not in the ranking.

Write your own by subclassing :class:`Part` and implementing :meth:`Part.inputs`.
"""

from __future__ import annotations

import copy
import io
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

EmbedInput = str | PILImage.Image
Getter = str | Callable[[Mapping[str, Any]], Any]

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000  # Voyage's per-image limit; larger images are downscaled


def get_field(record: Mapping[str, Any], getter: Getter) -> Any:
    """``"a.b"`` walks nested dicts; a callable receives the whole record."""
    if callable(getter):
        return getter(record)
    value: Any = record
    for key in getter.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


class Part:
    """One ingredient of a joint vector. Subclasses implement :meth:`inputs`."""

    #: Short label used in ``repr`` and by :meth:`Joint.describe`.
    kind = "part"

    def __init__(self, field: Getter):
        self.field = field

    def inputs(self, record: Mapping[str, Any]) -> list[EmbedInput]:
        """The strings and/or images this part contributes for ``record`` (``[]`` if absent)."""
        raise NotImplementedError

    def as_query(self) -> list[EmbedInput]:
        """Inputs when this part *is* the query: ``search(Image("q.jpg"))`` holds a value, not a field."""
        clone = copy.copy(self)
        clone.field = "_"
        return clone.inputs({"_": self.field})

    def __add__(self, other: Part | Joint) -> Joint:
        return Joint([self]) + other

    def __repr__(self) -> str:
        name = self.field if isinstance(self.field, str) else getattr(self.field, "__name__", "fn")
        return f"{type(self).__name__}({name!r})"


class Text(Part):
    """A text field. Lists of strings are joined; long text is truncated to ``max_chars``.

    ``chunk=800`` splits long text into pieces of at most that many characters (paragraphs first) and
    stores one joint vector per piece, each embedded *together with the record's other parts*. Search
    keeps each record's best piece. One vector per record stops working once a part is long (bench/:
    0.93 → 0.54 Hit@1 with the answer 1/32 of the text). Fusing per chunk recovers it: 0.94 with ideal
    boundaries, 0.74 with 1,600-character packed chunks, so prefer chunks close to one idea each.
    """

    kind = "text"

    def __init__(self, field: Getter, *, max_chars: int = 8000, label: str | None = None, chunk: int | None = None):
        super().__init__(field)
        self.max_chars = max_chars
        self.label = label  # e.g. label="Title" embeds "Title: ..." to give the model context
        if chunk is not None and (not isinstance(field, str) or "." in field):
            raise ValueError("chunk= needs a top-level field name, e.g. Text('body', chunk=1500)")
        if chunk is not None and chunk < 100:
            raise ValueError(f"chunk must be at least 100 characters, got {chunk}")
        self.chunk = chunk

    def inputs(self, record: Mapping[str, Any]) -> list[EmbedInput]:
        value = get_field(record, self.field)
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value if v)
        text = str(value).strip()[: self.max_chars]
        if not text:
            return []
        return [f"{self.label}: {text}" if self.label else text]


class Image(Part):
    """An image field: a PIL image, bytes, a local path, or an http(s) URL."""

    kind = "image"

    def __init__(self, field: Getter, *, timeout_s: float = 20):
        super().__init__(field)
        self.timeout_s = timeout_s

    def inputs(self, record: Mapping[str, Any]) -> list[EmbedInput]:
        value = get_field(record, self.field)
        if value is None:
            return []
        image = load_image(value, timeout_s=self.timeout_s)
        return [image] if image is not None else []


def load_image(value: Any, *, timeout_s: float = 20) -> PILImage.Image | None:
    """Open ``value`` as an RGB PIL image, downscaled to Voyage's pixel limit."""
    if isinstance(value, PILImage.Image):
        image = value
    elif isinstance(value, (bytes, bytearray)):
        image = PILImage.open(io.BytesIO(value))
    elif isinstance(value, (str, Path)) and str(value).startswith(("http://", "https://")):
        request = urllib.request.Request(safe_url(str(value)), headers={"User-Agent": "cinematlas"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (http(s) only)
            data = response.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image larger than {MAX_IMAGE_BYTES // 2**20} MiB: {value}")
        image = PILImage.open(io.BytesIO(data))
    elif isinstance(value, (str, Path)) and Path(value).is_file():
        image = PILImage.open(value)
    else:
        return None
    image = image.convert("RGB")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (image.width * image.height)) ** 0.5
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return image


def safe_url(url: str) -> str:
    """Percent-encode characters like spaces that real-world image URLs contain but HTTP forbids."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(parts._replace(path=urllib.parse.quote(parts.path, safe="/%~:@!$&'()*+,;="),
                                                  query=urllib.parse.quote(parts.query, safe="=&%/?:+,;")))


class Joint:
    """Parts embedded together into one vector, in order."""

    def __init__(self, parts: Iterable[Part]):
        self.parts = list(parts)
        if not all(isinstance(p, Part) for p in self.parts):
            raise TypeError("Joint takes Part instances, e.g. Text('title') + Image('photo')")

    def __add__(self, other: Part | Joint) -> Joint:
        extra = other.parts if isinstance(other, Joint) else [other]
        return Joint([*self.parts, *extra])

    def inputs(self, record: Mapping[str, Any]) -> list[EmbedInput]:
        return [item for part in self.parts for item in part.inputs(record)]

    @property
    def chunked(self) -> Text | None:
        """The part that splits long text into chunks, if any (at most one)."""
        found = [p for p in self.parts if isinstance(p, Text) and p.chunk]
        if len(found) > 1:
            raise ValueError("Only one part can use chunk=")
        return found[0] if found else None

    def describe(self) -> str:
        return " + ".join(repr(p) for p in self.parts)

    def __repr__(self) -> str:
        return f"Joint({self.describe()})"


_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_chunks(text: str, size: int) -> list[str]:
    """Pack paragraphs (then sentences, then hard cuts) into pieces of at most ``size`` characters."""
    pieces: list[str] = []
    for para in (p.strip() for p in _PARAGRAPH.split(text or "")):
        units = [para] if len(para) <= size else [s for s in _SENTENCE_END.split(para) if s]
        for unit in units:
            while len(unit) > size:
                pieces.append(unit[:size])
                unit = unit[size:]
            if unit:
                pieces.append(unit)
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 2 + len(piece) <= size:
            chunks[-1] += "\n\n" + piece
        else:
            chunks.append(piece)
    return chunks


def as_joint(spec: Part | Joint) -> Joint:
    return spec if isinstance(spec, Joint) else Joint([spec])
