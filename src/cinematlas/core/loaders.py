"""Loaders: turn files into records a collection can add.

A loader is any iterable of dicts. Subclass :class:`Loader` to get a suggested ``embed`` spec and
``moment`` field, so ``atlas.collection("decks", like=PDFPages)`` needs no further configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..exceptions import DependencyError
from .parts import Image, Joint, Part, Text

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


class Loader:
    """Yields records. Class attributes suggest how to embed them."""

    #: Parts that make up the joint vector for these records.
    embed: Part | Joint | None = None
    #: Text field the reranker searches for the best sentence.
    moment: str | None = None
    #: Field that identifies a record, so re-adding replaces it.
    key: str | None = None
    #: Fields worth filtering on.
    filters: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


class PDFPages(Loader):
    """One record per page: the rendered page image and its text. Slides, papers, manuals.

    Needs PyMuPDF: ``pip install 'cinematlas[pdf]'``.
    """

    embed = Text("text") + Image("image")
    moment = "text"
    key = "id"
    filters = ("source",)

    def __init__(self, path: str | Path, *, dpi: int = 110):
        self.path = Path(path)
        self.dpi = dpi

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import pymupdf
        except ImportError as e:
            raise DependencyError("PDFPages needs PyMuPDF: pip install 'cinematlas[pdf]'") from e
        from PIL import Image as PILImage

        with pymupdf.open(self.path) as doc:
            for number, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=self.dpi)
                image = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
                yield {"id": f"{self.path.name}#{number}", "source": self.path.name, "page": number,
                       "text": page.get_text().strip(), "image": image}


class ImageFolder(Loader):
    """One record per image file, with an optional caption from a same-named ``.txt`` file."""

    embed = Image("path") + Text("caption")
    moment = "caption"
    key = "path"
    filters = ("folder",)

    def __init__(self, folder: str | Path, *, recursive: bool = True):
        self.folder = Path(folder)
        self.recursive = recursive

    def __iter__(self) -> Iterator[dict[str, Any]]:
        files = self.folder.rglob("*") if self.recursive else self.folder.glob("*")
        for path in sorted(files):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            caption_file = path.with_suffix(".txt")
            yield {"path": str(path), "name": path.stem, "folder": path.parent.name,
                   "caption": caption_file.read_text().strip() if caption_file.is_file() else None}


class JSONLines(Loader):
    """One record per line of a ``.jsonl`` file. Pair it with your own ``embed`` spec."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
