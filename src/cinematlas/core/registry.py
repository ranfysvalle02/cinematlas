"""Plugin discovery: parts and loaders from other packages.

A package ships a plugin by subclassing :class:`~cinematlas.core.Part` or
:class:`~cinematlas.core.Loader` and declaring an entry point::

    # pyproject.toml of cinematlas-audio
    [project.entry-points."cinematlas.plugins"]
    audio = "cinematlas_audio:Audio"

Then ``cinematlas.core.plugins()`` lists it and ``cinematlas.core.plugin("audio")`` returns it.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from .loaders import ImageFolder, JSONLines, Loader, PDFPages, Screenshots, Slides
from .parts import Image, Part, Text

logger = logging.getLogger("cinematlas")
GROUP = "cinematlas.plugins"
BUILTIN: dict[str, type[Part] | type[Loader]] = {
    "text": Text, "image": Image, "pdf_pages": PDFPages, "image_folder": ImageFolder, "jsonl": JSONLines,
    "slides": Slides, "screenshots": Screenshots,
}


def plugins() -> dict[str, type[Part] | type[Loader]]:
    """Every available part and loader: built-ins plus installed ``cinematlas.plugins`` entry points."""
    found = dict(BUILTIN)
    for ep in entry_points(group=GROUP):
        try:
            obj = ep.load()
        except Exception as e:  # a broken third-party plugin shouldn't break the library
            logger.warning(f"Could not load cinematlas plugin {ep.name!r}: {e}")
            continue
        if isinstance(obj, type) and issubclass(obj, (Part, Loader)):
            found[ep.name] = obj
        else:
            logger.warning(f"cinematlas plugin {ep.name!r} is not a Part or Loader subclass; ignored")
    return found


def plugin(name: str) -> type[Part] | type[Loader]:
    available = plugins()
    if name not in available:
        raise KeyError(f"No cinematlas plugin {name!r}. Available: {sorted(available)}")
    return available[name]
