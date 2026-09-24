"""Joint-vector search for any records on MongoDB Atlas + Voyage AI.

Fuse in the embedding, not in the ranking: every record's parts (text, images, pages, …) are
embedded together into one vector, and search ranks by it.

    from cinematlas.core import Atlas, Text, Image

    photos = Atlas().collection("nasa.photos", embed=Text("title") + Image("image"), moment="description")
    photos.setup()
    photos.add(records)
    photos.search("astronaut repairing a telescope").top

Extend it with your own :class:`Part` (what goes into the vector) or :class:`Loader` (where records
come from); see :mod:`cinematlas.core.registry`.
"""

from .chunking import Paragraphs, Semantic
from .collection import AddResult, Atlas, Collection, split_sentences
from .evaluate import EvalReport, mcnemar
from .loaders import ImageFolder, JSONLines, Loader, PDFPages, Screenshots, Slides
from .parts import Image, Joint, Part, Text, get_field, load_image
from .registry import plugin, plugins
from .results import Hit, Hits

__all__ = [
    "AddResult",
    "Atlas",
    "Collection",
    "EvalReport",
    "Hit",
    "Hits",
    "Image",
    "ImageFolder",
    "JSONLines",
    "Joint",
    "Loader",
    "PDFPages",
    "Paragraphs",
    "Part",
    "Screenshots",
    "Semantic",
    "Slides",
    "Text",
    "get_field",
    "load_image",
    "mcnemar",
    "plugin",
    "plugins",
    "split_sentences",
]
