"""Atlas and Collection: joint-vector search over any records, in a few lines.

    from cinematlas.core import Atlas, Text, Image

    atlas = Atlas()                                   # MONGODB_URI and VOYAGE_API_KEY from the environment
    photos = atlas.collection("nasa.photos", embed=Text("title") + Image("image"),
                              moment="description", filters=["center"], key="nasa_id")
    photos.setup()                                    # the joint vector index; idempotent
    photos.add(records)                               # any iterable of dicts, or a Loader
    photos.search("astronaut repairing a telescope", where={"center": "GSFC"}).top

Every record's parts are embedded together into one vector. Search ranks by that vector; when a
``moment`` field is set, a reranker picks each hit's best sentence without changing the order.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bson.binary import Binary, BinaryVectorDtype
from PIL import Image as PILImage
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from ..exceptions import CinematlasError, DependencyError, SearchError
from ..indexes import definition_drift, wait_until_queryable
from .loaders import Loader
from .parts import EmbedInput, Joint, Part, as_joint, get_field
from .results import Hit, Hits

logger = logging.getLogger("cinematlas")

DEFAULT_MODEL = "voyage-multimodal-3.5"
DEFAULT_RERANK_MODEL = "rerank-2.5"
DEFAULT_DB = "cinematlas"
DIMENSIONS = 1024
VECTOR_PATH = "embedding"
MAX_CANDIDATES = 10_000
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class AddResult:
    """What :meth:`Collection.add` stored."""

    added: int
    skipped: int = 0  # records with nothing to embed
    failed: int = 0  # records whose batch failed every retry
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        extra = "".join(f", {n} {label}" for n, label in ((self.skipped, "skipped"), (self.failed, "failed")) if n)
        return f"Added {self.added} records{extra} in {self.seconds:.1f}s"


class Atlas:
    """A MongoDB Atlas cluster plus a Voyage AI client. Hands out :class:`Collection` objects."""

    def __init__(
        self,
        mongo_uri: str | None = None,
        *,
        voyage_api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        rerank_model: str | None = DEFAULT_RERANK_MODEL,
        db: str = DEFAULT_DB,
        mongo_client: MongoClient | None = None,
        voyage_client: Any = None,
    ):
        self.model = model
        self.rerank_model = rerank_model
        self.db = db
        if mongo_client is None:
            uri = mongo_uri or os.getenv("MONGODB_URI") or os.getenv("MDB_URI")
            if not uri:
                raise CinematlasError("No MongoDB URI: pass mongo_uri or set MONGODB_URI.")
            mongo_client = MongoClient(uri, appname="cinematlas-core")
            self._owns_client = True
        else:
            self._owns_client = False
        self.client = mongo_client
        if voyage_client is None:
            try:
                import voyageai
            except ImportError as e:  # pragma: no cover - core dependency
                raise DependencyError("voyageai is required: pip install voyageai") from e
            voyage_client = voyageai.Client(api_key=voyage_api_key or os.getenv("VOYAGE_API_KEY"))
        self.vo = voyage_client

    def collection(
        self,
        name: str,
        *,
        embed: Part | Joint | None = None,
        moment: str | None = None,
        key: str | None = None,
        filters: Sequence[str] = (),
        like: Loader | type[Loader] | None = None,
        display: Sequence[str] = (),
    ) -> Collection:
        """A searchable collection. ``name`` is ``"collection"`` or ``"db.collection"``.

        ``like=PDFPages`` borrows a loader's suggested ``embed``, ``moment``, ``key`` and ``filters``;
        anything you pass explicitly wins.
        """
        if like is not None:
            embed = embed or like.embed
            moment = moment or like.moment
            key = key or like.key
            filters = filters or like.filters
        if embed is None:
            raise ValueError("Say what to embed, e.g. embed=Text('title') + Image('photo'), or pass like=<Loader>.")
        db, _, coll = name.rpartition(".")
        return Collection(self, self.client[db or self.db][coll], as_joint(embed), moment=moment, key=key,
                          filters=tuple(filters), display=tuple(display))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Atlas:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Collection:
    """Records with one joint vector each, searchable by text or image."""

    def __init__(self, atlas: Atlas, collection: Any, embed: Joint, *, moment: str | None, key: str | None,
                 filters: tuple[str, ...], display: tuple[str, ...]):
        self.atlas = atlas
        self.mongo = collection
        self.embed = embed
        self.moment = moment
        self.key = key
        self.filters = filters
        self.display = display or tuple(f for f in ("title", "name", "source", key) if f)
        self.index_name = f"{collection.name}_joint"

    def __repr__(self) -> str:
        return f"<Collection {self.mongo.database.name}.{self.mongo.name} · {self.embed.describe()}>"

    # ------------------------------------------------------------------ setup
    def index_definition(self, quantization: str | None = "scalar") -> dict[str, Any]:
        vector: dict[str, Any] = {"type": "vector", "path": VECTOR_PATH, "numDimensions": DIMENSIONS,
                                  "similarity": "cosine"}
        if quantization:
            vector["quantization"] = quantization
        return {"fields": [vector, *({"type": "filter", "path": f} for f in self.filters)]}

    def setup(self, *, wait: bool = True, timeout_s: float = 600) -> Collection:
        """Create (or update in place) the joint vector index. Safe to call every time."""
        db = self.mongo.database
        if self.mongo.name not in db.list_collection_names():
            db.create_collection(self.mongo.name)
        if self.key:
            self.mongo.create_index("_key", unique=True, name="cinematlas_key")
        want = self.index_definition()
        existing = {ix["name"]: ix for ix in self.mongo.list_search_indexes()}
        if self.index_name not in existing:
            self.mongo.create_search_index(SearchIndexModel(definition=want, name=self.index_name, type="vectorSearch"))
        elif definition_drift(existing[self.index_name].get("latestDefinition"), want):
            self.mongo.update_search_index(self.index_name, want)
        if wait:
            wait_until_queryable(self.mongo, [self.index_name], timeout_s=timeout_s)
        return self

    # ------------------------------------------------------------------ writing
    def add(self, records: Iterable[Mapping[str, Any]], *, batch_size: int = 16, retries: int = 3,
            progress: Progress | None = None) -> AddResult:
        """Embed and store records (any iterable of dicts, or a :class:`Loader`).

        With a ``key``, adding a record whose key already exists replaces it.
        """
        t0 = time.monotonic()
        added = skipped = failed = done = 0
        errors: list[str] = []
        batch: list[tuple[Mapping[str, Any], list[EmbedInput]]] = []

        def flush() -> None:
            nonlocal added, failed
            vectors, error = self._embed([inputs for _, inputs in batch], "document", retries)
            if vectors is None:
                failed += len(batch)
                errors.append(error or "embedding failed")
            else:
                self._write([self._document(rec, vec) for (rec, _), vec in zip(batch, vectors, strict=True)])
                added += len(batch)
            batch.clear()

        for record in records:
            try:
                inputs = self.embed.inputs(record)
            except Exception as e:  # an unreadable image or bad field fails this record, not the whole add
                failed += 1
                if len(errors) < 20:
                    errors.append(f"{self._label(record)}: {type(e).__name__}: {e}")
                logger.warning(f"Skipping record {self._label(record)}: {e}")
                done += 1
                continue
            if not inputs:
                skipped += 1
            else:
                batch.append((record, inputs))
                if len(batch) >= batch_size:
                    flush()
            done += 1
            if progress:
                progress(done, added)
        if batch:
            flush()
        return AddResult(added, skipped, failed, time.monotonic() - t0, errors)

    def _label(self, record: Mapping[str, Any]) -> str:
        return str(get_field(record, self.key) if self.key else next(iter(record.values()), "?"))[:80]

    def _embed(self, inputs: list[list[EmbedInput]], input_type: str, retries: int) -> tuple[list | None, str | None]:
        error = None
        for attempt in range(1, retries + 1):
            try:
                response = self.atlas.vo.multimodal_embed(inputs=inputs, model=self.atlas.model, input_type=input_type)
                if len(response.embeddings) != len(inputs):
                    got = len(response.embeddings)
                    raise CinematlasError(f"Voyage returned {got} vectors for {len(inputs)} inputs")
                return list(response.embeddings), None
            except Exception as e:  # rate limits and transient network errors: back off and retry
                error = f"{type(e).__name__}: {e}"
                logger.warning(f"Voyage embedding failed (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(1.5**attempt)
        return None, error

    def _document(self, record: Mapping[str, Any], vector: Sequence[float]) -> dict[str, Any]:
        doc = {k: v for k, v in record.items() if _storable(v)}
        doc[VECTOR_PATH] = Binary.from_vector([float(x) for x in vector], BinaryVectorDtype.FLOAT32)
        if self.key:
            value = get_field(record, self.key)
            if value is None:
                raise CinematlasError(f"Record has no {self.key!r} (the collection's key): {list(record)[:8]}")
            doc["_key"] = value
        return doc

    def _write(self, docs: list[dict[str, Any]]) -> None:
        if self.key:
            self.mongo.bulk_write([ReplaceOne({"_key": d["_key"]}, d, upsert=True) for d in docs], ordered=False)
        else:
            self.mongo.insert_many(docs)

    def wait_until_searchable(self, *, timeout_s: float = 120, poll_s: float = 2) -> Collection:
        """Block until every stored record is visible to vector search (Atlas syncs asynchronously)."""
        target = min(self.count({VECTOR_PATH: {"$exists": True}}), MAX_CANDIDATES)
        probe = self.mongo.find_one({VECTOR_PATH: {"$exists": True}}, {VECTOR_PATH: 1})
        if not target or probe is None:
            return self
        stage = {"index": self.index_name, "path": VECTOR_PATH, "queryVector": probe[VECTOR_PATH],
                 "numCandidates": target, "limit": target}
        deadline = time.monotonic() + timeout_s
        while len(list(self.mongo.aggregate([{"$vectorSearch": stage}, {"$project": {"_id": 1}}]))) < target:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Only part of {self.index_name!r} is searchable after {timeout_s}s")
            time.sleep(poll_s)
        return self

    def delete(self, where: Mapping[str, Any] | None = None) -> int:
        """Delete matching records (all of them with no filter). Returns how many."""
        return self.mongo.delete_many(dict(where or {})).deleted_count

    def count(self, where: Mapping[str, Any] | None = None) -> int:
        return self.mongo.count_documents(dict(where or {}))

    # ------------------------------------------------------------------ reading
    def search(self, query: str | Part | Joint | Sequence[str | Part], k: int = 5, *,
               where: Mapping[str, Any] | None = None, moment: bool = True,
               candidates: int | None = None) -> Hits:
        """The ``k`` records whose joint vector best matches ``query``.

        ``query`` can be text, ``Image("photo.jpg")``, or a mix: ``["red shoes", Image("q.jpg")]``.
        ``where`` filters on the collection's ``filters`` fields, e.g. ``{"brand": "Nike"}``.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        inputs = _query_inputs(query)
        if not inputs:
            return Hits(display=self.display)
        vectors, error = self._embed([inputs], "query", retries=2)
        if vectors is None:
            raise SearchError(f"Could not embed the query: {error}")
        stage: dict[str, Any] = {"index": self.index_name, "path": VECTOR_PATH, "queryVector": list(vectors[0]),
                                 "numCandidates": min(candidates or max(k * 20, 100), MAX_CANDIDATES), "limit": k}
        if where:
            unknown = set(where) - set(self.filters)
            if unknown:
                raise ValueError(f"Can only filter on {list(self.filters)} (declared filters); got {sorted(unknown)}")
            stage["filter"] = dict(where)
        pipeline = [{"$vectorSearch": stage}, {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                    {"$project": {VECTOR_PATH: 0}}]
        try:
            rows = list(self.mongo.aggregate(pipeline))
        except PyMongoError as e:
            raise SearchError(f"Vector search failed on {self.index_name!r}: {e}. Did you run .setup()?") from e
        hits = Hits([Hit({**row, "rank": i}) for i, row in enumerate(rows, 1)], display=self.display)
        text = " ".join(x for x in inputs if isinstance(x, str))
        if moment and self.moment and self.atlas.rerank_model and text and hits:
            self._attach_moments(text, hits)
        return hits

    def _attach_moments(self, query: str, hits: Hits) -> None:
        """Rerank every sentence of the hits' moment field; each hit keeps its best. Order is unchanged."""
        owners, sentences = [], []
        for i, hit in enumerate(hits):
            for s in split_sentences(get_field(hit, self.moment)):
                owners.append(i)
                sentences.append(s)
        if not sentences:
            return
        try:
            ranked = self.atlas.vo.rerank(query, sentences, model=self.atlas.rerank_model).results
        except Exception as e:
            logger.warning(f"Rerank failed; results have no moments: {e}")
            return
        for r in ranked:  # best first, so the first sentence seen for a hit is its best
            hit = hits[owners[r.index]]
            if "moment" not in hit:
                hit["moment"] = {"text": sentences[r.index], "relevance": r.relevance_score}

    @classmethod
    def extend(cls, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Add a method to every collection, jQuery ``$.fn`` style::

            @Collection.extend
            def similar(self, record, k=5): ...
        """
        setattr(cls, fn.__name__, fn)
        return fn


def split_sentences(value: Any, *, max_sentences: int = 60, max_chars: int = 500) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    parts = (s.strip() for s in _SENTENCE.split(str(value)))
    return [s[:max_chars] for s in parts if len(s) > 2][:max_sentences]


def _query_inputs(query: Any) -> list[EmbedInput]:
    items = query if isinstance(query, (list, tuple)) else [query]
    out: list[EmbedInput] = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                out.append(item.strip())
        elif isinstance(item, Joint):
            out.extend(x for p in item.parts for x in p.as_query())
        elif isinstance(item, Part):
            out.extend(item.as_query())
        elif isinstance(item, PILImage.Image):
            out.append(item.convert("RGB"))
        else:
            raise TypeError(f"Query items must be text, a Part, or a PIL image; got {type(item).__name__}")
    return out


def _storable(value: Any) -> bool:
    """Keep plain data; drop images and raw bytes (they're in the vector, not the document)."""
    return not isinstance(value, (PILImage.Image, bytes, bytearray))
