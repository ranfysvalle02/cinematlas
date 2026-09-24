"""Test support: synthetic video + recording fakes.

Design notes
------------
* Unit tests never touch the network. External services are replaced with
  small hand-written fakes that *record* what they were asked to do, so tests
  assert on behaviour (what got embedded, what got persisted, in what order)
  rather than on mock call plumbing.
* Video processing is NOT faked: we synthesise a real MP4 with known,
  hard-cut colour scenes, so OpenCV + PySceneDetect run for real and we can
  assert that the right frame ended up in the right scene.
"""

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from pymongo.errors import OperationFailure

FPS = 24
# Distinct BGR colours. NB: ContentDetector weights hue/sat/lum equally, so a pure hue
# cut (red->green) scores ~20 and is *missed* at the default threshold of 27;
# fixtures alternate saturated and white frames to get unambiguous cuts.
COLOURS_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
}


def write_colour_video(path: Path, scenes: Sequence[tuple[str, float]], size=(160, 120)) -> Path:
    """Write an MP4 made of solid-colour segments, e.g. [("red", 1.5), ("blue", 1.5)]."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    assert writer.isOpened(), "OpenCV could not open an mp4v writer"
    for colour, seconds in scenes:
        frame = np.full((size[1], size[0], 3), COLOURS_BGR[colour], dtype=np.uint8)
        for _ in range(int(seconds * FPS)):
            writer.write(frame)
    writer.release()
    return path


def dominant_colour(pil_image) -> str:
    """Name of the COLOURS_BGR entry closest to the image's mean RGB colour."""
    r, g, b = np.asarray(pil_image.convert("RGB")).reshape(-1, 3).mean(axis=0)
    return min(
        COLOURS_BGR,
        key=lambda n: sum((a - c) ** 2 for a, c in zip((b, g, r), COLOURS_BGR[n], strict=True)),
    )


# ---------------------------------------------------------------- fakes


@dataclass
class _EmbedResponse:
    embeddings: list[list[float]]


@dataclass
class FakeVoyage:
    """Records each multimodal_embed call; returns a vector encoding the input's colour.

    ``failures`` = number of calls that raise before succeeding.
    """

    failures: int = 0
    drop_one: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    def multimodal_embed(self, inputs, model, input_type):
        self.calls.append({"inputs": inputs, "model": model, "input_type": input_type,
                           "colours": [dominant_colour(i[0]) if not isinstance(i[0], str) else i[0]
                                       for i in inputs]})
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("429 rate limited")
        vecs = [[float(list(COLOURS_BGR).index(c)) if c in COLOURS_BGR else -1.0]
                for c in self.calls[-1]["colours"]]
        return _EmbedResponse(vecs[:-1] if self.drop_one else vecs)

    rerank_error: Exception | None = None

    def rerank(self, query, documents, model):
        """Relevance = share of query content words present in the document (deterministic)."""
        self.calls.append({"rerank": query, "documents": list(documents), "model": model})
        if self.rerank_error:
            raise self.rerank_error
        q = {w for w in query.lower().split() if len(w) > 3}
        scored = [
            SimpleNamespace(index=i, relevance_score=len(q & set(d.lower().replace(".", "").split())) / max(len(q), 1))
            for i, d in enumerate(documents)
        ]
        return SimpleNamespace(results=sorted(scored, key=lambda r: -r.relevance_score))

    def embed(self, texts, model, input_type):
        """Text embeddings: a 1-d vector = len(text), so tests can tell which text got which vector."""
        self.calls.append({"texts": list(texts), "model": model, "input_type": input_type})
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("429 rate limited")
        return _EmbedResponse([[float(len(t))] for t in texts])


@dataclass
class _InsertManyResult:
    inserted_ids: list[int]


class FakeCollection:
    """Minimal in-memory collection that keeps an ordered operation log.

    Search indexes are simulated; ``autoembed_supported=False`` makes autoEmbed index
    creation fail exactly like Atlas Local does.
    """

    name = "scenes"

    def __init__(self, autoembed_supported: bool = True):
        self.autoembed_supported = autoembed_supported
        self.search_indexes: dict[str, dict] = {}
        self.database = SimpleNamespace(list_collection_names=lambda: ["scenes"], create_collection=lambda n: None)
        self.docs: list[dict[str, Any]] = []
        self.ops: list[tuple[str, Any]] = []
        self.aggregate_error: Exception | None = None
        self.results_by_index: dict[str, list[dict]] = {}  # canned $vectorSearch results per index
        self.index_errors: dict[str, Exception] = {}
        self.native_fusion_rows: list[dict] | None = None  # set to simulate $rankFusion support
        self.native_rerank_rows: list[dict] | None = None  # set to simulate native $rerank support
        self.pipelines: list[list] = []

    def delete_many(self, flt):
        self.ops.append(("delete_many", flt))
        self.docs = [d for d in self.docs if not _matches(d, flt)]

    def insert_many(self, docs):
        self.ops.append(("insert_many", len(docs)))
        self.docs.extend(docs)
        return _InsertManyResult(list(range(len(docs))))

    def insert_one(self, doc):
        self.ops.append(("insert_one", doc.get("status")))
        self.docs.append(doc)

    def create_index(self, keys, **kwargs):
        self.ops.append(("create_index", kwargs.get("name")))

    def list_search_indexes(self, name=None):
        return [{"name": n, "queryable": True, "status": "READY", "latestDefinition": d}
                for n, d in self.search_indexes.items()]

    def update_search_index(self, name, definition):
        self.ops.append(("update_search_index", name))
        self.search_indexes[name] = definition

    def find_one(self, flt=None, projection=None):
        return next((d for d in self.docs if _matches(d, flt or {})), None)

    def count_documents(self, flt):
        return sum(_matches(d, flt) for d in self.docs)

    def distinct(self, key, flt=None):
        return sorted({d.get(key) for d in self.docs if _matches(d, flt or {})})

    def create_search_index(self, model):
        doc = model.document
        if not self.autoembed_supported and any(f["type"] == "autoEmbed" for f in doc["definition"].get("fields", [])):
            raise OperationFailure("CanonicalModel: voyage-4 not registered yet, supported models are: []", code=8)
        self.ops.append(("create_search_index", doc["name"]))
        self.search_indexes[doc["name"]] = doc["definition"]

    def aggregate(self, pipeline):
        self.pipelines.append(pipeline)
        if self.aggregate_error:
            raise self.aggregate_error
        first = pipeline[0]
        if "$rankFusion" in first:
            if self.native_fusion_rows is None:  # behave like a server without $rankFusion
                raise OperationFailure("Unrecognized pipeline stage name: '$rankFusion'", code=40324)
            return iter([dict(r) for r in self.native_fusion_rows])
        if any("$rerank" in stage for stage in pipeline):
            if self.native_rerank_rows is None:  # behave like Atlas with $rerank disabled / Atlas Local
                raise OperationFailure("$rerank is not enabled", code=8000)
            return iter([dict(r) for r in self.native_rerank_rows])
        index = (first.get("$vectorSearch") or first.get("$search") or {}).get("index")
        if index in self.index_errors:
            raise self.index_errors[index]
        if self.results_by_index:
            return iter([dict(d) for d in self.results_by_index.get(index, [])])
        return iter([{"video_id": "v", "scene_id": 0, "score": 0.9}])


def _matches(doc, flt):
    for k, v in flt.items():
        if isinstance(v, dict) and "$exists" in v:
            if (k in doc) != v["$exists"]:
                return False
        elif isinstance(v, dict) and "$ne" in v:
            if doc.get(k) == v["$ne"]:
                return False
        elif doc.get(k) != v:
            return False
    return True


class FakeMongoClient:
    def __init__(self):
        self.collection = FakeCollection()
        self.closed = False
        self.admin = SimpleNamespace(command=lambda cmd: {"version": "9.0.2"} if cmd == "buildInfo" else {"ok": 1})

    def __getitem__(self, _db):
        return _AnyKey(self.collection)

    def close(self):
        self.closed = True


class _AnyKey:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, _):
        return self._value


def env_or_none(*names: str):
    for n in names:
        if os.getenv(n):
            return os.getenv(n)
    return None


def eventually(fn, *, timeout=180, interval=3):
    """autoEmbed and mongot replication are asynchronous: poll until fn() is truthy."""
    deadline = time.monotonic() + timeout
    while True:
        result = fn()
        if result or time.monotonic() > deadline:
            return result
        time.sleep(interval)


# Real-media fixture committed to the repo (847 KiB, public domain). Built by
# tests/fixtures/build_fixture.py from NASA's "The Quiet Crew | Joe Dussling": three
# sentence-aligned segments on unrelated topics joined by hard cuts, so each topic's
# speech lies entirely inside its own scene -> ground truth for scene alignment.
FIXTURES = Path(__file__).parent / "fixtures"
REAL_CLIP = FIXTURES / "x59_quiet_crew.mp4"
REAL_CLIP_SHA256 = "531af1e2db2baaa832a677672001461e229d5d21f9f2833bf30ea1dfb2e44103"
REAL_CLIP_SECONDS = 39.7
REAL_CLIP_TOPICS = {  # topic -> phrases Whisper `small` transcribes verbatim
    "aircraft": ("x-59", "aerodynamics", "sonic boom"),
    "outdoors": ("hiking, kayaking", "disc golf"),
    "beer": ("roasted jalapeno", "medals"),
}


def fake_resolver(table: dict[str, str], default: str = "93.184.216.34"):
    """socket.getaddrinfo stand-in: host -> IP from ``table``, else a public address."""
    import socket

    def getaddrinfo(host, port, *args, **kwargs):
        ip = table.get(host, default)
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 443))]

    return getaddrinfo


def as_list(vec):
    """Decode a stored vector (BSON float32 Binary or plain list) for assertions."""
    from bson.binary import Binary

    if isinstance(vec, Binary):
        return [round(x, 5) for x in vec.as_vector().data]
    return vec
