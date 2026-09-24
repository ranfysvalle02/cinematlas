"""cinematlas.core: parts compose into one joint vector; collections store, index and search it."""

import io
import json
from types import SimpleNamespace

import pytest
from bson.binary import Binary
from PIL import Image as PILImage
from pymongo import ReplaceOne
from pymongo.errors import OperationFailure

import cinematlas.core.collection as core_collection
import cinematlas.core.registry as core_plugins
from cinematlas import SearchError
from cinematlas.core import (
    Atlas,
    Collection,
    Image,
    ImageFolder,
    Joint,
    JSONLines,
    Loader,
    PDFPages,
    Text,
    get_field,
    load_image,
    plugin,
    plugins,
    split_sentences,
)


class Voyage:
    """Vector = [number of strings, number of images] per input; rerank = word overlap."""

    def __init__(self, fail_times=0):
        self.calls, self.fail_times = [], fail_times

    def multimodal_embed(self, inputs, model, input_type):
        self.calls.append(("embed", input_type, inputs))
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("429")
        return SimpleNamespace(embeddings=[[float(sum(isinstance(x, str) for x in i)),
                                            float(sum(not isinstance(x, str) for x in i))] for i in inputs])

    def rerank(self, query, documents, model):
        self.calls.append(("rerank", query, list(documents)))
        q = set(query.lower().split())
        scored = [SimpleNamespace(index=i, relevance_score=len(q & set(d.lower().strip(".").split())) / len(q))
                  for i, d in enumerate(documents)]
        return SimpleNamespace(results=sorted(scored, key=lambda r: -r.relevance_score))


class Mongo:
    """In-memory stand-in for a pymongo collection."""

    def __init__(self, rows=()):
        self.name, self.rows, self.docs, self.ops, self.indexes, self.pipelines = "photos", list(rows), [], [], {}, []
        self.database = SimpleNamespace(name="nasa", list_collection_names=lambda: [],
                                        create_collection=lambda n: self.ops.append(("create_collection", n)))
        self.aggregate_error = None

    def create_index(self, key, **kw):
        self.ops.append(("create_index", key, kw))

    def list_search_indexes(self):
        return [{"name": n, "queryable": True, "status": "READY", "latestDefinition": d}
                for n, d in self.indexes.items()]

    def create_search_index(self, model):
        self.ops.append(("create_search_index", model.document["name"]))
        self.indexes[model.document["name"]] = model.document["definition"]

    def update_search_index(self, name, definition):
        self.ops.append(("update_search_index", name))
        self.indexes[name] = definition

    def insert_many(self, docs):
        self.docs.extend(docs)

    def bulk_write(self, ops, ordered):
        for op in ops:
            assert isinstance(op, ReplaceOne)
            key = op._filter["_key"]
            self.docs = [d for d in self.docs if d.get("_key") != key] + [op._doc]

    def aggregate(self, pipeline):
        self.pipelines.append(pipeline)
        if self.aggregate_error:
            raise self.aggregate_error
        return iter([dict(r) for r in self.rows])

    def delete_many(self, flt):
        return SimpleNamespace(deleted_count=len(self.docs))

    def count_documents(self, flt):
        return len(self.docs)


class Client:
    def __init__(self, coll):
        self.coll = coll

    def __getitem__(self, db):
        self.db = db
        return _Any(self.coll)


class _Any:
    def __init__(self, coll):
        self.coll = coll

    def __getitem__(self, name):
        return self.coll


def png(w=4, h=3, colour="red") -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), colour).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def voyage():
    return Voyage()


@pytest.fixture
def mongo():
    return Mongo()


@pytest.fixture
def atlas(mongo, voyage, monkeypatch):
    monkeypatch.setattr(core_collection.time, "sleep", lambda _s: None)
    return Atlas(mongo_client=Client(mongo), voyage_client=voyage)


@pytest.fixture
def photos(atlas):
    return atlas.collection("nasa.photos", embed=Text("title") + Image("image"), moment="description",
                            key="nasa_id", filters=["center"])


# ------------------------------------------------------------------ parts
def test_parts_compose_into_one_ordered_joint_input():
    spec = Text("title", label="Title") + Image("image") + Text("tags")
    assert isinstance(spec, Joint) and [type(p).__name__ for p in spec.parts] == ["Text", "Image", "Text"]
    inputs = spec.inputs({"title": "Hubble", "image": png(), "tags": ["space", "telescope"]})
    assert inputs[0] == "Title: Hubble" and isinstance(inputs[1], PILImage.Image) and inputs[2] == "space telescope"


def test_missing_or_empty_fields_contribute_nothing():
    spec = Text("title") + Image("image")
    assert spec.inputs({"title": "  "}) == [] and spec.inputs({}) == []


def test_text_is_truncated_and_fields_can_be_nested_or_computed():
    assert Text("a.b", max_chars=5).inputs({"a": {"b": "abcdefgh"}}) == ["abcde"]
    assert Text(lambda r: f"{r['x']}!").inputs({"x": "hi"}) == ["hi!"]
    assert get_field({"a": {"b": 1}}, "a.c") is None


def test_images_load_from_pil_bytes_or_path_and_are_downscaled(tmp_path, monkeypatch):
    path = tmp_path / "p.png"
    path.write_bytes(png(10, 10))
    assert load_image(str(path)).size == (10, 10) and load_image(png()).mode == "RGB"
    monkeypatch.setattr("cinematlas.core.parts.MAX_IMAGE_PIXELS", 25)
    assert load_image(PILImage.new("RGB", (10, 10))).size == (5, 5)
    assert load_image("not a file") is None


def test_a_part_can_be_the_query_itself(tmp_path):
    path = tmp_path / "q.png"
    path.write_bytes(png())
    assert isinstance(Image(str(path)).as_query()[0], PILImage.Image)
    assert Text("red shoes").as_query() == ["red shoes"]


# ------------------------------------------------------------------ collections
def test_collection_needs_an_embed_spec_or_a_loader_to_borrow_from(atlas):
    with pytest.raises(ValueError, match="Say what to embed"):
        atlas.collection("x")
    decks = atlas.collection("decks", like=PDFPages)
    assert decks.moment == "text" and decks.key == "id" and decks.filters == ("source",)
    assert decks.embed.describe() == "Text('text') + Image('image')"


def test_setup_creates_the_joint_index_with_filters_and_a_unique_key(photos, mongo):
    photos.setup()
    assert mongo.ops[0] == ("create_collection", "photos")
    assert ("create_index", "_key", {"unique": True, "name": "cinematlas_key"}) in mongo.ops
    fields = mongo.indexes["photos_joint"]["fields"]
    assert fields[0]["path"] == "embedding" and fields[0]["quantization"] == "scalar"
    assert {"type": "filter", "path": "center"} in fields


def test_setup_is_idempotent_and_updates_a_drifted_index_in_place(photos, mongo):
    photos.setup()
    photos.setup()
    assert [op for op in mongo.ops if op[0] == "create_search_index"] == [("create_search_index", "photos_joint")]
    photos.filters = ("center", "year")
    photos.setup()
    assert ("update_search_index", "photos_joint") in mongo.ops


def test_add_embeds_every_part_together_and_stores_plain_fields(photos, mongo, voyage):
    result = photos.add([{"nasa_id": "a", "title": "Hubble", "image": png(), "description": "A telescope."},
                         {"nasa_id": "b", "title": "", "image": None}])  # nothing to embed
    assert (result.added, result.skipped, result.failed) == (1, 1, 0)
    (_, input_type, inputs), = voyage.calls
    assert input_type == "document" and len(inputs) == 1 and len(inputs[0]) == 2  # one joint input, 2 parts
    doc = mongo.docs[0]
    assert "image" not in doc and doc["_key"] == "a" and isinstance(doc["embedding"], Binary)


def test_adding_the_same_key_replaces_instead_of_duplicating(photos, mongo):
    photos.add([{"nasa_id": "a", "title": "old"}])
    photos.add([{"nasa_id": "a", "title": "new"}])
    assert [d["title"] for d in mongo.docs] == ["new"]


def test_records_missing_the_key_are_rejected_clearly(photos):
    with pytest.raises(Exception, match="nasa_id"):
        photos.add([{"title": "no id"}])


def test_add_batches_and_retries_then_counts_failures(atlas, mongo):
    coll = atlas.collection("photos", embed=Text("title"))
    atlas.vo.fail_times = 1  # first call fails, retry succeeds
    assert coll.add([{"title": str(i)} for i in range(5)], batch_size=2).added == 5
    assert len([c for c in atlas.vo.calls if c[0] == "embed"]) == 4  # 3 batches + 1 retry
    atlas.vo.fail_times = 99
    result = coll.add([{"title": "x"}], retries=2)
    assert (result.added, result.failed) == (0, 1) and "429" in result.errors[0]


def test_add_accepts_a_loader(tmp_path, atlas, mongo):
    (tmp_path / "rows.jsonl").write_text(json.dumps({"title": "a"}) + "\n\n" + json.dumps({"title": "b"}) + "\n")
    assert atlas.collection("photos", embed=Text("title")).add(JSONLines(tmp_path / "rows.jsonl")).added == 2


# ------------------------------------------------------------------ search
ROWS = [{"_key": "a", "title": "Hubble repair", "score": 0.9,
         "description": "Astronauts service the telescope. They replace a gyroscope."},
        {"_key": "b", "title": "Launch", "score": 0.8, "description": "The shuttle lifts off at dawn."}]


def test_search_ranks_by_the_joint_vector_and_filters(atlas, voyage):
    mongo = Mongo(ROWS)
    coll = Atlas(mongo_client=Client(mongo), voyage_client=voyage).collection(
        "photos", embed=Text("title"), filters=["center"])
    hits = coll.search("telescope repair", k=2, where={"center": "GSFC"})
    stage = mongo.pipelines[0][0]["$vectorSearch"]
    assert stage["filter"] == {"center": "GSFC"} and stage["limit"] == 2 and stage["path"] == "embedding"
    assert [h["_key"] for h in hits] == ["a", "b"] and [h.rank for h in hits] == [1, 2]
    assert voyage.calls[0][1] == "query"


def test_filtering_on_an_undeclared_field_fails_fast(photos):
    with pytest.raises(ValueError, match="declared filters"):
        photos.search("x", where={"year": 1990})


def test_the_reranker_picks_each_hits_best_sentence_without_reordering(atlas, voyage):
    coll = Atlas(mongo_client=Client(Mongo(ROWS)), voyage_client=voyage).collection(
        "photos", embed=Text("title"), moment="description")
    hits = coll.search("replace a gyroscope", k=2)
    assert [h["_key"] for h in hits] == ["a", "b"]  # vector order kept
    assert hits[0].text == "They replace a gyroscope." and hits[0].moment["relevance"] == 1.0
    assert "vector #1" in hits[0].explain() and "relevance" in hits[0].explain()


def test_image_queries_skip_moments_and_reranker_outages_degrade(atlas, voyage, tmp_path):
    coll = Atlas(mongo_client=Client(Mongo(ROWS)), voyage_client=voyage).collection(
        "photos", embed=Image("image"), moment="description")
    path = tmp_path / "q.png"
    path.write_bytes(png())
    assert all("moment" not in h for h in coll.search(Image(str(path))))
    voyage.rerank = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503"))
    assert all("moment" not in h for h in coll.search("telescope"))


def test_search_errors_point_at_setup(photos, mongo):
    mongo.aggregate_error = OperationFailure("index not found")
    with pytest.raises(SearchError, match="setup"):
        photos.search("x")


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_invalid_k(photos, bad):
    with pytest.raises(ValueError):
        photos.search("x", k=bad)


def test_empty_query_returns_nothing_without_calling_voyage(photos, voyage):
    assert photos.search("   ") == [] and voyage.calls == []


def test_hits_print_and_become_llm_context(atlas, voyage):
    coll = Atlas(mongo_client=Client(Mongo(ROWS)), voyage_client=voyage).collection(
        "photos", embed=Text("title"), moment="description")
    hits = coll.search("replace a gyroscope", k=2)
    assert str(hits).splitlines()[0].startswith(" 1. Hubble repair")
    assert hits.to_context().startswith("[1] Hubble repair\nThey replace a gyroscope.")


def test_sentences_split_on_punctuation_and_newlines():
    assert split_sentences("One. Two!\nThree? x") == ["One.", "Two!", "Three?"]
    assert split_sentences(None) == []


# ------------------------------------------------------------------ extension
def test_collections_can_be_extended_jquery_style(photos):
    @Collection.extend
    def shout(self):
        return self.index_name.upper()

    try:
        assert photos.shout() == "PHOTOS_JOINT"
    finally:
        del Collection.shout


def test_plugins_include_builtins_and_valid_entry_points(monkeypatch):
    class Audio(Loader):
        def __iter__(self):
            yield {}

    eps = [SimpleNamespace(name="audio", load=lambda: Audio),
           SimpleNamespace(name="broken", load=lambda: (_ for _ in ()).throw(ImportError("nope"))),
           SimpleNamespace(name="junk", load=lambda: 42)]
    monkeypatch.setattr(core_plugins, "entry_points", lambda group: eps)
    found = plugins()
    assert {"text", "image", "pdf_pages", "image_folder", "jsonl", "audio"} <= set(found)
    assert "broken" not in found and "junk" not in found and plugin("audio") is Audio
    with pytest.raises(KeyError, match="Available"):
        plugin("nope")


def test_image_folder_pairs_images_with_caption_files(tmp_path):
    (tmp_path / "a.png").write_bytes(png())
    (tmp_path / "a.txt").write_text("A red square.\n")
    (tmp_path / "notes.md").write_text("ignored")
    (record,) = list(ImageFolder(tmp_path))
    assert record["caption"] == "A red square." and record["name"] == "a"
    assert len(ImageFolder.embed.inputs(record)) == 2


def test_pdf_pages_yields_page_text_and_image(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly results. Revenue grew.")
    doc.save(tmp_path / "deck.pdf")
    (record,) = list(PDFPages(tmp_path / "deck.pdf"))
    assert record["id"] == "deck.pdf#1" and "Revenue grew" in record["text"]
    assert isinstance(record["image"], PILImage.Image)


def test_one_unreadable_record_fails_alone(atlas, mongo):
    coll = atlas.collection("photos", embed=Text("title") + Image("image"), key="id")
    result = coll.add([{"id": "ok", "title": "fine"}, {"id": "bad", "title": "x", "image": b"not an image"},
                       {"id": "ok2", "title": "also fine"}])
    assert (result.added, result.failed) == (2, 1) and result.errors[0].startswith("bad:")


def test_wait_until_searchable_polls_until_every_record_is_visible(atlas, mongo):
    coll = atlas.collection("photos", embed=Text("title"))
    mongo.find_one = lambda flt, proj: {"embedding": [1.0]}
    mongo.count_documents = lambda flt: 3
    seen = iter([[{}], [{}, {}, {}]])
    mongo.aggregate = lambda pipeline: next(seen)
    assert coll.wait_until_searchable(poll_s=0) is coll
