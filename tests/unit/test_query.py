"""The search builder: immutable, lazy, and filterable by video and declared metadata."""

import shutil

import pytest
from support import fake_resolver

from cinematlas import Cinematlas
from cinematlas._utils import build_match, build_text_search_stage
from cinematlas.indexes import definition_drift, desired_indexes, text_index_definition, visual_index_definition
from cinematlas.query import Search


@pytest.fixture
def filtered(fake_mongo, fake_voyage, monkeypatch):
    monkeypatch.setattr("cinematlas.usage.time.sleep", lambda _s: None)
    monkeypatch.setattr("cinematlas.urlsafety.socket.getaddrinfo", fake_resolver({}))
    return Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False, filters=("genre", "lang"))


# ---------------------------------------------------------------- immutability and laziness
def test_refining_a_query_never_changes_the_base(filtered):
    base = filtered.search("sonic boom")
    three, one_video = base.limit(3), base.video("v")
    assert isinstance(base, Search) and (base.k, base.videos) == (5, ())
    assert (three.k, three.videos) == (3, ()) and (one_video.k, one_video.videos) == (5, ("v",))


def test_nothing_runs_until_the_query_is_read(filtered, fake_voyage, fake_mongo):
    query = filtered.search("sonic boom").only("visual").limit(2)
    assert fake_voyage.calls == [] and fake_mongo.collection.pipelines == []
    first = list(query)
    assert len(fake_mongo.collection.pipelines) == 1
    assert query.run() is query.run() and list(query) == first and len(query) == len(first)
    assert len(fake_mongo.collection.pipelines) == 1  # later reads reuse the results


def test_refined_copies_do_not_share_results(filtered, fake_mongo):
    base = filtered.search("sonic boom").only("visual")
    base.run()
    base.limit(1).run()
    assert len(fake_mongo.collection.pipelines) == 2


def test_video_calls_accumulate_without_duplicates(filtered):
    assert filtered.search("q").video("a").video("b", "a").videos == ("a", "b")
    assert filtered.search("q").video("").videos == ()  # an empty id means "any video" (CLI, demo)


# ---------------------------------------------------------------- where()
def test_where_requires_declared_fields(filtered):
    with pytest.raises(ValueError, match="filters=\\('year'"):
        filtered.search("q").where(year="1999")


@pytest.mark.parametrize("value", [["a", 3], [], 3, None, b"bytes"])
def test_where_takes_strings_only(filtered, value):
    with pytest.raises(ValueError, match="string"):
        filtered.search("q").where(genre=value)


def test_where_merges_and_later_values_win(filtered):
    query = filtered.search("q").where(genre="action").where(lang=["en", "es"]).where(genre="drama")
    assert query.match == {"metadata.genre": "drama", "metadata.lang": {"$in": ["en", "es"]}}


def test_build_match():
    assert build_match() == {}
    assert build_match(["a"]) == {"video_id": "a"}
    assert build_match(["a", "b"], {"genre": "x", "lang": ["en", "es"]}) == {
        "video_id": {"$in": ["a", "b"]}, "metadata.genre": "x", "metadata.lang": {"$in": ["en", "es"]}}


def test_text_stage_filters_with_equals_and_in():
    stage = build_text_search_stage("ix", "q", {"video_id": "a", "metadata.lang": {"$in": ["en", "es"]}})
    assert stage["$search"]["compound"]["filter"] == [
        {"equals": {"path": "video_id", "value": "a"}},
        {"in": {"path": "metadata.lang", "value": ["en", "es"]}},
    ]


def test_every_source_is_pre_filtered(filtered, fake_mongo):
    filtered.search("sonic boom").video("v").where(genre="action").using("visual", "text").rerank(False).run()
    stages = [p[0] for p in fake_mongo.collection.pipelines]
    vector = next(s["$vectorSearch"] for s in stages if "$vectorSearch" in s)
    text = next(s["$search"] for s in stages if "$search" in s)
    assert vector["filter"] == {"video_id": "v", "metadata.genre": "action"}
    assert text["compound"]["filter"] == [{"equals": {"path": "video_id", "value": "v"}},
                                          {"equals": {"path": "metadata.genre", "value": "action"}}]


def test_unknown_filter_names_are_rejected_at_construction(fake_mongo, fake_voyage):
    with pytest.raises(ValueError, match="identifiers"):
        Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False, filters=("not a name",))


# ---------------------------------------------------------------- ingest metadata
def test_ingest_metadata_lands_on_every_scene(filtered, fake_mongo, colour_video, monkeypatch):
    monkeypatch.setattr(filtered, "_download_and_extract_media",
                        lambda url, d: (shutil.copy(colour_video, f"{d}/in.mp4"), None))
    monkeypatch.setattr(filtered, "_transcribe_audio_safe", lambda _p: [])
    result = filtered.ingest("www.b.com/v.mp4", video_id="talk", metadata={"genre": "action"})
    scenes = [d for d in fake_mongo.collection.docs if d.get("video_id") == "talk"]
    assert result.scenes == len(scenes) == 3
    assert all(d["metadata"] == {"genre": "action"} for d in scenes)


def test_file_ingest_metadata_too(filtered, fake_mongo, colour_video, monkeypatch):
    monkeypatch.setattr(filtered, "_transcribe_audio_safe", lambda _p: [])
    filtered.ingest(str(colour_video), metadata={"genre": "doc"})
    assert {d["metadata"]["genre"] for d in fake_mongo.collection.docs} == {"doc"}


# ---------------------------------------------------------------- indexes
def test_vector_indexes_declare_metadata_filter_fields():
    fields = visual_index_definition(filters=("genre",))["fields"]
    assert {"type": "filter", "path": "metadata.genre"} in fields
    assert {"type": "filter", "path": "video_id"} in fields
    for _kind, definition in desired_indexes("client", filters=("genre",)).values():
        if "fields" in definition:
            assert {"type": "filter", "path": "metadata.genre"} in definition["fields"]


def test_text_index_maps_metadata_as_tokens():
    fields = text_index_definition(("genre", "lang"))["mappings"]["fields"]
    assert fields["metadata"] == {"type": "document", "dynamic": False,
                                  "fields": {"genre": {"type": "token"}, "lang": {"type": "token"}}}
    assert "metadata" not in text_index_definition()["mappings"]["fields"]


def test_server_added_nested_options_are_not_drift():
    want = text_index_definition(("genre",))
    have = {"mappings": {"dynamic": False, "fields": {
        "transcript": {"type": "string", "analyzer": "lucene.english", "norms": "include"},
        "video_id": {"type": "token", "normalizer": "none"},
        "metadata": {"type": "document", "dynamic": False,
                     "fields": {"genre": {"type": "token", "normalizer": "none"}}},
    }}}
    assert definition_drift(have, want) == ()
    del have["mappings"]["fields"]["metadata"]["fields"]["genre"]
    assert definition_drift(have, want) == ("mappings.fields.metadata.fields.genre: None -> {'type': 'token'}",)


def test_ensure_indexes_uses_the_engine_filters(filtered, fake_mongo):
    filtered.ensure_indexes(wait=False)
    defs = fake_mongo.collection.search_indexes
    assert {"type": "filter", "path": "metadata.lang"} in defs["cinematlas_vector_index"]["fields"]
    assert "metadata" in defs["cinematlas_text_index"]["mappings"]["fields"]


def test_removing_a_declared_filter_is_reported_as_drift():
    from cinematlas.indexes import definition_drift, text_index_definition, visual_index_definition

    both, one = ("genre", "year"), ("genre",)
    assert definition_drift(visual_index_definition(filters=both), visual_index_definition(filters=one)) == (
        "metadata.year: filter no longer declared",)
    assert definition_drift(text_index_definition(filters=both), text_index_definition(filters=one)) == (
        "mappings.fields.metadata.fields.year: filter no longer declared",)
    assert definition_drift(visual_index_definition(filters=one), visual_index_definition(filters=one)) == ()
