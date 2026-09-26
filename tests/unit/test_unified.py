"""One connection, one retry path, one query builder: video and core share them."""

from types import SimpleNamespace

import pytest
from support import FakeMongoClient, FakeVoyage

from cinematlas import Cinematlas
from cinematlas.core import Atlas, Text
from cinematlas.query import Query, RecordSearch, Search
from cinematlas.usage import with_retries


@pytest.fixture
def atlas():
    return Atlas(mongo_client=FakeMongoClient(), voyage_client=FakeVoyage())


# ---------------------------------------------------------------- one connection
def test_videos_share_the_atlas_connection_voyage_client_and_usage(atlas):
    talks = atlas.videos("media.talks", filters=("course",), ping=False)
    photos = atlas.collection("media.photos", embed=Text("title"))
    assert talks.atlas is atlas and talks.mongo_client is atlas.client
    assert talks.vo is atlas.vo and talks.usage is atlas.usage is photos.atlas.usage
    assert (talks.db_name, talks.collection_name, talks.filters) == ("media", "talks", ("course",))
    assert atlas.videos("scenes", ping=False).db_name == atlas.db  # bare name: the Atlas's default db


def test_closing_a_shared_video_collection_leaves_the_connection_open(atlas):
    atlas.videos("talks", ping=False).close()
    assert not atlas.client.closed
    atlas.close()  # the Atlas didn't create the client, so it isn't its to close either
    assert not atlas.client.closed


def test_a_standalone_engine_owns_its_atlas(fake_mongo, fake_voyage):
    eng = Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False)
    assert isinstance(eng.atlas, Atlas) and eng.atlas.client is fake_mongo and eng.usage is eng.atlas.usage
    records = eng.atlas.collection("notes", embed=Text("body"))
    assert records.atlas.usage is eng.usage  # video and records bill to the same meter
    eng.close()
    assert fake_mongo.closed


# ---------------------------------------------------------------- one retry path
def test_with_retries_retries_failures_and_misaligned_batches(monkeypatch):
    monkeypatch.setattr("cinematlas.usage.time.sleep", lambda _s: None)
    answers = iter([RuntimeError("429"), SimpleNamespace(embeddings=[[1.0]]),  # misaligned: 1 of 2
                    SimpleNamespace(embeddings=[[1.0], [2.0]])])

    def call():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert with_retries(call, expect=2, retries=3) == ([[1.0], [2.0]], None)


def test_with_retries_reports_the_last_error(monkeypatch):
    monkeypatch.setattr("cinematlas.usage.time.sleep", lambda _s: None)
    calls = []

    def call():
        calls.append(1)
        raise RuntimeError("503 unavailable")

    vecs, error = with_retries(call, expect=1, retries=2)
    assert vecs is None and error == "RuntimeError: 503 unavailable" and len(calls) == 2


# ---------------------------------------------------------------- one builder
def test_video_and_record_searches_share_the_builder(atlas, fake_mongo, fake_voyage):
    eng = Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False, filters=("genre",))
    photos = atlas.collection("photos", embed=Text("title"), filters=["center", "year"])
    video, records = eng.search("q"), photos.search("q")
    assert isinstance(video, Search) and isinstance(records, RecordSearch)
    assert isinstance(video, Query) and isinstance(records, Query)
    for q in (video, records):
        assert q.limit(3).k == 3 and q.k == 5  # immutable: refinements are new queries
        assert q.rerank(False).use_rerank is False and q.candidates(40).n_candidates == 40
        with pytest.raises(ValueError, match="positive integer"):
            q.limit(0)
        with pytest.raises(ValueError, match="not a filter field"):
            q.where(nope="x")


def test_record_where_takes_any_mql_value_and_builds_lazily(atlas):
    photos = atlas.collection("photos", embed=Text("title"), filters=["center", "year"])
    q = photos.search("telescope").where(center="GSFC").where(year={"$gte": 2000}).limit(2)
    assert dict(q.filters) == {"center": "GSFC", "year": {"$gte": 2000}}
    assert atlas.vo.calls == []  # nothing ran yet
    assert "where=" in repr(q) and atlas.vo.calls == []  # repr never runs the query


def test_merged_validates_the_fusion_while_building(atlas):
    photos = atlas.collection("photos", embed=Text("title"), late=True)
    assert photos.search("q").merged("sum", depth=20).fusion == "sum"
    with pytest.raises(ValueError, match="fusion must be one of"):
        photos.search("q").merged("magic")


def test_private_attribute_probes_never_run_a_query(atlas):
    q = atlas.collection("photos", embed=Text("title")).search("q")
    assert not hasattr(q, "_ipython_canary_method_should_not_exist_")
    assert atlas.vo.calls == []
