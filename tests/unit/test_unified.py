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


# ---------------------------------------------------------------- one result type
def test_video_and_record_hits_share_one_base_and_stay_plain_json():
    import json

    from cinematlas import Hit, Hits, SearchHit, SearchResults
    from cinematlas.core import RecordHit, RecordHits

    scene = SearchResults([{"video_id": "v", "scene_id": 2, "score": 0.5, "rank": 1,
                            "moment": {"text": " hi ", "start": 4.0, "end": 5.0, "relevance": 0.9}}])
    record = RecordHits([{"_key": "k", "key": "own field", "score": 0.7, "rank": 1}])
    assert isinstance(scene, Hits) and isinstance(record, Hits)
    assert isinstance(scene.top, SearchHit) and isinstance(record.top, RecordHit) and isinstance(scene.top, Hit)
    top = scene.top
    assert (top.video_id, top.scene_id, top.text, top.moment["relevance"]) == ("v", 2, "hi", 0.9)
    assert scene.top.ranks == {} and scene.top.relevance is None
    assert record.top.key == "own field"  # a record's own fields are never shadowed
    assert (record.top.score, record.top.rank, record.top.moment) == (0.7, 1, None)
    assert json.loads(json.dumps(scene.top)) == dict(scene.top)  # plain dicts underneath


# ---------------------------------------------------------------- async
def test_awaiting_a_query_runs_the_same_search_off_the_event_loop(atlas):
    import asyncio
    import threading

    photos = atlas.collection("photos", embed=Text("title"))
    threads = []
    photos._search = lambda *a, **k: threads.append(threading.get_ident()) or ["hit"]

    async def main():
        q = photos.search("telescope").limit(2)
        first = await q
        again = await q  # cached: no second search
        others = await asyncio.gather(*(photos.search(f"q{i}") for i in range(5)))
        return first, again, others

    first, again, others = asyncio.run(main())
    assert first == again == ["hit"] and others == [["hit"]] * 5
    assert len(threads) == 6 and threading.get_ident() not in threads  # never on the loop's thread


def test_concurrent_reads_of_one_query_run_it_once(atlas):
    from concurrent.futures import ThreadPoolExecutor

    photos = atlas.collection("photos", embed=Text("title"))
    calls = []
    photos._search = lambda *a, **k: calls.append(1) or ["hit"]
    q = photos.search("telescope")
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(lambda _: q.run(), range(32))) == [["hit"]] * 32
    assert len(calls) == 1
