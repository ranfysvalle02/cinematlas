"""Transcript backend selection: Atlas autoEmbed when available, client-side voyage-4 otherwise."""

import shutil

import pytest
from pymongo.errors import OperationFailure
from support import FakeCollection, FakeMongoClient, FakeVoyage, as_list

from cinematlas import Cinematlas, ensure_search_indexes

URL = "https://www.youtube.com/watch?v=abcdefghijk"


def make_engine(monkeypatch, *, autoembed_supported=True, **kwargs):
    mongo = FakeMongoClient()
    mongo.collection = FakeCollection(autoembed_supported=autoembed_supported)
    voyage = FakeVoyage()
    monkeypatch.setattr("cinematlas.usage.time.sleep", lambda _s: None)
    return Cinematlas(mongo_client=mongo, voyage_client=voyage, ping=False, **kwargs), mongo.collection, voyage


# ---------------------------------------------------------------- index setup / detection
def test_auto_mode_uses_autoembed_when_cluster_supports_it(monkeypatch):
    eng, coll, _ = make_engine(monkeypatch)
    assert eng.ensure_indexes(wait=False) == "autoembed"
    assert set(coll.search_indexes) == {"cinematlas_vector_index", "cinematlas_scene_index", "cinematlas_text_index",
                                       "cinematlas_auto_index"}
    assert coll.search_indexes["cinematlas_auto_index"]["fields"][0]["model"] == "voyage-4"


def test_auto_mode_falls_back_to_client_vectors_when_autoembed_is_rejected(monkeypatch):
    eng, coll, _ = make_engine(monkeypatch, autoembed_supported=False)
    assert eng.ensure_indexes(wait=False) == "client"
    assert set(coll.search_indexes) == {"cinematlas_vector_index", "cinematlas_scene_index",
                                       "cinematlas_text_index", "cinematlas_transcript_index"}
    assert eng.transcript_mode == "client"


def test_explicit_autoembed_mode_does_not_silently_fall_back(monkeypatch):
    eng, _, _ = make_engine(monkeypatch, autoembed_supported=False, transcript_mode="autoembed")
    with pytest.raises(OperationFailure):
        eng.ensure_indexes(wait=False)


def test_existing_client_index_is_respected_even_if_autoembed_is_available(monkeypatch):
    # Switching backends under existing data would orphan stored vectors; auto must not do that.
    coll = FakeCollection()
    ensure_search_indexes(coll, transcript_mode="client", wait=False)
    assert ensure_search_indexes(coll, wait=False) == "client"
    assert "cinematlas_auto_index" not in coll.search_indexes


def test_mode_is_detected_from_existing_indexes(monkeypatch):
    eng, coll, _ = make_engine(monkeypatch)
    ensure_search_indexes(coll, transcript_mode="autoembed", wait=False)
    assert eng.transcript_mode == "autoembed"


def test_mode_defaults_to_client_before_any_index_exists_and_is_not_cached(monkeypatch):
    eng, coll, _ = make_engine(monkeypatch)
    assert eng.transcript_mode == "client"
    ensure_search_indexes(coll, transcript_mode="autoembed", wait=False)
    assert eng.transcript_mode == "autoembed"


def test_ensure_indexes_creates_the_video_lookup_index(monkeypatch):
    eng, coll, _ = make_engine(monkeypatch)
    eng.ensure_indexes(wait=False)
    assert ("create_index", "video_scene") in coll.ops


def test_ensure_indexes_is_idempotent(monkeypatch):
    eng, coll, _ = make_engine(monkeypatch, autoembed_supported=False)
    eng.ensure_indexes(wait=False)
    eng.ensure_indexes(wait=False)
    assert [op for op in coll.ops if op[0] == "create_search_index"] == [
        ("create_search_index", "cinematlas_vector_index"),
        ("create_search_index", "cinematlas_scene_index"),
        ("create_search_index", "cinematlas_text_index"),
        ("create_search_index", "cinematlas_transcript_index"),
    ]


def test_invalid_mode_is_rejected_up_front(monkeypatch):
    with pytest.raises(ValueError):
        make_engine(monkeypatch, transcript_mode="bogus")


# ---------------------------------------------------------------- ingest + search per mode
@pytest.fixture
def local_serve(colour_video):
    def install(eng):
        eng._download_and_extract_media = lambda url, d: (shutil.copy(colour_video, f"{d}/input.mp4"), None)
        eng._transcribe_audio_safe = lambda _p: [
            {"start": 0.1, "end": 1.0, "text": "short"},
            {"start": 3.1, "end": 4.0, "text": "a longer sentence"},
        ]
    return install


def test_client_mode_stores_transcript_vectors_aligned_per_scene(monkeypatch, local_serve):
    eng, coll, voyage = make_engine(monkeypatch, transcript_mode="client", text_model="voyage-4-lite")
    local_serve(eng)
    eng.ingest(URL)

    by_scene = {d["scene_id"]: d for d in coll.docs}
    # FakeVoyage.embed encodes len(text), so each vector proves which transcript it belongs to.
    assert as_list(by_scene[0]["transcript_embedding"]) == [float(len("short"))]
    assert by_scene[1]["transcript_embedding"] is None  # silent scene: no wasted API call
    assert as_list(by_scene[2]["transcript_embedding"]) == [float(len("a longer sentence"))]
    (text_call,) = [c for c in voyage.calls if "texts" in c]
    assert text_call["texts"] == ["short", "a longer sentence"]
    assert text_call["model"] == "voyage-4-lite" and text_call["input_type"] == "document"


def test_autoembed_mode_stores_no_client_vectors_and_makes_no_text_calls(monkeypatch, local_serve):
    eng, coll, voyage = make_engine(monkeypatch, transcript_mode="autoembed")
    local_serve(eng)
    eng.ingest(URL)
    assert all("transcript_embedding" not in d for d in coll.docs)
    assert not [c for c in voyage.calls if "texts" in c]


def test_transcript_embedding_failure_degrades_to_null_vectors(monkeypatch, local_serve):
    eng, coll, voyage = make_engine(monkeypatch, transcript_mode="client")
    local_serve(eng)

    def always_fail(*a, **k):
        raise RuntimeError("503")

    voyage.embed = always_fail
    assert eng.ingest(URL).scenes == 3
    assert all(d["transcript_embedding"] is None for d in coll.docs)


def test_search_transcript_dispatches_to_client_vectors(monkeypatch):
    eng, coll, voyage = make_engine(monkeypatch, transcript_mode="client")
    eng.search("elephants").only("transcript").limit(2).video("v").run()
    stage = coll.pipelines[-1][0]["$vectorSearch"]
    assert stage["index"] == "cinematlas_transcript_index"
    assert stage["path"] == "transcript_embedding"
    assert stage["queryVector"] == [float(len("elephants"))]
    assert stage["filter"] == {"video_id": "v"}
    assert voyage.calls[-1]["input_type"] == "query" and voyage.calls[-1]["model"] == "voyage-4"


def test_search_transcript_dispatches_to_autoembed_text_query(monkeypatch):
    eng, coll, voyage = make_engine(monkeypatch, transcript_mode="autoembed")
    eng.search("elephants").only("transcript").limit(2).run()
    stage = coll.pipelines[-1][0]["$vectorSearch"]
    assert stage["index"] == "cinematlas_auto_index" and stage["query"] == "elephants"
    assert voyage.calls == []  # Atlas embeds the query itself
