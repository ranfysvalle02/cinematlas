"""End-to-end ingestion against a real synthetic video; only network edges are faked."""

import shutil

import pytest
from support import COLOURS_BGR, as_list, write_colour_video

from cinematlas import IngestionError, IngestionStatus

URL = "https://www.youtube.com/watch?v=abcdefghijk"
INDEX = {name: float(i) for i, name in enumerate(COLOURS_BGR)}


@pytest.fixture
def serve(monkeypatch, engine):
    """Make _download_and_extract_media 'download' a local file (optionally with an audio file)."""

    def _serve(video_path, audio_bytes=None):
        def fake_download(url, temp_dir):
            dst = shutil.copy(video_path, f"{temp_dir}/input.mp4")
            audio = None
            if audio_bytes is not None:
                audio = f"{temp_dir}/input.mp3"
                with open(audio, "wb") as fh:
                    fh.write(audio_bytes)
            return dst, audio

        monkeypatch.setattr(engine, "_download_and_extract_media", fake_download)

    return _serve


def test_each_scene_is_indexed_with_its_own_keyframe_vector(engine, fake_mongo, colour_video, serve):
    serve(colour_video)
    assert engine.ingest_video(URL) == 3

    docs = sorted(fake_mongo.collection.docs, key=lambda d: d["scene_id"])
    # The vector encodes the keyframe colour, proving frame -> scene -> document alignment.
    assert [as_list(d["visual_embedding"]) for d in docs] == [[INDEX["red"]], [INDEX["white"]], [INDEX["blue"]]]
    assert [d["timestamp_start"] for d in docs] == pytest.approx([0.0, 1.5, 3.0], abs=0.1)
    assert all(d["timestamp_end"] > d["timestamp_start"] for d in docs)
    assert {d["video_id"] for d in docs} == {"abcdefghijk"}
    assert {d["status"] for d in docs} == {IngestionStatus.COMPLETED.value}
    assert docs[1]["deep_link"] == f"{URL}&t=1s"


def test_transcript_segments_land_in_the_overlapping_scene(engine, fake_mongo, colour_video, serve, monkeypatch):
    serve(colour_video, audio_bytes=b"fake-mp3")
    monkeypatch.setattr(engine, "_transcribe_audio_safe", lambda _p: [
        {"start": 0.2, "end": 1.0, "text": "a red apple"},
        {"start": 3.2, "end": 4.0, "text": "the blue sea"},
    ])
    engine.ingest_video(URL)
    by_scene = {d["scene_id"]: d["transcript"] for d in fake_mongo.collection.docs}
    assert by_scene == {0: "a red apple", 1: "", 2: "the blue sea"}


def test_video_without_cuts_becomes_a_single_full_length_scene(engine, fake_mongo, tmp_path, serve):
    serve(write_colour_video(tmp_path / "flat.mp4", [("white", 2.0)]))
    assert engine.ingest_video(URL) == 1
    (doc,) = fake_mongo.collection.docs
    assert doc["timestamp_start"] == 0.0
    assert doc["timestamp_end"] == pytest.approx(2.0, abs=0.1)
    assert as_list(doc["visual_embedding"]) == [INDEX["white"]]


def test_reingest_replaces_previous_documents(engine, fake_mongo, colour_video, serve):
    serve(colour_video)
    engine.ingest_video(URL)
    engine.ingest_video(URL)
    assert len(fake_mongo.collection.docs) == 3
    # New version is written *before* the old one is removed: no window where the video is missing.
    assert [op for op, _ in fake_mongo.collection.ops] == ["insert_many", "delete_many"] * 2
    assert len({d["ingest_id"] for d in fake_mongo.collection.docs}) == 1


def test_failed_insert_keeps_the_previous_version(engine, fake_mongo, colour_video, serve, monkeypatch):
    serve(colour_video)
    engine.ingest_video(URL)
    before = list(fake_mongo.collection.docs)

    def boom(docs):
        raise RuntimeError("primary stepped down")

    monkeypatch.setattr(fake_mongo.collection, "insert_many", boom)
    with pytest.raises(IngestionError):
        engine.ingest_video(URL)
    completed = [d for d in fake_mongo.collection.docs if d["status"] == IngestionStatus.COMPLETED.value]
    assert completed == before, "previous version must survive a failed re-ingest"


def test_failed_reingest_tombstone_does_not_hide_previous_version(engine, fake_mongo, colour_video, serve,
                                                                  monkeypatch):
    serve(colour_video)
    engine.ingest_video(URL)
    monkeypatch.setattr(engine, "_detect_scene_spans", lambda *a: (_ for _ in ()).throw(RuntimeError("codec")))
    with pytest.raises(IngestionError):
        engine.ingest_video(URL)
    statuses = sorted(d["status"] for d in fake_mongo.collection.docs)
    assert statuses == ["COMPLETED"] * 3 + ["FAILED"]
    # The next successful ingest cleans up both the old version and the tombstone.
    monkeypatch.undo()
    serve(colour_video)
    monkeypatch.setattr("cinematlas.media.time.sleep", lambda _s: None)
    engine.ingest_video(URL)
    assert sorted(d["status"] for d in fake_mongo.collection.docs) == ["COMPLETED"] * 3


def test_documents_record_embedding_provenance(engine, fake_mongo, colour_video, serve):
    serve(colour_video)
    engine.ingest_video(URL)
    assert {repr(d["embedding_models"]) for d in fake_mongo.collection.docs} == {
        repr({"visual": "voyage-multimodal-3.5", "transcript": "voyage-4", "transcript_mode": "client",
              "scene": "voyage-multimodal-3.5"})
    }


def test_long_uncut_footage_is_split_into_bounded_scenes(engine, fake_mongo, tmp_path, serve):
    # A 70s single shot (lecture / interview with dissolves) must not become one giant scene.
    serve(write_colour_video(tmp_path / "long.mp4", [("white", 70.0)], size=(32, 24)))
    assert engine.ingest_video(URL) == 3
    docs = sorted(fake_mongo.collection.docs, key=lambda d: d["scene_id"])
    assert all(d["duration"] <= 30 for d in docs)
    assert docs[0]["timestamp_start"] == 0.0 and docs[-1]["timestamp_end"] == pytest.approx(70, abs=0.1)
    assert all(a["timestamp_end"] == b["timestamp_start"] for a, b in zip(docs, docs[1:], strict=False))


def test_scene_cap_can_be_disabled(engine, fake_mongo, tmp_path, serve):
    engine.max_scene_seconds = None
    serve(write_colour_video(tmp_path / "long.mp4", [("white", 40.0)], size=(32, 24)))
    assert engine.ingest_video(URL) == 1


def test_reingest_does_not_touch_other_videos(engine, fake_mongo, colour_video, serve):
    fake_mongo.collection.docs.append({"video_id": "someone_else", "scene_id": 0})
    serve(colour_video)
    engine.ingest_video(URL)
    assert any(d["video_id"] == "someone_else" for d in fake_mongo.collection.docs)


def test_explicit_video_id_overrides_url_derivation(engine, fake_mongo, colour_video, serve):
    serve(colour_video)
    engine.ingest_video(URL, video_id="custom-id")
    assert {d["video_id"] for d in fake_mongo.collection.docs} == {"custom-id"}


def test_failure_writes_tombstone_and_raises_without_partial_data(engine, fake_mongo, monkeypatch):
    def boom(url, temp_dir):
        raise IngestionError("yt-dlp video download failed: 403")

    monkeypatch.setattr(engine, "_download_and_extract_media", boom)
    with pytest.raises(IngestionError, match="403"):
        engine.ingest_video(URL)

    (tomb,) = fake_mongo.collection.docs
    assert tomb["status"] == IngestionStatus.FAILED.value
    assert tomb["video_id"] == "abcdefghijk"
    assert "403" in tomb["error_message"]
    assert "delete_many" not in [op for op, _ in fake_mongo.collection.ops]


def test_failure_still_raises_if_tombstone_write_fails(engine, fake_mongo, monkeypatch):
    monkeypatch.setattr(engine, "_download_and_extract_media", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    monkeypatch.setattr(fake_mongo.collection, "insert_one", lambda _d: (_ for _ in ()).throw(RuntimeError("db")))
    with pytest.raises(IngestionError, match="disk"):
        engine.ingest_video(URL)


def test_temp_directory_is_removed_after_ingest(engine, colour_video, monkeypatch):
    seen = {}

    def fake_download(url, temp_dir):
        seen["dir"] = temp_dir
        return shutil.copy(colour_video, f"{temp_dir}/input.mp4"), None

    monkeypatch.setattr(engine, "_download_and_extract_media", fake_download)
    engine.ingest_video(URL)
    import os

    assert not os.path.exists(seen["dir"])


def test_vectors_are_stored_as_bson_float32(engine, fake_mongo, colour_video, serve):
    from bson.binary import Binary, BinaryVectorDtype

    serve(colour_video)
    engine.ingest_video(URL)
    vec = fake_mongo.collection.docs[0]["visual_embedding"]
    assert isinstance(vec, Binary) and vec.as_vector().dtype == BinaryVectorDtype.FLOAT32


def test_plain_array_storage_can_be_chosen(engine, fake_mongo, colour_video, serve):
    engine.bson_vectors = False
    serve(colour_video)
    engine.ingest_video(URL)
    assert isinstance(fake_mongo.collection.docs[0]["visual_embedding"], list)
