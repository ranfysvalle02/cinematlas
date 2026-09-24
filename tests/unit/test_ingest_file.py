"""File-upload entrypoint: every input shape a caller or web framework hands us."""

import hashlib
import io
import shutil
from types import SimpleNamespace

import pytest

from cinematlas import IngestionError, IngestionStatus
from cinematlas.engine import UPLOAD_CHUNK_BYTES

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def video_bytes(colour_video):
    return colour_video.read_bytes()


@pytest.fixture
def no_stt(engine, monkeypatch):
    monkeypatch.setattr(engine, "_transcribe_audio_safe", lambda _p: [])


def sha_id(data: bytes) -> str:
    return f"file_{hashlib.sha256(data).hexdigest()[:16]}"


def docs(fake_mongo, status=IngestionStatus.COMPLETED.value):
    return [d for d in fake_mongo.collection.docs if d["status"] == status]


# ---------------------------------------------------------------- input shapes
def test_path_upload_indexes_every_scene_with_file_metadata(engine, fake_mongo, colour_video, video_bytes, no_stt):
    assert engine.ingest_file(colour_video) == 3
    d = docs(fake_mongo)
    assert {x["video_id"] for x in d} == {sha_id(video_bytes)}
    assert {x["source_type"] for x in d} == {"file"}
    assert {x["filename"] for x in d} == {"rgb.mp4"}
    assert {x["content_sha256"] for x in d} == {hashlib.sha256(video_bytes).hexdigest()}
    assert {x["deep_link"] for x in d} == {None}
    assert {x["video_url"] for x in d} == {None}


def test_raw_bytes_upload(engine, fake_mongo, video_bytes, no_stt):
    assert engine.ingest_file(video_bytes, filename="clip.mp4") == 3
    assert {x["video_id"] for x in docs(fake_mongo)} == {sha_id(video_bytes)}


def test_same_content_is_the_same_video_regardless_of_name_or_shape(
    engine, fake_mongo, colour_video, video_bytes, tmp_path, no_stt
):
    # Idempotent re-uploads: content-addressed IDs mean a re-upload replaces, never duplicates.
    renamed = tmp_path / "renamed.mp4"
    shutil.copy(colour_video, renamed)
    engine.ingest_file(colour_video)
    engine.ingest_file(renamed)
    engine.ingest_file(io.BytesIO(video_bytes), filename="third.mp4")
    assert len(docs(fake_mongo)) == 3
    assert {x["filename"] for x in docs(fake_mongo)} == {"third.mp4"}


def test_file_object_is_rewound_before_reading(engine, fake_mongo, colour_video, video_bytes, no_stt):
    with open(colour_video, "rb") as fh:
        fh.read(100)  # e.g. a framework sniffed the header first
        engine.ingest_file(fh)
    assert {x["video_id"] for x in docs(fake_mongo)} == {sha_id(video_bytes)}
    assert {x["filename"] for x in docs(fake_mongo)} == {"rgb.mp4"}  # taken from fh.name


def test_non_seekable_stream_is_consumed_in_bounded_chunks(engine, fake_mongo, video_bytes, no_stt):
    reads = []

    class Pipe:  # like a socket / request body: read() only, no seek()
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def read(self, n=-1):
            reads.append(n)
            return self._buf.read(n)

    assert engine.ingest_file(Pipe(video_bytes), filename="stream.mp4") == 3
    assert reads and all(0 < n <= UPLOAD_CHUNK_BYTES for n in reads), "never slurp the whole upload into RAM"


def test_fastapi_uploadfile_shape(engine, fake_mongo, video_bytes, no_stt):
    upload = SimpleNamespace(file=io.BytesIO(video_bytes), filename="From Phone.MP4", content_type="video/mp4")
    assert engine.ingest_file(upload) == 3
    assert {x["filename"] for x in docs(fake_mongo)} == {"From Phone.MP4"}


def test_flask_filestorage_shape(engine, fake_mongo, video_bytes, no_stt):
    upload = SimpleNamespace(stream=io.BytesIO(video_bytes), filename="lecture.mp4")
    assert engine.ingest_file(upload, video_id="lecture-1") == 3
    assert {x["video_id"] for x in docs(fake_mongo)} == {"lecture-1"}


def test_explicit_filename_overrides_detected_one(engine, fake_mongo, colour_video, no_stt):
    engine.ingest_file(colour_video, filename="display-name.mp4")
    assert {x["filename"] for x in docs(fake_mongo)} == {"display-name.mp4"}


def test_ingest_video_with_local_path_routes_to_file_pipeline(engine, fake_mongo, colour_video, no_stt, monkeypatch):
    monkeypatch.setattr(engine, "_download_video", lambda *a: pytest.fail("must not download a local file"))
    engine.ingest_video(str(colour_video))
    assert {x["source_type"] for x in docs(fake_mongo)} == {"file"}


def test_uploaded_audio_reaches_transcription(engine, fake_mongo, colour_video, tmp_path, monkeypatch):
    import subprocess

    av = tmp_path / "av.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(colour_video),
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4.5", "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(av)], check=True)
    heard = []
    monkeypatch.setattr(engine, "_transcribe_audio_safe", lambda p: heard.append(p) or [
        {"start": 3.2, "end": 4.0, "text": "blue scene words"}])
    engine.ingest_file(io.BytesIO(av.read_bytes()), filename="av.mp4")
    assert heard and heard[0].endswith(".mp3")
    assert {x["scene_id"]: x["transcript"] for x in docs(fake_mongo)} == {0: "", 1: "", 2: "blue scene words"}


# ---------------------------------------------------------------- rejection
def test_empty_upload_is_rejected_without_tombstone(engine, fake_mongo):
    with pytest.raises(IngestionError, match="empty"):
        engine.ingest_file(b"", filename="x.mp4")
    assert fake_mongo.collection.docs == []  # no content -> no ID to hang a tombstone on


def test_corrupt_upload_is_rejected_and_tombstoned(engine, fake_mongo):
    junk = b"this is definitely not an mp4" * 100
    with pytest.raises(IngestionError, match="Not a decodable video"):
        engine.ingest_file(junk, filename="evil.mp4")
    (tomb,) = fake_mongo.collection.docs
    assert tomb["status"] == IngestionStatus.FAILED.value
    assert tomb["video_id"] == sha_id(junk)
    assert tomb["filename"] == "evil.mp4"


def test_missing_path_is_rejected(engine, tmp_path):
    with pytest.raises(IngestionError, match="No such file"):
        engine.ingest_file(tmp_path / "nope.mp4")


def test_text_mode_file_is_rejected_with_actionable_error(engine, tmp_path):
    p = tmp_path / "t.mp4"
    p.write_text("x")
    with open(p) as fh, pytest.raises(IngestionError, match="binary mode"):
        engine.ingest_file(fh)


def test_unsupported_type_is_rejected(engine):
    with pytest.raises(IngestionError, match="Unsupported"):
        engine.ingest_file(12345)
