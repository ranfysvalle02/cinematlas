"""Edges to external services: construction, transcription, S3 streaming, search error mapping."""

import io
import os
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws
from PIL import Image
from pymongo.errors import OperationFailure

from cinematlas import Cinematlas, CinematlasError, SearchError


# ------------------------------------------------------------- construction
def test_missing_mongo_uri_fails_fast(monkeypatch, fake_voyage):
    for var in ("MONGODB_URI", "MDB_URI"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(CinematlasError, match="MongoDB URI"):
        Cinematlas(voyage_client=fake_voyage)


def test_unreachable_mongo_is_reported_as_cinematlas_error(fake_voyage):
    with pytest.raises(CinematlasError, match="connect"):
        Cinematlas("mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=200", voyage_client=fake_voyage)


def test_context_manager_closes_client(fake_mongo, fake_voyage):
    with Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False):
        pass
    assert fake_mongo.closed


# ------------------------------------------------------------- transcription
@pytest.fixture
def audio_file(tmp_path):
    p = tmp_path / "a.mp3"
    p.write_bytes(b"mp3")
    return str(p)


def _openai_returning(segments=None, exc=None):
    def create(**kwargs):
        assert kwargs["model"] == "whisper-1" and kwargs["response_format"] == "verbose_json"
        if exc:
            raise exc
        return SimpleNamespace(segments=segments)

    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


def test_cloud_whisper_segments_are_normalized_and_audio_deleted(engine, audio_file):
    engine.openai_client = _openai_returning([SimpleNamespace(start=0, end=1.5, text=" hi ")])
    assert engine._transcribe_audio_safe(audio_file) == [{"start": 0.0, "end": 1.5, "text": "hi"}]
    assert not os.path.exists(audio_file)


def test_transcription_failure_degrades_to_empty_and_still_deletes_audio(engine, audio_file):
    engine.openai_client = _openai_returning(exc=TimeoutError("whisper down"))
    assert engine._transcribe_audio_safe(audio_file) == []
    assert not os.path.exists(audio_file)


@pytest.mark.parametrize("path", [None, "/nonexistent/a.mp3"])
def test_no_audio_means_no_transcript(engine, path):
    assert engine._transcribe_audio_safe(path) == []


# ------------------------------------------------------------- S3 streaming
@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="frames")
        yield client


def test_keyframe_is_streamed_to_s3_as_jpeg(engine, s3):
    engine.s3_client, engine.s3_bucket = s3, "frames"
    url = engine._upload_keyframe_stream(Image.new("RGB", (16, 16), "red"), "keyframes/v/scene_0.jpg")

    assert url == "https://frames.s3.amazonaws.com/keyframes/v/scene_0.jpg"
    obj = s3.get_object(Bucket="frames", Key="keyframes/v/scene_0.jpg")
    assert obj["ContentType"] == "image/jpeg"
    assert Image.open(io.BytesIO(obj["Body"].read())).format == "JPEG"


def test_s3_failure_degrades_to_empty_url(engine, s3):
    engine.s3_client, engine.s3_bucket = s3, "no-such-bucket"
    assert engine._upload_keyframe_stream(Image.new("RGB", (4, 4)), "k.jpg") == ""


def test_no_s3_configured_is_a_noop(engine):
    assert engine._upload_keyframe_stream(Image.new("RGB", (4, 4)), "k.jpg") == ""


# ------------------------------------------------------------- search
def test_visual_search_embeds_query_as_query_type(engine, fake_voyage, fake_mongo):
    engine.search("a red car").only("visual").limit(2).video("v").run()
    assert fake_voyage.calls[-1]["input_type"] == "query"
    stage = fake_mongo.collection.pipelines[-1][0]["$vectorSearch"]
    assert stage["path"] == "visual_embedding" and stage["filter"] == {"video_id": "v"}


@pytest.mark.parametrize("source", ["visual", "transcript"])
def test_empty_query_short_circuits(engine, fake_voyage, fake_mongo, source):
    engine._resolved_transcript_mode = "autoembed"
    assert engine.search("").only(source).run() == []
    assert fake_voyage.calls == [] and fake_mongo.collection.pipelines == []


@pytest.mark.parametrize("source", ["visual", "transcript"])
def test_database_errors_surface_as_search_error(engine, fake_mongo, source):
    engine._resolved_transcript_mode = "autoembed"
    fake_mongo.collection.aggregate_error = OperationFailure("index not found")
    with pytest.raises(SearchError, match="index not found"):
        engine.search("q").only(source).run()


def test_voyage_errors_surface_as_search_error(engine, fake_voyage):
    fake_voyage.failures = 1
    with pytest.raises(SearchError, match="vectorization"):
        engine.search("q").only("visual").run()
