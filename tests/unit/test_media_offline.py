"""Real ffmpeg / OpenCV behaviour on locally generated media (no network)."""

import json
import shutil
import subprocess
import sys
import types

import pytest
from support import write_colour_video

from cinematlas import IngestionError

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["streams"]


@pytest.fixture
def video_with_audio(tmp_path, colour_video):
    out = tmp_path / "av.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(colour_video),
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4.5:sample_rate=44100",
         "-ac", "2", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )
    return out


def test_audio_is_extracted_as_16khz_mono_mp3(engine, video_with_audio, tmp_path):
    audio = engine._extract_audio(str(video_with_audio), str(tmp_path))
    (stream,) = probe(audio)
    assert stream["codec_name"] == "mp3"
    assert stream["channels"] == 1
    assert stream["sample_rate"] == "16000"
    assert float(stream["duration"]) == pytest.approx(4.5, abs=0.3)


def test_video_without_audio_track_yields_none(engine, colour_video, tmp_path):
    assert engine._extract_audio(str(colour_video), str(tmp_path)) is None


def test_missing_ffmpeg_degrades_to_visual_only(engine, video_with_audio, tmp_path, monkeypatch):
    monkeypatch.setattr("cinematlas.engine.shutil.which", lambda _n: None)
    assert engine._extract_audio(str(video_with_audio), str(tmp_path)) is None


def test_local_file_is_ingested_without_download(engine, fake_mongo, video_with_audio, monkeypatch):
    monkeypatch.setattr(engine, "_download_video", lambda *a: pytest.fail("must not download a local file"))
    heard = {}

    def fake_stt(path):
        heard["streams"] = probe(path)
        return [{"start": 0.0, "end": 4.5, "text": "tone"}]

    monkeypatch.setattr(engine, "_transcribe_audio_safe", fake_stt)
    assert engine.ingest_video(str(video_with_audio), video_id="local") == 3
    assert heard["streams"][0]["codec_name"] == "mp3"  # the real extracted track reached STT
    assert {d["deep_link"] for d in fake_mongo.collection.docs} == {None}  # no URL to deep-link into


def test_download_failure_is_wrapped_as_ingestion_error(engine, tmp_path, monkeypatch):
    class Boom:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            raise RuntimeError("HTTP Error 403: Forbidden")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Boom))
    with pytest.raises(IngestionError, match="403"):
        engine._download_video("https://example.com/x.mp4", str(tmp_path))


def test_transient_download_failures_are_retried(engine, tmp_path, monkeypatch):
    attempts = []

    class Flaky:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            attempts.append(urls)
            if len(attempts) < 3:
                raise RuntimeError("HTTP Error 403: Forbidden")
            (tmp_path / "input.mp4").write_bytes(b"ok")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Flaky))
    path = engine._download_video("https://youtu.be/x", str(tmp_path))
    assert path.endswith("input.mp4") and len(attempts) == 3


def test_download_that_writes_nothing_is_an_error(engine, tmp_path, monkeypatch):
    class Silent:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            (tmp_path / "input.mp4.part").write_bytes(b"partial")  # leftovers must not count

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Silent))
    with pytest.raises(IngestionError, match="No video file was downloaded"):
        engine._download_video("https://example.com/x.mp4", str(tmp_path))


def test_local_whisper_backend_is_used_when_no_openai_key(engine, tmp_path, monkeypatch):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    seg = types.SimpleNamespace(start=0.0, end=1.0, text=" hello ")
    loads = []

    class FakeWhisperModel:
        def __init__(self, name, **kw):
            loads.append(name)

        def transcribe(self, path, **kw):
            assert kw.get("vad_filter") is True
            return iter([seg]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    assert engine._transcribe_audio_safe(str(audio)) == [{"start": 0.0, "end": 1.0, "text": "hello"}]
    audio.write_bytes(b"x")
    engine._transcribe_audio_safe(str(audio))
    assert loads == ["small"], "default model loaded once and reused across videos"


def test_colour_video_fixture_is_valid(tmp_path):
    # Sanity check on our own fixture so failures elsewhere aren't fixture bugs.
    path = write_colour_video(tmp_path / "x.mp4", [("red", 1.0)])
    (stream,) = probe(path)
    assert stream["codec_type"] == "video" and int(stream["nb_frames"]) == 24


def test_missing_speech_backend_gives_an_actionable_error(engine, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_faster_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_faster_whisper)
    from cinematlas import DependencyError

    with pytest.raises(DependencyError, match=r"cinematlas\[whisper\]"):
        engine._transcribe_locally("audio.mp3")
