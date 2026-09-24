"""Real media, real tools, no database, no internet.

Runs against the committed public-domain fixture (tests/fixtures/x59_quiet_crew.mp4).
URL ingestion is exercised through the real yt-dlp code path via a loopback HTTP server.
Because the fixture's scene structure is known, these tests assert per-scene truth,
not just "something was transcribed".
"""

import hashlib
import json
import shutil
import subprocess
import sys

import cv2
import pytest
from support import REAL_CLIP_SECONDS, REAL_CLIP_SHA256, REAL_CLIP_TOPICS, FakeMongoClient, FakeVoyage

from cinematlas import Cinematlas
from cinematlas._utils import assign_transcripts

pytestmark = [
    pytest.mark.media,
    pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed"),
]

# Cut points baked into the fixture by build_fixture.py (segment lengths 22.1s, 9.6s, 8.0s),
# plus one real cut inside the "outdoors" segment from the source footage.
EXPECTED_CUTS = (22.13, 29.2, 31.73)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def engine(test_whisper_model):
    """Engine with no DB traffic: only the media stages are exercised."""
    pytest.importorskip("faster_whisper")
    eng = Cinematlas(mongo_client=FakeMongoClient(), voyage_client=FakeVoyage(), ping=False,
                     whisper_model=test_whisper_model)
    eng.openai_client = None  # these tests are about local faster-whisper, whatever the environment says
    return eng


@pytest.fixture(scope="module")
def segments(engine, real_clip, tmp_path_factory):
    d = tmp_path_factory.mktemp("stt")
    return engine._transcribe_audio_safe(Cinematlas._extract_audio(str(real_clip), str(d)))


def test_ytdlp_downloads_a_direct_url_byte_for_byte(clip_server, tmp_path):
    # Real yt-dlp (generic extractor + our format selector), zero dependence on YouTube.
    path = Cinematlas._download_video(f"{clip_server}/x59_quiet_crew.mp4", str(tmp_path))
    with open(path, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == REAL_CLIP_SHA256


def test_audio_is_extracted_for_whisper(real_clip, tmp_path):
    info = probe(Cinematlas._extract_audio(str(real_clip), str(tmp_path)))
    (stream,) = info["streams"]
    assert (stream["codec_name"], stream["channels"], stream["sample_rate"]) == ("mp3", 1, "16000")
    assert float(info["format"]["duration"]) == pytest.approx(REAL_CLIP_SECONDS, abs=0.5)


def test_scene_detection_finds_exactly_the_known_cuts(real_clip):
    spans = Cinematlas._detect_scene_spans(str(real_clip), 27.0)
    assert [round(a, 1) for a, _ in spans[1:]] == [round(c, 1) for c in EXPECTED_CUTS]
    assert spans[-1][1] == pytest.approx(REAL_CLIP_SECONDS, abs=0.5)

    cap = cv2.VideoCapture(str(real_clip))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
        ok, frame = cap.read()
        assert ok and frame.std() > 5, "decoded frame should have real content"
    finally:
        cap.release()


def test_transcription_contains_every_known_phrase(segments):
    text = " ".join(s["text"] for s in segments).lower()
    for phrases in REAL_CLIP_TOPICS.values():
        for phrase in phrases:
            assert phrase in text, f"{phrase!r} missing from transcript: {text}"
    assert [s["start"] for s in segments] == sorted(s["start"] for s in segments)


def test_each_topic_lands_in_its_own_scene(real_clip, segments):
    """The pipeline's core promise on real speech: dialogue is attached to the scene it was spoken in."""
    spans = Cinematlas._detect_scene_spans(str(real_clip), 27.0)
    per_scene = [t.lower() for t in assign_transcripts(spans, segments)]

    assert all(p in per_scene[0] for p in REAL_CLIP_TOPICS["aircraft"])
    assert all(p in per_scene[1] for p in REAL_CLIP_TOPICS["outdoors"])
    assert all(p in per_scene[-1] for p in REAL_CLIP_TOPICS["beer"])
    # No bleed across cuts (the sonic boom sentence ends ~0.07s after the first cut).
    assert "sonic" not in " ".join(per_scene[1:])
    assert "jalapeno" not in " ".join(per_scene[:-1])


def test_transcription_deletes_its_input(engine, real_clip, tmp_path):
    audio = Cinematlas._extract_audio(str(real_clip), str(tmp_path))
    engine._transcribe_audio_safe(audio)
    assert not (tmp_path / "audio.mp3").exists()


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="needs macOS `say` for TTS")
def test_timestamps_skip_leading_silence(engine, tmp_path):
    """Speech after 3s of silence must be stamped after ~3s; scene alignment depends on it.

    This test is about *timing*; the synthetic voice is only a signal source.
    """
    aiff = tmp_path / "s.aiff"
    subprocess.run(["say", "-o", str(aiff), "Testing one two three. The fox jumps over the dog."], check=True)
    padded = tmp_path / "s.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "3",
                    "-i", "anullsrc=r=16000:cl=mono", "-i", str(aiff),
                    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", "-ac", "1", "-ar", "16000", str(padded)],
                   check=True)

    segs = engine._transcribe_audio_safe(str(padded))
    assert segs, "speech should be detected"
    assert segs[0]["start"] >= 2.5, segs
