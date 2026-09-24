"""Shared fixtures. See support.py for the design notes."""

import functools
import hashlib
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from support import REAL_CLIP, REAL_CLIP_SHA256, FakeMongoClient, FakeVoyage, fake_resolver, write_colour_video

from cinematlas import Cinematlas


@pytest.fixture
def colour_video(tmp_path) -> Path:
    return write_colour_video(tmp_path / "rgb.mp4", [("red", 1.5), ("white", 1.5), ("blue", 1.5)])


@pytest.fixture
def fake_voyage():
    return FakeVoyage()


@pytest.fixture
def fake_mongo():
    return FakeMongoClient()


@pytest.fixture
def engine(fake_mongo, fake_voyage, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("cinematlas.engine.time.sleep", lambda _s: None)  # no real backoff in unit tests
    # Unit tests never touch DNS: every host "resolves" to a public documentation address.
    monkeypatch.setattr("cinematlas.engine.socket.getaddrinfo", fake_resolver({}))
    return Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False)


@pytest.fixture(scope="session")
def test_whisper_model():
    """Override with CINEMATLAS_TEST_WHISPER_MODEL; defaults to the engine default."""
    import os

    from cinematlas.engine import DEFAULT_WHISPER_MODEL

    return os.getenv("CINEMATLAS_TEST_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)


@pytest.fixture(scope="session")
def real_clip() -> Path:
    """The committed real-speech fixture; the hash check makes accidental edits fail loudly."""
    digest = hashlib.sha256(REAL_CLIP.read_bytes()).hexdigest()
    assert digest == REAL_CLIP_SHA256, "fixture changed: rebuild with tests/fixtures/build_fixture.py and update hash"
    return REAL_CLIP


@pytest.fixture(scope="session")
def clip_server(real_clip):
    """Serve tests/fixtures over loopback HTTP so URL ingestion runs the real yt-dlp path offline."""
    handler = functools.partial(_QuietHandler, directory=str(real_clip.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
