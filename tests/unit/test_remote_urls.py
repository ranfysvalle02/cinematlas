"""Remote URL ingestion: normalization, credential hygiene, deep links, SSRF guard, limits."""

from urllib.parse import parse_qs, urlparse

import pytest
from support import fake_resolver

from cinematlas import IngestionError
from cinematlas._utils import build_deep_link, normalize_source_url, redact_url

PRESIGNED = (
    "https://bkt.s3.us-east-1.amazonaws.com/talks/keynote.mp4"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA%2F20260923&X-Amz-Signature=deadbeef"
    "&X-Amz-Expires=900&version=3"
)


# ---------------------------------------------------------------- pure helpers
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("www.b.com/v.mp4", "https://www.b.com/v.mp4"),
        ("  cdn.example.co.uk:8443/a.mp4?x=1 ", "https://cdn.example.co.uk:8443/a.mp4?x=1"),
        ("http://b.com/v.mp4", "http://b.com/v.mp4"),
        ("videos/talk.mp4", "videos/talk.mp4"),  # relative path: not a host
    ],
)
def test_normalize_source_url(raw, expected):
    assert normalize_source_url(raw) == expected


def test_redaction_strips_signatures_tokens_and_userinfo_but_keeps_identity():
    safe = redact_url("https://user:pw@" + PRESIGNED.removeprefix("https://"))
    assert "user" not in safe and "pw@" not in safe
    assert parse_qs(urlparse(safe).query) == {"version": ["3"]}
    assert urlparse(safe).path == "/talks/keynote.mp4"


@pytest.mark.parametrize(
    "param", ["sig", "se", "sp", "Signature", "Key-Pair-Id", "Policy", "token", "access_token", "api_key"]
)
def test_redaction_covers_azure_cloudfront_and_token_conventions(param):
    assert param.lower() not in redact_url(f"https://h.com/v.mp4?{param}=secret&keep=1").lower()


def test_direct_file_deep_link_uses_media_fragment_and_preserves_signed_query():
    link = build_deep_link(PRESIGNED, 42.9)
    assert link == PRESIGNED + "#t=42"  # query untouched -> signature still valid


def test_direct_file_deep_link_replaces_existing_fragment():
    assert build_deep_link("https://b.com/v.mp4#t=5", 9) == "https://b.com/v.mp4#t=9"


# ---------------------------------------------------------------- engine behaviour
@pytest.fixture
def downloads(engine, monkeypatch):
    """Capture the URL actually fetched, then abort the pipeline cheaply."""
    seen = []

    def fake(url, temp_dir):
        seen.append(url)
        raise IngestionError("stop after fetch")

    monkeypatch.setattr(engine, "_download_and_extract_media", fake)
    return seen


def test_schemeless_url_is_fetched_over_https(engine, downloads):
    with pytest.raises(IngestionError, match="stop after fetch"):
        engine.ingest_video("www.b.com/v.mp4")
    assert downloads == ["https://www.b.com/v.mp4"]


def test_signed_url_is_used_to_fetch_but_never_persisted(engine, fake_mongo, downloads):
    with pytest.raises(IngestionError):
        engine.ingest_video(PRESIGNED)
    assert downloads == [PRESIGNED]
    (tomb,) = fake_mongo.collection.docs
    assert "Signature" not in tomb["video_url"] and "Credential" not in tomb["video_url"]


def test_signed_url_video_id_is_stable_across_fresh_signatures(engine, fake_mongo, downloads):
    # A re-signed URL for the same object must map to the same video (so re-ingest replaces).
    with pytest.raises(IngestionError):
        engine.ingest_video(PRESIGNED)
    with pytest.raises(IngestionError):
        engine.ingest_video(PRESIGNED.replace("deadbeef", "cafef00d"))
    ids = {d["video_id"] for d in fake_mongo.collection.docs}
    assert len(ids) == 1


@pytest.mark.parametrize(
    ("host", "ip"),
    [
        ("metadata.internal", "169.254.169.254"),  # cloud metadata endpoint
        ("localhost", "127.0.0.1"),
        ("intranet.corp", "10.1.2.3"),
        ("v6.local", "::1"),
        ("evil-rebind.com", "192.168.0.10"),  # public-looking name, private answer
    ],
)
def test_ssrf_guard_refuses_non_public_addresses(engine, downloads, monkeypatch, host, ip):
    monkeypatch.setattr("cinematlas.urlsafety.socket.getaddrinfo", fake_resolver({host: ip}))
    with pytest.raises(IngestionError, match="non-public address"):
        engine.ingest_video(f"https://{host}/v.mp4")
    assert downloads == [], "must refuse before any request is made"


def test_private_urls_can_be_explicitly_allowed(engine, downloads, monkeypatch):
    monkeypatch.setattr("cinematlas.urlsafety.socket.getaddrinfo", fake_resolver({"nas.lan": "192.168.1.5"}))
    engine.allow_private_urls = True
    with pytest.raises(IngestionError, match="stop after fetch"):
        engine.ingest_video("http://nas.lan/media/v.mp4")
    assert downloads == ["http://nas.lan/media/v.mp4"]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://b.com/v.mp4", "gopher://b.com/x"])
def test_only_http_schemes_are_fetched(engine, downloads, url):
    with pytest.raises(IngestionError, match="scheme"):
        engine.ingest_video(url)
    assert downloads == []


@pytest.mark.parametrize("src", ["clip.mp4", "missing_video.mov", "not a url"])
def test_missing_local_file_is_not_mistaken_for_a_host(engine, downloads, src):
    with pytest.raises(IngestionError, match="No such file"):
        engine.ingest_video(src)
    assert downloads == []


def test_unresolvable_host_is_a_clear_error(engine, downloads, monkeypatch):
    import socket

    def fail(*a, **k):
        raise socket.gaierror("nodename nor servname provided")

    monkeypatch.setattr("cinematlas.urlsafety.socket.getaddrinfo", fail)
    with pytest.raises(IngestionError, match="Cannot resolve"):
        engine.ingest_video("https://no-such-host.example/v.mp4")


def test_download_size_cap_is_passed_to_ytdlp(engine, monkeypatch, tmp_path):
    import sys
    import types

    captured = {}

    class Recorder:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            (tmp_path / "input.mp4").write_bytes(b"x")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Recorder))
    engine.max_download_mb = 50
    monkeypatch.setattr(engine, "_extract_audio", lambda *a: None)
    engine._download_and_extract_media("https://b.com/watch/keynote", str(tmp_path))  # a page: yt-dlp's job
    assert captured["max_filesize"] == 50 * 1024 * 1024
