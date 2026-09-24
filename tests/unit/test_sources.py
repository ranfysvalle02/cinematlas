"""Remote sources without yt-dlp: s3://, gs://, and direct video links (with SSRF-checked redirects)."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from cinematlas import IngestionError, media


class FakeS3:
    def __init__(self, size=10):
        self.size, self.calls = size, []

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Bucket, Key))
        return {"ContentLength": self.size}

    def download_file(self, bucket, key, path):
        self.calls.append(("download", bucket, key))
        open(path, "wb").write(b"s3 video")


class FakeGCS:
    def __init__(self, size=10):
        self.blob_obj = SimpleNamespace(size=size, reload=lambda: None,
                                        download_to_filename=lambda path: open(path, "wb").write(b"gcs video"))

    def bucket(self, name):
        return SimpleNamespace(blob=lambda key: self.blob_obj)


@pytest.fixture
def server():
    """Tiny local server: /v.mp4 (video), /page.mp4 (HTML), /big.mp4 (huge Content-Length), /hop.mp4 (redirect)."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/hop.mp4":
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/v.mp4")
                self.end_headers()
                return
            body, kind, length = b"video bytes", "video/mp4", None
            if self.path == "/page.mp4":
                body, kind = b"<html>login</html>", "text/html; charset=utf-8"
            if self.path == "/big.mp4":
                length = str(5 * 1024 * 1024 * 1024)
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", length or str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


# ------------------------------------------------------------------ routing
@pytest.mark.parametrize("url,cloud,direct", [
    ("s3://bucket/talks/keynote.mp4", True, False),
    ("gs://bucket/keynote.mov", True, False),
    ("https://cdn.example.com/v.mp4", False, True),
    ("https://bucket.s3.amazonaws.com/v.mp4?X-Amz-Signature=abc", False, True),  # presigned
    ("https://www.youtube.com/watch?v=x", False, False),                         # a page: yt-dlp
])
def test_sources_are_routed_by_kind(url, cloud, direct):
    assert (media.is_cloud_uri(url), media.is_direct_file(url)) == (cloud, direct)


def test_engine_uses_the_cloud_sdk_for_s3_without_ytdlp(engine, monkeypatch, tmp_path):
    engine.s3_client = FakeS3()
    monkeypatch.setattr(engine, "_download_video", lambda *a, **k: pytest.fail("s3 must not use yt-dlp"))
    monkeypatch.setattr(engine, "_extract_audio", lambda *a: None)
    path, _ = engine._download_and_extract_media("s3://media/talks/keynote.mov", str(tmp_path))
    assert path.endswith("input.mov") and open(path, "rb").read() == b"s3 video"
    key = ("media", "talks/keynote.mov")
    assert engine.s3_client.calls == [("head", *key), ("download", *key)]


def test_cloud_objects_over_the_size_cap_are_refused_before_downloading(tmp_path):
    s3 = FakeS3(size=3 * 1024**3)
    with pytest.raises(IngestionError, match="max_download_mb"):
        media.download_object("s3://b/k.mp4", str(tmp_path), max_download_mb=100, s3_client=s3)
    assert ("download", "b", "k.mp4") not in s3.calls
    with pytest.raises(IngestionError, match="max_download_mb"):
        media.download_object("gs://b/k.mp4", str(tmp_path), max_download_mb=100, gcs_client=FakeGCS(3 * 1024**3))


def test_gs_objects_download_with_the_gcs_client(tmp_path):
    path = media.download_object("gs://b/clips/a.webm", str(tmp_path), gcs_client=FakeGCS())
    assert path.endswith("input.webm") and open(path, "rb").read() == b"gcs video"


def test_cloud_uris_need_a_bucket_and_key(engine):
    for bad in ("s3://bucket", "gs:///key"):
        with pytest.raises(IngestionError, match="bucket/key"):
            engine._validate_remote_url(bad)
    assert engine._validate_remote_url("s3://bucket/a.mp4") == "s3://bucket/a.mp4"


# ------------------------------------------------------------------ direct links
def test_direct_links_stream_to_disk(server, tmp_path):
    path = media.download_direct(f"{server}/v.mp4", str(tmp_path))
    assert open(path, "rb").read() == b"video bytes"


def test_a_web_page_behind_a_video_url_is_refused(server, tmp_path):
    with pytest.raises(IngestionError, match="web page"):
        media.download_direct(f"{server}/page.mp4", str(tmp_path))


def test_direct_downloads_over_the_size_cap_are_refused(server, tmp_path):
    with pytest.raises(IngestionError, match="max_download_mb"):
        media.download_direct(f"{server}/big.mp4", str(tmp_path), max_download_mb=100)


def test_redirects_are_ssrf_checked(server, tmp_path):
    """The redirect points at the cloud metadata address; the real SSRF guard (real DNS, not the unit-test
    stub) must refuse it before any connection is made."""
    from cinematlas.urlsafety import validate_remote_url

    seen = []

    def guard(url):
        seen.append(url)
        return validate_remote_url(url)

    with pytest.raises(IngestionError, match="non-public"):
        media.download_direct(f"{server}/hop.mp4", str(tmp_path), validate=guard, timeout_s=5)
    assert seen == ["http://169.254.169.254/latest/meta-data/v.mp4"]


def test_s3_with_a_real_boto3_client_against_mocked_s3(tmp_path, monkeypatch):
    """Not a hand-rolled fake: real boto3 calls (head_object, download_file) against moto's S3."""
    moto = pytest.importorskip("moto")
    import boto3

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with moto.mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="media")
        s3.put_object(Bucket="media", Key="talks/keynote.mp4", Body=b"real s3 bytes")
        path = media.download_object("s3://media/talks/keynote.mp4", str(tmp_path), s3_client=s3)
        assert open(path, "rb").read() == b"real s3 bytes"
        with pytest.raises(IngestionError, match="Could not fetch"):
            media.download_object("s3://media/missing.mp4", str(tmp_path), s3_client=s3)
