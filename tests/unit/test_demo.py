"""cinematlas demo: the page, the API contract, background ingest jobs, playback sources, upload safety."""

import time
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cinematlas.demo import create_app, player, youtube_id  # noqa: E402
from cinematlas.results import IngestResult, SearchHit, SearchResults  # noqa: E402


class StubEngine:
    db_name, collection_name, transcript_mode = "demo", "scenes", "autoembed"

    def __init__(self, fail_ingest=False):
        self.fail_ingest = fail_ingest
        self.searches = []
        self.collection = SimpleNamespace(aggregate=lambda pipeline: iter([
            {"_id": "yt1", "scenes": 12, "video_url": "https://www.youtube.com/watch?v=abc123", "duration": 79.0},
        ]))

    def ingest(self, source, filename=None, progress=None):
        if self.fail_ingest:
            raise RuntimeError("not a video")
        progress("fetched", {"video_id": "file_1", "audio": True})
        progress("scenes", {"count": 4})
        return IngestResult(video_id="file_1", scenes=4, source_type="file", transcript_mode="autoembed", seconds=2.0)

    def search(self, q):
        return StubQuery(self, q)

    def _run(self, q, top_k, video_id, routing):
        self.searches.append((q, top_k, video_id, routing))
        hit = SearchHit({"video_id": "yt1", "video_url": "https://www.youtube.com/watch?v=abc123",
                         "timestamp_start": 50.0, "timestamp_end": 60.0, "rank": 1, "ranks": {"scene": 1},
                         "moment": {"start": 56.0, "end": 58.0, "text": "about as loud as a balloon popping"},
                         "moment_link": "https://www.youtube.com/watch?v=abc123&t=56s", "score": 0.02})
        return SearchResults([hit])


class StubQuery:
    def __init__(self, eng, q, k=5, video=None, routing=None):
        self.eng, self.q, self.k, self.vid, self.mode = eng, q, k, video, routing

    def limit(self, k):
        return StubQuery(self.eng, self.q, k, self.vid, self.mode)

    def video(self, vid):
        return StubQuery(self.eng, self.q, self.k, vid or None, self.mode)

    def adaptive(self):
        return StubQuery(self.eng, self.q, self.k, self.vid, "adaptive")

    def run(self):
        return self.eng._run(self.q, self.k, self.vid, self.mode)


@pytest.fixture
def client(tmp_path):
    engine = StubEngine()
    return TestClient(create_app(engine, media_dir=tmp_path)), engine, tmp_path


def wait(client, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_page_and_info(client):
    c, _, _ = client
    assert "Get the second" in c.get("/").text
    assert c.get("/api/info").json() == {"collection": "demo.scenes", "transcript_mode": "autoembed"}


def test_library_lists_videos(client):
    c, _, _ = client
    (video,) = c.get("/api/videos").json()
    assert video["video_id"] == "yt1" and video["scenes"] == 12 and video["playable"]


def test_search_returns_the_moment_and_how_to_play_it(client):
    c, engine, _ = client
    res = c.get("/api/search", params={"q": "how loud", "k": 3, "mode": "adaptive", "video_id": "yt1"}).json()
    (hit,) = res["hits"]
    assert hit["text"] == "about as loud as a balloon popping" and hit["at"] == "0:56"
    assert "relevance" in hit  # the page flags weak matches from it
    assert hit["play"] == {"kind": "youtube", "src": "https://www.youtube.com/embed/abc123", "start": 56.0}
    assert engine.searches == [("how loud", 3, "yt1", "adaptive")]
    assert c.get("/api/search", params={"q": "x", "mode": "magic"}).status_code == 400


def test_upload_runs_in_the_background_and_becomes_playable(client):
    c, _, media = client
    job = c.post("/api/videos", files={"file": ("talk.mp4", b"fake video bytes", "video/mp4")}).json()["job"]
    done = wait(c, job)
    assert done["state"] == "done" and done["result"]["scenes"] == 4
    assert [s["stage"] for s in done["stages"]] == ["fetched", "scenes"]
    stored = [p for p in media.iterdir() if p.suffix == ".mp4"]
    assert len(stored) == 1 and c.get(f"/media/{stored[0].name}").content == b"fake video bytes"


def test_failed_ingest_is_reported_not_raised(tmp_path):
    c = TestClient(create_app(StubEngine(fail_ingest=True), media_dir=tmp_path))
    job = c.post("/api/videos", json={"url": "https://example.com/v.mp4"}).json()["job"]
    assert wait(c, job)["state"] == "failed" and "not a video" in wait(c, job)["error"]


def test_bad_requests_are_rejected(client):
    c, _, _ = client
    assert c.post("/api/videos", json={"url": ""}).status_code == 400
    assert c.post("/api/videos", files={"file": ("notes.txt", b"x", "text/plain")}).status_code == 400
    assert c.get("/api/jobs/nope").status_code == 404


def test_media_route_cannot_escape_its_folder(client):
    c, _, media = client
    (media.parent / "secret.txt").write_text("no")
    assert c.get("/media/..%2Fsecret.txt").status_code == 404
    assert c.get("/media/missing.mp4").status_code == 404


@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=5NhYvbMdbBU", "5NhYvbMdbBU"),
    ("youtu.be/5NhYvbMdbBU", "5NhYvbMdbBU"),
    ("https://youtube.com/shorts/abc", "abc"),
    ("https://cdn.example.com/v.mp4", None),
    (None, None),
])
def test_youtube_ids(url, expected):
    assert youtube_id(url) == expected


def test_player_prefers_local_uploads_then_youtube_then_direct_links():
    assert player("https://youtu.be/x1", "abc.mp4", 3)["src"] == "/media/abc.mp4"
    assert player("https://youtu.be/x1", None, 3)["kind"] == "youtube"
    assert player("https://cdn.example.com/v.mp4", None, 3) == {"kind": "file", "src": "https://cdn.example.com/v.mp4",
                                                                "start": 3}
    assert player(None, None, 0)["kind"] == "none"
