"""``cinematlas demo``: a local web app. Add a video, ask a question, jump to the second.

    pip install "cinematlas[demo,whisper]"
    cinematlas demo                                              # a fresh collection
    cinematlas --db cinematlas_bench --collection scenes_autoembed demo   # browse an existing one

The server is FastAPI; the page is one HTML file with plain JavaScript. Ingestion runs in a background
thread and reports progress; uploads are kept in a local media folder so the player can seek in them.
"""


import json
import logging
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from ..engine import Cinematlas
from ..exceptions import CinematlasError

logger = logging.getLogger("cinematlas")
STATIC = Path(__file__).parent / "static"
DEFAULT_MEDIA_DIR = Path.home() / ".cache" / "cinematlas-demo" / "media"
_VIDEO_TYPES = {".mp4", ".mov", ".webm", ".m4v", ".mkv", ".avi"}


def youtube_id(url: str | None) -> str | None:
    """The video id of a YouTube watch/short/embed URL, else ``None``."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0] or None
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            return (urllib.parse.parse_qs(parsed.query).get("v") or [None])[0]
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
            return parts[1]
    return None


def player(video_url: str | None, local: str | None, start: float) -> dict[str, Any]:
    """How the page should play this moment: a YouTube embed, a direct file, or a local upload."""
    if local:
        return {"kind": "file", "src": f"/media/{local}", "start": start}
    yid = youtube_id(video_url)
    if yid:
        return {"kind": "youtube", "src": f"https://www.youtube.com/embed/{yid}", "start": start}
    if video_url:
        return {"kind": "file", "src": video_url, "start": start}
    return {"kind": "none", "src": None, "start": start}


class Jobs:
    """Background ingests and their progress, safe to read from request threads."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, label: str, work: Any) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "label": label, "state": "running", "stages": [], "result": None,
                                  "error": None}

        def progress(stage: str, info: dict[str, Any]) -> None:
            with self._lock:
                self._jobs[job_id]["stages"].append({"stage": stage, **{k: v for k, v in info.items()
                                                                        if isinstance(v, (str, int, float, bool))}})

        def run() -> None:
            try:
                result = work(progress)
                with self._lock:
                    self._jobs[job_id].update(state="done", result=result)
            except Exception as e:  # reported to the page, not raised in a worker thread
                logger.warning(f"demo ingest failed: {e}")
                with self._lock:
                    self._jobs[job_id].update(state="failed", error=str(e))

        threading.Thread(target=run, daemon=True, name=f"ingest-{job_id}").start()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return json.loads(json.dumps(job, default=str)) if job else None


class Library:
    """Which uploaded file backs which video id, so uploads can be played back."""

    def __init__(self, media_dir: Path):
        self.media_dir = media_dir
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._index = self.media_dir / "index.json"
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        return json.loads(self._index.read_text()) if self._index.is_file() else {}

    def local_file(self, video_id: str) -> str | None:
        with self._lock:
            return self._read().get(video_id)

    def remember(self, video_id: str, filename: str) -> None:
        with self._lock:
            index = self._read()
            index[video_id] = filename
            self._index.write_text(json.dumps(index, indent=1))


def create_app(engine: Cinematlas, *, media_dir: Path = DEFAULT_MEDIA_DIR) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse
    except ImportError as e:
        raise CinematlasError('The demo needs FastAPI: pip install "cinematlas[demo]"') from e

    app = FastAPI(title="Cinematlas demo", docs_url="/api/docs")
    jobs, library = Jobs(), Library(media_dir)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {"collection": f"{engine.db_name}.{engine.collection_name}", "transcript_mode": engine.transcript_mode}

    @app.get("/api/videos")
    def videos() -> list[dict[str, Any]]:
        rows = engine.collection.aggregate([
            {"$match": {"status": "COMPLETED"}},
            {"$group": {"_id": "$video_id", "scenes": {"$sum": 1}, "video_url": {"$first": "$video_url"},
                        "filename": {"$first": "$filename"}, "updated_at": {"$max": "$updated_at"},
                        "duration": {"$max": "$timestamp_end"}}},
            {"$sort": {"updated_at": -1}},
        ])
        return [{"video_id": r["_id"], "scenes": r["scenes"],
                 "label": r.get("filename") or r.get("video_url") or r["_id"], "duration": r.get("duration"),
                 "playable": bool(r.get("video_url") or library.local_file(r["_id"]))} for r in rows]

    @app.post("/api/videos")
    async def add_video(request: Request) -> dict[str, str]:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload: UploadFile | None = form.get("file")  # type: ignore[assignment]
            if upload is None or not upload.filename:
                raise HTTPException(400, "No file in the upload.")
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in _VIDEO_TYPES:
                raise HTTPException(400, f"Not a video file type: {suffix or 'none'}")
            stored = f"{uuid.uuid4().hex}{suffix}"
            path = media_dir / stored
            with path.open("wb") as out:
                while chunk := await upload.read(1 << 20):
                    out.write(chunk)

            def work(progress: Any) -> dict[str, Any]:
                result = engine.ingest(str(path), filename=upload.filename, progress=progress)
                library.remember(result.video_id, stored)
                return {"video_id": result.video_id, "scenes": result.scenes, "seconds": result.seconds}

            return {"job": jobs.start(upload.filename, work)}

        body = await request.json()
        url = (body or {}).get("url", "").strip()
        if not url:
            raise HTTPException(400, "Send a video URL, or upload a file.")

        def work_url(progress: Any) -> dict[str, Any]:
            result = engine.ingest(url, progress=progress)
            return {"video_id": result.video_id, "scenes": result.scenes, "seconds": result.seconds}

        return {"job": jobs.start(url, work_url)}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        found = jobs.get(job_id)
        if found is None:
            raise HTTPException(404, "No such job.")
        return found

    @app.get("/api/search")
    def search(q: str, k: int = 6, video_id: str | None = None, mode: str = "scene") -> dict[str, Any]:
        if mode not in ("scene", "adaptive"):
            raise HTTPException(400, "mode is 'scene' or 'adaptive'")
        results = engine.search(q, top_k=max(1, min(k, 20)), video_id=video_id or None,
                                routing=None if mode == "scene" else "adaptive")
        return {
            "query": q,
            "speech_confidence": results.speech_confidence,
            "hits": [{
                "video_id": h.get("video_id"),
                "label": h.get("filename") or h.get("video_url") or h.get("video_id"),
                "scene_start": h.get("timestamp_start"), "scene_end": h.get("timestamp_end"),
                "at": h.timestamp, "start": h.start, "text": h.text, "link": h.link, "why": h.explain(),
                "relevance": h.get("relevance"),
                "play": player(h.get("video_url"), library.local_file(h.get("video_id", "")), h.start),
            } for h in results],
        }

    @app.get("/media/{name}")
    def media(name: str) -> Any:
        path = (media_dir / name).resolve()
        if path.parent != media_dir.resolve() or not path.is_file():
            raise HTTPException(404, "Not found.")
        return FileResponse(path)

    return app


def serve(engine: Cinematlas, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          media_dir: Path = DEFAULT_MEDIA_DIR) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise CinematlasError('The demo needs uvicorn: pip install "cinematlas[demo]"') from e
    app = create_app(engine, media_dir=media_dir)
    url = f"http://{host}:{port}"
    print(f"Cinematlas demo on {url}  ({engine.db_name}.{engine.collection_name})  · Ctrl+C to stop")
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
