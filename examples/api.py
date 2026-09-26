"""A video search API: upload a video, search every video, get links to the second.

    cd examples
    uv run --with fastapi --with uvicorn --with python-multipart uvicorn api:app
    curl -F file=@talk.mp4 localhost:8000/videos
    curl "localhost:8000/search?q=when+do+they+announce+pricing"

Interactive docs at http://localhost:8000/docs.
"""

from _env import YOURS, engine
from fastapi import FastAPI, HTTPException, UploadFile

from cinematlas import CinematlasError

eng = engine(YOURS)
eng.ensure_indexes()
app = FastAPI(title="Video moment search")


@app.post("/videos")
def upload(file: UploadFile) -> dict:
    """Index an uploaded video. The id is a content hash, so re-uploading replaces instead of duplicating."""
    try:
        result = eng.ingest(file)
    except CinematlasError as e:
        raise HTTPException(422, str(e)) from e
    return {"video_id": result.video_id, "scenes": result.scenes, "seconds": round(result.seconds, 1)}


@app.get("/search")
def search(q: str, k: int = 5, video_id: str | None = None) -> list[dict]:
    """The best moments for a question, each with a link to the exact second."""
    return [{"video_id": h.video_id, "at": h.timestamp, "text": h.text, "link": h.link, "why": h.explain()}
            for h in eng.search(q).video(video_id or "").limit(min(k, 20))]
