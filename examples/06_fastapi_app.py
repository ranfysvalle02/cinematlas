"""Advanced: a video search API in about 20 lines. Upload a video, search it, get deep links.

    cd examples
    uv run --with fastapi --with uvicorn --with python-multipart uvicorn 06_fastapi_app:app
    curl -F file=@talk.mp4 localhost:8000/videos
    curl "localhost:8000/search?q=when+do+they+announce+pricing"
"""

from _env import demo_engine  # noqa: F401  (loads .env)
from fastapi import FastAPI, UploadFile

from cinematlas import Cinematlas

engine = Cinematlas(db_name="cinematlas_examples", collection_name="scenes")
engine.ensure_indexes()
app = FastAPI(title="Video search")


@app.post("/videos")
def upload(file: UploadFile) -> dict:
    result = engine.ingest(file)  # streams to disk; the id is a content hash, so re-uploads replace
    return {"video_id": result.video_id, "scenes": result.scenes}


@app.get("/search")
def search(q: str, k: int = 5, video_id: str | None = None) -> list[dict]:
    return [{"text": h.text, "at": h.timestamp, "link": h.link, "video_id": h.video_id}
            for h in engine.search(q, top_k=k, video_id=video_id)]
