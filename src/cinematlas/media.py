"""Getting frames and sound out of a video: download, uploads, audio, scene cuts, keyframes, S3."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from typing import IO, Any, Union

from PIL import Image

from .exceptions import DependencyError, IngestionError

logger = logging.getLogger("cinematlas")

UPLOAD_CHUNK_BYTES = 1 << 20
DEFAULT_MAX_DOWNLOAD_MB = 2048
# Prefer H.264 (decodable by every OpenCV build) + a real audio track; YouTube no longer
# serves progressive MP4s with audio, so a plain "mp4" selector yields silent AV1 video.
YTDLP_FORMAT = (
    "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=720][vcodec^=avc1]+ba"
    "/b[height<=720][vcodec^=avc1]/bv*[height<=720]+ba/b[height<=720]/b"
)

VideoFile = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, IO[bytes], Any]


_PACKAGES = {"cv2": "opencv-python-headless", "yt_dlp": "yt-dlp"}


def require(module: str) -> Any:
    """Import a video dependency, or say which extra installs it (plain ``cinematlas`` is search-only)."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        package = _PACKAGES.get(module.split(".")[0], module)
        raise DependencyError(f'Ingesting video needs {package}: pip install "cinematlas[video]"') from e


def upload_keyframe(s3_client: Any, bucket: str | None, pil_image: Image.Image, s3_key: str) -> str:
    """Upload a keyframe straight from memory; returns its URL, or ``""`` without S3 or on failure."""
    if not s3_client or not bucket:
        return ""
    try:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=80, optimize=True)
        buffer.seek(0)
        s3_client.upload_fileobj(buffer, bucket, s3_key,
                                 ExtraArgs={"ContentType": "image/jpeg", "CacheControl": "max-age=31536000"})
        return f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    except Exception as e:
        logger.warning(f"S3 upload failed for '{s3_key}': {e}. Continuing without CDN URL.")
        return ""


def materialize_file(file: VideoFile, temp_dir: str, filename: str | None) -> tuple[str, str]:
    """Resolve an upload to a path on disk plus its SHA-256, streaming in chunks (constant RAM).

    Accepts a path, raw bytes, a binary file-like object, or a web-framework upload (anything with
    ``.file``/``.stream`` + ``.filename``, e.g. FastAPI ``UploadFile``, Flask/Werkzeug ``FileStorage``).
    """
    digest = hashlib.sha256()

    if isinstance(file, (str, os.PathLike)):
        path = os.fspath(file)
        if not os.path.isfile(path):
            raise IngestionError(f"No such file: {path}")
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(UPLOAD_CHUNK_BYTES), b""):
                digest.update(chunk)
        return path, digest.hexdigest()

    ext = os.path.splitext(filename or "")[1].lower() or ".mp4"
    dst = os.path.join(temp_dir, f"upload{ext}")
    with open(dst, "wb") as out:
        if isinstance(file, (bytes, bytearray, memoryview)):
            data = bytes(file)
            digest.update(data)
            out.write(data)
        elif hasattr(file, "read"):
            if hasattr(file, "seek"):
                try:
                    file.seek(0)
                except (OSError, ValueError):
                    pass  # non-seekable stream: read from current position
            for chunk in iter(lambda: file.read(UPLOAD_CHUNK_BYTES), b""):
                if isinstance(chunk, str):
                    raise IngestionError("File object must be opened in binary mode ('rb').")
                digest.update(chunk)
                out.write(chunk)
        else:
            raise TypeError(f"Unsupported file type for ingest: {type(file).__name__}")

    if os.path.getsize(dst) == 0:
        raise IngestionError("Uploaded file is empty.")
    return dst, digest.hexdigest()


def unwrap_upload(file: Any, filename: str | None) -> tuple[Any, str | None]:
    """Unwrap framework upload objects to their underlying binary stream."""
    inner = getattr(file, "file", None) or getattr(file, "stream", None)
    if inner is not None and hasattr(inner, "read") and hasattr(file, "filename"):
        return inner, filename or getattr(file, "filename", None)
    if filename is None and isinstance(getattr(file, "name", None), str):
        filename = os.path.basename(file.name)
    if filename is None and isinstance(file, (str, os.PathLike)):
        filename = os.path.basename(os.fspath(file))
    return file, filename


def assert_decodable(video_path: str) -> None:
    cv2 = require("cv2")

    cap = cv2.VideoCapture(video_path)
    try:
        ok = cap.isOpened() and cap.read()[0]
    finally:
        cap.release()
    if not ok:
        raise IngestionError(f"Not a decodable video file: {os.path.basename(video_path)}")


def download_video(video_url: str, temp_dir: str, max_attempts: int = 3,
                   max_download_mb: int | None = DEFAULT_MAX_DOWNLOAD_MB) -> str:
    """Download with yt-dlp, retrying transient failures (YouTube intermittently answers 403)."""
    yt_dlp = require("yt_dlp")

    ydl_opts = {
        "format": YTDLP_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(temp_dir, "input.%(ext)s"),
        "noplaylist": True,
        "overwrites": True,
        "retries": 3,
        "fragment_retries": 3,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if max_download_mb:
        ydl_opts["max_filesize"] = max_download_mb * 1024 * 1024
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Downloading video from source: {video_url} (attempt {attempt}/{max_attempts})")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            break
        except Exception as e:
            if attempt == max_attempts:
                raise IngestionError(f"yt-dlp video download failed for '{video_url}': {e}") from e
            logger.warning(f"Download attempt {attempt} failed ({e}); retrying.")
            time.sleep(2**attempt)

    candidates = sorted(f for f in os.listdir(temp_dir)
                        if f.startswith("input.") and not f.endswith((".part", ".ytdl")))
    if not candidates:
        limit = f" (or it exceeds max_download_mb={max_download_mb})" if max_download_mb else ""
        raise IngestionError(f"No video file was downloaded from '{video_url}'{limit}")
    return os.path.join(temp_dir, candidates[0])


def extract_audio(video_path: str, temp_dir: str) -> str | None:
    """Extract 16 kHz mono MP3 (what Whisper consumes; ~10x smaller than WAV). ``None`` if no audio."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found on PATH. Ingesting visual content only.")
        return None

    audio_path = os.path.join(temp_dir, "audio.mp3")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k", audio_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        logger.warning(f"No audio extracted ({proc.stderr.strip()[-200:]}). Ingesting visual content only.")
        return None
    return audio_path


def detect_scene_spans(video_path: str, scene_threshold: float) -> list[tuple[float, float]]:
    """Return ``(start_sec, end_sec)`` per detected scene; whole video as one scene if no cuts."""
    cv2 = require("cv2")
    scenedetect = require("scenedetect")
    ContentDetector = require("scenedetect.detectors").ContentDetector
    SceneManager, open_video = scenedetect.SceneManager, scenedetect.open_video

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=scene_threshold))
    scene_manager.detect_scenes(video)

    def secs(tc: Any) -> float:  # FrameTimecode.seconds (>=0.7) replaced get_seconds() (0.6.x)
        return float(tc.seconds if hasattr(type(tc), "seconds") else tc.get_seconds())

    spans = [(secs(s), secs(e)) for s, e in scene_manager.get_scene_list()]
    if spans:
        return spans

    logger.warning("No visual scenes detected. Treating entire video as single scene.")
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    finally:
        cap.release()
    return [(0.0, total_frames / fps)]


def extract_keyframes(video_path: str, spans: Sequence[tuple[float, float]], vid: str,
                      upload: Callable[[Image.Image, str], str]) -> list[dict[str, Any]]:
    """Grab the middle frame of each span as a PIL image; ``upload`` returns its thumbnail URL (or "")."""
    cv2 = require("cv2")

    cap = cv2.VideoCapture(video_path)
    scenes: list[dict[str, Any]] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for scene_id, (start_sec, end_sec) in enumerate(spans):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((start_sec + end_sec) / 2.0 * fps))
            ret, frame = cap.read()

            pil_img = None
            thumbnail_url = ""
            if ret:
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    thumbnail_url = upload(pil_img, f"keyframes/{vid}/scene_{scene_id}.jpg")
                except Exception as img_err:
                    logger.error(f"Failed processing frame for scene {scene_id}: {img_err}")

            scenes.append({
                "scene_id": scene_id,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "duration_sec": round(end_sec - start_sec, 2),
                "keyframe_image": pil_img,
                "thumbnail_url": thumbnail_url,
            })
    finally:
        cap.release()
    return scenes


# ---------------------------------------------------------------------- remote sources without yt-dlp
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpg", ".mpeg", ".ts", ".flv", ".wmv")
CLOUD_SCHEMES = ("s3", "gs")


def is_cloud_uri(url: str) -> bool:
    return urllib.parse.urlparse(url).scheme in CLOUD_SCHEMES


def is_direct_file(url: str) -> bool:
    """An http(s) link to a video file (including presigned cloud URLs), not a page for yt-dlp."""
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.path.lower().endswith(VIDEO_SUFFIXES)


def _target(temp_dir: str, name: str) -> str:
    suffix = os.path.splitext(name)[1].lower()
    return os.path.join(temp_dir, f"input{suffix if suffix in VIDEO_SUFFIXES else '.mp4'}")


def _too_big(size: int | None, max_download_mb: int | None) -> bool:
    return bool(max_download_mb and size and size > max_download_mb * 1024 * 1024)


def download_direct(url: str, temp_dir: str, *, max_download_mb: int | None = DEFAULT_MAX_DOWNLOAD_MB,
                    validate: Callable[[str], str] | None = None, timeout_s: float = 60) -> str:
    """Stream a direct video link to disk, capped at ``max_download_mb``.

    Every redirect is re-checked with ``validate`` (the SSRF guard), which the yt-dlp path can't do.
    """
    class CheckedRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
            if validate:
                validate(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(CheckedRedirects)
    request = urllib.request.Request(url, headers={"User-Agent": "cinematlas"})
    try:
        with opener.open(request, timeout=timeout_s) as response:
            if "text/html" in (response.headers.get("Content-Type") or ""):
                raise IngestionError(f"'{url}' returned a web page, not a video file")
            size = int(response.headers.get("Content-Length") or 0) or None
            if _too_big(size, max_download_mb):
                raise IngestionError(f"'{url}' is {size / 2**20:.0f} MiB, over max_download_mb={max_download_mb}")
            path = _target(temp_dir, urllib.parse.urlparse(url).path)
            written, limit = 0, (max_download_mb or 0) << 20
            with open(path, "wb") as out:
                while chunk := response.read(UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if limit and written > limit:
                        raise IngestionError(f"'{url}' exceeds max_download_mb={max_download_mb}")
                    out.write(chunk)
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Download failed for '{url}': {e}") from e
    if written == 0:
        raise IngestionError(f"'{url}' returned no data")
    return path


def download_object(uri: str, temp_dir: str, *, max_download_mb: int | None = DEFAULT_MAX_DOWNLOAD_MB,
                    s3_client: Any = None, gcs_client: Any = None) -> str:
    """Fetch ``s3://bucket/key`` or ``gs://bucket/object`` with the cloud SDK, using ambient credentials."""
    parsed = urllib.parse.urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    if not bucket or not key:
        raise IngestionError(f"Expected {parsed.scheme}://bucket/key, got {uri!r}")
    path = _target(temp_dir, key)
    try:
        if parsed.scheme == "s3":
            if s3_client is None:
                try:
                    import boto3
                except ImportError as e:
                    raise IngestionError("s3:// sources need boto3: pip install 'cinematlas[s3]'") from e
                s3_client = boto3.client("s3")
            size = s3_client.head_object(Bucket=bucket, Key=key).get("ContentLength")
            if _too_big(size, max_download_mb):
                raise IngestionError(f"{uri} is {size / 2**20:.0f} MiB, over max_download_mb={max_download_mb}")
            s3_client.download_file(bucket, key, path)
        else:
            if gcs_client is None:
                try:
                    from google.cloud import storage
                except ImportError as e:
                    raise IngestionError(
                        "gs:// sources need google-cloud-storage: pip install 'cinematlas[gcs]'") from e
                gcs_client = storage.Client()
            blob = gcs_client.bucket(bucket).blob(key)
            blob.reload()
            if _too_big(blob.size, max_download_mb):
                raise IngestionError(f"{uri} is {blob.size / 2**20:.0f} MiB, over max_download_mb={max_download_mb}")
            blob.download_to_filename(path)
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Could not fetch {uri}: {e}") from e
    return path
