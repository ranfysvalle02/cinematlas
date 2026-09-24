"""Cinematlas: Multimodal Video Search Engine.

Combines yt-dlp, PySceneDetect, Whisper (faster-whisper locally, or the OpenAI API),
Voyage AI embeddings, and MongoDB Atlas Vector Search.

Voyage model roles:
  * keyframes  -> ``voyage-multimodal-3.5`` (image + text in one space, so text queries find frames)
  * transcripts -> a ``voyage-4`` family text model, either via Atlas autoEmbed or client-side
"""

import gc
import hashlib
import io
import ipaddress
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
import uuid
from collections.abc import Callable, Sequence
from typing import IO, Any, Union

from bson.binary import Binary, BinaryVectorDtype
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from ._cache import LRUCache
from ._utils import (
    SEARCH_PROJECTION,
    assign_segments,
    build_deep_link,
    build_native_rerank_pipeline,
    build_rank_fusion_pipeline,
    build_text_search_stage,
    build_vector_search_pipeline,
    extract_video_id,
    normalize_segments,
    normalize_source_url,
    redact_url,
    split_long_spans,
)
from .doctor import Check, Diagnosis, explain_native_failure
from .exceptions import CinematlasError, DependencyError, IngestionError, IngestionStatus, SearchError
from .indexes import (
    DEFAULT_AUTO_INDEX,
    DEFAULT_SCENE_INDEX,
    DEFAULT_TEXT_INDEX,
    DEFAULT_TEXT_MODEL,
    DEFAULT_TRANSCRIPT_INDEX,
    DEFAULT_VISUAL_INDEX,
    IndexStatus,
    TranscriptMode,
    desired_indexes,
    detect_transcript_mode,
    ensure_search_indexes,
    inspect_indexes,
)
from .results import IngestResult, SearchHit, SearchResults
from .retrieval import lexical_moment, rerank_segments, rrf_fuse, scene_key

logger = logging.getLogger("cinematlas")
logger.addHandler(logging.NullHandler())

DEFAULT_VOYAGE_MODEL = "voyage-multimodal-3.5"
# "small" is the accuracy floor we measured: "base" mishears ordinary words ("long trunks" -> "long hunts").
DEFAULT_WHISPER_MODEL = "small"
UPLOAD_CHUNK_BYTES = 1 << 20
DEFAULT_MAX_DOWNLOAD_MB = 2048
DEFAULT_MAX_SCENE_SECONDS = 30.0
DEFAULT_RERANK_MODEL = "rerank-2.5"
SOURCES = ("visual", "scene", "transcript", "text")
# Tuned on bench/ (30 labelled questions, 6 videos): the keyframe list is noisy for questions
# about speech, so it acts as a tie-breaker; the sentence-level reranker carries the most signal.
DEFAULT_WEIGHTS = {"visual": 0.25, "scene": 1.0, "transcript": 1.0, "text": 1.0, "rerank": 2.0}
# Adaptive routing (default). Questions are usually about what was *said* or what was *shown*, and
# fusing a speech specialist with a visual one loses to whichever specialist fits (bench/). The
# sentence reranker's top relevance says which it is: speech confidence
# f = clamp((relevance - lo) / (hi - lo), 0, 1) scales the speech group by f, the visual group by 1 - f.
SPEECH_WEIGHTS = {"transcript": 1.0, "text": 0.25, "rerank": 2.0}
VISUAL_WEIGHTS = {"scene": 1.0, "visual": 0.25}
# Thresholds live on the reranker's score scale, so they're only valid for the model they were
# calibrated on (bench/: speech questions median 0.66, visual 0.42 with rerank-2.5). Uncalibrated
# rerankers step down to fixed fusion rather than guess; pass routing_thresholds=(lo, hi) to calibrate.
ROUTING_CALIBRATION: dict[str, tuple[float, float]] = {"rerank-2.5": (0.45, 0.55)}
ROUTING_THRESHOLDS = ROUTING_CALIBRATION["rerank-2.5"]  # backwards-compatible alias
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts", ".flv", ".wmv"}

# Prefer H.264 (decodable by every OpenCV build) + a real audio track; YouTube no longer
# serves progressive MP4s with audio, so a plain "mp4" selector yields silent AV1 video.
YTDLP_FORMAT = (
    "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=720][vcodec^=avc1]+ba"
    "/b[height<=720][vcodec^=avc1]/bv*[height<=720]+ba/b[height<=720]/b"
)


VideoFile = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, IO[bytes], Any]


Progress = Callable[[str, dict[str, Any]], None]
_WARNED: set[tuple[str, str]] = set()  # native-stage fallbacks already announced in this process


def _warn_once(stage: str, error: Exception, server_version: str | None = None) -> None:
    reason, fix = explain_native_failure(stage, error, server_version)
    if (stage, reason) not in _WARNED:
        _WARNED.add((stage, reason))
        fallback = "Voyage rerank API (same model)" if stage == "$rerank" else "client-side fusion (same ranking)"
        logger.warning(f"{reason}; using {fallback}. Fix: {fix}")


def _resolve_mongo_uri(mongo_uri: str | None) -> str | None:
    return mongo_uri or os.getenv("MONGODB_URI") or os.getenv("MDB_URI")


class Cinematlas:
    """Video Search & Intelligence Engine.

    Every external client can be injected (``mongo_client``, ``voyage_client``,
    ``s3_client``, ``openai_client``); otherwise it is constructed from
    arguments / environment variables.
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "cinematlas_enterprise",
        collection_name: str = "multimodal_scenes",
        voyage_model: str = DEFAULT_VOYAGE_MODEL,
        s3_bucket_name: str | None = None,
        voyage_api_key: str | None = None,
        openai_api_key: str | None = None,
        *,
        text_model: str = DEFAULT_TEXT_MODEL,
        transcript_mode: str = "auto",
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        allow_private_urls: bool = False,
        max_download_mb: int | None = DEFAULT_MAX_DOWNLOAD_MB,
        max_scene_seconds: float | None = DEFAULT_MAX_SCENE_SECONDS,
        scene_embeddings: bool = True,
        rerank_model: str | None = DEFAULT_RERANK_MODEL,
        routing_thresholds: tuple[float, float] | None = None,
        query_cache_size: int = 256,
        native_fusion: bool | None = None,
        native_rerank: bool | None = None,
        bson_vectors: bool = True,
        progress: Progress | None = None,
        mongo_client: MongoClient | None = None,
        voyage_client: Any = None,
        s3_client: Any = None,
        openai_client: Any = None,
        ping: bool = True,
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.model = voyage_model
        self.text_model = text_model
        self.whisper_model = whisper_model
        self.allow_private_urls = allow_private_urls
        self.max_download_mb = max_download_mb
        self.max_scene_seconds = max_scene_seconds
        self.scene_embeddings = scene_embeddings
        self.rerank_model = rerank_model
        if routing_thresholds is not None and not 0 <= routing_thresholds[0] < routing_thresholds[1] <= 1:
            raise ValueError(f"routing_thresholds must satisfy 0 <= lo < hi <= 1, got {routing_thresholds!r}")
        self.routing_thresholds = routing_thresholds
        self._query_cache = LRUCache(query_cache_size)  # repeated queries skip the Voyage round trip
        # None = auto-detect: try the Atlas-native stage once, remember if the cluster rejects it.
        self.native_fusion = native_fusion
        self.native_rerank = native_rerank
        self.bson_vectors = bson_vectors
        self.progress = progress  # default ingest progress callback: progress(stage, info)
        self._server_version: str | None = None
        self.s3_bucket = s3_bucket_name
        if transcript_mode not in ("auto", "autoembed", "client"):
            raise ValueError(f"transcript_mode must be 'auto', 'autoembed' or 'client', got {transcript_mode!r}")
        self._transcript_mode_setting = transcript_mode
        self._resolved_transcript_mode: TranscriptMode | None = (
            None if transcript_mode == "auto" else transcript_mode  # type: ignore[assignment]
        )
        self._local_whisper: Any = None

        # Voyage AI
        if voyage_client is not None:
            self.vo = voyage_client
        else:
            try:
                import voyageai
            except ImportError as e:  # pragma: no cover - core dependency
                raise DependencyError("voyageai is required: pip install voyageai") from e
            api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
            if not api_key:
                logger.warning("No Voyage API key provided. Visual vector generation will fail.")
            try:
                self.vo = voyageai.Client(api_key=api_key)
            except Exception as e:
                raise CinematlasError(f"Failed to initialize Voyage AI client: {e}") from e

        # MongoDB
        if mongo_client is not None:
            self.mongo_client = mongo_client
        else:
            uri = _resolve_mongo_uri(mongo_uri)
            if not uri:
                raise CinematlasError("No MongoDB URI provided (pass mongo_uri or set MONGODB_URI).")
            self.mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="cinematlas")
        self.collection = self.mongo_client[db_name][collection_name]
        if ping:
            try:
                self.mongo_client.admin.command("ping")
            except PyMongoError as e:
                raise CinematlasError(f"Failed to connect to MongoDB Atlas: {e}") from e

        # Optional S3
        self.s3_client = s3_client
        if self.s3_client is None and self.s3_bucket:
            try:
                import boto3

                self.s3_client = boto3.client("s3")
            except ImportError:
                logger.warning("boto3 not installed (pip install 'cinematlas[s3]'). S3 keyframe uploads disabled.")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}. Keyframe uploads disabled.")

        # Optional OpenAI (cloud Whisper)
        self.openai_client = openai_client
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_client is None and openai_key:
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=openai_key)
            except ImportError:
                logger.warning("openai not installed (pip install 'cinematlas[openai]'). Using local Whisper.")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back to local Whisper.")

    def __repr__(self) -> str:
        # Side-effect free: notebooks, debuggers and loggers call repr() implicitly, so no I/O here.
        mode = self._resolved_transcript_mode or self._transcript_mode_setting
        rerank = self.rerank_model or "off"
        return f"<Cinematlas {self.db_name}.{self.collection_name} · transcripts={mode} · rerank={rerank}>"

    def close(self) -> None:
        self.mongo_client.close()

    # ==========================================
    # Index management / transcript backend
    # ==========================================
    @property
    def transcript_mode(self) -> TranscriptMode:
        """``autoembed`` or ``client``.

        In ``auto`` mode this is inferred from the collection's search indexes. With no
        transcript index yet it reports ``client`` (without caching), so ingested
        documents carry their own transcript vectors and remain searchable either way.
        """
        if self._resolved_transcript_mode is None:
            try:
                detected = detect_transcript_mode(self.collection)
            except PyMongoError as e:
                logger.debug(f"Could not list search indexes: {e}")
                detected = None
            if detected is None:
                return "client"
            self._resolved_transcript_mode = detected
        return self._resolved_transcript_mode

    def ensure_indexes(self, **kwargs: Any) -> TranscriptMode:
        """Create the Vector Search indexes this engine needs; returns the transcript mode.

        On clusters without Atlas autoEmbed, ``auto`` mode falls back to a client-side
        transcript vector index. Extra kwargs go to :func:`ensure_search_indexes`.
        """
        # Serves re-ingest replacement and per-video lookups.
        self.collection.create_index([("video_id", 1), ("scene_id", 1)], name="video_scene")
        kwargs.setdefault("scene_index_name", DEFAULT_SCENE_INDEX if self.scene_embeddings else None)
        mode = ensure_search_indexes(
            self.collection, transcript_mode=self._transcript_mode_setting, text_model=self.text_model, **kwargs
        )
        self._resolved_transcript_mode = mode
        return mode

    @property
    def server_version(self) -> str | None:
        """MongoDB server version (cached; ``None`` if it can't be determined)."""
        if self._server_version is None:
            try:
                self._server_version = str(self.mongo_client.admin.command("buildInfo").get("version") or "")
            except Exception:
                self._server_version = ""
        return self._server_version or None

    @property
    def routing_calibration(self) -> tuple[float, float] | None:
        """``(lo, hi)`` routing thresholds for the configured reranker, or ``None`` if uncalibrated.

        Uncalibrated rerankers fall back to fixed fusion (announced once per process), because a
        threshold only means something on the score scale it was measured on.
        """
        if self.routing_thresholds is not None:
            return self.routing_thresholds
        if not self.rerank_model:
            return None
        calibrated = ROUTING_CALIBRATION.get(self.rerank_model)
        if calibrated is None and ("routing", self.rerank_model) not in _WARNED:
            _WARNED.add(("routing", self.rerank_model))
            logger.warning(
                f"No routing calibration for reranker {self.rerank_model!r}; search() uses fixed fusion. "
                "Calibrate with bench/ and pass routing_thresholds=(lo, hi), or use rerank-2.5."
            )
        return calibrated

    def desired_indexes(self) -> dict[str, tuple[str, dict[str, Any]]]:
        """The search indexes this engine's configuration calls for: ``{name: (kind, definition)}``."""
        return desired_indexes(self.transcript_mode, text_model=self.text_model, scene=self.scene_embeddings)

    def inspect_indexes(self) -> list[IndexStatus]:
        """Each desired index: ``ready`` / ``building`` / ``missing`` / ``failed``, plus definition drift."""
        return inspect_indexes(self.collection, self.desired_indexes())

    def _probe(self, stage: str, pipeline: list[dict[str, Any]]) -> tuple[bool, Exception | None]:
        try:
            list(self.collection.aggregate(pipeline))
            return True, None
        except PyMongoError as e:
            return False, e

    def doctor(self, *, check_voyage: bool = True) -> Diagnosis:
        """Check everything Cinematlas depends on and say exactly how to fix what isn't right.

        Non-destructive and cheap: a ping, index metadata, two tiny probe queries, counts,
        and (optionally) one tiny Voyage call. Probe results are remembered, so the first
        real search skips capability discovery. ``print(engine.doctor())`` for a report.
        """
        checks: list[Check] = []
        add = checks.append

        # MongoDB
        try:
            info = self.mongo_client.admin.command("buildInfo")
            version = info.get("version", "?")
            self._server_version = str(version)
            add(Check("MongoDB", "ok", f"connected · server {version} · {self.db_name}.{self.collection_name}"))
        except PyMongoError as e:
            add(Check("MongoDB", "fail", f"cannot reach the cluster: {str(e)[:120]}",
                      "Check MONGODB_URI / MDB_URI, network access list, and credentials."))
            return Diagnosis(checks)

        # Transcript backend
        mode = self.transcript_mode
        add(Check("Transcript search", "ok" if mode == "autoembed" else "info",
                  "Atlas autoEmbed (voyage-4, embedded server-side)" if mode == "autoembed"
                  else "client-side voyage-4 vectors (works on any deployment)"))

        # Indexes
        try:
            statuses = self.inspect_indexes()
        except PyMongoError as e:
            statuses = []
            add(Check("Indexes", "fail", f"cannot list search indexes: {str(e)[:120]}",
                      "Search indexes need Atlas or Atlas Local (mongot)."))
        for ix in statuses:
            if ix.state == "missing":
                add(Check(f"Index {ix.name}", "fail", "missing", "engine.ensure_indexes()  ·  cinematlas setup"))
            elif ix.state == "failed":
                add(Check(f"Index {ix.name}", "fail", "build failed", "Inspect it in the Atlas UI, then recreate it."))
            elif ix.drift:
                add(Check(f"Index {ix.name}", "warn", "outdated: " + "; ".join(ix.drift),
                          "engine.ensure_indexes(update=True)  ·  cinematlas setup --update  (in place, no downtime)"))
            elif ix.state == "building":
                add(Check(f"Index {ix.name}", "warn", "building (queries may miss recent data)",
                          "Wait; engine.ensure_indexes() blocks until ready."))
            else:
                add(Check(f"Index {ix.name}", "ok", "ready"))

        # Atlas-native stages
        ok, err = self._probe("$rankFusion", [
            {"$rankFusion": {"input": {"pipelines": {"probe": [{"$sort": {"_id": 1}}, {"$limit": 1}]}}}},
            {"$limit": 1}])
        self.native_fusion = ok
        if ok:
            add(Check("$rankFusion", "ok", "hybrid search runs as one native query"))
        else:
            reason, fix = explain_native_failure("$rankFusion", err, self.server_version)
            add(Check("$rankFusion", "warn", f"{reason}; using client-side fusion (same ranking)", fix))

        has_docs = self.collection.find_one({}, {"_id": 1}) is not None
        if not self.rerank_model:
            add(Check("$rerank", "info", "reranking disabled (rerank_model=None)"))
        elif not has_docs:
            add(Check("$rerank", "info", "not probed yet (collection is empty)"))
        else:
            ok, err = self._probe("$rerank", [
                {"$limit": 1}, {"$set": {"_probe": "probe"}},
                {"$rerank": {"query": {"text": "probe"}, "path": "_probe", "numDocsToRerank": 1,
                             "model": self.rerank_model}}])
            self.native_rerank = ok
            if ok:
                add(Check("$rerank", "ok", f"sentence reranking runs inside Atlas ({self.rerank_model})"))
            else:
                reason, fix = explain_native_failure("$rerank", err, self.server_version)
                add(Check("$rerank", "warn", f"{reason}; using the Voyage rerank API (same model)", fix))

        # Routing
        if not self.rerank_model:
            add(Check("Routing", "info", "off (no reranker): search() uses fixed fusion"))
        elif self.routing_thresholds is not None:
            lo, hi = self.routing_thresholds
            add(Check("Routing", "ok", f"adaptive, custom calibration ({lo:.2f}–{hi:.2f})"))
        elif self.rerank_model in ROUTING_CALIBRATION:
            lo, hi = ROUTING_CALIBRATION[self.rerank_model]
            add(Check("Routing", "ok", f"adaptive, calibrated for {self.rerank_model} ({lo:.2f}–{hi:.2f})"))
        else:
            add(Check("Routing", "warn", f"no calibration for {self.rerank_model}; using fixed fusion",
                      "Calibrate on your data with bench/ and pass routing_thresholds=(lo, hi), "
                      "or use rerank_model='rerank-2.5'."))

        # Voyage
        if check_voyage:
            try:
                self.vo.embed(["ping"], model=self.text_model, input_type="query")
                add(Check("Voyage AI", "ok", f"API key works ({self.model}, {self.text_model})"))
            except Exception as e:
                add(Check("Voyage AI", "fail", f"embedding call failed: {str(e)[:120]}",
                          "Set VOYAGE_API_KEY (https://dashboard.voyageai.com)."))

        # Data
        if has_docs:
            done = {"status": IngestionStatus.COMPLETED.value}
            scenes = self.collection.count_documents(done)
            videos = len(self.collection.distinct("video_id", done))
            add(Check("Data", "ok", f"{scenes} scenes across {videos} videos"))
            legacy = self.collection.count_documents({**done, "segments": {"$exists": False}})
            if self.scene_embeddings:
                legacy = max(legacy, self.collection.count_documents({**done, "scene_embedding": {"$exists": False}}))
            if legacy:
                add(Check("Legacy scenes", "warn",
                          f"{legacy} scenes predate 0.2 (no moments / joint vectors); still searchable",
                          "Re-ingest those videos: engine.ingest(url_or_file, video_id=...)"))
            failed = self.collection.distinct("video_id", {"status": IngestionStatus.FAILED.value})
            if failed:
                shown = ", ".join(map(str, failed[:5])) + (" …" if len(failed) > 5 else "")
                add(Check("Failed ingests", "warn", f"{len(failed)} video(s) with FAILED tombstones: {shown}",
                          "See error_message on the tombstone; a successful re-ingest clears it."))
        else:
            add(Check("Data", "info", "no scenes yet", "engine.ingest('https://…/video.mp4')"))

        # Local media tooling
        add(Check("ffmpeg", "ok", shutil.which("ffmpeg")) if shutil.which("ffmpeg") else
            Check("ffmpeg", "fail", "not on PATH (needed to extract audio)",
                  "brew install ffmpeg  ·  apt-get install -y ffmpeg"))
        if self.openai_client:
            add(Check("Speech-to-text", "ok", "OpenAI Whisper API"))
        else:
            try:
                import faster_whisper  # noqa: F401

                add(Check("Speech-to-text", "ok", f"faster-whisper ({self.whisper_model}), runs locally"))
            except ImportError:
                add(Check("Speech-to-text", "warn", "no backend installed: videos index visually only",
                          'pip install "cinematlas[whisper]"  (or set OPENAI_API_KEY with cinematlas[openai])'))
        return Diagnosis(checks)

    def __enter__(self) -> "Cinematlas":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ==========================================
    # Utility Helpers & Memory-Safe Operations
    # ==========================================

    def _upload_keyframe_stream(self, pil_image: Image.Image, s3_key: str) -> str:
        """Uploads PIL Image directly from RAM using io.BytesIO without writing to disk."""
        if not self.s3_client or not self.s3_bucket:
            return ""

        try:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=80, optimize=True)
            buffer.seek(0)

            self.s3_client.upload_fileobj(
                buffer,
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": "image/jpeg",
                    "CacheControl": "max-age=31536000",
                },
            )
            return f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
        except Exception as e:
            logger.warning(f"S3 upload failed for '{s3_key}': {e}. Continuing without CDN URL.")
            return ""

    def _download_and_extract_media(self, video_url: str, temp_dir: str) -> tuple[str, str | None]:
        """Download with yt-dlp and extract a speech-ready audio track."""
        video_path = self._download_video(video_url, temp_dir, max_download_mb=self.max_download_mb)
        return video_path, self._extract_audio(video_path, temp_dir)

    def _materialize_file(self, file: VideoFile, temp_dir: str, filename: str | None) -> tuple[str, str]:
        """Resolve an upload to a path on disk plus its SHA-256, streaming in chunks (constant RAM).

        Accepts a path, raw bytes, a binary file-like object, or a web-framework upload
        (anything with ``.file``/``.stream`` + ``.filename``, e.g. FastAPI ``UploadFile``,
        Flask/Werkzeug ``FileStorage``).
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
                raise TypeError(f"Unsupported file type for ingest_file: {type(file).__name__}")

        if os.path.getsize(dst) == 0:
            raise IngestionError("Uploaded file is empty.")
        return dst, digest.hexdigest()

    @staticmethod
    def _unwrap_upload(file: Any, filename: str | None) -> tuple[Any, str | None]:
        """Unwrap framework upload objects to their underlying binary stream."""
        inner = getattr(file, "file", None) or getattr(file, "stream", None)
        if inner is not None and hasattr(inner, "read") and hasattr(file, "filename"):
            return inner, filename or getattr(file, "filename", None)
        if filename is None and isinstance(getattr(file, "name", None), str):
            filename = os.path.basename(file.name)
        if filename is None and isinstance(file, (str, os.PathLike)):
            filename = os.path.basename(os.fspath(file))
        return file, filename

    @staticmethod
    def _assert_decodable(video_path: str) -> None:
        import cv2

        cap = cv2.VideoCapture(video_path)
        try:
            ok = cap.isOpened() and cap.read()[0]
        finally:
            cap.release()
        if not ok:
            raise IngestionError(f"Not a decodable video file: {os.path.basename(video_path)}")

    @staticmethod
    def _download_video(
        video_url: str, temp_dir: str, max_attempts: int = 3, max_download_mb: int | None = DEFAULT_MAX_DOWNLOAD_MB
    ) -> str:
        """Download with yt-dlp, retrying transient failures (YouTube intermittently answers 403)."""
        import yt_dlp

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

        candidates = sorted(
            f for f in os.listdir(temp_dir) if f.startswith("input.") and not f.endswith((".part", ".ytdl"))
        )
        if not candidates:
            limit = f" (or it exceeds max_download_mb={max_download_mb})" if max_download_mb else ""
            raise IngestionError(f"No video file was downloaded from '{video_url}'{limit}")
        return os.path.join(temp_dir, candidates[0])

    @staticmethod
    def _extract_audio(video_path: str, temp_dir: str) -> str | None:
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

    def _transcribe_audio_safe(self, audio_path: str | None) -> list[dict[str, Any]]:
        """Transcribes audio with automatic cloud/local failover and guarantees immediate disk cleanup."""
        if not audio_path or not os.path.exists(audio_path):
            return []

        segments: list[dict[str, Any]] = []
        try:
            if self.openai_client:
                logger.info("Transcribing audio via OpenAI Whisper API...")
                with open(audio_path, "rb") as af:
                    tx_res = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=af,
                        response_format="verbose_json",
                    )
                segments = normalize_segments(getattr(tx_res, "segments", None))
            else:
                segments = self._transcribe_locally(audio_path)
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}. Continuing with empty transcript.", exc_info=True)
            segments = []
        finally:
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except OSError as cleanup_err:
                logger.warning(f"Failed to delete temp audio file '{audio_path}': {cleanup_err}")

        return segments

    def _transcribe_locally(self, audio_path: str) -> list[dict[str, Any]]:
        """Local speech-to-text with faster-whisper (CPU-friendly, no PyTorch).

        VAD trims silence so segment starts land on actual speech (without it, a sentence after 3s of
        silence is stamped 0.0 and lands in the wrong scene); word timestamps let sentences that
        straddle a cut be split precisely.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise DependencyError(
                "No speech-to-text backend: pip install 'cinematlas[whisper]', or set OPENAI_API_KEY "
                "with 'cinematlas[openai]'."
            ) from e
        logger.info(f"Transcribing audio via faster-whisper ({self.whisper_model})...")
        if self._local_whisper is None:
            self._local_whisper = WhisperModel(self.whisper_model, device="auto", compute_type="int8")
        segs, _info = self._local_whisper.transcribe(audio_path, vad_filter=True, word_timestamps=True)
        return normalize_segments(list(segs))

    def _embed_transcripts(
        self, transcripts: Sequence[str], batch_size: int = 64, max_retries: int = 3
    ) -> list[list[float] | None]:
        """Client-side transcript vectors (``voyage-4`` family); ``None`` for empty text or failed batches."""
        out: list[list[float] | None] = [None] * len(transcripts)
        todo = [i for i, t in enumerate(transcripts) if t.strip()]
        for b in range(0, len(todo), batch_size):
            idx = todo[b : b + batch_size]
            for attempt in range(1, max_retries + 1):
                try:
                    res = self.vo.embed([transcripts[i] for i in idx], model=self.text_model, input_type="document")
                    if len(res.embeddings) != len(idx):
                        raise CinematlasError(f"Voyage returned {len(res.embeddings)} embeddings for {len(idx)} inputs")
                    for i, vec in zip(idx, res.embeddings, strict=True):
                        out[i] = vec
                    break
                except Exception as e:
                    logger.warning(f"Voyage transcript embedding failed (Attempt {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        time.sleep(1.5**attempt)
        return out

    def _store_vector(self, vec: list[float] | None) -> Binary | list[float] | None:
        """Persist as a BSON float32 vector (3.2x smaller than an array of doubles) when enabled."""
        if vec is None or not self.bson_vectors:
            return vec
        return Binary.from_vector([float(x) for x in vec], BinaryVectorDtype.FLOAT32)

    def _multimodal_embed_aligned(self, inputs: list[list[Any]], max_retries: int) -> list[Any] | None:
        """One multimodal_embed call with retries; ``None`` if every attempt failed or misaligned."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.vo.multimodal_embed(inputs=inputs, model=self.model, input_type="document")
                if len(response.embeddings) != len(inputs):
                    raise CinematlasError(
                        f"Voyage returned {len(response.embeddings)} embeddings for {len(inputs)} inputs"
                    )
                return list(response.embeddings)
            except Exception as e:
                logger.warning(f"Voyage AI batch embedding failed (Attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    logger.error("Max retries reached for batch. Populating null vector embeddings.")
                else:
                    time.sleep(1.5**attempt)
        return None

    def _embed_keyframes_batched(
        self,
        scenes: list[dict[str, Any]],
        batch_size: int = 16,
        max_retries: int = 3,
        transcripts: Sequence[str] | None = None,
    ) -> list[list[float] | None]:
        """Embed each scene's keyframe in low-RAM batches.

        Returns exactly one entry per scene, aligned by position; scenes with no
        usable keyframe (or whose batch exhausted its retries) get ``None``.

        With ``transcripts``, also computes a *joint* image+transcript vector per
        scene (voyage-multimodal-3.5 accepts interleaved inputs) and stores it as
        ``scene["scene_embedding"]``; scenes without speech skip the extra call.
        Keyframe images are closed and removed from the scene dicts as each batch completes.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        all_embeddings: list[list[float] | None] = []

        for i in range(0, len(scenes), batch_size):
            batch_scenes = scenes[i : i + batch_size]
            batch_embeddings: list[list[float] | None] = [None] * len(batch_scenes)
            with_image = [j for j, s in enumerate(batch_scenes) if s.get("keyframe_image") is not None]

            if with_image:
                vecs = self._multimodal_embed_aligned(
                    [[batch_scenes[j]["keyframe_image"]] for j in with_image], max_retries
                )
                for j, vec in zip(with_image, vecs or [None] * len(with_image), strict=True):
                    batch_embeddings[j] = vec

                if transcripts is not None:
                    spoken = [j for j in with_image if transcripts[i + j].strip()]
                    if spoken:
                        joint = self._multimodal_embed_aligned(
                            [[batch_scenes[j]["keyframe_image"], transcripts[i + j]] for j in spoken], max_retries
                        )
                        for j, vec in zip(spoken, joint or [None] * len(spoken), strict=True):
                            batch_scenes[j]["scene_embedding"] = vec

            all_embeddings.extend(batch_embeddings)

            # Free RAM: Close and discard Pillow images for this batch immediately
            for s in batch_scenes:
                img = s.pop("keyframe_image", None)
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
            gc.collect()

        return all_embeddings

    @staticmethod
    def _detect_scene_spans(video_path: str, scene_threshold: float) -> list[tuple[float, float]]:
        """Return ``(start_sec, end_sec)`` per detected scene; whole video as one scene if no cuts."""
        import cv2
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=scene_threshold))
        scene_manager.detect_scenes(video)
        # FrameTimecode.seconds (>=0.7) replaced get_seconds() (0.6.x).
        def secs(tc: Any) -> float:
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

    def _extract_keyframes(
        self, video_path: str, spans: Sequence[tuple[float, float]], vid: str
    ) -> list[dict[str, Any]]:
        """Grab the middle frame of each span as a PIL image (and upload to S3 if configured)."""
        import cv2

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
                        thumbnail_url = self._upload_keyframe_stream(pil_img, f"keyframes/{vid}/scene_{scene_id}.jpg")
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

    # ==========================================
    # Main Pipeline Execution
    # ==========================================
    def ingest(
        self,
        source: VideoFile,
        *,
        video_id: str | None = None,
        filename: str | None = None,
        scene_threshold: float = 27.0,
        batch_size: int = 16,
        progress: Progress | None = None,
    ) -> IngestResult:
        """Ingest anything: a URL (YouTube, file link, scheme optional), a local path, ``bytes``,
        a binary file object, or a web upload (FastAPI ``UploadFile``, Flask ``FileStorage``).

        Returns an :class:`IngestResult` (video id, scene counts, timings). ``progress`` is
        called as ``progress(stage, info)`` for ``fetched``, ``scenes``, ``transcribed``,
        ``embedded`` and ``stored``.
        """
        kwargs = {"video_id": video_id, "scene_threshold": scene_threshold, "batch_size": batch_size,
                  "progress": progress}
        if isinstance(source, str) and not os.path.isfile(source):
            return self._ingest_url(source, **kwargs)
        return self._ingest_file(source, filename=filename, **kwargs)

    def ingest_video(
        self,
        video_url: str,
        video_id: str | None = None,
        scene_threshold: float = 27.0,
        batch_size: int = 16,
        progress: Progress | None = None,
    ) -> int:
        """Ingest from a URL; returns the number of scenes indexed. See :meth:`ingest` for details."""
        if os.path.isfile(video_url):
            return self.ingest_file(video_url, video_id=video_id, scene_threshold=scene_threshold,
                                    batch_size=batch_size, progress=progress)
        return self._ingest_url(video_url, video_id=video_id, scene_threshold=scene_threshold,
                                batch_size=batch_size, progress=progress).scenes

    def _ingest_url(
        self,
        video_url: str,
        video_id: str | None = None,
        scene_threshold: float = 27.0,
        batch_size: int = 16,
        progress: Progress | None = None,
    ) -> IngestResult:
        url = self._validate_remote_url(video_url)
        safe_url = redact_url(url)  # never persist signatures/tokens; the raw URL is only used to fetch
        vid = video_id or extract_video_id(safe_url)
        return self._run_pipeline(
            vid,
            fetch=lambda temp_dir: self._download_and_extract_media(url, temp_dir),
            source={"source_type": "url", "video_url": safe_url},
            deep_link_base=safe_url,
            scene_threshold=scene_threshold,
            batch_size=batch_size,
            progress=progress,
        )

    def _validate_remote_url(self, source: str) -> str:
        """Normalize a remote source and refuse schemes/hosts that are unsafe to fetch.

        Private, loopback, link-local and reserved addresses are refused unless
        ``allow_private_urls=True``: when user-supplied URLs reach ``ingest_video``,
        this blocks SSRF against internal services and cloud metadata endpoints
        (e.g. 169.254.169.254). Redirects followed by yt-dlp are not re-checked;
        put an egress proxy in front if you accept arbitrary URLs from the public.
        """
        url = normalize_source_url(source)
        parsed = urllib.parse.urlparse(url)
        if "://" not in url:
            raise IngestionError(f"No such file or URL: {source!r}")
        if os.path.splitext(parsed.hostname or "")[1].lower() in _VIDEO_EXTENSIONS and "/" not in source:
            raise IngestionError(f"No such file: {source!r}")  # "clip.mp4" is a missing file, not a host
        if parsed.scheme not in ("http", "https"):
            raise IngestionError(f"Unsupported URL scheme {parsed.scheme!r}; use http(s) or ingest_file()")
        if not parsed.hostname:
            raise IngestionError(f"URL has no host: {source!r}")
        if not self.allow_private_urls:
            try:
                infos = socket.getaddrinfo(parsed.hostname, parsed.port or None, proto=socket.IPPROTO_TCP)
            except socket.gaierror as e:
                raise IngestionError(f"Cannot resolve host {parsed.hostname!r}: {e}") from e
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if not ip.is_global:
                    raise IngestionError(
                        f"Refusing to fetch {parsed.hostname!r} ({ip}): non-public address. "
                        "Pass allow_private_urls=True for trusted internal sources."
                    )
        return url

    def ingest_file(
        self,
        file: VideoFile,
        video_id: str | None = None,
        filename: str | None = None,
        scene_threshold: float = 27.0,
        batch_size: int = 16,
        progress: Progress | None = None,
    ) -> int:
        """Ingest an uploaded or local file; returns the number of scenes indexed. See :meth:`ingest`."""
        return self._ingest_file(file, video_id=video_id, filename=filename, scene_threshold=scene_threshold,
                                 batch_size=batch_size, progress=progress).scenes

    def _ingest_file(
        self,
        file: VideoFile,
        video_id: str | None = None,
        filename: str | None = None,
        scene_threshold: float = 27.0,
        batch_size: int = 16,
        progress: Progress | None = None,
    ) -> IngestResult:
        """Ingest an uploaded or local video file. Returns scenes indexed.

        ``file`` may be a path, ``bytes``, a binary file object, or a web upload
        (FastAPI ``UploadFile``, Flask ``FileStorage``). Uploads are streamed to a temp
        file in chunks. Without ``video_id`` the ID is derived from the content hash,
        so re-uploading the same file replaces it instead of duplicating it.
        """
        stream, filename = self._unwrap_upload(file, filename)
        materialized: dict[str, str] = {}

        def fetch(temp_dir: str) -> tuple[str, str | None]:
            path, sha256 = self._materialize_file(stream, temp_dir, filename)
            materialized["sha256"] = sha256
            return path, self._extract_audio(path, temp_dir)

        # The content hash is only known after streaming, so resolve the ID lazily.
        return self._run_pipeline(
            video_id,
            fetch=fetch,
            source={"source_type": "file", "video_url": None, "filename": filename},
            deep_link_base=None,
            scene_threshold=scene_threshold,
            batch_size=batch_size,
            id_from_content=lambda: f"file_{materialized['sha256'][:16]}",
            extra=lambda: {"content_sha256": materialized.get("sha256")},
            progress=progress,
        )

    def _run_pipeline(
        self,
        vid: str | None,
        *,
        fetch: Any,
        source: dict[str, Any],
        deep_link_base: str | None,
        scene_threshold: float,
        batch_size: int,
        id_from_content: Any = None,
        extra: Any = None,
        progress: Progress | None = None,
    ) -> IngestResult:
        """Shared ingest pipeline: fetch -> scenes -> keyframes -> STT -> embeddings -> replace docs.

        Re-ingesting a video ID replaces its previous version without a gap: the new
        scenes are inserted under a fresh ``ingest_id`` first, then older versions are
        deleted, so a crash mid-way never leaves the video missing. On failure a
        ``FAILED`` tombstone is written (the previous version stays searchable) and
        :class:`IngestionError` is raised.
        """
        ingest_id = uuid.uuid4().hex
        label = source.get("video_url") or source.get("filename") or "<upload>"
        report = progress or self.progress
        started = time.perf_counter()
        stages: dict[str, float] = {}
        clock = [started]

        def stage(name: str, **info: Any) -> None:
            now = time.perf_counter()
            stages[name] = round(now - clock[0], 3)
            clock[0] = now
            if report:
                try:
                    report(name, info)
                except Exception:  # a broken progress callback must never break ingestion
                    logger.debug("progress callback raised", exc_info=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                video_path, audio_path = fetch(temp_dir)
                if vid is None:
                    vid = id_from_content()
                logger.info(f"Starting ingestion pipeline for Video ID: {vid} ({label})")
                self._assert_decodable(video_path)
                stage("fetched", video_id=vid, source=label, audio=audio_path is not None)

                logger.info("Executing PySceneDetect visual cut analysis...")
                spans = split_long_spans(self._detect_scene_spans(video_path, scene_threshold),
                                         self.max_scene_seconds)
                scenes = self._extract_keyframes(video_path, spans, vid)
                logger.info(f"Extracted {len(scenes)} visual scenes.")
                stage("scenes", count=len(scenes))

                segments = self._transcribe_audio_safe(audio_path)
                scene_segments = assign_segments([(s["start_sec"], s["end_sec"]) for s in scenes], segments)
                transcripts = [" ".join(seg["text"] for seg in segs) for segs in scene_segments]
                stage("transcribed", sentences=len(segments), spoken_scenes=sum(bool(t) for t in transcripts))

                logger.info("Generating multimodal vectors via Voyage AI...")
                embeddings = self._embed_keyframes_batched(
                    scenes, batch_size=batch_size, transcripts=transcripts if self.scene_embeddings else None
                )
                mode = self.transcript_mode
                client_side = mode == "client"
                transcript_vecs = self._embed_transcripts(transcripts) if client_side else [None] * len(scenes)
                stage("embedded", vectors=sum(v is not None for v in embeddings))

                now = time.time()
                common = {
                    **source,
                    **(extra() if extra else {}),
                    "ingest_id": ingest_id,
                    # Provenance: which models produced these vectors (needed for re-embedding migrations).
                    "embedding_models": {"visual": self.model, "transcript": self.text_model,
                                         "transcript_mode": mode,
                                         "scene": self.model if self.scene_embeddings else None},
                }
                documents = [
                    {
                        "video_id": vid,
                        **common,
                        "scene_id": scene["scene_id"],
                        "timestamp_start": scene["start_sec"],
                        "timestamp_end": scene["end_sec"],
                        "duration": scene["duration_sec"],
                        "transcript": transcript,
                        # Timestamped sentences: lets search return the exact moment, not just the scene.
                        "segments": [{"start": round(g["start"], 2), "end": round(g["end"], 2), "text": g["text"]}
                                     for g in segs],
                        "thumbnail_url": scene["thumbnail_url"],
                        "deep_link": build_deep_link(deep_link_base, scene["start_sec"]) if deep_link_base else None,
                        "visual_embedding": self._store_vector(vec),
                        "status": IngestionStatus.COMPLETED.value,
                        "updated_at": now,
                        **({"transcript_embedding": self._store_vector(tvec)} if client_side else {}),
                        # Joint image+speech vector; silent scenes reuse the keyframe vector.
                        **({"scene_embedding": self._store_vector(scene.get("scene_embedding") or vec)}
                           if self.scene_embeddings else {}),
                    }
                    for scene, transcript, segs, vec, tvec in zip(
                        scenes, transcripts, scene_segments, embeddings, transcript_vecs, strict=True
                    )
                ]

                count = 0
                if documents:
                    result = self.collection.insert_many(documents)
                    self.collection.delete_many({"video_id": vid, "ingest_id": {"$ne": ingest_id}})
                    count = len(result.inserted_ids)
                    logger.info(f"Successfully indexed {count} scenes into MongoDB Atlas!")
                stage("stored", scenes=count)
                return IngestResult(
                    video_id=vid, scenes=count, source_type=source["source_type"], transcript_mode=mode,
                    seconds=round(time.perf_counter() - started, 2), spoken_scenes=sum(bool(t) for t in transcripts),
                    stages=stages,
                )

            except Exception as e:
                logger.error(f"Ingestion failed for '{label}': {e}", exc_info=True)
                if vid is not None:
                    try:
                        self.collection.insert_one({
                            "video_id": vid,
                            **source,
                            "ingest_id": ingest_id,
                            "status": IngestionStatus.FAILED.value,
                            "error_message": str(e),
                            "updated_at": time.time(),
                        })
                    except Exception:
                        logger.error("Failed to write FAILED tombstone record.", exc_info=True)
                if isinstance(e, IngestionError):
                    raise
                raise IngestionError(f"Video ingestion failed: {e}") from e

    # ==========================================
    # Search
    # ==========================================
    def search_transcript(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> SearchResults:
        """Semantic search over spoken dialogue using whichever backend this deployment has."""
        if self.transcript_mode == "autoembed":
            return self.search_auto_embedded_transcript(query_text, top_k=top_k, video_id=video_id)
        return self.search_client_embedded_transcript(query_text, top_k=top_k, video_id=video_id)

    def search_client_embedded_transcript(
        self,
        query_text: str,
        top_k: int = 5,
        index_name: str = DEFAULT_TRANSCRIPT_INDEX,
        video_id: str | None = None,
    ) -> SearchResults:
        """Transcript search with client-side ``voyage-4`` query vectors (no autoEmbed required)."""
        if not query_text:
            return SearchResults()
        query_vec = self._embed_text_query(query_text)
        pipeline = build_vector_search_pipeline(
            index_name=index_name, path="transcript_embedding", top_k=top_k, query_vector=query_vec, video_id=video_id
        )
        try:
            return SearchResults(self.collection.aggregate(pipeline))
        except PyMongoError as e:
            raise SearchError(f"Atlas Vector Search failed: {e}") from e

    def search_auto_embedded_transcript(
        self,
        query_text: str,
        top_k: int = 5,
        index_name: str = DEFAULT_AUTO_INDEX,
        video_id: str | None = None,
    ) -> SearchResults:
        """Semantic search over transcripts via Atlas Automated Embedding (autoEmbed)."""
        if not query_text:
            return SearchResults()
        pipeline = build_vector_search_pipeline(
            index_name=index_name, path="transcript", top_k=top_k, query_text=query_text, video_id=video_id
        )
        try:
            return SearchResults(self.collection.aggregate(pipeline))
        except PyMongoError as e:
            raise SearchError(f"Atlas Vector Search failed: {e}") from e

    def _embed_multimodal_query(self, query_text: str) -> list[float]:
        """Query vector for the keyframe and scene sources (cached; see ``query_cache_size``)."""
        def compute() -> list[float]:
            return self.vo.multimodal_embed(inputs=[[query_text]], model=self.model, input_type="query").embeddings[0]
        try:
            return self._query_cache.get_or_compute(("multimodal", self.model, query_text), compute)
        except Exception as e:
            raise SearchError(f"Voyage AI query vectorization failed: {e}") from e

    def _embed_text_query(self, query_text: str) -> list[float]:
        """Query vector for client-side transcript search (cached; see ``query_cache_size``)."""
        def compute() -> list[float]:
            return self.vo.embed([query_text], model=self.text_model, input_type="query").embeddings[0]
        try:
            return self._query_cache.get_or_compute(("text", self.text_model, query_text), compute)
        except Exception as e:
            raise SearchError(f"Voyage AI query vectorization failed: {e}") from e

    def query_cache_info(self) -> dict[str, int]:
        """Query-embedding cache stats: ``hits``, ``misses``, ``size``, ``maxsize``."""
        return self._query_cache.info()

    def clear_query_cache(self) -> None:
        self._query_cache.clear()

    def _vector_search(
        self, query_vec: list[float], *, index_name: str, path: str, top_k: int, video_id: str | None
    ) -> SearchResults:
        pipeline = build_vector_search_pipeline(
            index_name=index_name, path=path, top_k=top_k, query_vector=query_vec, video_id=video_id
        )
        try:
            return SearchResults(self.collection.aggregate(pipeline))
        except PyMongoError as e:
            raise SearchError(f"MongoDB vector search on {index_name!r} failed: {e}") from e

    def search_visual_vector(
        self,
        query_text: str,
        top_k: int = 5,
        index_name: str = DEFAULT_VISUAL_INDEX,
        video_id: str | None = None,
    ) -> SearchResults:
        """Text-to-image search over keyframes using Voyage AI multimodal vectors."""
        if not query_text:
            return SearchResults()
        return self._vector_search(
            self._embed_multimodal_query(query_text),
            index_name=index_name, path="visual_embedding", top_k=top_k, video_id=video_id,
        )

    def search_scene_vector(
        self,
        query_text: str,
        top_k: int = 5,
        index_name: str = DEFAULT_SCENE_INDEX,
        video_id: str | None = None,
    ) -> SearchResults:
        """Search joint keyframe+transcript vectors (what was shown *and* said, in one vector)."""
        if not query_text:
            return SearchResults()
        return self._vector_search(
            self._embed_multimodal_query(query_text),
            index_name=index_name, path="scene_embedding", top_k=top_k, video_id=video_id,
        )

    def _rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        res = self.vo.rerank(query, texts, model=self.rerank_model)
        return [(r.index, r.relevance_score) for r in res.results]

    def search_text(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> SearchResults:
        """Full-text (BM25) transcript search via Atlas Search; best for exact names and numbers."""
        if not query_text:
            return SearchResults()
        pipeline = [build_text_search_stage(DEFAULT_TEXT_INDEX, query_text, video_id), {"$limit": top_k},
                    {"$project": {**{k: v for k, v in SEARCH_PROJECTION.items() if k != "score"},
                                  "score": {"$meta": "searchScore"}}}]
        try:
            return SearchResults(self.collection.aggregate(pipeline))
        except PyMongoError as e:
            raise SearchError(f"Atlas full-text search failed: {e}") from e

    def _source_pipeline(
        self, name: str, query_text: str, n: int, video_id: str | None, mm_vec: list[float] | None
    ) -> list[dict[str, Any]]:
        """The retrieval pipeline for one source (usable standalone or inside ``$rankFusion``)."""
        if name == "text":
            return [build_text_search_stage(DEFAULT_TEXT_INDEX, query_text, video_id), {"$limit": n}]
        if name == "transcript":
            if self.transcript_mode == "autoembed":
                stage = build_vector_search_pipeline(index_name=DEFAULT_AUTO_INDEX, path="transcript", top_k=n,
                                                     query_text=query_text, video_id=video_id)[0]
            else:
                qv = self._embed_text_query(query_text)
                stage = build_vector_search_pipeline(index_name=DEFAULT_TRANSCRIPT_INDEX, path="transcript_embedding",
                                                     top_k=n, query_vector=qv, video_id=video_id)[0]
            return [stage]
        index, path = ((DEFAULT_VISUAL_INDEX, "visual_embedding") if name == "visual"
                       else (DEFAULT_SCENE_INDEX, "scene_embedding"))
        return [build_vector_search_pipeline(index_name=index, path=path, top_k=n, query_vector=mm_vec,
                                             video_id=video_id)[0]]

    def _retrieve_native(
        self, pipelines: dict[str, list[dict[str, Any]]], weights: dict[str, float], n: int
    ) -> tuple[dict[str, list[tuple[str, int]]], dict[tuple[str, int], dict[str, Any]]]:
        """One round trip: Atlas ``$rankFusion`` returns the union plus each document's per-source rank."""
        rows = list(self.collection.aggregate(build_rank_fusion_pipeline(pipelines, weights, n * len(pipelines))))
        per_source: dict[str, list[tuple[int, tuple[str, int]]]] = {name: [] for name in pipelines}
        docs: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = scene_key(row)
            docs[key] = {k: v for k, v in row.items() if k not in ("score", "fusion")}
            for detail in (row.get("fusion") or {}).get("details", []):
                rank = detail.get("rank")
                if isinstance(rank, int):
                    per_source.setdefault(detail["inputPipelineName"], []).append((rank, key))
        return {name: [k for _, k in sorted(v)] for name, v in per_source.items()}, docs

    def _rerank_scenes(
        self, query_text: str, docs: dict[tuple[str, int], dict[str, Any]]
    ) -> tuple[list[tuple[str, int]], dict[tuple[str, int], tuple[float, dict[str, Any]]]]:
        """Sentence-level rerank: Atlas-native ``$rerank`` when enabled, Voyage API otherwise."""
        if self.native_rerank is not False:
            try:
                rows = list(self.collection.aggregate(
                    build_native_rerank_pipeline(list(docs), query_text, self.rerank_model)))
                best: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
                for row in rows:
                    key, seg = scene_key(row), row.get("segment") or {
                        "start": row.get("timestamp_start", 0.0), "end": row.get("timestamp_end", 0.0),
                        "text": row.get("transcript", "")}
                    if key not in best or row["relevance"] > best[key][0]:
                        best[key] = (row["relevance"], seg)
                self.native_rerank = True
                return sorted(best, key=lambda k: -best[k][0]), best
            except PyMongoError as e:
                if self.native_rerank:
                    raise
                _warn_once("$rerank", e, self.server_version)
                self.native_rerank = False
        return rerank_segments(self._rerank, query_text, list(docs.values()))

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        video_id: str | None = None,
        *,
        sources: Sequence[str] = SOURCES,
        rerank: bool = True,
        candidates: int | None = None,
        weights: dict[str, float] | None = None,
        routing: str = "adaptive",
    ) -> SearchResults:
        """Hybrid search over what was shown and what was said, down to the exact moment.

        Sources, fused with Reciprocal Rank Fusion (``weights``, default :data:`DEFAULT_WEIGHTS`):

        * ``visual``: keyframe vectors; ``scene``: joint keyframe+speech vectors;
        * ``transcript``: semantic (autoEmbed or client voyage-4); ``text``: full-text BM25;
        * ``rerank``: a Voyage reranker scoring every candidate *sentence*.

        Runs as one native ``$rankFusion`` query when the cluster supports it (8.0+), else
        per-source queries fused client-side; both yield identical rankings. Reranking uses
        native ``$rerank`` when enabled (8.3+), else the Voyage API. Every result carries
        ``ranks`` (per source), ``relevance``, ``moment`` (best sentence) and ``moment_link``.
        A failing source is skipped; ``SearchError`` only if every source fails.
        """
        if not query_text:
            return SearchResults()
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        unknown = set(sources) - set(SOURCES)
        if unknown:
            raise ValueError(f"Unknown sources: {sorted(unknown)}")
        n = candidates or max(top_k * 4, 20)
        if routing not in ("adaptive", "fixed"):
            raise ValueError(f"routing must be 'adaptive' or 'fixed', got {routing!r}")
        user_weights = weights
        weights = {**DEFAULT_WEIGHTS, **(weights or {})}

        mm_vec = self._embed_multimodal_query(query_text) if {"visual", "scene"} & set(sources) else None
        pipelines: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, Exception] = {}
        for name in sources:
            try:
                pipelines[name] = self._source_pipeline(name, query_text, n, video_id, mm_vec)
            except SearchError as e:
                errors[name] = e

        ranked: dict[str, list[tuple[str, int]]] = {}
        docs: dict[tuple[str, int], dict[str, Any]] = {}
        if pipelines and self.native_fusion is not False:
            try:
                ranked, docs = self._retrieve_native(pipelines, weights, n)
                self.native_fusion = True
            except PyMongoError as e:
                if self.native_fusion:
                    raise SearchError(f"$rankFusion failed: {e}") from e
                _warn_once("$rankFusion", e, self.server_version)
                self.native_fusion = False
        if self.native_fusion is False:
            for name, pipeline in pipelines.items():
                try:
                    hits = list(self.collection.aggregate(
                        [*pipeline, {"$project": {k: v for k, v in SEARCH_PROJECTION.items() if k != "score"}}]))
                except PyMongoError as e:
                    errors[name] = e
                    logger.warning(f"search source {name!r} failed, continuing without it: {e}")
                    continue
                ranked[name] = [scene_key(h) for h in hits]
                for h in hits:
                    docs.setdefault(scene_key(h), h)
        if errors and len(errors) == len(sources):
            raise SearchError(f"All search sources failed: {errors}")

        moments: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
        if rerank and self.rerank_model and docs:
            try:
                ranked["rerank"], moments = self._rerank_scenes(query_text, docs)
            except Exception as e:
                logger.warning(f"Rerank failed, using retrieval ranks only: {e}")

        speech_confidence = None
        thresholds = self.routing_calibration if routing == "adaptive" and user_weights is None else None
        if thresholds and moments:
            lo, hi = thresholds
            top_relevance = max(rel for rel, _ in moments.values())
            speech_confidence = min(1.0, max(0.0, (top_relevance - lo) / (hi - lo)))
            weights = {**{k: w * speech_confidence for k, w in SPEECH_WEIGHTS.items()},
                       **{k: w * (1.0 - speech_confidence) for k, w in VISUAL_WEIGHTS.items()}}

        results = SearchResults()
        results.speech_confidence = speech_confidence
        results.weights = {k: round(v, 3) for k, v in weights.items() if k in ranked}
        for key, score in rrf_fuse(ranked, weights)[:top_k]:
            doc = dict(docs[key])
            if key in moments:
                relevance, moment = moments[key]
            else:
                relevance, moment = None, lexical_moment(query_text, doc.get("segments") or [])
            start = moment["start"] if moment else doc.get("timestamp_start", 0.0)
            url = doc.get("video_url")
            results.append(SearchHit({
                **doc,
                "score": score,
                "ranks": {name: keys.index(key) + 1 for name, keys in ranked.items() if key in keys},
                "relevance": relevance,
                "moment": dict(moment) if moment else None,
                "moment_link": build_deep_link(url, start) if url else None,
            }))
        return results
