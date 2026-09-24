"""Cinematlas: ask a question, get the second in the video that answers it.

This module is the facade: construction, configuration and the public API. The work happens in
focused modules, each with one reason to change:

    media.py         download, uploads, audio, scene cuts, keyframes, S3
    urlsafety.py     remote URLs are untrusted input (SSRF guard)
    transcribe.py    speech to timestamped sentences
    embed.py         keyframe, joint, transcript and query vectors
    ingest.py        the pipeline and its gapless replace
    search.py        single sources, fusion, reranking, routing
    capabilities.py  native-stage fallbacks and routing calibration, said once
    doctor.py        what's wrong and how to fix it

Private ``_methods`` below are the seams between them. They stay on the facade on purpose: the ingest
pipeline and search call steps through it, so any one step can be replaced (or stubbed in tests)
without touching the rest.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from . import ingest as _ingest
from . import media
from . import search as _search
from ._utils import extract_video_id, redact_url
from .capabilities import ROUTING_CALIBRATION, routing_calibration
from .doctor import Diagnosis, run_doctor
from .embed import Embedder
from .exceptions import CinematlasError, DependencyError
from .indexes import (
    DEFAULT_AUTO_INDEX,
    DEFAULT_SCENE_INDEX,
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
from .ingest import Progress
from .media import DEFAULT_MAX_DOWNLOAD_MB, UPLOAD_CHUNK_BYTES, YTDLP_FORMAT, VideoFile  # noqa: F401 (re-exported)
from .results import IngestResult, SearchResults
from .search import DEFAULT_WEIGHTS, SOURCES, SPEECH_WEIGHTS, VISUAL_WEIGHTS  # noqa: F401 (re-exported)
from .transcribe import DEFAULT_WHISPER_MODEL, Transcriber
from .urlsafety import validate_remote_url

logger = logging.getLogger("cinematlas")
logger.addHandler(logging.NullHandler())

DEFAULT_VOYAGE_MODEL = "voyage-multimodal-3.5"
DEFAULT_MAX_SCENE_SECONDS = 30.0
DEFAULT_RERANK_MODEL = "rerank-2.5"
ROUTING_THRESHOLDS = ROUTING_CALIBRATION["rerank-2.5"]  # backwards-compatible alias


def _resolve_mongo_uri(mongo_uri: str | None) -> str | None:
    return mongo_uri or os.getenv("MONGODB_URI") or os.getenv("MDB_URI")


class Cinematlas:
    """Video Search & Intelligence Engine.

    Every external client can be injected (``mongo_client``, ``voyage_client``, ``s3_client``,
    ``openai_client``); otherwise it is constructed from arguments / environment variables.
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
        if routing_thresholds is not None and not 0 <= routing_thresholds[0] < routing_thresholds[1] <= 1:
            raise ValueError(f"routing_thresholds must satisfy 0 <= lo < hi <= 1, got {routing_thresholds!r}")
        if transcript_mode not in ("auto", "autoembed", "client"):
            raise ValueError(f"transcript_mode must be 'auto', 'autoembed' or 'client', got {transcript_mode!r}")
        self.db_name = db_name
        self.collection_name = collection_name
        self.allow_private_urls = allow_private_urls
        self.max_download_mb = max_download_mb
        self.max_scene_seconds = max_scene_seconds
        self.scene_embeddings = scene_embeddings
        self.rerank_model = rerank_model
        self.routing_thresholds = routing_thresholds
        # None = auto-detect: try the Atlas-native stage once, remember if the cluster rejects it.
        self.native_fusion = native_fusion
        self.native_rerank = native_rerank
        self.progress = progress  # default ingest progress callback: progress(stage, info)
        self.s3_bucket = s3_bucket_name
        self._server_version: str | None = None
        self._transcript_mode_setting = transcript_mode
        self._resolved_transcript_mode: TranscriptMode | None = (
            None if transcript_mode == "auto" else transcript_mode  # type: ignore[assignment]
        )

        self.vo = voyage_client if voyage_client is not None else _voyage_client(voyage_api_key)
        self._embedder = Embedder(self.vo, model=voyage_model, text_model=text_model, bson_vectors=bson_vectors,
                                  query_cache_size=query_cache_size)
        self._transcriber = Transcriber(openai_client=openai_client or _openai_client(openai_api_key),
                                        whisper_model=whisper_model)

        if mongo_client is None:
            uri = _resolve_mongo_uri(mongo_uri)
            if not uri:
                raise CinematlasError("No MongoDB URI provided (pass mongo_uri or set MONGODB_URI).")
            mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="cinematlas")
        self.mongo_client = mongo_client
        self.collection = self.mongo_client[db_name][collection_name]
        if ping:
            try:
                self.mongo_client.admin.command("ping")
            except PyMongoError as e:
                raise CinematlasError(f"Failed to connect to MongoDB Atlas: {e}") from e

        self.s3_client = s3_client if s3_client is not None or not self.s3_bucket else _s3_client()

    # ------------------------------------------------------------------ configuration (writes go through)
    @property
    def model(self) -> str:
        return self._embedder.model

    @model.setter
    def model(self, value: str) -> None:
        self._embedder.model = value

    @property
    def text_model(self) -> str:
        return self._embedder.text_model

    @text_model.setter
    def text_model(self, value: str) -> None:
        self._embedder.text_model = value

    @property
    def bson_vectors(self) -> bool:
        return self._embedder.bson_vectors

    @bson_vectors.setter
    def bson_vectors(self, value: bool) -> None:
        self._embedder.bson_vectors = value

    @property
    def openai_client(self) -> Any:
        return self._transcriber.openai_client

    @openai_client.setter
    def openai_client(self, value: Any) -> None:
        self._transcriber.openai_client = value

    @property
    def whisper_model(self) -> str:
        return self._transcriber.whisper_model

    @whisper_model.setter
    def whisper_model(self, value: str) -> None:
        self._transcriber.whisper_model = value

    @property
    def _local_whisper(self) -> Any:
        return self._transcriber.local_model

    @_local_whisper.setter
    def _local_whisper(self, value: Any) -> None:
        self._transcriber.local_model = value

    @property
    def _query_cache(self) -> Any:
        return self._embedder.query_cache

    def __repr__(self) -> str:
        # Side-effect free: notebooks, debuggers and loggers call repr() implicitly, so no I/O here.
        mode = self._resolved_transcript_mode or self._transcript_mode_setting
        rerank = self.rerank_model or "off"
        return f"<Cinematlas {self.db_name}.{self.collection_name} · transcripts={mode} · rerank={rerank}>"

    def close(self) -> None:
        self.mongo_client.close()

    def __enter__(self) -> Cinematlas:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ deployment
    @property
    def transcript_mode(self) -> TranscriptMode:
        """``autoembed`` or ``client``.

        In ``auto`` mode this is inferred from the collection's search indexes. With no transcript index
        yet it reports ``client`` (without caching), so ingested documents carry their own transcript
        vectors and remain searchable either way.
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

        On clusters without Atlas autoEmbed, ``auto`` mode falls back to a client-side transcript
        vector index. Extra kwargs go to :func:`ensure_search_indexes`.
        """
        self.collection.create_index([("video_id", 1), ("scene_id", 1)], name="video_scene")
        kwargs.setdefault("scene_index_name", DEFAULT_SCENE_INDEX if self.scene_embeddings else None)
        mode = ensure_search_indexes(self.collection, transcript_mode=self._transcript_mode_setting,
                                     text_model=self.text_model, **kwargs)
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
        return routing_calibration(self.rerank_model, self.routing_thresholds)

    def desired_indexes(self) -> dict[str, tuple[str, dict[str, Any]]]:
        """The search indexes this engine's configuration calls for: ``{name: (kind, definition)}``."""
        return desired_indexes(self.transcript_mode, text_model=self.text_model, scene=self.scene_embeddings)

    def inspect_indexes(self) -> list[IndexStatus]:
        """Each desired index: ``ready`` / ``building`` / ``missing`` / ``failed``, plus definition drift."""
        return inspect_indexes(self.collection, self.desired_indexes())

    def doctor(self, *, check_voyage: bool = True) -> Diagnosis:
        """Check everything Cinematlas depends on and say exactly how to fix what isn't right.

        Non-destructive and cheap: a ping, index metadata, two tiny probe queries, counts, and
        (optionally) one tiny Voyage call. Probe results are remembered, so the first real search
        skips capability discovery. ``print(engine.doctor())`` for a report.
        """
        return run_doctor(self, check_voyage=check_voyage)

    # ------------------------------------------------------------------ ingest
    def ingest(self, source: VideoFile, *, video_id: str | None = None, filename: str | None = None,
               scene_threshold: float = 27.0, batch_size: int = 16, progress: Progress | None = None) -> IngestResult:
        """Ingest anything: a URL (YouTube, file link, scheme optional), a local path, ``bytes``,
        a binary file object, or a web upload (FastAPI ``UploadFile``, Flask ``FileStorage``).

        Returns an :class:`IngestResult` (video id, scene counts, timings). ``progress`` is called as
        ``progress(stage, info)`` for ``fetched``, ``scenes``, ``transcribed``, ``embedded`` and ``stored``.
        """
        kwargs = {"video_id": video_id, "scene_threshold": scene_threshold, "batch_size": batch_size,
                  "progress": progress}
        if isinstance(source, str) and not os.path.isfile(source):
            return self._ingest_url(source, **kwargs)
        return self._ingest_file(source, filename=filename, **kwargs)

    def ingest_video(self, video_url: str, video_id: str | None = None, scene_threshold: float = 27.0,
                     batch_size: int = 16, progress: Progress | None = None) -> int:
        """Ingest from a URL; returns the number of scenes indexed. See :meth:`ingest` for details."""
        if os.path.isfile(video_url):
            return self.ingest_file(video_url, video_id=video_id, scene_threshold=scene_threshold,
                                    batch_size=batch_size, progress=progress)
        return self._ingest_url(video_url, video_id=video_id, scene_threshold=scene_threshold,
                                batch_size=batch_size, progress=progress).scenes

    def ingest_file(self, file: VideoFile, video_id: str | None = None, filename: str | None = None,
                    scene_threshold: float = 27.0, batch_size: int = 16, progress: Progress | None = None) -> int:
        """Ingest an uploaded or local file; returns the number of scenes indexed. See :meth:`ingest`."""
        return self._ingest_file(file, video_id=video_id, filename=filename, scene_threshold=scene_threshold,
                                 batch_size=batch_size, progress=progress).scenes

    def _ingest_url(self, video_url: str, video_id: str | None = None, scene_threshold: float = 27.0,
                    batch_size: int = 16, progress: Progress | None = None) -> IngestResult:
        url = self._validate_remote_url(video_url)
        safe_url = redact_url(url)  # never persist signatures/tokens; the raw URL is only used to fetch
        return self._run_pipeline(
            video_id or extract_video_id(safe_url),
            fetch=lambda temp_dir: self._download_and_extract_media(url, temp_dir),
            source={"source_type": "url", "video_url": safe_url},
            deep_link_base=safe_url, scene_threshold=scene_threshold, batch_size=batch_size, progress=progress,
        )

    def _ingest_file(self, file: VideoFile, video_id: str | None = None, filename: str | None = None,
                     scene_threshold: float = 27.0, batch_size: int = 16,
                     progress: Progress | None = None) -> IngestResult:
        """Ingest an uploaded or local video file.

        Uploads are streamed to a temp file in chunks. Without ``video_id`` the ID is derived from the
        content hash, so re-uploading the same file replaces it instead of duplicating it.
        """
        stream, filename = self._unwrap_upload(file, filename)
        materialized: dict[str, str] = {}

        def fetch(temp_dir: str) -> tuple[str, str | None]:
            path, sha256 = self._materialize_file(stream, temp_dir, filename)
            materialized["sha256"] = sha256
            return path, self._extract_audio(path, temp_dir)

        return self._run_pipeline(  # the content hash is only known after streaming: resolve the ID lazily
            video_id, fetch=fetch, source={"source_type": "file", "video_url": None, "filename": filename},
            deep_link_base=None, scene_threshold=scene_threshold, batch_size=batch_size,
            id_from_content=lambda: f"file_{materialized['sha256'][:16]}",
            extra=lambda: {"content_sha256": materialized.get("sha256")}, progress=progress,
        )

    def _run_pipeline(self, vid: str | None, **kwargs: Any) -> IngestResult:
        return _ingest.run_pipeline(self, vid, **kwargs)

    def _validate_remote_url(self, source: str) -> str:
        return validate_remote_url(source, allow_private=self.allow_private_urls)

    # ------------------------------------------------------------------ ingest steps (seams; see module doc)
    def _download_and_extract_media(self, video_url: str, temp_dir: str) -> tuple[str, str | None]:
        """Download with yt-dlp and extract a speech-ready audio track."""
        video_path = self._download_video(video_url, temp_dir, max_download_mb=self.max_download_mb)
        return video_path, self._extract_audio(video_path, temp_dir)

    _download_video = staticmethod(media.download_video)
    _extract_audio = staticmethod(media.extract_audio)
    _detect_scene_spans = staticmethod(media.detect_scene_spans)
    _assert_decodable = staticmethod(media.assert_decodable)
    _unwrap_upload = staticmethod(media.unwrap_upload)
    _materialize_file = staticmethod(media.materialize_file)

    def _upload_keyframe_stream(self, pil_image: Any, s3_key: str) -> str:
        return media.upload_keyframe(self.s3_client, self.s3_bucket, pil_image, s3_key)

    def _extract_keyframes(self, video_path: str, spans: Sequence[tuple[float, float]],
                           vid: str) -> list[dict[str, Any]]:
        return media.extract_keyframes(video_path, spans, vid, upload=self._upload_keyframe_stream)

    def _transcribe_audio_safe(self, audio_path: str | None) -> list[dict[str, Any]]:
        return self._transcriber.transcribe_safe(audio_path)

    def _transcribe_locally(self, audio_path: str) -> list[dict[str, Any]]:
        return self._transcriber.transcribe_locally(audio_path)

    def _embed_keyframes_batched(self, scenes: list[dict[str, Any]], batch_size: int = 16, max_retries: int = 3,
                                 transcripts: Sequence[str] | None = None) -> list[list[float] | None]:
        return self._embedder.embed_keyframes_batched(scenes, batch_size, max_retries, transcripts)

    def _multimodal_embed_aligned(self, inputs: list[list[Any]], max_retries: int) -> list[Any] | None:
        return self._embedder.multimodal_aligned(inputs, max_retries)

    def _embed_transcripts(self, transcripts: Sequence[str], batch_size: int = 64,
                           max_retries: int = 3) -> list[list[float] | None]:
        return self._embedder.embed_transcripts(transcripts, batch_size, max_retries)

    def _store_vector(self, vec: list[float] | None) -> Any:
        return self._embedder.store_vector(vec)

    # ------------------------------------------------------------------ search
    def search(self, query_text: str, top_k: int = 5, video_id: str | None = None, *,
               sources: Sequence[str] = SOURCES, rerank: bool = True, candidates: int | None = None,
               weights: dict[str, float] | None = None, routing: str | None = None) -> SearchResults:
        """Search what was shown and what was said, down to the exact moment.

        ``routing`` picks the strategy:

        * ``"scene"`` (default): the joint keyframe+speech vector ranks scenes in one query; the
          reranker scores only those scenes' sentences to pick each one's moment.
        * ``"adaptive"``: all sources fused, weighted per question by the reranker's confidence that
          it's about speech. Leans ahead on questions about what was said, behind on what was shown.
        * ``"fixed"``: all sources fused with ``weights``.

        Passing ``weights`` or ``sources`` without ``routing`` keeps fusion: adaptive routing, or fixed
        fusion when ``weights`` is given.

        Sources, fused with Reciprocal Rank Fusion (``weights``, default :data:`DEFAULT_WEIGHTS`):

        * ``visual``: keyframe vectors; ``scene``: joint keyframe+speech vectors;
        * ``transcript``: semantic (autoEmbed or client voyage-4); ``text``: full-text BM25;
        * ``rerank``: a Voyage reranker scoring every candidate *sentence*.

        Runs as one native ``$rankFusion`` query when the cluster supports it (8.0+), else per-source
        queries fused client-side; both yield identical rankings. Reranking uses native ``$rerank``
        when enabled (8.3+), else the Voyage API. Every result carries ``ranks`` (per source),
        ``relevance``, ``moment`` (best sentence) and ``moment_link``. A failing source is skipped;
        ``SearchError`` only if every source fails.
        """
        return _search.search(self, query_text, top_k, video_id, sources=sources, rerank=rerank,
                              candidates=candidates, weights=weights, routing=routing)

    def search_transcript(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> SearchResults:
        """Semantic search over spoken dialogue using whichever backend this deployment has."""
        if self.transcript_mode == "autoembed":
            return self.search_auto_embedded_transcript(query_text, top_k=top_k, video_id=video_id)
        return self.search_client_embedded_transcript(query_text, top_k=top_k, video_id=video_id)

    def search_client_embedded_transcript(self, query_text: str, top_k: int = 5,
                                          index_name: str = DEFAULT_TRANSCRIPT_INDEX,
                                          video_id: str | None = None) -> SearchResults:
        """Transcript search with client-side ``voyage-4`` query vectors (no autoEmbed required)."""
        return _search.search_client_transcript(self, query_text, top_k, index_name, video_id)

    def search_auto_embedded_transcript(self, query_text: str, top_k: int = 5, index_name: str = DEFAULT_AUTO_INDEX,
                                        video_id: str | None = None) -> SearchResults:
        """Semantic search over transcripts via Atlas Automated Embedding (autoEmbed)."""
        return _search.search_auto_transcript(self, query_text, top_k, index_name, video_id)

    def search_visual_vector(self, query_text: str, top_k: int = 5, index_name: str = DEFAULT_VISUAL_INDEX,
                             video_id: str | None = None) -> SearchResults:
        """Text-to-image search over keyframes using Voyage AI multimodal vectors."""
        if not query_text:
            return SearchResults()
        return self._vector_search(self._embed_multimodal_query(query_text), index_name=index_name,
                                   path="visual_embedding", top_k=top_k, video_id=video_id)

    def search_scene_vector(self, query_text: str, top_k: int = 5, index_name: str = DEFAULT_SCENE_INDEX,
                            video_id: str | None = None) -> SearchResults:
        """Search joint keyframe+transcript vectors (what was shown *and* said, in one vector)."""
        if not query_text:
            return SearchResults()
        return self._vector_search(self._embed_multimodal_query(query_text), index_name=index_name,
                                   path="scene_embedding", top_k=top_k, video_id=video_id)

    def search_text(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> SearchResults:
        """Full-text (BM25) transcript search via Atlas Search; best for exact names and numbers."""
        return _search.search_text(self, query_text, top_k, video_id)

    def query_cache_info(self) -> dict[str, int]:
        """Query-embedding cache stats: ``hits``, ``misses``, ``size``, ``maxsize``."""
        return self._embedder.query_cache.info()

    def clear_query_cache(self) -> None:
        self._embedder.query_cache.clear()

    # ------------------------------------------------------------------ search steps (seams)
    def _embed_multimodal_query(self, query_text: str) -> list[float]:
        return self._embedder.multimodal_query(query_text)

    def _embed_text_query(self, query_text: str) -> list[float]:
        return self._embedder.text_query(query_text)

    def _vector_search(self, query_vec: list[float], *, index_name: str, path: str, top_k: int,
                       video_id: str | None) -> SearchResults:
        return _search.vector_search(self, query_vec, index_name=index_name, path=path, top_k=top_k,
                                     video_id=video_id)

    def _rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        res = self.vo.rerank(query, texts, model=self.rerank_model)
        return [(r.index, r.relevance_score) for r in res.results]

    def _source_pipeline(self, name: str, query_text: str, n: int, video_id: str | None,
                         mm_vec: list[float] | None) -> list[dict[str, Any]]:
        return _search.source_pipeline(self, name, query_text, n, video_id, mm_vec)

    def _retrieve_native(self, pipelines: dict[str, list[dict[str, Any]]], weights: dict[str, float], n: int) -> Any:
        return _search.retrieve_native(self, pipelines, weights, n)

    def _rerank_scenes(self, query_text: str, docs: dict[tuple[str, int], dict[str, Any]]) -> Any:
        return _search.rerank_scenes(self, query_text, docs)


# ---------------------------------------------------------------------- client construction
def _voyage_client(api_key: str | None) -> Any:
    try:
        import voyageai
    except ImportError as e:  # pragma: no cover - core dependency
        raise DependencyError("voyageai is required: pip install voyageai") from e
    api_key = api_key or os.getenv("VOYAGE_API_KEY")
    if not api_key:
        logger.warning("No Voyage API key provided. Visual vector generation will fail.")
    try:
        return voyageai.Client(api_key=api_key)
    except Exception as e:
        raise CinematlasError(f"Failed to initialize Voyage AI client: {e}") from e


def _openai_client(api_key: str | None) -> Any:
    """Cloud Whisper, if a key is configured; ``None`` means local faster-whisper."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=key)
    except ImportError:
        logger.warning("openai not installed (pip install 'cinematlas[openai]'). Using local Whisper.")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back to local Whisper.")
    return None


def _s3_client() -> Any:
    try:
        import boto3

        return boto3.client("s3")
    except ImportError:
        logger.warning("boto3 not installed (pip install 'cinematlas[s3]'). S3 keyframe uploads disabled.")
    except Exception as e:
        logger.warning(f"Failed to initialize S3 client: {e}. Keyframe uploads disabled.")
    return None
