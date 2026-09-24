"""Ingestion: fetch -> scenes -> keyframes -> speech -> vectors -> a gapless replace of the video's scenes.

``run_pipeline`` orchestrates; each step is a method on an :class:`IngestHost` (the engine facade), so
steps stay individually replaceable: a custom transcriber, a stubbed download in tests.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ._utils import assign_segments, build_deep_link, split_long_spans
from .exceptions import IngestionError, IngestionStatus
from .results import IngestResult

logger = logging.getLogger("cinematlas")

Progress = Callable[[str, dict[str, Any]], None]


class IngestHost(Protocol):
    collection: Any
    model: str
    text_model: str
    scene_embeddings: bool
    max_scene_seconds: float | None
    progress: Progress | None

    @property
    def transcript_mode(self) -> str: ...
    def _assert_decodable(self, video_path: str) -> None: ...
    def _detect_scene_spans(self, video_path: str, scene_threshold: float) -> list[tuple[float, float]]: ...
    def _extract_keyframes(self, video_path: str, spans: Sequence[tuple[float, float]],
                           vid: str) -> list[dict[str, Any]]: ...
    def _transcribe_audio_safe(self, audio_path: str | None) -> list[dict[str, Any]]: ...
    def _embed_keyframes_batched(self, scenes: list[dict[str, Any]], batch_size: int = 16, max_retries: int = 3,
                                 transcripts: Sequence[str] | None = None) -> list[list[float] | None]: ...
    def _embed_transcripts(self, transcripts: Sequence[str]) -> list[list[float] | None]: ...
    def _store_vector(self, vec: list[float] | None) -> Any: ...


def run_pipeline(
    host: IngestHost,
    vid: str | None,
    *,
    fetch: Callable[[str], tuple[str, str | None]],
    source: dict[str, Any],
    deep_link_base: str | None,
    scene_threshold: float,
    batch_size: int,
    id_from_content: Callable[[], str] | None = None,
    extra: Callable[[], dict[str, Any]] | None = None,
    progress: Progress | None = None,
) -> IngestResult:
    """Run the ingest pipeline for one video.

    Re-ingesting a video ID replaces its previous version without a gap: the new scenes are inserted
    under a fresh ``ingest_id`` first, then older versions are deleted, so a crash mid-way never leaves
    the video missing. On failure a ``FAILED`` tombstone is written (the previous version stays
    searchable) and :class:`IngestionError` is raised.
    """
    ingest_id = uuid.uuid4().hex
    label = source.get("video_url") or source.get("filename") or "<upload>"
    report = progress or host.progress
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
            host._assert_decodable(video_path)
            stage("fetched", video_id=vid, source=label, audio=audio_path is not None)

            logger.info("Executing PySceneDetect visual cut analysis...")
            spans = split_long_spans(host._detect_scene_spans(video_path, scene_threshold), host.max_scene_seconds)
            scenes = host._extract_keyframes(video_path, spans, vid)
            logger.info(f"Extracted {len(scenes)} visual scenes.")
            stage("scenes", count=len(scenes))

            segments = host._transcribe_audio_safe(audio_path)
            scene_segments = assign_segments([(s["start_sec"], s["end_sec"]) for s in scenes], segments)
            transcripts = [" ".join(seg["text"] for seg in segs) for segs in scene_segments]
            stage("transcribed", sentences=len(segments), spoken_scenes=sum(bool(t) for t in transcripts))

            logger.info("Generating multimodal vectors via Voyage AI...")
            embeddings = host._embed_keyframes_batched(
                scenes, batch_size=batch_size, transcripts=transcripts if host.scene_embeddings else None)
            mode = host.transcript_mode
            client_side = mode == "client"
            transcript_vecs = host._embed_transcripts(transcripts) if client_side else [None] * len(scenes)
            stage("embedded", vectors=sum(v is not None for v in embeddings))

            documents = _documents(host, vid, ingest_id, mode, source, extra, deep_link_base, scenes, transcripts,
                                   scene_segments, embeddings, transcript_vecs)
            count = 0
            if documents:
                result = host.collection.insert_many(documents)
                host.collection.delete_many({"video_id": vid, "ingest_id": {"$ne": ingest_id}})
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
                    host.collection.insert_one({
                        "video_id": vid, **source, "ingest_id": ingest_id,
                        "status": IngestionStatus.FAILED.value, "error_message": str(e), "updated_at": time.time(),
                    })
                except Exception:
                    logger.error("Failed to write FAILED tombstone record.", exc_info=True)
            if isinstance(e, IngestionError):
                raise
            raise IngestionError(f"Video ingestion failed: {e}") from e


def _documents(host: IngestHost, vid: str, ingest_id: str, mode: str, source: dict[str, Any],
               extra: Callable[[], dict[str, Any]] | None, deep_link_base: str | None, scenes: list[dict[str, Any]],
               transcripts: list[str], scene_segments: list[list[dict[str, Any]]], embeddings: list[Any],
               transcript_vecs: list[Any]) -> list[dict[str, Any]]:
    now = time.time()
    client_side = mode == "client"
    common = {
        **source,
        **(extra() if extra else {}),
        "ingest_id": ingest_id,
        # Provenance: which models produced these vectors (needed for re-embedding migrations).
        "embedding_models": {"visual": host.model, "transcript": host.text_model, "transcript_mode": mode,
                             "scene": host.model if host.scene_embeddings else None},
    }
    return [
        {
            "video_id": vid,
            **common,
            "scene_id": scene["scene_id"],
            "timestamp_start": scene["start_sec"],
            "timestamp_end": scene["end_sec"],
            "duration": scene["duration_sec"],
            "transcript": transcript,
            # Timestamped sentences: lets search return the exact moment, not just the scene.
            "segments": [{"start": round(g["start"], 2), "end": round(g["end"], 2), "text": g["text"]} for g in segs],
            "thumbnail_url": scene["thumbnail_url"],
            "deep_link": build_deep_link(deep_link_base, scene["start_sec"]) if deep_link_base else None,
            "visual_embedding": host._store_vector(vec),
            "status": IngestionStatus.COMPLETED.value,
            "updated_at": now,
            **({"transcript_embedding": host._store_vector(tvec)} if client_side else {}),
            # Joint image+speech vector; silent scenes reuse the keyframe vector.
            **({"scene_embedding": host._store_vector(scene.get("scene_embedding") or vec)}
               if host.scene_embeddings else {}),
        }
        for scene, transcript, segs, vec, tvec in zip(scenes, transcripts, scene_segments, embeddings,
                                                      transcript_vecs, strict=True)
    ]
