"""Vectors: keyframe and joint keyframe+speech embeddings, transcript embeddings, query embeddings."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from typing import Any

from bson.binary import Binary, BinaryVectorDtype

from ._cache import LRUCache
from .exceptions import SearchError
from .usage import with_retries

logger = logging.getLogger("cinematlas")


class Embedder:
    """Voyage calls with retries and alignment checks, BSON vector storage, and a query cache."""

    def __init__(self, vo: Any, *, model: str, text_model: str, bson_vectors: bool = True,
                 query_cache_size: int = 256):
        self.vo = vo
        self.model = model
        self.text_model = text_model
        self.bson_vectors = bson_vectors
        self.query_cache = LRUCache(query_cache_size)  # repeated queries skip the Voyage round trip

    # ------------------------------------------------------------------ storage
    def store_vector(self, vec: list[float] | None) -> Binary | list[float] | None:
        """Persist as a BSON float32 vector (3.2x smaller than an array of doubles) when enabled."""
        if vec is None or not self.bson_vectors:
            return vec
        return Binary.from_vector([float(x) for x in vec], BinaryVectorDtype.FLOAT32)

    # ------------------------------------------------------------------ documents
    def embed_transcripts(self, transcripts: Sequence[str], batch_size: int = 64,
                          max_retries: int = 3) -> list[list[float] | None]:
        """Client-side transcript vectors (``voyage-4`` family); ``None`` for empty text or failed batches."""
        out: list[list[float] | None] = [None] * len(transcripts)
        todo = [i for i, t in enumerate(transcripts) if t.strip()]
        for b in range(0, len(todo), batch_size):
            idx = todo[b : b + batch_size]
            texts = [transcripts[i] for i in idx]
            vecs, _ = with_retries(lambda t=texts: self.vo.embed(t, model=self.text_model, input_type="document"),
                                   expect=len(idx), retries=max_retries, what="Voyage transcript embedding")
            for i, vec in zip(idx, vecs or [None] * len(idx), strict=True):
                out[i] = vec
        return out

    def multimodal_aligned(self, inputs: list[list[Any]], max_retries: int) -> list[Any] | None:
        """One multimodal_embed call with retries; ``None`` if every attempt failed or misaligned."""
        vecs, _ = with_retries(lambda: self.vo.multimodal_embed(inputs=inputs, model=self.model, input_type="document"),
                               expect=len(inputs), retries=max_retries, what="Voyage multimodal embedding")
        if vecs is None:
            logger.error("Max retries reached for batch. Populating null vector embeddings.")
        return vecs

    def embed_keyframes_batched(self, scenes: list[dict[str, Any]], batch_size: int = 16, max_retries: int = 3,
                                transcripts: Sequence[str] | None = None) -> list[list[float] | None]:
        """Embed each scene's keyframe in low-RAM batches.

        Returns exactly one entry per scene, aligned by position; scenes with no usable keyframe (or
        whose batch exhausted its retries) get ``None``. With ``transcripts``, also computes a *joint*
        image+transcript vector per scene (voyage-multimodal-3.5 accepts interleaved inputs) and stores
        it as ``scene["scene_embedding"]``; scenes without speech skip the extra call. Keyframe images
        are closed and removed from the scene dicts as each batch completes.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        all_embeddings: list[list[float] | None] = []
        for i in range(0, len(scenes), batch_size):
            batch = scenes[i : i + batch_size]
            vectors: list[list[float] | None] = [None] * len(batch)
            with_image = [j for j, s in enumerate(batch) if s.get("keyframe_image") is not None]
            if with_image:
                vecs = self.multimodal_aligned([[batch[j]["keyframe_image"]] for j in with_image], max_retries)
                for j, vec in zip(with_image, vecs or [None] * len(with_image), strict=True):
                    vectors[j] = vec
                if transcripts is not None:
                    spoken = [j for j in with_image if transcripts[i + j].strip()]
                    if spoken:
                        joint = self.multimodal_aligned(
                            [[batch[j]["keyframe_image"], transcripts[i + j]] for j in spoken], max_retries)
                        for j, vec in zip(spoken, joint or [None] * len(spoken), strict=True):
                            batch[j]["scene_embedding"] = vec
            all_embeddings.extend(vectors)
            for s in batch:  # free RAM: close and discard this batch's images now
                img = s.pop("keyframe_image", None)
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
            gc.collect()
        return all_embeddings

    # ------------------------------------------------------------------ queries
    def multimodal_query(self, query_text: str) -> list[float]:
        """Query vector for the keyframe and scene sources (cached)."""
        def compute() -> list[float]:
            return self.vo.multimodal_embed(inputs=[[query_text]], model=self.model, input_type="query").embeddings[0]
        try:
            return self.query_cache.get_or_compute(("multimodal", self.model, query_text), compute)
        except Exception as e:
            raise SearchError(f"Voyage AI query vectorization failed: {e}") from e

    def text_query(self, query_text: str) -> list[float]:
        """Query vector for client-side transcript search (cached)."""
        def compute() -> list[float]:
            return self.vo.embed([query_text], model=self.text_model, input_type="query").embeddings[0]
        try:
            return self.query_cache.get_or_compute(("text", self.text_model, query_text), compute)
        except Exception as e:
            raise SearchError(f"Voyage AI query vectorization failed: {e}") from e
