"""Speech to timestamped sentences: the OpenAI Whisper API if configured, else faster-whisper locally."""

from __future__ import annotations

import logging
import os
from typing import Any

from ._utils import normalize_segments
from .exceptions import DependencyError

logger = logging.getLogger("cinematlas")

# "small" is the accuracy floor we measured: "base" mishears ordinary words ("long trunks" -> "long hunts").
DEFAULT_WHISPER_MODEL = "small"


class Transcriber:
    """Owns the speech-to-text backend and caches the local model across videos."""

    def __init__(self, *, openai_client: Any = None, whisper_model: str = DEFAULT_WHISPER_MODEL):
        self.openai_client = openai_client
        self.whisper_model = whisper_model
        self.local_model: Any = None  # faster-whisper model, loaded on first use

    def transcribe_safe(self, audio_path: str | None) -> list[dict[str, Any]]:
        """Transcribe with cloud/local failover; always deletes the audio file; ``[]`` on failure."""
        if not audio_path or not os.path.exists(audio_path):
            return []
        segments: list[dict[str, Any]] = []
        try:
            if self.openai_client:
                logger.info("Transcribing audio via OpenAI Whisper API...")
                with open(audio_path, "rb") as af:
                    tx_res = self.openai_client.audio.transcriptions.create(
                        model="whisper-1", file=af, response_format="verbose_json")
                segments = normalize_segments(getattr(tx_res, "segments", None))
            else:
                segments = self.transcribe_locally(audio_path)
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

    def transcribe_locally(self, audio_path: str) -> list[dict[str, Any]]:
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
        if self.local_model is None:
            self.local_model = WhisperModel(self.whisper_model, device="auto", compute_type="int8")
        segs, _info = self.local_model.transcribe(audio_path, vad_filter=True, word_timestamps=True)
        return normalize_segments(list(segs))
