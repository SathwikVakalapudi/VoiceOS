"""faster-whisper STT implementation.

Inference is CPU/GPU-bound and synchronous, so it runs in a thread
executor to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from voiceos.config.settings import STTSettings
from voiceos.stt.base import BaseSTT, TranscriptionResult

logger = logging.getLogger(__name__)


class FasterWhisperSTT(BaseSTT):
    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._model = None

    async def load(self) -> None:
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_sync)
        logger.info(
            "faster-whisper loaded (model=%s, device=%s)",
            self._settings.model,
            self._settings.device,
        )

    def _load_sync(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._settings.model,
            device=self._settings.device,
            compute_type=self._settings.compute_type,
        )

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("FasterWhisperSTT.load() must be called first")
        if sample_rate != 16000:
            raise ValueError(f"whisper expects 16 kHz audio, got {sample_rate}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> TranscriptionResult:
        segments, info = self._model.transcribe(
            audio,
            language=self._settings.language,
            beam_size=self._settings.beam_size,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_s=info.duration,
        )
