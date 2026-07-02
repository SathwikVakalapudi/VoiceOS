"""Streaming (partial) transcription.

Phase 1 transcribed complete utterances only. This adds *partial*
transcripts produced while the user is still speaking, which the speech
detector feeds to an endpoint predictor to guess when a turn is done and
close it early (predictive endpointing).

`RollingTranscriber` is honest pseudo-streaming: it re-runs an ordinary
`BaseSTT` over the growing utterance buffer. faster-whisper is not a true
streaming model, so this trades CPU for partials rather than using an
incremental decoder. Give it its own STT instance so partial and final
transcription never hit one model concurrently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from voiceos.interfaces.stt import BaseSTT


class StreamingTranscriber(ABC):
    @abstractmethod
    async def partial(self, audio: np.ndarray, sample_rate: int) -> str:
        """Best transcript of the utterance-so-far (mono float32 in [-1, 1])."""

    async def load(self) -> None:
        """Warm the underlying model. Called once at startup."""

    def reset(self) -> None:
        """Clear any per-utterance state."""

    async def close(self) -> None:
        """Release resources."""


class RollingTranscriber(StreamingTranscriber):
    def __init__(self, stt: BaseSTT) -> None:
        self._stt = stt

    async def load(self) -> None:
        await self._stt.load()

    async def partial(self, audio: np.ndarray, sample_rate: int) -> str:
        result = await self._stt.transcribe(audio, sample_rate)
        return result.text

    async def close(self) -> None:
        await self._stt.close()
