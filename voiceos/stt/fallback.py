"""Fallback STT.

Wraps an ordered list of STT providers and behaves like any other
`BaseSTT`. `transcribe` is a single call per utterance, so on failure we
simply try the next provider. Drops in at the `create_stt` factory with
no changes to the STT worker.
"""

from __future__ import annotations

import logging

import numpy as np

from voiceos.stt.base import BaseSTT, TranscriptionResult

logger = logging.getLogger(__name__)


class FallbackSTT(BaseSTT):
    def __init__(self, providers: list[BaseSTT]) -> None:
        if not providers:
            raise ValueError("FallbackSTT needs at least one provider")
        self._providers = providers

    async def load(self) -> None:
        for provider in self._providers:
            try:
                await provider.load()
            except Exception:
                logger.exception(
                    "STT provider failed to load: %s", type(provider).__name__
                )

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                return await provider.transcribe(audio, sample_rate)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "STT %s failed — falling back", type(provider).__name__
                )
        assert last_error is not None  # providers is non-empty
        raise last_error

    async def close(self) -> None:
        for provider in self._providers:
            try:
                await provider.close()
            except Exception:
                logger.exception(
                    "STT provider failed to close: %s", type(provider).__name__
                )
