"""Fallback TTS.

Wraps an ordered list of TTS providers and behaves like any other
`BaseTTS`. If a provider fails *before emitting any audio*, the next one
is tried; once audio has started the sentence can't be safely restarted
on a different voice, so the error propagates. This drops in at the
`create_tts` factory — no worker ever knows there is more than one voice.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import numpy as np

from voiceos.tts.base import BaseTTS

logger = logging.getLogger(__name__)


class FallbackTTS(BaseTTS):
    def __init__(self, providers: list[BaseTTS]) -> None:
        if not providers:
            raise ValueError("FallbackTTS needs at least one provider")
        self._providers = providers

    @property
    def sample_rate(self) -> int:
        # The pipeline opens the speaker once at a fixed rate, so every
        # provider must agree; the primary's rate is authoritative.
        return self._providers[0].sample_rate

    async def load(self) -> None:
        for provider in self._providers:
            try:
                await provider.load()
            except Exception:
                logger.exception(
                    "TTS provider failed to load: %s", type(provider).__name__
                )

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        last_error: Exception | None = None
        for provider in self._providers:
            emitted = False
            try:
                async for chunk in provider.synthesize(text):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                if emitted:
                    # Mid-sentence: another provider would repeat audio.
                    raise
                logger.warning(
                    "TTS %s failed before audio — falling back", type(provider).__name__
                )
        if last_error is not None:
            raise last_error

    async def close(self) -> None:
        for provider in self._providers:
            try:
                await provider.close()
            except Exception:
                logger.exception(
                    "TTS provider failed to close: %s", type(provider).__name__
                )
