"""Svara-TTS via its self-hosted inference server.

The Svara inference server (github.com/Kenpath/svara-tts-inference)
exposes an OpenAI-compatible endpoint:

    POST /v1/audio/speech  {model, voice, input, response_format, stream}

With response_format="pcm" it streams raw signed 16-bit LE mono audio
at 24 kHz, which we re-chunk into int16 numpy arrays.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx
import numpy as np

from voiceos.config.settings import TTSSettings
from voiceos.tts.base import BaseTTS

logger = logging.getLogger(__name__)


class SvaraTTS(BaseTTS):
    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def sample_rate(self) -> int:
        return self._settings.sample_rate

    async def load(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            headers={"Authorization": f"Bearer {self._settings.api_key}"},
            timeout=httpx.Timeout(self._settings.timeout_s, connect=10.0),
        )
        try:
            # The Svara server exposes /health at the root, not under /v1.
            root = self._settings.base_url.rsplit("/v1", 1)[0]
            response = await self._client.get(f"{root}/health")
            response.raise_for_status()
            logger.info("TTS endpoint reachable at %s", self._settings.base_url)
        except httpx.HTTPError as exc:
            logger.warning(
                "TTS endpoint %s not reachable yet (%s) — is the Svara server running?",
                self._settings.base_url,
                exc,
            )

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        if self._client is None:
            raise RuntimeError("SvaraTTS.load() must be called first")
        payload = {
            "model": self._settings.model,
            "voice": self._settings.voice,
            "input": text,
            "response_format": "pcm",
            "stream": True,
        }
        async with self._client.stream("POST", "/audio/speech", json=payload) as response:
            response.raise_for_status()
            pending = b""
            async for chunk in response.aiter_bytes():
                pending += chunk
                usable = len(pending) - (len(pending) % 2)  # int16 alignment
                if usable:
                    yield np.frombuffer(pending[:usable], dtype="<i2")
                    pending = pending[usable:]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
