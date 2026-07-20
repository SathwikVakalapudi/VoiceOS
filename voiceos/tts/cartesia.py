"""Cartesia Sonic TTS.

The lowest-latency hosted TTS available — built for voice agents.
Streams raw PCM (s16le) so playback starts as soon as the first bytes
arrive, with no container decoding step at all.

    POST https://api.cartesia.ai/tts/bytes
    headers: Authorization: Bearer <key>, Cartesia-Version
    body: {model_id, transcript, voice: {mode, id}, output_format, language}
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx
import numpy as np

from voiceos.config.settings import TTSSettings
from voiceos.tts.base import BaseTTS
from voiceos.utils.http import RETRYABLE_STATUS, error_detail

logger = logging.getLogger(__name__)

_API_BASE = "https://api.cartesia.ai"
_API_VERSION = "2026-03-01"


class CartesiaTTS(BaseTTS):
    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def sample_rate(self) -> int:
        return self._settings.sample_rate

    async def load(self) -> None:
        if not self._settings.cartesia_api_key:
            raise RuntimeError(
                "Cartesia TTS needs an API key: set VOICEOS_TTS__CARTESIA_API_KEY"
            )
        self._client = httpx.AsyncClient(
            base_url=_API_BASE,
            headers={
                "Authorization": f"Bearer {self._settings.cartesia_api_key}",
                "Cartesia-Version": _API_VERSION,
            },
            timeout=httpx.Timeout(self._settings.cartesia_timeout_s, connect=3.0),
        )
        logger.info(
            "Cartesia TTS ready (model=%s, voice=%s)",
            self._settings.cartesia_model,
            self._settings.cartesia_voice_id,
        )

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        if self._client is None:
            raise RuntimeError("CartesiaTTS.load() must be called first")
        payload = {
            "model_id": self._settings.cartesia_model,
            "transcript": text,
            "voice": {"mode": "id", "id": self._settings.cartesia_voice_id},
            "language": self._settings.cartesia_language,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self._settings.sample_rate,
            },
        }

        # Retry with backoff while nothing has been emitted; once audio is
        # flowing, a dropped stream ends the sentence with what we have.
        for attempt in range(4):
            emitted = False
            try:
                async with self._client.stream(
                    "POST", "/tts/bytes", json=payload
                ) as response:
                    if response.status_code >= 400:
                        detail = await error_detail(response)
                        if response.status_code in RETRYABLE_STATUS and attempt < 3:
                            logger.warning(
                                "Cartesia %s (%s); retrying",
                                response.status_code, detail,
                            )
                            await asyncio.sleep(0.3 * 2**attempt)
                            continue
                        logger.error(
                            "Cartesia failed: %s — %s", response.status_code, detail
                        )
                        response.raise_for_status()
                    pending = b""
                    async for chunk in response.aiter_bytes():
                        pending += chunk
                        usable = len(pending) - (len(pending) % 2)  # int16 alignment
                        if usable:
                            emitted = True
                            yield np.frombuffer(pending[:usable], dtype="<i2")
                            pending = pending[usable:]
                return
            except httpx.HTTPStatusError:
                raise  # non-retryable: bad key, quota exhausted, bad voice id
            except httpx.HTTPError as exc:
                if emitted:
                    logger.warning(
                        "Cartesia stream dropped mid-sentence (%s); "
                        "continuing with partial audio",
                        type(exc).__name__,
                    )
                    return
                if attempt < 3:
                    logger.warning(
                        "Cartesia attempt %d/4 failed (%s); retrying",
                        attempt + 1,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(0.3 * 2**attempt)  # 0.3, 0.6, 1.2s
                    continue
                raise

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
