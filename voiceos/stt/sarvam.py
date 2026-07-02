"""Sarvam AI hosted STT (Saaras / Saarika models).

Strong on Indian accents and all major Indian languages, and moves
transcription off the local CPU entirely.

    POST https://api.sarvam.ai/speech-to-text
    header: api-subscription-key
    multipart: file (wav), model, language_code (optional -> autodetect)
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave

import httpx
import numpy as np

from voiceos.config.settings import STTSettings
from voiceos.stt.base import BaseSTT, TranscriptionResult
from voiceos.utils.audio import float32_to_int16

logger = logging.getLogger(__name__)

_API_BASE = "https://api.sarvam.ai"


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(float32_to_int16(audio).tobytes())
    return buffer.getvalue()


class SarvamSTT(BaseSTT):
    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def load(self) -> None:
        if not self._settings.sarvam_api_key:
            raise RuntimeError(
                "Sarvam STT needs an API key: set VOICEOS_STT__SARVAM_API_KEY"
            )
        # Short connect timeout: on unstable networks a dead connection
        # attempt must fail in seconds, not stall the whole turn.
        self._client = httpx.AsyncClient(
            base_url=_API_BASE,
            headers={"api-subscription-key": self._settings.sarvam_api_key},
            timeout=httpx.Timeout(self._settings.sarvam_timeout_s, connect=3.0),
        )
        logger.info("Sarvam STT ready (model=%s)", self._settings.sarvam_model)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        if self._client is None:
            raise RuntimeError("SarvamSTT.load() must be called first")
        wav = _to_wav_bytes(audio, sample_rate)
        data = {"model": self._settings.sarvam_model}
        if self._settings.sarvam_language:
            data["language_code"] = self._settings.sarvam_language

        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                response = await self._client.post(
                    "/speech-to-text",
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                    data=data,
                )
                response.raise_for_status()
                payload = response.json()
                return TranscriptionResult(
                    text=(payload.get("transcript") or "").strip(),
                    language=payload.get("language_code"),
                    duration_s=len(audio) / sample_rate,
                )
            except httpx.HTTPStatusError:
                raise  # auth/quota errors won't improve on retry
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Sarvam STT attempt %d/4 failed (%s); retrying",
                    attempt + 1,
                    type(exc).__name__,
                )
                await asyncio.sleep(0.3 * 2**attempt)  # 0.3, 0.6, 1.2s
        raise last_exc  # type: ignore[misc]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
