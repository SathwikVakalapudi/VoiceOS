"""edge-tts: free Microsoft neural voices.

Zero-cost fallback for machines that can't run the Svara server.
Needs internet; no API key. The service streams MP3, which we decode
to PCM with PyAV (already installed as a faster-whisper dependency).

Indian voices: en-IN-NeerjaNeural / en-IN-PrabhatNeural,
hi-IN-SwaraNeural / hi-IN-MadhurNeural, plus hundreds more
(`edge-tts --list-voices`).
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import AsyncIterator

import numpy as np

from voiceos.config.settings import TTSSettings
from voiceos.tts.base import BaseTTS

logger = logging.getLogger(__name__)


def _mp3_to_pcm(mp3_bytes: bytes, target_rate: int) -> np.ndarray:
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(mp3_bytes)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):  # flush
        chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks).astype(np.int16, copy=False)


class EdgeTTS(BaseTTS):
    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings

    @property
    def sample_rate(self) -> int:
        return self._settings.sample_rate

    async def load(self) -> None:
        try:
            import av  # noqa: F401
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts provider needs the 'edge-tts' and 'av' packages: "
                "pip install edge-tts av"
            ) from exc
        logger.info("edge-tts ready (voice=%s)", self._settings.edge_voice)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        import edge_tts

        # The Edge service connection is easily dropped on unstable
        # networks; each attempt reconnects from scratch.
        mp3 = bytearray()
        last_exc: Exception | None = None
        for attempt in range(4):
            mp3.clear()
            try:
                communicate = edge_tts.Communicate(
                    text,
                    voice=self._settings.edge_voice,
                    connect_timeout=3,  # fail fast on flaky networks; we retry
                )
                async for message in communicate.stream():
                    if message["type"] == "audio":
                        mp3.extend(message["data"])
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "edge-tts attempt %d/4 failed (%s); retrying",
                    attempt + 1,
                    type(exc).__name__,
                )
                await asyncio.sleep(0.3 * 2**attempt)  # 0.3, 0.6, 1.2s
        if last_exc is not None:
            raise last_exc
        if not mp3:
            return
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(
            None, _mp3_to_pcm, bytes(mp3), self._settings.sample_rate
        )
        if len(pcm):
            yield pcm
