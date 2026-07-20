"""Sarvam streaming STT over WebSocket.

Batch STT cannot start until the user stops talking, so its whole latency lands
inside the turn: audio ends, upload, transcribe, wait. Measured here that was
~400 ms, the largest slice of a ~1180 ms response.

Streaming moves that work *underneath* the speech. Audio is uploaded as it is
captured and transcribed progressively, so when the endpointer commits there is
only the tail left to finalise.

Unlike `RollingTranscriber` — which fakes streaming by re-running a batch model
over a growing buffer, at quadratic cost — this is a genuine streaming endpoint:
one connection per turn, each sample sent once.

    session = SarvamStreamingSTT(settings)
    await session.start()
    async for chunk in mic:
        await session.send(chunk)
    text, language = await session.finish()

Protocol: wss://api.sarvam.ai/speech-to-text/ws, `Api-Subscription-Key` header,
client sends {"audio": {"data": b64, ...}}, server replies {"type": "data",
"data": {"transcript": ...}}, and {"type": "flush"} finalises.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import numpy as np

from voiceos.config.settings import STTSettings
from voiceos.utils.audio import float32_to_int16

logger = logging.getLogger(__name__)

_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"


class SarvamStreamingSTT:
    """One streaming transcription session — start, send, finish, repeat."""

    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._parts: list[str] = []
        self._language: str | None = None
        self._error: str | None = None
        self._done = asyncio.Event()
        self._last_data = 0.0

    @property
    def partial(self) -> str:
        """Everything transcribed so far this turn."""
        return " ".join(self._parts).strip()

    async def start(self) -> None:
        import websockets

        if not self._settings.sarvam_api_key:
            raise RuntimeError("Sarvam streaming needs VOICEOS_STT__SARVAM_API_KEY")

        params = [
            f"model={self._settings.sarvam_streaming_model}",
            "mode=transcribe",          # saaras models also translate; we never want that
            f"sample_rate={self._settings.sarvam_streaming_sample_rate}",
            "input_audio_codec=pcm_s16le",
        ]
        if self._settings.sarvam_language:
            params.append(f"language-code={self._settings.sarvam_language}")

        self._parts, self._language, self._error = [], None, None
        self._done.clear()
        self._ws = await websockets.connect(
            f"{_WS_URL}?{'&'.join(params)}",
            additional_headers={"Api-Subscription-Key": self._settings.sarvam_api_key},
            open_timeout=self._settings.sarvam_timeout_s,
        )
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        """Collect transcript fragments until the socket closes."""
        try:
            async for message in self._ws:
                payload = json.loads(message)
                kind = payload.get("type")
                data = payload.get("data") or {}
                if kind == "data":
                    text = (data.get("transcript") or "").strip()
                    if text:
                        self._parts.append(text)
                    self._language = data.get("language_code") or self._language
                    self._last_data = asyncio.get_running_loop().time()
                elif kind == "error":
                    self._error = str(data.get("error") or payload)
                    logger.error("Sarvam streaming error: %s", self._error)
                    break
        except Exception as exc:            # socket closed mid-turn
            if not self._done.is_set():
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._done.set()

    async def send(self, audio: np.ndarray) -> None:
        """Push one chunk. int16 or float32, mono, at the configured rate."""
        if self._ws is None:
            raise RuntimeError("SarvamStreamingSTT.start() must be called first")
        pcm = audio if audio.dtype == np.int16 else float32_to_int16(audio)
        await self._ws.send(json.dumps({
            "audio": {
                "data": base64.b64encode(pcm.tobytes()).decode("ascii"),
                "sample_rate": str(self._settings.sarvam_streaming_sample_rate),
                "encoding": "audio/wav",
            }
        }))

    async def finish(self, quiet_ms: int = 350, hard_timeout_s: float = 3.0
                     ) -> tuple[str, str | None]:
        """Flush, wait briefly for the tail, and return (transcript, language).

        The server does *not* close the socket after a flush, so waiting for
        socket-close hangs until the read timeout — an early version sat here
        for 12 s with the transcript already in hand. Instead, settle: return
        once no new fragment has arrived for `quiet_ms`, bounded by
        `hard_timeout_s` so a stalled socket cannot stall the turn.
        """
        if self._ws is None:
            return "", None
        loop = asyncio.get_running_loop()
        try:
            await self._ws.send(json.dumps({"type": "flush"}))
            deadline = loop.time() + hard_timeout_s
            while loop.time() < deadline and not self._done.is_set():
                # The quiet window only means "the tail has stopped arriving",
                # which is meaningless before anything has arrived at all. An
                # earlier version broke immediately in that case and returned an
                # empty transcript, silently falling back to batch every turn.
                if self._parts and (loop.time() - self._last_data) * 1000 >= quiet_ms:
                    break
                await asyncio.sleep(0.02)
        except Exception as exc:
            logger.warning("Sarvam streaming finish: %s", type(exc).__name__)
        finally:
            await self.close()
        if self._error and not self._parts:
            raise RuntimeError(f"Sarvam streaming failed: {self._error}")
        return self.partial, self._language

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
