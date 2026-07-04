"""Piper TTS — fully local, offline neural text-to-speech.

Piper (github.com/rhasspy/piper) runs on CPU, is fast, and has good voices for
many languages including Hindi/Indian languages — the standard local TTS for an
offline voice agent. This wraps the ``piper`` CLI: text in on stdin, raw 16-bit
mono PCM out on stdout, streamed in chunks (so playback can start immediately,
matching the cloud providers' streaming behaviour).

Setup (one time, on the machine running VoiceOS):
    pip install piper-tts                       # provides the `piper` command
    # download a voice (.onnx + .onnx.json) from
    # https://huggingface.co/rhasspy/piper-voices  e.g. hi_IN-* for Hindi
    # then point VOICEOS_TTS__PIPER_MODEL at the .onnx file.

Config (.env): VOICEOS_TTS__PROVIDER=piper, VOICEOS_TTS__PIPER_MODEL=/path/voice.onnx,
VOICEOS_TTS__PIPER_SAMPLE_RATE=22050 (match the voice's rate from its .onnx.json).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from voiceos.config.settings import TTSSettings
from voiceos.interfaces.tts import BaseTTS

logger = logging.getLogger(__name__)


class PiperTTS(BaseTTS):
    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings
        self._binary: str | None = None

    @property
    def sample_rate(self) -> int:
        return self._settings.piper_sample_rate

    async def load(self) -> None:
        self._binary = shutil.which(self._settings.piper_binary)
        if self._binary is None:
            raise RuntimeError(
                f"piper binary '{self._settings.piper_binary}' not found on PATH — "
                "install with `pip install piper-tts`"
            )
        if not self._settings.piper_model or not Path(self._settings.piper_model).exists():
            raise RuntimeError(
                "Piper voice model not found: set VOICEOS_TTS__PIPER_MODEL to a "
                ".onnx voice file (download from huggingface.co/rhasspy/piper-voices)"
            )
        logger.info(
            "Piper TTS ready (model=%s, %d Hz)",
            Path(self._settings.piper_model).name,
            self._settings.piper_sample_rate,
        )

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        text = text.strip()
        if not text or self._binary is None:
            return
        args = [self._binary, "--model", self._settings.piper_model, "--output-raw"]
        if self._settings.piper_speaker is not None:
            args += ["--speaker", str(self._settings.piper_speaker)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(text.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        pending = b""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            pending += chunk
            usable = len(pending) - (len(pending) % 2)  # int16 alignment
            if usable:
                yield np.frombuffer(pending[:usable], dtype="<i2")
                pending = pending[usable:]

        rc = await proc.wait()
        if rc != 0:
            err = (await proc.stderr.read()).decode("utf-8", "replace")[:300]
            logger.warning("piper exited %d: %s", rc, err)

    async def close(self) -> None:
        pass
