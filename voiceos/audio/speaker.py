"""Speaker output.

Owns the output stream and blocking writes. Writes run in a thread
executor so the event loop never blocks on audio hardware.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

from voiceos.config.settings import AudioSettings

logger = logging.getLogger(__name__)

# Write in small slices so a future barge-in can interrupt mid-chunk.
_WRITE_SLICE = 2400  # 100 ms @ 24 kHz


class Speaker:
    def __init__(self, settings: AudioSettings) -> None:
        self._settings = settings
        self._stream: sd.OutputStream | None = None
        self._interrupted = False

    def open(self, sample_rate: int) -> None:
        self._stream = sd.OutputStream(
            device=self._settings.output_device,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        self._stream.start()
        logger.info("speaker opened (%d Hz)", sample_rate)

    async def play(self, audio: np.ndarray) -> None:
        """Play mono int16 audio; returns when fully written to the device."""
        await asyncio.get_running_loop().run_in_executor(None, self._write, audio)

    def _write(self, audio: np.ndarray) -> None:
        if self._stream is None:
            return
        for start in range(0, len(audio), _WRITE_SLICE):
            if self._interrupted:
                break
            self._stream.write(audio[start : start + _WRITE_SLICE].reshape(-1, 1))

    def interrupt(self) -> None:
        """Abort in-flight playback (barge-in hook; unused in Phase 1)."""
        self._interrupted = True

    def resume(self) -> None:
        self._interrupted = False

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("speaker closed")
