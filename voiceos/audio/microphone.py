"""Microphone capture.

Streams fixed-size int16 frames from the system microphone into an
AudioQueue. The sounddevice callback runs on a PortAudio thread, so
frames are handed to the event loop via call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

from voiceos.audio.audio_queue import AudioQueue, make_frame
from voiceos.config.settings import AudioSettings

logger = logging.getLogger(__name__)


class Microphone:
    def __init__(self, settings: AudioSettings) -> None:
        self._settings = settings
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: AudioQueue | None = None

    def start(self, loop: asyncio.AbstractEventLoop, queue: AudioQueue) -> None:
        self._loop = loop
        self._queue = queue
        self._stream = sd.InputStream(
            device=self._settings.input_device,
            samplerate=self._settings.input_sample_rate,
            blocksize=self._settings.frame_size,
            channels=self._settings.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "microphone started (%d Hz, %d-sample frames)",
            self._settings.input_sample_rate,
            self._settings.frame_size,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("input stream status: %s", status)
        if self._loop is None or self._queue is None:
            return
        frame = make_frame(indata[:, 0].copy(), self._settings.input_sample_rate)
        self._loop.call_soon_threadsafe(self._queue.put_drop_oldest, frame)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("microphone stopped")

    @staticmethod
    def list_devices() -> str:
        return str(sd.query_devices())
