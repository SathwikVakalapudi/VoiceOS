"""Audio stream primitives shared across the pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AudioFrame:
    """One fixed-size chunk of mono PCM audio."""

    data: np.ndarray          # int16
    sample_rate: int
    timestamp: float = 0.0    # time.monotonic() at capture

    @property
    def duration_ms(self) -> float:
        return len(self.data) / self.sample_rate * 1000.0


@dataclass(slots=True)
class Utterance:
    """One complete stretch of user speech, ready for STT."""

    audio: np.ndarray         # int16, includes pre-roll padding
    sample_rate: int
    duration_s: float


class AudioQueue:
    """Bounded asyncio queue that drops the oldest frame when full.

    Real-time capture must never block on a slow consumer; losing the
    oldest audio is the least-bad failure mode.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def put_drop_oldest(self, frame: AudioFrame) -> None:
        """Synchronous put for use via loop.call_soon_threadsafe."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(frame)

    async def get(self) -> AudioFrame:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()


def make_frame(data: np.ndarray, sample_rate: int) -> AudioFrame:
    return AudioFrame(data=data, sample_rate=sample_rate, timestamp=time.monotonic())
