"""Transport contracts.

`AudioSource` produces input frames (mic, phone call, WebRTC track);
`AudioSink` consumes output audio (speaker, phone call, WebRTC track). An
`AudioTransport` bundles one of each. The existing Microphone and Speaker
already satisfy these contracts structurally — the abstraction just names
the seam so telephony can replace them without touching the pipeline.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np

from voiceos.audio.audio_queue import AudioQueue


class AudioSource(ABC):
    @abstractmethod
    def start(self, loop: asyncio.AbstractEventLoop, queue: AudioQueue) -> None:
        """Begin pushing captured int16 frames into `queue`."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capture and release the input device/stream."""


class AudioSink(ABC):
    @abstractmethod
    def open(self, sample_rate: int) -> None:
        """Prepare the output at the pipeline's TTS sample rate."""

    @abstractmethod
    async def play(self, audio: np.ndarray) -> None:
        """Play one mono int16 buffer; return when fully written."""

    def interrupt(self) -> None:
        """Abort in-flight playback (barge-in)."""

    def resume(self) -> None:
        """Clear the interrupt flag so playback can continue."""

    def close(self) -> None:
        """Release the output device/stream."""


class AudioTransport(ABC):
    @property
    @abstractmethod
    def source(self) -> AudioSource:
        """The input side."""

    @property
    @abstractmethod
    def sink(self) -> AudioSink:
        """The output side."""
