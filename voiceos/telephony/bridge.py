"""Media-stream transport.

A per-call AudioTransport whose audio flows over a media-server socket
(FreeSWITCH mod_audio_stream WebSocket, Asterisk AudioSocket, or Twilio
Media Streams) instead of a local mic/speaker. The wire protocol lives in
a thin server adapter; this class owns the reusable middle:

  inbound  telephony frame -> decode (mu-law 8k -> int16 16k) -> re-chunk
           to the VAD's fixed frame size -> pipeline audio queue
  outbound pipeline TTS (int16 24k) -> encode (-> mu-law 8k) -> send callback

Re-chunking matters: telephony frames are 20 ms (320 samples @16k) but
Silero VAD requires exactly 512-sample frames, so decoded audio is buffered
and re-emitted at the pipeline's frame size.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import numpy as np

from voiceos.audio.audio_queue import AudioQueue, make_frame
from voiceos.telephony.transcode import TelephonyDecoder, TelephonyEncoder
from voiceos.transport.base import AudioSink, AudioSource, AudioTransport

SendFrame = Callable[[bytes], Awaitable[None]]


class _Rechunker:
    """Buffers a stream of int16 samples and yields fixed-size frames."""

    def __init__(self, frame_size: int) -> None:
        self._frame_size = frame_size
        self._buffer = np.zeros(0, dtype=np.int16)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buffer = np.concatenate([self._buffer, samples])
        frames = []
        while len(self._buffer) >= self._frame_size:
            frames.append(self._buffer[: self._frame_size])
            self._buffer = self._buffer[self._frame_size :]
        return frames


class _StreamSource(AudioSource):
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: AudioQueue | None = None

    def start(self, loop, queue) -> None:
        self._loop, self._queue = loop, queue

    def stop(self) -> None:
        self._loop = self._queue = None

    def push(self, frame) -> None:
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_drop_oldest, frame)


class _StreamSink(AudioSink):
    def __init__(
        self,
        send: SendFrame,
        encoding: str,
        on_interrupt: Callable[[], None] | None = None,
    ) -> None:
        self._send = send
        self._encoding = encoding
        self._on_interrupt = on_interrupt
        self._encoder: TelephonyEncoder | None = None
        self._interrupted = False

    def open(self, sample_rate: int) -> None:
        # sample_rate is the pipeline's TTS rate (e.g. 24000).
        self._encoder = TelephonyEncoder(source_rate=sample_rate, encoding=self._encoding)
        self._interrupted = False

    async def play(self, audio: np.ndarray) -> None:
        if self._interrupted or self._encoder is None:
            return
        await self._send(self._encoder.encode(audio))

    def interrupt(self) -> None:
        # Stop emitting locally; `on_interrupt` also flushes the media server's
        # already-buffered audio (Twilio "clear" / stop streaming) for a clean cut.
        self._interrupted = True
        if self._on_interrupt is not None:
            self._on_interrupt()

    def resume(self) -> None:
        self._interrupted = False

    def close(self) -> None:
        self._encoder = None


class MediaStreamTransport(AudioTransport):
    def __init__(
        self,
        send: SendFrame,
        *,
        input_sample_rate: int = 16000,
        frame_size: int = 512,
        encoding: str = "mulaw",
        on_interrupt: Callable[[], None] | None = None,
    ) -> None:
        self._input_sample_rate = input_sample_rate
        self._decoder = TelephonyDecoder(target_rate=input_sample_rate, encoding=encoding)
        self._rechunker = _Rechunker(frame_size)
        self._source = _StreamSource()
        self._sink = _StreamSink(send, encoding, on_interrupt)

    @property
    def source(self) -> AudioSource:
        return self._source

    @property
    def sink(self) -> AudioSink:
        return self._sink

    def on_inbound_audio(self, payload: bytes) -> None:
        """Feed one raw telephony audio payload from the media server."""
        samples = self._decoder.decode(payload)
        for frame in self._rechunker.push(samples):
            self._source.push(make_frame(frame, self._input_sample_rate))
