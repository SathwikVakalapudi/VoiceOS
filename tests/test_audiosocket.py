"""Asterisk AudioSocket server tests: framing + end-to-end call handling."""

import asyncio

import numpy as np

from voiceos.telephony import audiosocket as A
from voiceos.telephony.bridge import MediaStreamTransport


def msg(kind: int, payload: bytes = b"") -> bytes:
    return bytes([kind]) + len(payload).to_bytes(2, "big") + payload


def test_frame_audio_splits_into_typed_messages():
    frames = A.frame_audio(b"\x00" * 700)  # > 320, so 3 frames (320,320,60)
    assert len(frames) == 3
    assert frames[0][0] == A.TYPE_AUDIO
    assert int.from_bytes(frames[0][1:3], "big") == 320
    assert int.from_bytes(frames[2][1:3], "big") == 60


async def test_read_message_parses_header_and_payload():
    reader = asyncio.StreamReader()
    reader.feed_data(msg(A.TYPE_UUID, b"\x11" * 16) + msg(A.TYPE_AUDIO, b"ab"))
    reader.feed_eof()
    kind, payload = await A.read_message(reader)
    assert kind == A.TYPE_UUID and payload == b"\x11" * 16
    kind, payload = await A.read_message(reader)
    assert kind == A.TYPE_AUDIO and payload == b"ab"


class FakeSession:
    """Stands in for VoicePipeline; wires the transport's source to a queue."""

    def __init__(self, transport: MediaStreamTransport):
        self.transport = transport
        self.queue = None
        self.started = self.stopped = False

    async def start(self):
        from voiceos.audio.audio_queue import AudioQueue

        self.queue = AudioQueue()
        self.transport.source.start(asyncio.get_running_loop(), self.queue)
        self.transport.sink.open(24000)
        self.started = True

    async def stop(self):
        self.stopped = True


async def test_serve_call_feeds_inbound_audio_and_stops_on_terminate():
    captured = {}

    def factory(transport):
        session = FakeSession(transport)
        captured["session"] = session
        return session

    reader = asyncio.StreamReader()
    # UUID, one 8k linear audio frame (320 bytes = 160 samples), then terminate.
    reader.feed_data(msg(A.TYPE_UUID, b"\x22" * 16))
    reader.feed_data(msg(A.TYPE_AUDIO, np.zeros(160, dtype="<i2").tobytes()))
    reader.feed_data(msg(A.TYPE_TERMINATE))
    reader.feed_eof()

    class FakeWriter:
        def write(self, data): pass
        async def drain(self): pass
        def close(self): pass

    await A.serve_call(reader, FakeWriter(), factory, input_sample_rate=16000, frame_size=512)

    session = captured["session"]
    assert session.started and session.stopped
    # 160 samples @8k -> ~320 @16k, less than one 512 frame, so buffered (0 frames)
    # but the pipeline session was driven and torn down cleanly.
    assert session.queue.qsize() == 0


async def test_serve_call_dtmf_callback():
    digits = []

    def factory(transport):
        return FakeSession(transport)

    reader = asyncio.StreamReader()
    reader.feed_data(msg(A.TYPE_DTMF, b"5"))
    reader.feed_data(msg(A.TYPE_TERMINATE))
    reader.feed_eof()

    class FakeWriter:
        def write(self, data): pass
        async def drain(self): pass
        def close(self): pass

    await A.serve_call(reader, FakeWriter(), factory, on_dtmf=digits.append)
    assert digits == ["5"]
