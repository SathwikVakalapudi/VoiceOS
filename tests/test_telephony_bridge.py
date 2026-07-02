"""MediaStreamTransport tests: inbound re-chunking + outbound encode/barge-in."""

import asyncio

import numpy as np

from voiceos.audio.audio_queue import AudioQueue
from voiceos.telephony.bridge import MediaStreamTransport, _Rechunker


def test_rechunker_emits_fixed_size_frames():
    rc = _Rechunker(frame_size=512)
    assert rc.push(np.zeros(300, dtype=np.int16)) == []          # buffered
    frames = rc.push(np.zeros(800, dtype=np.int16))              # 1100 total
    assert len(frames) == 2 and all(len(f) == 512 for f in frames)


def test_inbound_audio_is_decoded_rechunked_and_queued():
    sent: list[bytes] = []

    async def send(b):
        sent.append(b)

    t = MediaStreamTransport(send, input_sample_rate=16000, frame_size=512, encoding="pcm16")
    loop = asyncio.new_event_loop()
    queue = AudioQueue()
    t.source.start(loop, queue)

    # 512 samples @16k need ~256 @8k; feed enough linear-8k audio for >=1 frame.
    t.on_inbound_audio(np.zeros(400, dtype="<i2").tobytes())  # 800 bytes @8k
    loop.run_until_complete(asyncio.sleep(0))                 # flush call_soon
    assert queue.qsize() >= 1
    frame = loop.run_until_complete(queue.get())
    assert len(frame.data) == 512 and frame.sample_rate == 16000
    loop.close()


async def test_outbound_play_encodes_and_sends():
    sent: list[bytes] = []

    async def send(b):
        sent.append(b)

    t = MediaStreamTransport(send, encoding="pcm16")
    t.sink.open(24000)                       # pipeline TTS rate
    await t.sink.play(np.zeros(480, dtype="<i2"))  # 20 ms @24k
    assert len(sent) == 1 and len(sent[0]) > 0     # ~160 samples @8k


async def test_barge_in_stops_outbound_until_resumed():
    sent: list[bytes] = []

    async def send(b):
        sent.append(b)

    t = MediaStreamTransport(send, encoding="pcm16")
    t.sink.open(24000)
    t.sink.interrupt()
    await t.sink.play(np.zeros(480, dtype="<i2"))
    assert sent == []                        # dropped while interrupted
    t.sink.resume()
    await t.sink.play(np.zeros(480, dtype="<i2"))
    assert len(sent) == 1
