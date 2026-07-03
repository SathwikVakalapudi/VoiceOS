"""WebSocket media bridge tests: Twilio + binary protocols, no real socket."""

import base64
import json

import numpy as np

from voiceos.telephony.transcode import pcm16_to_ulaw
from voiceos.telephony.websocket import (
    BinaryProtocol,
    TwilioProtocol,
    WebSocketMediaSession,
)


class FakeSession:
    """A stand-in for VoicePipeline bound to the bridge's transport."""

    def __init__(self, transport):
        self.transport = transport
        self.started = self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def _twilio_media(payload_ulaw: bytes) -> str:
    return json.dumps(
        {"event": "media", "media": {"payload": base64.b64encode(payload_ulaw).decode()}}
    )


# --- protocol units --------------------------------------------------------

def test_twilio_decodes_media_and_wraps_outbound_with_stream_sid():
    proto = TwilioProtocol()
    proto.decode_inbound(True, json.dumps({"event": "start", "streamSid": "MZ123"}))
    ulaw = pcm16_to_ulaw(np.zeros(160, dtype="<i2").tobytes())
    assert proto.decode_inbound(True, _twilio_media(ulaw)) == ulaw

    is_text, data = proto.wrap_outbound(ulaw)
    msg = json.loads(data)
    assert is_text and msg["event"] == "media" and msg["streamSid"] == "MZ123"
    assert base64.b64decode(msg["media"]["payload"]) == ulaw


def test_twilio_clear_frame_carries_stream_sid():
    proto = TwilioProtocol()
    proto.decode_inbound(True, json.dumps({"event": "start", "streamSid": "MZ9"}))
    is_text, data = proto.clear_frame()
    assert is_text and json.loads(data) == {"event": "clear", "streamSid": "MZ9"}


def test_binary_passes_raw_audio_and_ignores_text_control():
    proto = BinaryProtocol()
    assert proto.decode_inbound(False, b"\x01\x02") == b"\x01\x02"
    assert proto.decode_inbound(True, '{"event":"start"}') is None
    assert proto.wrap_outbound(b"\xaa") == (False, b"\xaa")


# --- session integration ---------------------------------------------------

async def test_session_feeds_inbound_audio_into_transport_source():
    sent = []

    async def send(is_text, data):
        sent.append((is_text, data))

    session = WebSocketMediaSession(
        FakeSession, send, protocol="twilio", input_sample_rate=16000, frame_size=512
    )
    await session.start()

    # Enough 8k mu-law to produce >=1 rechunked 16k frame (512 @16k ~ 256 @8k).
    ulaw = pcm16_to_ulaw(np.zeros(400, dtype="<i2").tobytes())
    pushed = []
    session._transport.source.push = lambda frame: pushed.append(frame)
    await session.feed(True, _twilio_media(ulaw))
    assert len(pushed) >= 1


async def test_session_barge_in_emits_clear_on_next_inbound_frame():
    sent = []

    async def send(is_text, data):
        sent.append((is_text, data))

    session = WebSocketMediaSession(FakeSession, send, protocol="twilio")
    await session.start()
    session._transport.sink.open(24000)

    # Pipeline barge-in flags a pending clear (interrupt is sync).
    session._transport.sink.interrupt()
    assert sent == []
    # The clear frame is flushed to the wire on the next inbound frame.
    await session.feed(True, json.dumps({"event": "media", "media": {}}))
    assert any(t and json.loads(d).get("event") == "clear" for t, d in sent)


async def test_session_start_stop_delegate_to_pipeline():
    made = {}

    def factory(transport):
        made["session"] = FakeSession(transport)
        return made["session"]

    async def send(is_text, data):
        pass

    session = WebSocketMediaSession(factory, send, protocol="binary")
    await session.start()
    await session.stop()
    assert made["session"].started and made["session"].stopped
