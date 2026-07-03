"""WebSocket media bridge (FreeSWITCH mod_audio_stream / Twilio Media Streams).

The WebSocket counterpart to `audiosocket.py`. A media server streams a
call's audio to VoiceOS over a WebSocket; VoiceOS decodes it, runs the
pipeline, and streams TTS back on the same socket. Two wire protocols are
supported because the common media servers frame audio differently:

  * ``"twilio"`` — Twilio Media Streams and compatible gateways. Everything
    is JSON *text* frames; audio is base64 mu-law 8 kHz:
        <- {"event":"start","streamSid":"MZ..."}
        <- {"event":"media","media":{"payload":"<base64 mulaw>"}}
        <- {"event":"stop"}
        -> {"event":"media","streamSid":"MZ...","media":{"payload":"..."}}
        -> {"event":"clear","streamSid":"MZ..."}      # barge-in flush
  * ``"binary"`` — FreeSWITCH mod_audio_stream (and similar). Inbound audio
    arrives as raw *binary* frames; control/metadata arrives as JSON text
    frames. Outbound audio is sent as raw binary frames.

> The exact envelope varies by module version — confirm against your
> mod_audio_stream build (see docs/TELEPHONY.md). The protocol handlers
> below isolate that framing so only one small class needs adjusting.

The transport/session wiring is identical to AudioSocket: one WebSocket
connection == one call == one VoiceOS session bound to a
`MediaStreamTransport`. The session logic (`WebSocketMediaSession`) is
transport-framework-agnostic and unit-tested without a real socket; the
thin `websockets`-based server at the bottom is the runtime.
"""

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from voiceos.telephony.bridge import MediaStreamTransport

logger = logging.getLogger(__name__)

# A session is anything with async start()/stop() bound to a transport —
# VoicePipeline(settings, transport=...) in practice, a fake in tests.
SessionFactory = Callable[[MediaStreamTransport], "object"]

# What the adapter sends to the wire: (is_text, data). Text frames carry str,
# binary frames carry bytes. The runtime maps this onto the socket's
# send()/send(bytes) methods.
WireSend = Callable[[bool, "str | bytes"], Awaitable[None]]


class MediaProtocol(ABC):
    """Frames call audio for a specific media-server wire format."""

    encoding: str  # transcode encoding: "mulaw" or "pcm16"

    @abstractmethod
    def decode_inbound(self, is_text: bool, data: "str | bytes") -> bytes | None:
        """Extract a raw telephony audio payload from one wire frame.

        Returns the audio bytes to feed the pipeline, or None for control
        frames (start/stop/mark/etc.) that carry no audio.
        """

    @abstractmethod
    def wrap_outbound(self, payload: bytes) -> tuple[bool, "str | bytes"]:
        """Wrap encoded telephony audio into one wire frame (is_text, data)."""

    def clear_frame(self) -> tuple[bool, "str | bytes"] | None:
        """Barge-in frame that flushes the media server's buffer, if any."""
        return None

    def on_control(self, event: dict) -> None:
        """Hook for control events (e.g. capture Twilio streamSid)."""


class BinaryProtocol(MediaProtocol):
    """Raw binary audio frames (FreeSWITCH mod_audio_stream style)."""

    encoding = "mulaw"

    def __init__(self, encoding: str = "mulaw") -> None:
        self.encoding = encoding

    def decode_inbound(self, is_text, data):
        if is_text:
            # JSON control/metadata frame — no audio.
            return None
        return data  # raw audio payload

    def wrap_outbound(self, payload):
        return (False, payload)


class TwilioProtocol(MediaProtocol):
    """Twilio Media Streams: JSON text frames, base64 mu-law payloads."""

    encoding = "mulaw"

    def __init__(self) -> None:
        self._stream_sid: str | None = None

    def decode_inbound(self, is_text, data):
        if not is_text:
            return None
        try:
            msg = json.loads(data)
        except (ValueError, TypeError):
            return None
        event = msg.get("event")
        if event == "media":
            payload = msg.get("media", {}).get("payload")
            return base64.b64decode(payload) if payload else None
        if event in ("start", "connected"):
            self._stream_sid = msg.get("streamSid") or msg.get("start", {}).get("streamSid")
        self.on_control(msg)
        return None

    def wrap_outbound(self, payload):
        frame = {
            "event": "media",
            "media": {"payload": base64.b64encode(payload).decode("ascii")},
        }
        if self._stream_sid:
            frame["streamSid"] = self._stream_sid
        return (True, json.dumps(frame))

    def clear_frame(self):
        frame = {"event": "clear"}
        if self._stream_sid:
            frame["streamSid"] = self._stream_sid
        return (True, json.dumps(frame))


def make_protocol(protocol: str) -> MediaProtocol:
    if protocol == "twilio":
        return TwilioProtocol()
    if protocol == "binary":
        return BinaryProtocol()
    raise ValueError(f"unknown media protocol: {protocol!r} (use 'twilio' or 'binary')")


class WebSocketMediaSession:
    """One call over a WebSocket: decode inbound audio, run the pipeline,
    stream TTS back. Framework-agnostic — driven by `feed()` and `send`.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        send: WireSend,
        *,
        protocol: str = "twilio",
        input_sample_rate: int = 16000,
        frame_size: int = 512,
    ) -> None:
        self._proto = make_protocol(protocol)
        self._send = send
        self._pending_clear = False

        async def send_audio(payload: bytes) -> None:
            is_text, data = self._proto.wrap_outbound(payload)
            await self._send(is_text, data)

        self._transport = MediaStreamTransport(
            send_audio,
            input_sample_rate=input_sample_rate,
            frame_size=frame_size,
            encoding=self._proto.encoding,
            on_interrupt=self._flag_clear,
        )
        self._session = session_factory(self._transport)

    def _flag_clear(self) -> None:
        # interrupt() is sync (called from the pipeline); defer the clear frame
        # to the next inbound turn of the async loop.
        self._pending_clear = True

    async def start(self) -> None:
        await self._session.start()

    async def feed(self, is_text: bool, data: "str | bytes") -> None:
        """Handle one inbound wire frame from the media server."""
        if self._pending_clear:
            self._pending_clear = False
            frame = self._proto.clear_frame()
            if frame is not None:
                await self._send(*frame)
        payload = self._proto.decode_inbound(is_text, data)
        if payload:
            self._transport.on_inbound_audio(payload)

    async def stop(self) -> None:
        await self._session.stop()


async def serve_websocket(
    websocket,
    session_factory: SessionFactory,
    *,
    protocol: str = "twilio",
    input_sample_rate: int = 16000,
    frame_size: int = 512,
) -> None:
    """Handle one `websockets` connection end-to-end (one call == one session)."""

    async def send(is_text: bool, data) -> None:
        await websocket.send(data)  # str -> text frame, bytes -> binary frame

    session = WebSocketMediaSession(
        session_factory, send, protocol=protocol,
        input_sample_rate=input_sample_rate, frame_size=frame_size,
    )
    await session.start()
    try:
        async for message in websocket:
            await session.feed(isinstance(message, str), message)
    finally:
        await session.stop()


class WebSocketMediaServer:
    """Accepts WebSocket media connections, one VoiceOS session per call.

    Requires the `websockets` package (`pip install websockets`); it is an
    optional telephony dependency, imported lazily so the rest of VoiceOS
    runs without it.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        host: str = "0.0.0.0",
        port: int = 8091,
        *,
        protocol: str = "twilio",
        input_sample_rate: int = 16000,
        frame_size: int = 512,
    ) -> None:
        self._session_factory = session_factory
        self._host = host
        self._port = port
        self._protocol = protocol
        self._input_sample_rate = input_sample_rate
        self._frame_size = frame_size
        self._server = None

    async def _handler(self, websocket) -> None:
        try:
            await serve_websocket(
                websocket, self._session_factory, protocol=self._protocol,
                input_sample_rate=self._input_sample_rate, frame_size=self._frame_size,
            )
        except Exception:
            logger.exception("websocket call failed")

    async def serve_forever(self) -> None:
        try:
            import websockets
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "the WebSocket media bridge needs the `websockets` package; "
                "run `pip install websockets`"
            ) from exc
        self._server = await websockets.serve(self._handler, self._host, self._port)
        logger.info("WebSocket media server listening on %s:%d", self._host, self._port)
        await self._server.wait_closed()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
