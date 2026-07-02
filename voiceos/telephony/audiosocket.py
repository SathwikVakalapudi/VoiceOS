"""Asterisk AudioSocket server.

AudioSocket is a dead-simple, bidirectional TCP framing that Asterisk uses
to hand a call's audio to an external app and play audio back — ideal for
an AI agent. Each message is:

    [1 byte type][2 bytes big-endian length][payload]

    0x00 terminate (hang up)      length 0
    0x01 uuid (call id)           16-byte payload, sent first by Asterisk
    0x03 dtmf                     1-byte ASCII digit (newer Asterisk)
    0x10 audio                    payload = signed-linear 16-bit, 8 kHz, mono
    0xff error                    1-byte error code

Audio is *linear* 8 kHz (SLIN), not mu-law, so the transport uses
encoding="pcm16". Implemented on stdlib asyncio — no extra dependencies.

Asterisk side (per call):
    inbound  dialplan:  exten => _X.,1,Answer()
                        same  => n,AudioSocket(${UUID()},voiceos-host:8090)
    outbound Dial(AudioSocket/voiceos-host:8090/<uuid>) after originate.

The FreeSWITCH/mod_audio_stream WebSocket variant carries mu-law (encoding
="mulaw") in a JSON envelope; reuse MediaStreamTransport with a WS adapter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from voiceos.telephony.bridge import MediaStreamTransport

logger = logging.getLogger(__name__)

TYPE_TERMINATE = 0x00
TYPE_UUID = 0x01
TYPE_DTMF = 0x03
TYPE_AUDIO = 0x10
TYPE_ERROR = 0xFF

_AUDIO_FRAME_BYTES = 320  # 20 ms of 8 kHz signed-linear 16-bit mono

# A session is anything with async start()/stop() bound to a transport —
# VoicePipeline(settings, transport=...) in practice, a fake in tests.
SessionFactory = Callable[[MediaStreamTransport], "object"]


def frame_audio(pcm: bytes, max_payload: int = _AUDIO_FRAME_BYTES) -> list[bytes]:
    """Split linear-PCM bytes into AudioSocket audio messages (type 0x10)."""
    out = []
    for i in range(0, len(pcm), max_payload):
        chunk = pcm[i : i + max_payload]
        out.append(bytes([TYPE_AUDIO]) + len(chunk).to_bytes(2, "big") + chunk)
    return out


async def read_message(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one AudioSocket message; raises IncompleteReadError at EOF."""
    header = await reader.readexactly(3)
    kind = header[0]
    length = int.from_bytes(header[1:3], "big")
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


async def serve_call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    session_factory: SessionFactory,
    *,
    input_sample_rate: int = 16000,
    frame_size: int = 512,
    on_dtmf: Callable[[str], None] | None = None,
) -> None:
    """Handle one AudioSocket call end-to-end (one call == one session)."""

    async def send(pcm: bytes) -> None:
        for message in frame_audio(pcm):
            writer.write(message)
        await writer.drain()

    transport = MediaStreamTransport(
        send, input_sample_rate=input_sample_rate, frame_size=frame_size,
        encoding="pcm16",  # AudioSocket audio is linear, not mu-law
    )
    session = session_factory(transport)
    await session.start()
    try:
        while True:
            try:
                kind, payload = await read_message(reader)
            except asyncio.IncompleteReadError:
                break
            if kind == TYPE_TERMINATE:
                break
            if kind == TYPE_AUDIO:
                transport.on_inbound_audio(payload)
            elif kind == TYPE_UUID:
                logger.info("audiosocket call %s", payload.hex())
            elif kind == TYPE_DTMF and on_dtmf and payload:
                on_dtmf(chr(payload[0]))
            elif kind == TYPE_ERROR:
                logger.warning("audiosocket error frame: %s", payload.hex())
    finally:
        await session.stop()
        writer.close()


class AudioSocketServer:
    """Accepts AudioSocket connections, one VoiceOS session per call."""

    def __init__(
        self,
        session_factory: SessionFactory,
        host: str = "0.0.0.0",
        port: int = 8090,
        *,
        input_sample_rate: int = 16000,
        frame_size: int = 512,
    ) -> None:
        self._session_factory = session_factory
        self._host = host
        self._port = port
        self._input_sample_rate = input_sample_rate
        self._frame_size = frame_size
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._on_connect, self._host, self._port)
        logger.info("AudioSocket server listening on %s:%d", self._host, self._port)

    async def _on_connect(self, reader, writer) -> None:
        try:
            await serve_call(
                reader, writer, self._session_factory,
                input_sample_rate=self._input_sample_rate, frame_size=self._frame_size,
            )
        except Exception:
            logger.exception("audiosocket call failed")

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
