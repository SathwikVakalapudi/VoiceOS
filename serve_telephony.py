"""VoiceOS telephony media server — one AI session per phone call.

Bridges calls from a media server (Asterisk / FreeSWITCH) to VoiceOS. The
media server terminates SIP/RTP and streams each call's audio here; VoiceOS
runs VAD -> STT -> LLM -> TTS and streams speech back. See docs/TELEPHONY.md.

    python serve_telephony.py                       # AudioSocket (Asterisk), :8090
    python serve_telephony.py --bridge websocket \
        --protocol twilio --port 8091               # WebSocket (Twilio/FreeSWITCH)

Handles inbound calls (the media server dials in per call) and the answered
leg of outbound calls originated via outbound_campaign.py / originate.py.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from voiceos.config.settings import get_settings
from voiceos.pipeline.pipeline import VoicePipeline
from voiceos.utils.logging import setup_logging


async def run(args) -> None:
    settings = get_settings()

    def make_session(transport):
        # One independent pipeline per call, bound to that call's audio.
        return VoicePipeline(settings, transport=transport)

    if args.bridge == "audiosocket":
        from voiceos.telephony.audiosocket import AudioSocketServer

        server = AudioSocketServer(
            make_session, host=args.host, port=args.port,
            input_sample_rate=settings.audio.input_sample_rate,
            frame_size=settings.audio.frame_size,
        )
    else:
        from voiceos.telephony.websocket import WebSocketMediaServer

        server = WebSocketMediaServer(
            make_session, host=args.host, port=args.port, protocol=args.protocol,
            input_sample_rate=settings.audio.input_sample_rate,
            frame_size=settings.audio.frame_size,
        )

    print(f"VoiceOS {args.bridge} bridge listening on {args.host}:{args.port}")
    print("Point your media server here (see docs/TELEPHONY.md). Ctrl+C to quit.\n")
    try:
        await server.serve_forever()
    finally:
        await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceos-telephony")
    parser.add_argument(
        "--bridge", choices=["audiosocket", "websocket"], default="audiosocket",
        help="audiosocket = Asterisk (default); websocket = FreeSWITCH/Twilio",
    )
    parser.add_argument(
        "--protocol", choices=["twilio", "binary"], default="twilio",
        help="websocket wire format: twilio (JSON/base64 mu-law) or binary (raw)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.port is None:
        args.port = 8090 if args.bridge == "audiosocket" else 8091

    settings = get_settings()
    setup_logging(settings.log_level)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
