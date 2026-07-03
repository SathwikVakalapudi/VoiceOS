"""VoiceOS telephony media server — one AI session per phone call.

Bridges calls from a media server (Asterisk / FreeSWITCH) to VoiceOS. The
media server terminates SIP/RTP and streams each call's audio here; VoiceOS
runs VAD -> STT -> LLM -> TTS and streams speech back. See docs/TELEPHONY.md.

    python serve_telephony.py                       # AudioSocket (Asterisk), :8090
    python serve_telephony.py --bridge websocket \
        --protocol twilio --port 8091               # WebSocket (Twilio/FreeSWITCH)
    python serve_telephony.py \
        --campaign campaigns/rajasthan_survey.json  # run a persona/survey script

Handles inbound calls (the media server dials in per call) and the answered
leg of outbound calls originated via outbound_campaign.py / originate.py.

`--campaign` loads a persona JSON ({system_prompt, first_message,
error_message}) into every call this server answers — the assistant speaks
`first_message` on connect (Vapi firstMessage style) and follows the prompt.
Run one server per persona; point each campaign's DIDs/routes at it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from voiceos.config.settings import get_settings
from voiceos.pipeline.pipeline import VoicePipeline
from voiceos.utils.logging import setup_logging


async def run(args) -> None:
    settings = get_settings()
    if args.campaign:
        # Deep-copy so we don't mutate the cached global settings singleton;
        # each spawned pipeline's ConversationManager loads this campaign file.
        settings = settings.model_copy(deep=True)
        settings.conversation.campaign_file = args.campaign

    # Post-call survey extraction, if the campaign defines a `survey` block.
    collector = None
    if args.campaign:
        from voiceos.survey import ResultStore, SurveyCollector, SurveyDefinition

        survey = SurveyDefinition.from_campaign_file(args.campaign)
        if survey is not None:
            collector = SurveyCollector(
                survey, ResultStore(args.results), settings=settings
            )

    def make_session(transport):
        # One independent pipeline per call, bound to that call's audio.
        pipeline = VoicePipeline(settings, transport=transport)
        if collector is not None:
            from voiceos.survey import SurveySession

            return SurveySession(pipeline, collector)
        return pipeline

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
    if args.campaign:
        opening = (json.loads(Path(args.campaign).read_text(encoding="utf-8"))
                   .get("first_message", ""))
        print(f"Campaign: {args.campaign}")
        if opening:
            print(f"  opening line: {opening[:80]}{'…' if len(opening) > 80 else ''}")
    if collector is not None:
        print(f"  survey: {collector._survey.name} "
              f"({len(collector._survey.questions)} fields) -> {args.results}")
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
    parser.add_argument(
        "--campaign", metavar="PATH",
        help="persona JSON {system_prompt, first_message, error_message} run on every call",
    )
    parser.add_argument(
        "--results", metavar="PATH", default="results/survey.jsonl",
        help="where to append extracted survey results (if the campaign has a `survey` block)",
    )
    args = parser.parse_args()
    if args.port is None:
        args.port = 8090 if args.bridge == "audiosocket" else 8091

    # Fail fast on a bad campaign file rather than per call inside each pipeline.
    if args.campaign:
        try:
            json.loads(Path(args.campaign).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            sys.exit(f"cannot load campaign '{args.campaign}': {exc}")

    settings = get_settings()
    setup_logging(settings.log_level)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
