"""VoiceOS entry point.

    python main.py                # start talking
    python main.py --list-devices # show audio devices
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# Windows consoles default to legacy codepages that can't print Telugu,
# Hindi, etc. Force UTF-8 so multilingual transcripts display.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv  # .env authoritative over stale shell env vars

    load_dotenv(override=True)
except ImportError:
    pass

from voiceos.config.settings import get_settings
from voiceos.pipeline.events import Event, EventBus, EventType
from voiceos.pipeline.pipeline import VoicePipeline
from voiceos.utils.logging import setup_logging


def _attach_console(bus: EventBus) -> None:
    """Print the conversation as it happens."""

    def on_event(event: Event) -> None:
        if event.type is EventType.TRANSCRIPT_READY:
            print(f"\nYou:       {event.data['text']}")
        elif event.type is EventType.LLM_FINISHED:
            ttfb = event.data.get("latency_first_token_s")
            suffix = f"  ({ttfb:.2f}s to first token)" if ttfb else ""
            print(f"Assistant: {event.data['text']}{suffix}")
        elif event.type is EventType.ERROR:
            print(f"[error in {event.data.get('stage', '?')} — see logs]", file=sys.stderr)

    bus.subscribe(None, on_event)


async def run() -> None:
    settings = get_settings()
    bus = EventBus()
    _attach_console(bus)
    pipeline = VoicePipeline(settings, event_bus=bus)

    await pipeline.start()
    print("\nVoiceOS is listening — speak into your microphone. Ctrl+C to quit.\n")
    try:
        await pipeline.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceos", description="VoiceOS voice AI engine")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    if args.list_devices:
        from voiceos.audio.microphone import Microphone

        print(Microphone.list_devices())
        return

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
