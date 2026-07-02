"""Per-turn latency tracking.

Subscribes to pipeline events and, when a turn's audio finishes playing,
logs where every millisecond went:

    listen   — how long the user spoke
    stt      — utterance end -> transcript
    llm_ttft — transcript -> first LLM token
    llm      — full LLM generation
    tts      — first sentence handed to TTS -> first audio playing
    response — utterance end -> assistant starts speaking (perceived latency)
    speak    — playback duration
    total    — utterance end -> playback finished

No worker knows this exists — it's pure event observation, and the seed
of the future analytics module.
"""

from __future__ import annotations

import logging

from voiceos.pipeline.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class LatencyMonitor:
    def __init__(self, event_bus: EventBus) -> None:
        self._turn: dict[str, float | None] = {}
        event_bus.subscribe(None, self._on_event)

    def _on_event(self, event: Event) -> None:
        t = self._turn
        if event.type is EventType.SPEECH_ENDED and event.data.get("accepted"):
            self._turn = {
                "speech_ended": event.timestamp,
                "listen_s": event.data.get("duration_s"),
            }
        elif event.type is EventType.TRANSCRIPT_READY:
            t["transcript"] = event.timestamp
        elif event.type is EventType.LLM_FINISHED:
            t["llm_ttft_s"] = event.data.get("latency_first_token_s")
            t["llm_total_s"] = event.data.get("latency_total_s")
        elif event.type is EventType.TTS_STARTED and "tts_first" not in t:
            t["tts_first"] = event.timestamp
        elif event.type is EventType.PLAYBACK_STARTED and "playback_start" not in t:
            t["playback_start"] = event.timestamp
        elif event.type is EventType.PLAYBACK_FINISHED and "speech_ended" in t:
            self._log_turn(event.data.get("turn_id"), event.timestamp)
            self._turn = {}

    def _log_turn(self, turn_id, finished_at: float) -> None:
        t = self._turn

        def fmt(value: float | None) -> str:
            return f"{value:.2f}s" if value is not None else "-"

        def delta(end_key: str, start_key: str) -> float | None:
            end, start = t.get(end_key), t.get(start_key)
            return (end - start) if end is not None and start is not None else None

        stt = delta("transcript", "speech_ended")
        tts = delta("playback_start", "tts_first")
        response = delta("playback_start", "speech_ended")
        playback_start = t.get("playback_start")
        speak = (finished_at - playback_start) if playback_start is not None else None
        speech_ended = t.get("speech_ended")
        total = (finished_at - speech_ended) if speech_ended is not None else None

        logger.info(
            "turn %s timings | listen %s | stt %s | llm first token %s | "
            "llm total %s | tts %s | response %s (silence->voice) | "
            "speak %s | total %s",
            turn_id if turn_id is not None else "?",
            fmt(t.get("listen_s")),
            fmt(stt),
            fmt(t.get("llm_ttft_s")),
            fmt(t.get("llm_total_s")),
            fmt(tts),
            fmt(response),
            fmt(speak),
            fmt(total),
        )
