"""Event system.

Every module communicates outcomes through events on the EventBus.
Later this powers barge-in, analytics, debugging, and telephony hooks
without touching any worker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_STOPPED = "pipeline_stopped"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    BARGE_IN = "barge_in"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    TRANSCRIPT_READY = "transcript_ready"
    LLM_STARTED = "llm_started"
    TOOL_CALLED = "tool_called"
    LLM_FINISHED = "llm_finished"
    TTS_STARTED = "tts_started"
    TTS_FINISHED = "tts_finished"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_FINISHED = "playback_finished"
    ERROR = "error"


@dataclass(slots=True)
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


class EndOfTurn:
    """Control marker flushed through the TTS and playback queues.

    When the playback worker consumes it, the assistant's turn is fully
    spoken and the pipeline returns to IDLE.
    """

    __slots__ = ("turn_id",)

    def __init__(self, turn_id: int = 0) -> None:
        self.turn_id = turn_id

    def __repr__(self) -> str:
        return f"EndOfTurn(turn_id={self.turn_id})"


class SpeakSentence:
    """A sentence bound for TTS, tagged with its turn and position.

    Replaces the bare string that used to flow on the TTS queue so the
    playback side can report exactly which sentences were spoken — the
    conversation only records what the user actually heard.
    """

    __slots__ = ("turn_id", "index", "text")

    def __init__(self, turn_id: int, index: int, text: str) -> None:
        self.turn_id = turn_id
        self.index = index
        self.text = text

    def __repr__(self) -> str:
        return f"SpeakSentence(turn_id={self.turn_id}, index={self.index})"


class SentenceSpoken:
    """Marker pushed onto the playback queue once a sentence's audio has
    been fully written to the speaker.

    Because it trails all of that sentence's audio frames, the playback
    worker consuming it means those frames have finished playing.
    """

    __slots__ = ("turn_id", "index")

    def __init__(self, turn_id: int, index: int) -> None:
        self.turn_id = turn_id
        self.index = index

    def __repr__(self) -> str:
        return f"SentenceSpoken(turn_id={self.turn_id}, index={self.index})"


Handler = Union[Callable[[Event], None], Callable[[Event], Awaitable[None]]]


class EventBus:
    """Minimal async pub/sub. Subscribe with an EventType, or None for all."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType | None, list[Handler]] = {}

    def subscribe(self, event_type: EventType | None, handler: Handler) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType | None, handler: Handler) -> None:
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        event = Event(type=event_type, data=data or {})
        logger.debug("event %s %s", event_type.value, event.data)
        handlers = [
            *self._subscribers.get(event_type, []),
            *self._subscribers.get(None, []),
        ]
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("event handler failed for %s", event_type.value)
        return event
