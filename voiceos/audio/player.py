"""Playback worker.

Consumes audio from the playback queue and writes it to the speaker.
An EndOfTurn marker means the assistant's turn is fully spoken: emit
PLAYBACK_FINISHED and return the pipeline to IDLE.
"""

from __future__ import annotations

import asyncio
import logging

from voiceos.audio.audio_queue import AudioFrame
from voiceos.audio.speaker import Speaker
from voiceos.pipeline.events import EndOfTurn, EventBus, EventType, SentenceSpoken
from voiceos.pipeline.state import PipelineState, StateMachine

logger = logging.getLogger(__name__)


class PlaybackWorker:
    def __init__(
        self,
        speaker: Speaker,
        playback_queue: asyncio.Queue,
        event_bus: EventBus,
        state: StateMachine,
    ) -> None:
        self._speaker = speaker
        self._queue = playback_queue
        self._bus = event_bus
        self._state = state
        self._playing = False
        # How many sentences of the current turn have finished playing, and
        # which turn they belong to — read by the barge-in handler to record
        # only what was actually heard.
        self._sentences_spoken = 0
        self._current_turn_id: int | None = None

    @property
    def sentences_spoken(self) -> int:
        return self._sentences_spoken

    @property
    def current_turn_id(self) -> int | None:
        return self._current_turn_id

    def notify_interrupted(self) -> None:
        """Called on barge-in so the next turn re-emits PLAYBACK_STARTED."""
        self._playing = False
        self._sentences_spoken = 0
        self._current_turn_id = None

    async def run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if isinstance(item, SentenceSpoken):
                    # All of this sentence's frames precede the marker in the
                    # queue, so they have already been played by now.
                    self._current_turn_id = item.turn_id
                    self._sentences_spoken = item.index + 1
                    continue

                if isinstance(item, EndOfTurn):
                    self._playing = False
                    await self._bus.emit(
                        EventType.PLAYBACK_FINISHED,
                        {"turn_id": item.turn_id, "sentences_spoken": self._sentences_spoken},
                    )
                    self._sentences_spoken = 0
                    self._current_turn_id = None
                    # After a barge-in the user is already speaking (LISTENING);
                    # a stale end-of-turn must not yank the state back to IDLE.
                    if self._state.state is not PipelineState.LISTENING:
                        self._state.transition(PipelineState.IDLE)
                    continue

                frame: AudioFrame = item
                if not self._playing:
                    self._playing = True
                    await self._bus.emit(EventType.PLAYBACK_STARTED, {})
                await self._speaker.play(frame.data)
            except Exception:
                logger.exception("playback failed")
                await self._bus.emit(EventType.ERROR, {"stage": "playback"})
            finally:
                self._queue.task_done()
