"""TTS worker.

Consumes sentence chunks, synthesizes them, and streams the resulting
audio to the playback queue. EndOfTurn markers pass through untouched
so the playback worker knows when the turn is complete.
"""

from __future__ import annotations

import asyncio
import logging

from voiceos.audio.audio_queue import make_frame
from voiceos.interfaces.tts import BaseTTS
from voiceos.pipeline.events import (
    EndOfTurn,
    EventBus,
    EventType,
    SentenceSpoken,
    SpeakSentence,
)
from voiceos.pipeline.state import InterruptController, PipelineState, StateMachine

logger = logging.getLogger(__name__)


class TTSWorker:
    def __init__(
        self,
        tts: BaseTTS,
        tts_queue: asyncio.Queue,
        playback_queue: asyncio.Queue,
        event_bus: EventBus,
        state: StateMachine,
        interrupts: InterruptController | None = None,
    ) -> None:
        self._tts = tts
        self._tts_queue = tts_queue
        self._playback_queue = playback_queue
        self._bus = event_bus
        self._state = state
        self._interrupts = interrupts or InterruptController()

    async def run(self) -> None:
        while True:
            item = await self._tts_queue.get()
            try:
                if isinstance(item, EndOfTurn):
                    await self._playback_queue.put(item)
                    continue
                if isinstance(item, SpeakSentence):
                    turn_id, index, text = item.turn_id, item.index, item.text
                else:  # backward-compat: a bare string (e.g. the greeting)
                    turn_id, index, text = None, 0, item
                completed = await self._synthesize(text)
                # Trailing marker: when playback consumes it, every frame of
                # this sentence has been played, so the turn can record it.
                if completed and turn_id is not None:
                    await self._playback_queue.put(SentenceSpoken(turn_id, index))
            except Exception:
                logger.exception("synthesis failed for: %.80s", item)
                await self._bus.emit(EventType.ERROR, {"stage": "tts"})
            finally:
                self._tts_queue.task_done()

    async def _synthesize(self, text: str) -> bool:
        """Return True if the sentence streamed to completion, False if a
        barge-in cut it short."""
        generation = self._interrupts.generation
        await self._bus.emit(EventType.TTS_STARTED, {"text": text})
        # Second-level retry on top of the adapter's own: rides out network
        # dropouts that outlast the adapter's backoff window. Adapters only
        # raise when no audio was emitted, so a full retry never repeats audio.
        completed = False
        for attempt in range(2):
            try:
                completed = await self._stream_sentence(text, generation)
                break
            except Exception:
                if attempt == 0 and self._interrupts.generation == generation:
                    logger.warning(
                        "synthesis failed for %.60s — one more try in 1.5s", text
                    )
                    await asyncio.sleep(1.5)
                    continue
                raise
        await self._bus.emit(EventType.TTS_FINISHED, {"text": text})
        return completed

    async def _stream_sentence(self, text: str, generation: int) -> bool:
        first_chunk = True
        async for audio in self._tts.synthesize(text):
            if self._interrupts.generation != generation:
                logger.debug("barge-in during synthesis; dropping rest of sentence")
                return False
            if len(audio) == 0:
                continue
            if first_chunk:
                first_chunk = False
                self._state.transition(PipelineState.SPEAKING)
            await self._playback_queue.put(make_frame(audio, self._tts.sample_rate))
        return True
