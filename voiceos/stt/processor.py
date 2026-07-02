"""STT worker.

Consumes complete utterances, transcribes them, and forwards non-empty
transcripts to the conversation queue.
"""

from __future__ import annotations

import asyncio
import logging
import time

from voiceos.audio.audio_queue import Utterance
from voiceos.interfaces.stt import BaseSTT
from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.state import PipelineState, StateMachine
from voiceos.utils.audio import int16_to_float32

logger = logging.getLogger(__name__)


class STTWorker:
    def __init__(
        self,
        stt: BaseSTT,
        utterance_queue: asyncio.Queue[Utterance],
        transcript_queue: asyncio.Queue[str],
        event_bus: EventBus,
        state: StateMachine,
    ) -> None:
        self._stt = stt
        self._utterance_queue = utterance_queue
        self._transcript_queue = transcript_queue
        self._bus = event_bus
        self._state = state

    async def run(self) -> None:
        while True:
            utterance = await self._utterance_queue.get()
            try:
                await self._handle(utterance)
            except Exception:
                logger.exception("transcription failed")
                await self._bus.emit(EventType.ERROR, {"stage": "stt"})
                self._state.transition(PipelineState.IDLE)
            finally:
                self._utterance_queue.task_done()

    async def _handle(self, utterance: Utterance) -> None:
        started = time.monotonic()
        result = await self._stt.transcribe(
            int16_to_float32(utterance.audio), utterance.sample_rate
        )
        elapsed = time.monotonic() - started

        if not result.text:
            logger.debug("empty transcript, back to idle")
            self._state.transition(PipelineState.IDLE)
            return

        logger.info("transcript (%.2fs): %s", elapsed, result.text)
        await self._bus.emit(
            EventType.TRANSCRIPT_READY,
            {
                "text": result.text,
                "language": result.language,
                "audio_duration_s": utterance.duration_s,
                "stt_latency_s": elapsed,
            },
        )
        await self._transcript_queue.put(result.text)
