"""Backchanneling.

Fills the awkward gap between the user finishing and the assistant's
first words with a short human affirmation — "mm-hmm", "right" — so a
slow turn doesn't feel like dead air.

It fires only while the pipeline is THINKING (the user has stopped and
the mic is gated), so the filler is never captured as user speech, and
it is cancelled the instant real audio starts. The fillers are
pre-rendered through the same TTS voice once at startup. Pure
orchestration: it wires the event bus and playback queue, and performs
no inference of its own.
"""

from __future__ import annotations

import asyncio
import logging
import random

import numpy as np

from voiceos.audio.audio_queue import make_frame
from voiceos.config.settings import PipelineSettings
from voiceos.interfaces.tts import BaseTTS
from voiceos.pipeline.events import Event, EventBus, EventType
from voiceos.pipeline.state import PipelineState, StateMachine

logger = logging.getLogger(__name__)


class BackchannelWorker:
    def __init__(
        self,
        tts: BaseTTS,
        playback_queue: asyncio.Queue,
        event_bus: EventBus,
        state: StateMachine,
        settings: PipelineSettings,
    ) -> None:
        self._tts = tts
        self._playback_queue = playback_queue
        self._bus = event_bus
        self._state = state
        self._settings = settings
        self._clips: list[np.ndarray] = []
        self._task: asyncio.Task | None = None

    async def load(self) -> None:
        """Pre-render each filler once. If synthesis is unavailable the
        feature quietly disables itself rather than breaking startup."""
        for phrase in self._settings.backchannel_phrases:
            try:
                chunks = [chunk async for chunk in self._tts.synthesize(phrase)]
            except Exception:
                logger.warning("backchannel disabled — could not pre-render fillers")
                self._clips = []
                return
            if chunks:
                self._clips.append(np.concatenate(chunks))
        logger.info("backchannel ready (%d fillers)", len(self._clips))

    def start(self) -> None:
        if not self._clips:
            return
        self._bus.subscribe(EventType.LLM_STARTED, self._arm)
        self._bus.subscribe(EventType.PLAYBACK_STARTED, self._cancel)
        self._bus.subscribe(EventType.BARGE_IN, self._cancel)

    async def stop(self) -> None:
        self._cancel()

    def _arm(self, event: Event) -> None:
        self._cancel_task()
        self._task = asyncio.create_task(self._maybe_fill())

    def _cancel(self, event: Event | None = None) -> None:
        self._cancel_task()

    def _cancel_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _maybe_fill(self) -> None:
        try:
            await asyncio.sleep(self._settings.backchannel_delay_ms / 1000.0)
        except asyncio.CancelledError:
            return
        # Only if the assistant is still thinking — real audio would have
        # flipped the state to SPEAKING and cancelled this task already.
        if self._state.state is not PipelineState.THINKING:
            return
        clip = random.choice(self._clips)
        await self._playback_queue.put(make_frame(clip, self._tts.sample_rate))
