"""Backchannel tests: pre-render fillers, fire while thinking, cancel on audio."""

import asyncio

import numpy as np

from voiceos.config.settings import PipelineSettings
from voiceos.pipeline.backchannel import BackchannelWorker
from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.state import PipelineState, StateMachine


class FakeTTS:
    @property
    def sample_rate(self):
        return 24000

    async def synthesize(self, text):
        yield np.ones(4, dtype="<i2")


def build(delay_ms):
    bus = EventBus()
    state = StateMachine()
    queue: asyncio.Queue = asyncio.Queue()
    settings = PipelineSettings(
        backchannel=True, backchannel_delay_ms=delay_ms, backchannel_phrases=["mm-hmm"]
    )
    worker = BackchannelWorker(FakeTTS(), queue, bus, state, settings)
    return worker, bus, state, queue


async def test_fires_a_filler_while_still_thinking():
    worker, bus, state, queue = build(delay_ms=0)
    await worker.load()
    worker.start()

    state.transition(PipelineState.THINKING)
    await bus.emit(EventType.LLM_STARTED, {"turn_id": 1})
    await asyncio.sleep(0.05)  # let the armed timer task run

    assert not queue.empty()


async def test_cancelled_when_real_audio_starts():
    worker, bus, state, queue = build(delay_ms=50)
    await worker.load()
    worker.start()

    state.transition(PipelineState.THINKING)
    await bus.emit(EventType.LLM_STARTED, {"turn_id": 1})
    await bus.emit(EventType.PLAYBACK_STARTED, {})  # assistant started speaking
    await asyncio.sleep(0.12)

    assert queue.empty()


async def test_does_not_fire_if_no_longer_thinking():
    worker, bus, state, queue = build(delay_ms=10)
    await worker.load()
    worker.start()

    state.transition(PipelineState.THINKING)
    await bus.emit(EventType.LLM_STARTED, {"turn_id": 1})
    state.transition(PipelineState.SPEAKING)  # real audio flipped the state
    await asyncio.sleep(0.05)

    assert queue.empty()
