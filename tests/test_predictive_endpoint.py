"""Predictive endpointing in the detector: a partial that looks complete
closes the turn after a short pause instead of the full trailing silence."""

import asyncio

import numpy as np

from voiceos.audio.audio_queue import AudioQueue, make_frame
from voiceos.config.settings import VADSettings
from voiceos.interfaces.vad import BaseVAD
from voiceos.pipeline.events import EventBus
from voiceos.pipeline.state import PipelineState, StateMachine
from voiceos.vad.detector import SpeechDetector
from voiceos.vad.endpoint import EndpointPredictor

SR = 16000
FRAME = 512  # 32 ms


class FakeVAD(BaseVAD):
    def __init__(self, probs):
        self.probs = list(probs)

    def process(self, frame, sample_rate):
        return self.probs.pop(0) if self.probs else 0.0


class FakePartial:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    async def partial(self, audio, sample_rate):
        self.calls += 1
        return self.text

    async def load(self):
        pass

    async def close(self):
        pass


async def feed(queue, n):
    # Deliver frames the way the mic does — spaced in time — so the async
    # partial-transcription task gets scheduled between frames (a burst of
    # pre-queued frames would starve it, which never happens with a live mic).
    for _ in range(n):
        queue.put_drop_oldest(make_frame(np.zeros(FRAME, dtype=np.int16), SR))
        await asyncio.sleep(0.01)


# Full silence 400ms; predicted-complete only needs 64ms (2 frames).
SETTINGS = VADSettings(
    threshold=0.5, min_speech_ms=50, min_silence_ms=400, pre_roll_ms=64,
    predictive_endpointing=True, partial_interval_ms=32,
    min_partial_speech_ms=64, predicted_silence_ms=64, min_partial_chars=3,
)


def build(partial_text, probs):
    audio_q = AudioQueue()
    utt_q: asyncio.Queue = asyncio.Queue()
    bus = EventBus()
    state = StateMachine()
    transcriber = FakePartial(partial_text)
    detector = SpeechDetector(
        FakeVAD(probs), SETTINGS, audio_q, utt_q, bus, state, frame_ms=32.0,
        partial_transcriber=transcriber, endpoint_predictor=EndpointPredictor(min_chars=3),
    )
    return detector, audio_q, utt_q, state, transcriber


async def test_predicted_complete_closes_after_short_silence():
    # 6 speech + 3 silence (96ms): far below the 400ms full silence, but above
    # the 64ms predicted-complete threshold — so it should close.
    detector, audio_q, utt_q, state, transcriber = build("I am done.", [0.9] * 6 + [0.0] * 3)
    task = asyncio.create_task(detector.run())
    try:
        await feed(audio_q, 9)
        await asyncio.wait_for(utt_q.get(), timeout=1.0)
    finally:
        task.cancel()

    assert state.state is PipelineState.THINKING
    assert transcriber.calls >= 1


async def test_incomplete_partial_waits_for_full_silence():
    import pytest

    # Same audio, but the partial never looks complete -> 96ms of silence is
    # not enough (needs the full 400ms), so no utterance is emitted.
    detector, audio_q, utt_q, state, _ = build("I would like to", [0.9] * 6 + [0.0] * 3)
    task = asyncio.create_task(detector.run())
    await feed(audio_q, 9)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(utt_q.get(), timeout=0.3)
    task.cancel()

    assert utt_q.empty()
