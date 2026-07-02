"""Speech detector tests using a scripted fake VAD — no models needed."""

import asyncio

import numpy as np
import pytest

from voiceos.audio.audio_queue import AudioQueue, make_frame
from voiceos.config.settings import VADSettings
from voiceos.interfaces.vad import BaseVAD
from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.state import PipelineState, StateMachine
from voiceos.vad.detector import SpeechDetector

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms


class FakeVAD(BaseVAD):
    """Returns pre-scripted probabilities, one per frame."""

    def __init__(self, probs: list[float]):
        self.probs = list(probs)

    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        return self.probs.pop(0) if self.probs else 0.0


def make_detector(probs: list[float], settings: VADSettings):
    audio_q = AudioQueue()
    utterance_q: asyncio.Queue = asyncio.Queue()
    bus = EventBus()
    state = StateMachine()
    detector = SpeechDetector(
        FakeVAD(probs), settings, audio_q, utterance_q, bus, state, frame_ms=32.0
    )
    return detector, audio_q, utterance_q, bus, state


def feed_frames(audio_q: AudioQueue, count: int):
    for _ in range(count):
        audio_q.put_drop_oldest(
            make_frame(np.zeros(FRAME, dtype=np.int16), SAMPLE_RATE)
        )


SETTINGS = VADSettings(
    threshold=0.5, min_speech_ms=100, min_silence_ms=90, pre_roll_ms=64
)


async def run_detector_until(detector, condition, timeout=2.0):
    task = asyncio.create_task(detector.run())
    try:
        await asyncio.wait_for(condition(), timeout)
    finally:
        task.cancel()


async def test_utterance_is_emitted_after_trailing_silence():
    # 6 speech frames (192 ms) then 4 silence frames (128 ms > 90 ms).
    probs = [0.9] * 6 + [0.0] * 4
    detector, audio_q, utterance_q, bus, state = make_detector(probs, SETTINGS)
    events: list[EventType] = []
    bus.subscribe(None, lambda e: events.append(e.type))

    feed_frames(audio_q, 10)
    await run_detector_until(detector, utterance_q.get)

    assert EventType.SPEECH_STARTED in events
    assert EventType.SPEECH_ENDED in events
    assert state.state is PipelineState.THINKING  # mic gated for downstream


async def test_short_noise_burst_is_discarded():
    # 2 speech frames (64 ms < min_speech_ms=100) then silence.
    probs = [0.9] * 2 + [0.0] * 4
    detector, audio_q, utterance_q, bus, state = make_detector(probs, SETTINGS)
    ended = asyncio.Event()
    bus.subscribe(EventType.SPEECH_ENDED, lambda e: ended.set())

    feed_frames(audio_q, 6)
    await run_detector_until(detector, ended.wait)

    assert utterance_q.empty()
    assert state.state is PipelineState.IDLE


async def test_frames_ignored_while_assistant_is_busy():
    probs = [0.9] * 10
    settings = SETTINGS.model_copy(update={"barge_in": False})
    detector, audio_q, utterance_q, bus, state = make_detector(probs, settings)
    state.transition(PipelineState.SPEAKING)
    started = asyncio.Event()
    bus.subscribe(EventType.SPEECH_STARTED, lambda e: started.set())

    feed_frames(audio_q, 10)
    task = asyncio.create_task(detector.run())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(started.wait(), timeout=0.3)
    task.cancel()

    assert utterance_q.empty()


ADAPTIVE_SETTINGS = VADSettings(
    threshold=0.5, min_speech_ms=50, min_silence_ms=90, pre_roll_ms=64,
    adaptive_silence=True, min_silence_short_ms=250, short_utterance_ms=1000,
)


async def test_adaptive_silence_waits_longer_for_a_short_utterance():
    # 3 speech frames (96 ms, still "short") then 4 silence frames (128 ms).
    # Fixed mode would close at 90 ms; adaptive demands 250 ms, so the
    # utterance is NOT emitted yet.
    probs = [0.9] * 3 + [0.0] * 4
    detector, audio_q, utterance_q, bus, state = make_detector(probs, ADAPTIVE_SETTINGS)

    feed_frames(audio_q, 7)
    task = asyncio.create_task(detector.run())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(utterance_q.get(), timeout=0.3)
    task.cancel()

    assert utterance_q.empty()


async def test_adaptive_silence_closes_after_a_longer_pause():
    # Same short utterance, but 9 silence frames (288 ms >= 250 ms) close it.
    probs = [0.9] * 3 + [0.0] * 9
    detector, audio_q, utterance_q, bus, state = make_detector(probs, ADAPTIVE_SETTINGS)

    feed_frames(audio_q, 12)
    await run_detector_until(detector, utterance_q.get)

    assert state.state is PipelineState.THINKING


def test_noise_gate_blocks_frames_below_the_rolling_floor():
    settings = VADSettings(adaptive_noise=True, noise_margin=3.0, noise_window_ms=320)
    detector, *_ = make_detector([], settings)
    quiet = np.full(FRAME, 0.01, dtype=np.float32)

    for _ in range(12):  # establish a ~0.01 noise floor
        detector._passes_noise_gate(quiet)

    assert detector._passes_noise_gate(np.full(FRAME, 0.1, dtype=np.float32)) is True
    assert detector._passes_noise_gate(quiet) is False


async def test_barge_in_interrupts_and_captures_utterance():
    # Assistant is SPEAKING; user talks over it: 10 speech frames (320 ms
    # > barge_in_speech_ms=250), then silence to close the utterance.
    probs = [0.9] * 10 + [0.0] * 4
    settings = SETTINGS.model_copy(
        update={"barge_in": True, "barge_in_threshold": 0.7, "barge_in_speech_ms": 250}
    )
    audio_q = AudioQueue()
    utterance_q: asyncio.Queue = asyncio.Queue()
    bus = EventBus()
    state = StateMachine()
    interrupted = asyncio.Event()

    async def on_barge_in():
        interrupted.set()

    detector = SpeechDetector(
        FakeVAD(probs), settings, audio_q, utterance_q, bus, state,
        frame_ms=32.0, on_barge_in=on_barge_in,
    )
    state.transition(PipelineState.SPEAKING)
    events: list[EventType] = []
    bus.subscribe(None, lambda e: events.append(e.type))

    feed_frames(audio_q, 14)
    await run_detector_until(detector, utterance_q.get)

    assert interrupted.is_set()
    assert EventType.BARGE_IN in events
    assert EventType.SPEECH_STARTED in events
    assert state.state is PipelineState.THINKING  # utterance accepted downstream
