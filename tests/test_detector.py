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


async def test_hysteresis_holds_speech_through_a_probability_dip():
    # 4 speech (0.9), 4 frames dipping to 0.4 (below start=0.5 but above the
    # 0.35 exit bar), 4 speech again, then silence. With hysteresis this is ONE
    # latched utterance; without it the dip would close the turn early and the
    # recovery would start a second one.
    probs = [0.9] * 4 + [0.4] * 4 + [0.9] * 4 + [0.0] * 6
    detector, audio_q, utterance_q, bus, state = make_detector(probs, SETTINGS)
    starts: list = []
    bus.subscribe(EventType.SPEECH_STARTED, lambda e: starts.append(e))

    feed_frames(audio_q, len(probs))
    got: dict = {}

    async def grab():
        got["u"] = await utterance_q.get()

    await run_detector_until(detector, grab)

    assert len(starts) == 1                 # single latched utterance, not two
    assert got["u"].duration_s >= 0.35      # spans the dip (~12 frames), not ~4


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


async def test_speech_during_thinking_is_captured_and_flushed_on_idle():
    # Pipeline is THINKING (assistant silent). User speaks 6 frames (~192 ms >
    # min_speech) then 4 silence frames close the capture -> held as pending.
    # Nothing is emitted until the pipeline returns to IDLE.
    probs = [0.9] * 6 + [0.0] * 4 + [0.0]  # final frame is processed at IDLE
    detector, audio_q, utterance_q, bus, state = make_detector(probs, SETTINGS)
    state.transition(PipelineState.THINKING)

    feed_frames(audio_q, 10)
    task = asyncio.create_task(detector.run())
    try:
        async def pending_set():
            while detector._pending is None:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(pending_set(), timeout=1.0)
        assert utterance_q.empty()  # not emitted while THINKING

        state.transition(PipelineState.IDLE)
        feed_frames(audio_q, 1)     # an IDLE frame triggers the flush
        utterance = await asyncio.wait_for(utterance_q.get(), timeout=1.0)
    finally:
        task.cancel()

    assert utterance.duration_s > 0
    assert detector._pending is None
    assert state.state is PipelineState.THINKING  # flush re-gates the mic


async def test_short_blip_during_thinking_is_not_captured():
    # 2 speech frames (64 ms < min_speech_ms=100) during THINKING must not be
    # captured as a pending utterance.
    probs = [0.9] * 2 + [0.0] * 6
    detector, audio_q, utterance_q, bus, state = make_detector(probs, SETTINGS)
    state.transition(PipelineState.THINKING)

    feed_frames(audio_q, 8)
    task = asyncio.create_task(detector.run())
    await asyncio.sleep(0.2)
    task.cancel()

    assert detector._pending is None
    assert utterance_q.empty()


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


SMART_TURN_SETTINGS = VADSettings(
    threshold=0.5, min_speech_ms=100, min_silence_ms=700, pre_roll_ms=64,
    smart_turn=True, smart_turn_pause_ms=90, smart_turn_threshold=0.5,
    predicted_silence_ms=90,
)


def _smart_turn_detector(probs, predict):
    audio_q = AudioQueue()
    utterance_q: asyncio.Queue = asyncio.Queue()
    bus = EventBus()
    state = StateMachine()
    detector = SpeechDetector(
        FakeVAD(probs), SMART_TURN_SETTINGS, audio_q, utterance_q, bus, state,
        frame_ms=32.0, turn_predictor=predict,
    )
    return detector, audio_q, utterance_q, bus, state


async def test_smart_turn_closes_turn_early_when_complete():
    # 6 speech + 6 silence (192 ms) — far short of min_silence_ms=700. Only Smart
    # Turn saying "complete" at the 90 ms pause lets this close at all. Feed in two
    # phases with an await between so the async prediction task can run (a live mic
    # yields between 32 ms frames; a pre-filled queue would not).
    probs = [0.9] * 6 + [0.0] * 6

    async def predict(_audio):
        return 0.9

    detector, audio_q, utterance_q, bus, state = _smart_turn_detector(probs, predict)
    task = asyncio.create_task(detector.run())
    try:
        feed_frames(audio_q, 12)
        await asyncio.sleep(0.1)             # let the queue drain and predict run
        assert utterance_q.empty()           # not closed yet on silence timer alone
        feed_frames(audio_q, 1)              # one more frame now that "complete" is set
        await asyncio.wait_for(utterance_q.get(), timeout=1.0)
    finally:
        task.cancel()

    assert state.state is PipelineState.THINKING


async def test_smart_turn_waits_when_incomplete():
    # Same short pause, but Smart Turn says "not done" -> the turn must NOT close
    # (it would need the full 700 ms silence, which never arrives here).
    probs = [0.9] * 6 + [0.0] * 6

    async def predict(_audio):
        return 0.1

    detector, audio_q, utterance_q, bus, state = _smart_turn_detector(probs, predict)
    feed_frames(audio_q, 12)
    task = asyncio.create_task(detector.run())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(utterance_q.get(), timeout=0.4)
    task.cancel()

    assert utterance_q.empty()


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


async def test_resumed_speech_clears_a_stale_endpoint_prediction_without_smart_turn():
    # Predictive (text) endpointing sets `_endpoint_predicted` with no turn
    # predictor wired. If resumed speech doesn't clear it, the flag latches and
    # the next pause anywhere in the turn commits after `predicted_silence_ms`
    # instead of the full timer — cutting the user off mid-sentence.
    settings = SETTINGS.model_copy(update={"predictive_endpointing": True})
    detector, *_ = make_detector([0.9, 0.9], settings)
    assert detector._turn_predictor is None

    frame = make_frame(np.zeros(FRAME, dtype=np.int16), SAMPLE_RATE)
    await detector._process_frame(frame)  # latch speech
    detector._endpoint_predicted = True   # as _transcribe_partial would
    detector._turn_checked = True

    await detector._process_frame(frame)  # user keeps talking

    assert detector._endpoint_predicted is False
    assert detector._turn_checked is False
    assert detector._required_silence_ms() == settings.min_silence_ms


async def test_noise_floor_window_receives_silent_frames():
    # The gate maintains the rolling floor as a side effect, so it must be
    # evaluated for every frame. Short-circuiting on `prob >= threshold` would
    # feed it only frames the VAD already called speech, leaving it measuring
    # the quiet end of *speech* rather than the background.
    settings = VADSettings(threshold=0.5, adaptive_noise=True, noise_window_ms=3200)
    detector, *_ = make_detector([0.1] * 20, settings)
    quiet = (np.full(FRAME, 30)).astype(np.int16)

    for _ in range(20):
        await detector._process_frame(make_frame(quiet, SAMPLE_RATE))

    assert detector._recorder.recording is False  # all frames were sub-threshold
    assert len(detector._rms_window) == 20
