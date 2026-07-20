"""Echo gate: the assistant's own voice must not trigger barge-in."""

import asyncio

import numpy as np
import pytest

from voiceos.audio.audio_queue import AudioQueue, make_frame
from voiceos.config.settings import VADSettings
from voiceos.interfaces.vad import BaseVAD
from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.state import PipelineState, StateMachine
from voiceos.vad.detector import SpeechDetector
from voiceos.vad.echo import EchoGate

SR = 16000
FRAME = 512


def _voice(seed: int, f0: float, n: int = FRAME * 40) -> np.ndarray:
    """Speech-like signal: harmonic stack with a wandering envelope.

    `f0` is explicit because an earlier version derived it from the seed and
    two "different speakers" landed on 130.47 Hz and 130.24 Hz — the same
    signal with a phase offset, correlating at 0.94. Real voices differ in
    formants and timing; measured against actual recordings the gate scores a
    different speaker at ~0.24.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    sig = sum(np.sin(2 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 10))
    sig *= 1 + 0.5 * np.sin(2 * np.pi * (2 + 3 * rng.random()) * t)
    sig += 0.15 * rng.standard_normal(n)      # breath, decorrelates the tail
    return (sig / np.abs(sig).max() * 0.6).astype(np.float32)


ASSISTANT_F0, USER_F0 = 120.0, 205.0          # clearly different speakers


def test_echo_is_recognised_after_playback():
    gate = EchoGate(sample_rate=SR, window_ms=500, threshold=0.35)
    played = _voice(1, ASSISTANT_F0)
    delay = int(SR * 0.15)                     # 150 ms speaker -> room -> mic

    for i in range(FRAME, played.size - delay, FRAME):
        gate.push_reference(played[i - FRAME:i], SR)

    echo = played[:FRAME] * 0.4                # attenuated copy of what we played
    assert gate.is_echo(echo)


def test_a_different_voice_is_not_echo():
    gate = EchoGate(sample_rate=SR, window_ms=500, threshold=0.35)
    played, other = _voice(1, ASSISTANT_F0), _voice(99, USER_F0)
    for i in range(FRAME, played.size, FRAME):
        gate.push_reference(played[i - FRAME:i], SR)

    assert not gate.is_echo(other[:FRAME])


def test_empty_reference_never_suppresses():
    # Nothing has been played, so nothing can be echo — a genuine barge-in
    # must never be blocked just because the gate has no data yet.
    gate = EchoGate(sample_rate=SR)
    assert gate.correlation(_voice(1, ASSISTANT_F0)[:FRAME]) == 0.0
    assert not gate.is_echo(_voice(1, ASSISTANT_F0)[:FRAME])


def test_reference_is_resampled_from_the_tts_rate():
    # TTS runs at 24 kHz while the mic is 16 kHz; without resampling the
    # correlation is computed against a time-stretched signal and fails.
    gate = EchoGate(sample_rate=SR, window_ms=500, threshold=0.35)
    played16 = _voice(1, ASSISTANT_F0)
    played24 = np.interp(
        np.linspace(0, played16.size - 1, int(played16.size * 24000 / SR)),
        np.arange(played16.size), played16,
    ).astype(np.float32)

    step = int(FRAME * 24000 / SR)
    for i in range(step, played24.size, step):
        gate.push_reference(played24[i - step:i], 24000)

    assert gate.is_echo(played16[:FRAME] * 0.4)


class _AlwaysSpeech(BaseVAD):
    def process(self, frame, sample_rate):
        return 0.99


def _detector(settings, echo_gate, on_barge_in):
    return SpeechDetector(
        _AlwaysSpeech(), settings, AudioQueue(), asyncio.Queue(),
        EventBus(), StateMachine(), frame_ms=32.0,
        on_barge_in=on_barge_in, echo_gate=echo_gate,
    )


@pytest.mark.asyncio
async def test_barge_in_is_suppressed_while_the_mic_hears_the_assistant():
    # The VAD is certain this is speech. Only the echo gate can stop it, which
    # is the whole point: on a speakerphone the assistant's own voice looks
    # exactly like a user talking over it.
    settings = VADSettings(barge_in=True, barge_in_threshold=0.7, barge_in_speech_ms=100)
    played = _voice(1, ASSISTANT_F0)
    gate = EchoGate(sample_rate=SR, window_ms=500, threshold=0.35)
    for i in range(FRAME, played.size, FRAME):
        gate.push_reference(played[i - FRAME:i], SR)

    fired = asyncio.Event()
    detector = _detector(settings, gate, lambda: (fired.set(), asyncio.sleep(0))[1])
    detector._state.transition(PipelineState.SPEAKING)

    echo_pcm = (played[:FRAME] * 0.4 * 32767).astype(np.int16)
    for _ in range(10):                        # 320 ms, well past the trigger
        await detector._process_frame(make_frame(echo_pcm, SR))

    assert not fired.is_set()
    assert detector._state.state is PipelineState.SPEAKING


@pytest.mark.asyncio
async def test_a_real_interruption_still_triggers_barge_in():
    settings = VADSettings(barge_in=True, barge_in_threshold=0.7, barge_in_speech_ms=100)
    played, other = _voice(1, ASSISTANT_F0), _voice(99, USER_F0)
    gate = EchoGate(sample_rate=SR, window_ms=500, threshold=0.35)
    for i in range(FRAME, played.size, FRAME):
        gate.push_reference(played[i - FRAME:i], SR)

    fired = asyncio.Event()

    async def on_barge_in():
        fired.set()

    detector = _detector(settings, gate, on_barge_in)
    detector._state.transition(PipelineState.SPEAKING)

    user_pcm = (other[:FRAME] * 32767).astype(np.int16)
    for _ in range(10):
        await detector._process_frame(make_frame(user_pcm, SR))

    assert fired.is_set()
