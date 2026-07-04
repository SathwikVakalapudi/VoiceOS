"""StreamingEndpointer tests with a deterministic fake VAD (no model needed)."""

import numpy as np

from voiceos.dashboard.streaming_vad import StreamingEndpointer

_FRAME = 512


class FakeVAD:
    """Returns a scripted speech probability per 512-sample frame."""

    def __init__(self, probs):
        self.probs = probs
        self.i = 0

    def process(self, frame, sample_rate):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset(self):
        pass


def _pcm(n_frames):
    return np.ones(_FRAME * n_frames, dtype=np.int16)


def test_emits_one_utterance_after_speech_then_silence():
    # 3 silence, 12 speech (~384ms), 15 silence (~480ms > min_silence)
    probs = [0.0] * 3 + [0.9] * 12 + [0.0] * 15
    ep = StreamingEndpointer(FakeVAD(probs), min_silence_ms=300, min_speech_ms=150)
    out = ep.push(_pcm(len(probs)))
    assert len(out) == 1
    # utterance includes pre-roll + speech + trailing pad -> longer than raw speech
    assert len(out[0]) >= _FRAME * 12


def test_short_blip_is_dropped():
    # 3 silence, 2 speech (~64ms < min_speech), 12 silence
    probs = [0.0] * 3 + [0.9] * 2 + [0.0] * 12
    ep = StreamingEndpointer(FakeVAD(probs), min_silence_ms=300, min_speech_ms=200)
    out = ep.push(_pcm(len(probs)))
    assert out == []


def test_hysteresis_keeps_speech_through_a_dip():
    # dip to 0.4 (below on=0.5 but above off=0.35) must NOT end the turn
    probs = [0.0] * 2 + [0.9] * 5 + [0.4] * 5 + [0.9] * 5 + [0.0] * 15
    ep = StreamingEndpointer(FakeVAD(probs), on_threshold=0.5, off_threshold=0.35,
                             min_silence_ms=300, min_speech_ms=150)
    out = ep.push(_pcm(len(probs)))
    assert len(out) == 1  # one continuous utterance, not two


def test_in_speech_flag_tracks_state():
    ep = StreamingEndpointer(FakeVAD([0.0, 0.9, 0.9]), min_silence_ms=300)
    ep.push(_pcm(1))
    assert ep.in_speech is False
    ep.push(_pcm(2))
    assert ep.in_speech is True


def test_two_separate_utterances():
    probs = ([0.9] * 8 + [0.0] * 12) + ([0.9] * 8 + [0.0] * 12)
    ep = StreamingEndpointer(FakeVAD(probs), min_silence_ms=300, min_speech_ms=150)
    out = ep.push(_pcm(len(probs)))
    assert len(out) == 2


# ---- SmartTurnEndpointer (fake VAD + fake semantic predictor) ----
import pytest
from voiceos.dashboard.streaming_vad import SmartTurnEndpointer


@pytest.mark.asyncio
async def test_smart_turn_ends_immediately_when_complete():
    probs = [0.0] * 2 + [0.9] * 10 + [0.0] * 8   # speech, then a short pause

    async def predict(audio):
        return 0.9                                # "you're done"

    ep = SmartTurnEndpointer(FakeVAD(probs), predict, pause_ms=200,
                             max_silence_ms=3000, min_speech_ms=150)
    out = await ep.push(_pcm(len(probs)))
    assert len(out) == 1                          # ended at the short pause


@pytest.mark.asyncio
async def test_smart_turn_waits_when_incomplete_then_forces_on_timeout():
    probs = [0.0] * 2 + [0.9] * 10 + [0.0] * 60   # speech, then long silence

    async def predict(audio):
        return 0.05                               # "not done — keep listening"

    ep = SmartTurnEndpointer(FakeVAD(probs), predict, pause_ms=200,
                             max_silence_ms=800, min_speech_ms=150)
    out = await ep.push(_pcm(len(probs)))
    assert len(out) == 1                          # forced only at max_silence
