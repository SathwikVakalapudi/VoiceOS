"""Fallback provider tests: a failing primary transparently rolls over."""

import numpy as np
import pytest

from voiceos.stt.base import TranscriptionResult
from voiceos.stt.fallback import FallbackSTT
from voiceos.tts.fallback import FallbackTTS


class FakeTTS:
    def __init__(self, sample_rate=24000, fail=False, emit_then_fail=False):
        self._sample_rate = sample_rate
        self._fail = fail
        self._emit_then_fail = emit_then_fail

    @property
    def sample_rate(self):
        return self._sample_rate

    async def load(self):
        pass

    async def synthesize(self, text):
        if self._fail:
            raise RuntimeError("provider down")
        yield np.ones(4, dtype="<i2")
        if self._emit_then_fail:
            raise RuntimeError("died mid-sentence")

    async def close(self):
        pass


class FakeSTT:
    def __init__(self, text=None, fail=False):
        self._text = text
        self._fail = fail

    async def load(self):
        pass

    async def transcribe(self, audio, sample_rate):
        if self._fail:
            raise RuntimeError("provider down")
        return TranscriptionResult(text=self._text)

    async def close(self):
        pass


async def _collect(tts, text):
    return [chunk async for chunk in tts.synthesize(text)]


async def test_tts_falls_back_when_primary_fails_before_audio():
    tts = FallbackTTS([FakeTTS(fail=True), FakeTTS()])
    chunks = await _collect(tts, "hello")
    assert len(chunks) == 1


async def test_tts_does_not_fall_back_after_audio_emitted():
    # Once audio is out, switching voices would repeat it — must raise.
    tts = FallbackTTS([FakeTTS(emit_then_fail=True), FakeTTS()])
    with pytest.raises(RuntimeError):
        await _collect(tts, "hello")


async def test_tts_reports_primary_sample_rate():
    tts = FallbackTTS([FakeTTS(sample_rate=24000), FakeTTS(sample_rate=16000)])
    assert tts.sample_rate == 24000


async def test_stt_falls_back_to_next_provider():
    stt = FallbackSTT([FakeSTT(fail=True), FakeSTT(text="from backup")])
    result = await stt.transcribe(np.zeros(8, dtype="float32"), 16000)
    assert result.text == "from backup"


async def test_stt_raises_when_all_fail():
    stt = FallbackSTT([FakeSTT(fail=True), FakeSTT(fail=True)])
    with pytest.raises(RuntimeError):
        await stt.transcribe(np.zeros(8, dtype="float32"), 16000)
