"""Piper (local TTS) provider tests — mocked subprocess, no Piper install needed."""

import asyncio

import numpy as np
import pytest

from voiceos.config.settings import Settings, TTSSettings
from voiceos.pipeline.pipeline import create_tts
from voiceos.tts.piper import PiperTTS


def test_factory_selects_piper():
    s = Settings()
    s.tts.provider = "piper"
    assert isinstance(create_tts(s), PiperTTS)


async def test_load_errors_clearly_without_binary():
    tts = PiperTTS(TTSSettings(provider="piper", piper_binary="no-such-binary-xyz123"))
    with pytest.raises(RuntimeError, match="piper binary"):
        await tts.load()


async def test_load_errors_without_voice_model(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/piper")  # binary present
    tts = PiperTTS(TTSSettings(provider="piper", piper_model=""))
    with pytest.raises(RuntimeError, match="voice model"):
        await tts.load()


async def test_synthesize_streams_int16_chunks(monkeypatch):
    pcm = np.arange(-500, 500, dtype="<i2").tobytes()

    class FakeStdout:
        def __init__(self, data):
            self.data, self.i = data, 0

        async def read(self, n):
            if self.i >= len(self.data):
                return b""
            chunk = self.data[self.i : self.i + n]
            self.i += n
            return chunk

    class FakeStdin:
        def write(self, b):
            pass

        async def drain(self):
            pass

        def close(self):
            pass

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStdout(pcm)
            self.stderr = FakeStdout(b"")

        async def wait(self):
            return 0

    async def fake_exec(*a, **k):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    tts = PiperTTS(TTSSettings(provider="piper"))
    tts._binary = "piper"  # normally set by load()
    chunks = [c async for c in tts.synthesize("नमस्ते जी")]
    assert chunks and all(c.dtype == np.int16 for c in chunks)
    assert np.concatenate(chunks).tobytes() == pcm  # exact audio round-trips


def test_sample_rate_from_settings():
    tts = PiperTTS(TTSSettings(provider="piper", piper_sample_rate=16000))
    assert tts.sample_rate == 16000
