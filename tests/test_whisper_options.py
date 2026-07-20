"""faster-whisper is configured with the hallucination guards it needs."""

import numpy as np

from voiceos.config.settings import STTSettings
from voiceos.stt.whisper import FasterWhisperSTT


class _CapturingModel:
    def __init__(self):
        self.kwargs = {}

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs

        class _Info:
            language = "en"
            duration = 1.0

        return [], _Info()


async def test_hallucination_guards_are_passed_through():
    # Whisper loops the previous phrase when conditioned on it, and emits
    # subtitle artifacts ("Thank you for watching") on near-silence. These
    # three options are the documented mitigations; they only work if they
    # actually reach the model.
    stt = FasterWhisperSTT(STTSettings())
    stt._model = _CapturingModel()

    await stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)

    assert stt._model.kwargs["condition_on_previous_text"] is False
    assert stt._model.kwargs["no_speech_threshold"] == 0.6
    assert stt._model.kwargs["compression_ratio_threshold"] == 2.4


async def test_guards_are_overridable():
    stt = FasterWhisperSTT(STTSettings(condition_on_previous_text=True))
    stt._model = _CapturingModel()

    await stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)

    assert stt._model.kwargs["condition_on_previous_text"] is True
