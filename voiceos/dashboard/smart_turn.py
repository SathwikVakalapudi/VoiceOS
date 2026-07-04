"""Smart Turn v3 — semantic end-of-turn detection (pipecat-ai/smart-turn-v3).

Silero VAD tells us *speech stopped*; Smart Turn tells us whether the speaker
actually *finished their thought* or is just pausing mid-sentence. It's a tiny
Whisper-tiny-encoder classifier (8 MB ONNX, ~12 ms on CPU) that reads the raw
waveform and outputs a "turn complete" probability. Runs locally, Hindi-capable.

Pipeline role: on a short (~250 ms) Silero-detected pause, run Smart Turn on the
utterance-so-far. If complete → end the turn now (fast, no long trailing wait);
if not → keep listening (the user paused to think). Falls back to a hard silence
timeout so a trailing-off answer still ends.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_SR = 16000
_MAX = 8 * _SR   # model input window: 8 seconds


class SmartTurn:
    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        self._path = model_path
        self._threshold = threshold
        self._sess = None
        self._fe = None

    def load(self) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        self._sess = ort.InferenceSession(
            self._path, providers=["CPUExecutionProvider"]
        )
        # chunk_length=8 -> 80 mel bins x 800 frames, matching the model input.
        self._fe = WhisperFeatureExtractor(chunk_length=8)
        logger.info("Smart Turn loaded (%s, threshold=%.2f)", self._path, self._threshold)

    def complete_prob(self, audio_int16: np.ndarray) -> float:
        """Probability that the turn is complete, from int16 16 kHz audio.
        Matches pipecat's reference inference.py exactly (do_normalize=True; the
        model's output is already a probability — no extra sigmoid)."""
        if self._sess is None:
            raise RuntimeError("SmartTurn.load() must be called first")
        audio = np.asarray(audio_int16, dtype=np.float32) / 32768.0
        if len(audio) > _MAX:
            audio = audio[-_MAX:]               # keep the most recent 8 s
        feats = self._fe(
            audio, sampling_rate=_SR, return_tensors="np",
            padding="max_length", max_length=_MAX, truncation=True, do_normalize=True,
        ).input_features.astype(np.float32)
        out = self._sess.run(None, {"input_features": feats})[0]
        return float(np.asarray(out).reshape(-1)[0])

    def is_complete(self, audio_int16: np.ndarray) -> bool:
        return self.complete_prob(audio_int16) >= self._threshold
