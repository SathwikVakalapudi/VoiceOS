"""Audio format helpers."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert PCM int16 to float32 in [-1, 1]."""
    return audio.astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 in [-1, 1] to PCM int16."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono int16 audio to a WAV file (debugging aid)."""
    if audio.dtype != np.int16:
        audio = float32_to_int16(audio)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())
