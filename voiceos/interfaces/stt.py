"""STT interface. Input: audio. Output: transcript. Nothing else."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    duration_s: float = 0.0


class BaseSTT(ABC):
    async def load(self) -> None:
        """Load model weights. Called once before the pipeline starts."""

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe one complete utterance (mono float32 in [-1, 1])."""

    async def close(self) -> None:
        """Release resources."""
