"""VAD interface. Only ever answers one question: speaking or silent?"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVAD(ABC):
    async def load(self) -> None:
        """Load model weights. Called once before the pipeline starts."""

    @abstractmethod
    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        """Return speech probability (0.0–1.0) for one audio frame.

        ``frame`` is mono float32 in [-1, 1]. Frame length is fixed by the
        implementation (Silero v5: 512 samples @ 16 kHz).
        """

    def reset(self) -> None:
        """Clear internal state between utterances."""
