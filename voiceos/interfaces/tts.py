"""TTS interface. Input: text. Output: streamed PCM audio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np


class BaseTTS(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""

    async def load(self) -> None:
        """Open connections / warm the model. Called once at startup."""

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """Stream synthesized audio as mono int16 chunks."""

    async def close(self) -> None:
        """Release connections."""
