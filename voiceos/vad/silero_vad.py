"""Silero VAD implementation.

Uses the `silero-vad` pip package. Silero v5 requires exactly
512-sample frames at 16 kHz (or 256 at 8 kHz).
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from voiceos.config.settings import VADSettings
from voiceos.vad.base import BaseVAD

logger = logging.getLogger(__name__)


class SileroVAD(BaseVAD):
    def __init__(self, settings: VADSettings) -> None:
        self._settings = settings
        self._model = None

    async def load(self) -> None:
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_sync)
        logger.info("Silero VAD loaded (onnx=%s)", self._settings.use_onnx)

    def _load_sync(self):
        from silero_vad import load_silero_vad

        return load_silero_vad(onnx=self._settings.use_onnx)

    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        if self._model is None:
            raise RuntimeError("SileroVAD.load() must be called first")
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        return float(self._model(tensor, sample_rate).item())

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()
