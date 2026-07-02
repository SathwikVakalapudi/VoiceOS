"""Utterance recorder.

Accumulates frames while the user speaks, keeping a rolling pre-roll
buffer so the first syllable (spoken just before VAD triggers) is not
clipped. Pure buffering — no AI here.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from voiceos.audio.audio_queue import AudioFrame, Utterance


class UtteranceRecorder:
    def __init__(self, pre_roll_frames: int) -> None:
        self._pre_roll: deque[AudioFrame] = deque(maxlen=max(1, pre_roll_frames))
        self._frames: list[AudioFrame] = []
        self.recording = False

    def push_idle(self, frame: AudioFrame) -> None:
        """Feed a frame while no speech is active (fills the pre-roll)."""
        self._pre_roll.append(frame)

    def start(self) -> None:
        """Begin an utterance, seeding it with the pre-roll buffer."""
        self._frames = list(self._pre_roll)
        self._pre_roll.clear()
        self.recording = True

    def push(self, frame: AudioFrame) -> None:
        self._frames.append(frame)

    @property
    def duration_ms(self) -> float:
        return sum(f.duration_ms for f in self._frames)

    def snapshot(self) -> tuple[np.ndarray, int] | None:
        """Concatenated audio recorded so far, for partial transcription.
        Returns (audio, sample_rate) or None if nothing has been recorded."""
        if not self._frames:
            return None
        return np.concatenate([f.data for f in self._frames]), self._frames[0].sample_rate

    def finish(self) -> Utterance | None:
        """Close the utterance and return it (None if nothing was recorded)."""
        self.recording = False
        if not self._frames:
            return None
        sample_rate = self._frames[0].sample_rate
        audio = np.concatenate([f.data for f in self._frames])
        self._frames = []
        return Utterance(
            audio=audio,
            sample_rate=sample_rate,
            duration_s=len(audio) / sample_rate,
        )

    def reset(self) -> None:
        self._frames = []
        self._pre_roll.clear()
        self.recording = False
