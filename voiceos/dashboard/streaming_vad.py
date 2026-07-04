"""Server-side streaming endpointer built on Silero VAD.

Replaces the browser's crude energy-VAD with a trained speech model. Feed it
16 kHz int16 PCM as it streams in; it runs Silero on exact 512-sample frames
(32 ms), tracks speech with hysteresis (enter at `on`, stay at the lower `off`
to avoid flicker), keeps a short pre-roll so the onset is never clipped, and
emits a complete utterance once speech is followed by `min_silence_ms` of quiet.

This is the "Silero VAD" box of the Gen-2 pipeline; Smart Turn v3 (semantic
end-of-turn) plugs in on top of the utterances this emits.
"""

from __future__ import annotations

from typing import Awaitable, Callable

import numpy as np

_FRAME = 512          # Silero v5 requires exactly 512 samples @ 16 kHz
_FRAME_MS = 1000 * _FRAME / 16000   # 32 ms


class StreamingEndpointer:
    def __init__(
        self,
        vad,
        *,
        on_threshold: float = 0.5,
        off_threshold: float = 0.35,
        min_speech_ms: int = 200,
        min_silence_ms: int = 600,
        preroll_ms: int = 250,
    ) -> None:
        self._vad = vad
        self._on = on_threshold
        self._off = off_threshold
        self._min_speech = min_speech_ms
        self._min_silence = min_silence_ms
        self._preroll_frames = max(1, int(preroll_ms / _FRAME_MS))
        self._buf = np.zeros(0, dtype=np.int16)
        self._in_speech = False
        self._speech: list[np.ndarray] = []
        self._preroll: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    @property
    def in_speech(self) -> bool:
        """True while the user is actively speaking (used for barge-in)."""
        return self._in_speech

    def reset(self) -> None:
        try:
            self._vad.reset()
        except Exception:
            pass
        self._buf = np.zeros(0, dtype=np.int16)
        self._in_speech = False
        self._speech = []
        self._preroll = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def push(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Feed int16 PCM @16k. Returns any completed utterances (int16 arrays)."""
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.int16)])
        done: list[np.ndarray] = []
        while len(self._buf) >= _FRAME:
            frame = self._buf[:_FRAME]
            self._buf = self._buf[_FRAME:]
            prob = self._vad.process(frame.astype(np.float32) / 32768.0, 16000)
            threshold = self._off if self._in_speech else self._on
            if prob >= threshold:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech = list(self._preroll)   # keep the onset
                    self._speech_ms = 0.0
                self._speech.append(frame)
                self._speech_ms += _FRAME_MS
                self._silence_ms = 0.0
            else:
                if self._in_speech:
                    self._speech.append(frame)           # trailing pad
                    self._silence_ms += _FRAME_MS
                    if self._silence_ms >= self._min_silence:
                        self._in_speech = False
                        if self._speech_ms >= self._min_speech:
                            done.append(np.concatenate(self._speech))
                        self._speech = []
                        self._speech_ms = 0.0
                        self._silence_ms = 0.0
                else:
                    self._preroll.append(frame)
                    if len(self._preroll) > self._preroll_frames:
                        self._preroll.pop(0)
        return done


class SmartTurnEndpointer:
    """Silero VAD + Smart Turn v3 semantic end-of-turn.

    Silero segments speech; on a short pause it asks Smart Turn "did they finish
    their thought?". Complete -> end now (snappy). Incomplete -> keep listening
    (they paused mid-sentence). A hard `max_silence_ms` guarantees termination
    if they simply trail off. `predict` is an async callable(int16 audio)->P.
    """

    def __init__(
        self,
        vad,
        predict: Callable[[np.ndarray], Awaitable[float]],
        *,
        on_threshold: float = 0.5,
        off_threshold: float = 0.35,
        pause_ms: int = 300,
        max_silence_ms: int = 1600,
        min_speech_ms: int = 200,
        preroll_ms: int = 300,
        turn_threshold: float = 0.5,
    ) -> None:
        self._vad = vad
        self._predict = predict
        self._on = on_threshold
        self._off = off_threshold
        self._pause_ms = pause_ms
        self._max_silence = max_silence_ms
        self._min_speech = min_speech_ms
        self._preroll_frames = max(1, int(preroll_ms / _FRAME_MS))
        self._turn_threshold = turn_threshold
        self._buf = np.zeros(0, dtype=np.int16)
        self._in_speech = False
        self._audio: list[np.ndarray] = []
        self._preroll: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._checked = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._audio = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._checked = False

    def reset(self) -> None:
        try:
            self._vad.reset()
        except Exception:
            pass
        self._buf = np.zeros(0, dtype=np.int16)
        self._preroll = []
        self._reset_utterance()

    async def push(self, pcm: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.int16)])
        done: list[np.ndarray] = []
        while len(self._buf) >= _FRAME:
            frame = self._buf[:_FRAME]
            self._buf = self._buf[_FRAME:]
            prob = self._vad.process(frame.astype(np.float32) / 32768.0, 16000)
            threshold = self._off if self._in_speech else self._on
            if prob >= threshold:
                if not self._in_speech:
                    self._in_speech = True
                    self._audio = list(self._preroll)
                    self._speech_ms = 0.0
                self._audio.append(frame)
                self._speech_ms += _FRAME_MS
                self._silence_ms = 0.0
                self._checked = False          # user is talking again → re-check next pause
            elif self._in_speech:
                self._audio.append(frame)
                self._silence_ms += _FRAME_MS
                if self._speech_ms >= self._min_speech and self._silence_ms >= self._max_silence:
                    done.append(np.concatenate(self._audio))   # trailed off → force end
                    self._reset_utterance()
                elif (
                    not self._checked
                    and self._speech_ms >= self._min_speech
                    and self._silence_ms >= self._pause_ms
                ):
                    self._checked = True
                    audio = np.concatenate(self._audio)
                    if await self._predict(audio) >= self._turn_threshold:
                        done.append(audio)                     # semantically complete
                        self._reset_utterance()
            else:
                self._preroll.append(frame)
                if len(self._preroll) > self._preroll_frames:
                    self._preroll.pop(0)
        return done
