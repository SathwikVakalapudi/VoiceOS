"""Speech detector worker.

Consumes microphone frames, asks the VAD "speaking or silent?", and
segments the stream into complete utterances:

- speech starts when probability crosses the threshold (pre-roll included)
- speech ends after `min_silence_ms` of trailing silence
- bursts shorter than `min_speech_ms` are discarded as noise
- utterances longer than `max_utterance_s` are force-closed

While the pipeline is THINKING or SPEAKING, frames are dropped so the
assistant never hears itself. Barge-in later relaxes this gate.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Awaitable, Callable

import numpy as np

from voiceos.audio.audio_queue import AudioFrame, AudioQueue, Utterance
from voiceos.audio.recorder import UtteranceRecorder
from voiceos.config.settings import VADSettings
from voiceos.interfaces.vad import BaseVAD
from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.state import PipelineState, StateMachine
from voiceos.stt.streaming import StreamingTranscriber
from voiceos.utils.audio import int16_to_float32
from voiceos.vad.endpoint import EndpointPredictor

logger = logging.getLogger(__name__)


class SpeechDetector:
    def __init__(
        self,
        vad: BaseVAD,
        settings: VADSettings,
        audio_queue: AudioQueue,
        utterance_queue: asyncio.Queue[Utterance],
        event_bus: EventBus,
        state: StateMachine,
        frame_ms: float = 32.0,
        on_barge_in: Callable[[], Awaitable[None]] | None = None,
        partial_transcriber: StreamingTranscriber | None = None,
        endpoint_predictor: EndpointPredictor | None = None,
    ) -> None:
        self._vad = vad
        self._settings = settings
        self._audio_queue = audio_queue
        self._utterance_queue = utterance_queue
        self._bus = event_bus
        self._state = state
        self._on_barge_in = on_barge_in
        self._partial_transcriber = partial_transcriber
        self._endpoint_predictor = endpoint_predictor
        pre_roll_frames = max(1, int(settings.pre_roll_ms / frame_ms))
        self._recorder = UtteranceRecorder(pre_roll_frames=pre_roll_frames)
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._barge_ms = 0.0
        noise_frames = max(1, int(settings.noise_window_ms / max(1.0, frame_ms)))
        self._rms_window: deque[float] = deque(maxlen=noise_frames)
        # Predictive-endpointing state (only used when a transcriber is wired).
        self._since_partial_ms = 0.0
        self._partial_task: asyncio.Task | None = None
        self._endpoint_predicted = False

    async def run(self) -> None:
        while True:
            frame = await self._audio_queue.get()
            try:
                await self._process_frame(frame)
            except Exception:
                logger.exception("speech detection failed")
                await self._bus.emit(EventType.ERROR, {"stage": "vad"})
                self._reset()

    async def _process_frame(self, frame: AudioFrame) -> None:
        state = self._state.state

        if state is PipelineState.SPEAKING and self._settings.barge_in and self._on_barge_in:
            await self._watch_for_barge_in(frame)
            return

        if state in (PipelineState.THINKING, PipelineState.SPEAKING):
            # Assistant is busy and barge-in doesn't apply: ignore the mic.
            if self._recorder.recording:
                self._reset()
            return

        signal = int16_to_float32(frame.data)
        prob = self._vad.process(signal, frame.sample_rate)
        is_speech = prob >= self._settings.threshold and self._passes_noise_gate(signal)

        if not self._recorder.recording:
            self._recorder.push_idle(frame)
            if is_speech:
                self._recorder.start()
                self._silence_ms = 0.0
                self._speech_ms = frame.duration_ms
                self._state.transition(PipelineState.LISTENING)
                await self._bus.emit(EventType.SPEECH_STARTED, {"probability": prob})
            return

        self._recorder.push(frame)
        if is_speech:
            self._speech_ms += frame.duration_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame.duration_ms

        self._since_partial_ms += frame.duration_ms
        self._maybe_transcribe_partial()

        utterance_done = self._silence_ms >= self._required_silence_ms()
        too_long = self._recorder.duration_ms >= self._settings.max_utterance_s * 1000
        if utterance_done or too_long:
            await self._finish_utterance()

    def _required_silence_ms(self) -> float:
        """Trailing silence needed to end the turn. A predicted-complete turn
        closes after only a short pause; otherwise, when adaptive, a still-
        short utterance waits longer (likely mid-thought) and a substantial
        one closes at the normal, snappier threshold."""
        base = (
            self._settings.min_silence_short_ms
            if self._settings.adaptive_silence
            and self._speech_ms < self._settings.short_utterance_ms
            else self._settings.min_silence_ms
        )
        if self._endpoint_predicted:
            return min(self._settings.predicted_silence_ms, base)
        return base

    def _maybe_transcribe_partial(self) -> None:
        """Kick off a rolling transcription of the utterance-so-far (at most
        one in flight) so the endpoint predictor can guess turn completion."""
        if not (self._settings.predictive_endpointing and self._partial_transcriber):
            return
        if self._speech_ms < self._settings.min_partial_speech_ms:
            return
        if self._since_partial_ms < self._settings.partial_interval_ms:
            return
        if self._partial_task is not None and not self._partial_task.done():
            return
        snapshot = self._recorder.snapshot()
        if snapshot is None:
            return
        self._since_partial_ms = 0.0
        audio, sample_rate = snapshot
        self._partial_task = asyncio.create_task(
            self._transcribe_partial(int16_to_float32(audio), sample_rate)
        )

    async def _transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            text = await self._partial_transcriber.partial(audio, sample_rate)
        except Exception:
            logger.debug("partial transcription failed", exc_info=True)
            return
        if not text or len(text.strip()) < self._settings.min_partial_chars:
            return
        await self._bus.emit(EventType.PARTIAL_TRANSCRIPT, {"text": text})
        if self._endpoint_predictor and self._endpoint_predictor.looks_complete(text):
            logger.debug("endpoint predicted complete: %.60s", text)
            self._endpoint_predicted = True

    def _cancel_partial(self) -> None:
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        self._partial_task = None
        self._since_partial_ms = 0.0
        self._endpoint_predicted = False

    def _passes_noise_gate(self, signal: np.ndarray) -> bool:
        """Track a rolling noise floor and, when enabled, require this frame's
        energy to clear it — so steady background noise or echo that fools the
        VAD probability still doesn't register as speech."""
        rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
        self._rms_window.append(rms)
        if not self._settings.adaptive_noise or len(self._rms_window) < 10:
            return True
        # 20th percentile approximates the quiet floor even while speech is in
        # the window; a real voice sits well above it.
        floor = float(np.percentile(self._rms_window, 20))
        return rms >= floor * self._settings.noise_margin

    async def _watch_for_barge_in(self, frame: AudioFrame) -> None:
        """While the assistant speaks, listen for the user talking over it."""
        prob = self._vad.process(int16_to_float32(frame.data), frame.sample_rate)
        self._recorder.push_idle(frame)  # keep the start of their sentence
        if prob >= self._settings.barge_in_threshold:
            self._barge_ms += frame.duration_ms
        else:
            self._barge_ms = 0.0
        if self._barge_ms < self._settings.barge_in_speech_ms:
            return

        logger.info("barge-in detected — interrupting assistant")
        await self._on_barge_in()
        # Seamlessly switch into recording the user's utterance.
        self._recorder.start()
        self._speech_ms = self._barge_ms
        self._silence_ms = 0.0
        self._barge_ms = 0.0
        self._since_partial_ms = 0.0
        self._endpoint_predicted = False
        self._state.transition(PipelineState.LISTENING)
        await self._bus.emit(EventType.BARGE_IN, {})
        await self._bus.emit(
            EventType.SPEECH_STARTED, {"probability": prob, "barge_in": True}
        )

    async def _finish_utterance(self) -> None:
        utterance = self._recorder.finish()
        accepted = (
            utterance is not None and self._speech_ms >= self._settings.min_speech_ms
        )
        await self._bus.emit(
            EventType.SPEECH_ENDED,
            {
                "duration_s": utterance.duration_s if utterance else 0.0,
                "accepted": accepted,
            },
        )
        if accepted and utterance is not None:
            # Gate the mic while downstream stages work on this utterance.
            self._state.transition(PipelineState.THINKING)
            await self._utterance_queue.put(utterance)
        else:
            self._state.transition(PipelineState.IDLE)
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._cancel_partial()
        self._vad.reset()

    def _reset(self) -> None:
        self._recorder.reset()
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._barge_ms = 0.0
        self._cancel_partial()
        self._vad.reset()
