"""Echo gate — suppress barge-in when the microphone is hearing the assistant.

On a speakerphone the assistant's own output re-enters the microphone, the VAD
sees energy, and barge-in fires. The assistant then interrupts itself,
transcribes its own voice, and replies to that. It is self-sustaining: one
observed session produced four turns of an agent talking to itself.

Full acoustic echo cancellation (WebRTC AEC3) solves this properly, but it
needs the far-end reference time-aligned into an adaptive filter. Here we only
need to *decide* whether a frame is echo, not remove it — and the reference is
already available, because `PlaybackWorker` has the exact samples it sent to
the speaker.

So: keep a short window of recently played audio and cross-correlate each
microphone frame against it. Echo is the same signal delayed and attenuated, so
it correlates strongly at some lag; the user's voice is uncorrelated. That is a
few hundred lines less than an adaptive filter and kills the dominant
false-trigger.

Limits worth knowing. Correlation only detects the *linear* part of the echo
path, so heavy speaker distortion at high volume weakens it, and someone
talking simultaneously with loud playback raises the mic's own correlation.
Both fail safe: a missed detection just leaves today's threshold-and-duration
gate in charge.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EchoGate:
    """Cross-correlates mic frames against recently played audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: int = 500,
        threshold: float = 0.35,
    ) -> None:
        # The window must span the echo delay — speaker to room to mic is
        # typically 100-500 ms once buffering is included. Too short and the
        # echo has not arrived yet; too long and unrelated audio drifts in.
        self._window = max(1, int(sample_rate * window_ms / 1000))
        self._threshold = threshold
        self._rate = sample_rate
        self._ref = np.zeros(0, dtype=np.float32)

    def push_reference(self, audio: np.ndarray, sample_rate: int) -> None:
        """Record what was just sent to the speaker."""
        if audio.size == 0:
            return
        pcm = audio.astype(np.float32)
        if pcm.max(initial=0) > 1.0 or pcm.min(initial=0) < -1.0:
            pcm = pcm / 32768.0
        if sample_rate != self._rate:
            # Cheap linear decimation. Correlation cares about envelope
            # alignment, not fidelity, so a polyphase filter buys nothing here.
            n = int(pcm.size * self._rate / sample_rate)
            if n <= 0:
                return
            pcm = np.interp(np.linspace(0, pcm.size - 1, n),
                            np.arange(pcm.size), pcm).astype(np.float32)
        self._ref = np.concatenate([self._ref, pcm])[-self._window:]

    def reset(self) -> None:
        """Forget the reference — call when playback stops."""
        self._ref = np.zeros(0, dtype=np.float32)

    def correlation(self, frame: np.ndarray) -> float:
        """Peak normalised cross-correlation of `frame` against the reference.

        Returns 0.0 when there is nothing to compare against, so an empty
        reference can never suppress a genuine barge-in.
        """
        if self._ref.size < frame.size or frame.size == 0:
            return 0.0
        x = frame.astype(np.float32)
        x = x / 32768.0 if np.abs(x).max() > 1.0 else x
        ref = self._ref

        x = x - x.mean()
        norm_x = float(np.linalg.norm(x))
        if norm_x < 1e-6:
            return 0.0

        # FFT correlation: direct would be ~4M multiply-adds per 32 ms frame,
        # which is a real cost inside the frame budget. This is ~0.1 ms.
        n = 1 << int(np.ceil(np.log2(ref.size + x.size)))
        corr = np.fft.irfft(np.fft.rfft(ref, n) * np.conj(np.fft.rfft(x, n)), n)
        corr = corr[: ref.size - x.size + 1]
        if corr.size == 0:
            return 0.0

        # Normalise per lag by the reference window's energy there, so a loud
        # passage cannot masquerade as a good match.
        energy = np.convolve(ref * ref, np.ones(x.size, dtype=np.float32), mode="valid")
        denom = np.sqrt(np.maximum(energy, 1e-12)) * norm_x
        return float(np.max(np.abs(corr[: denom.size]) / denom))

    def is_echo(self, frame: np.ndarray) -> bool:
        return self.correlation(frame) >= self._threshold
