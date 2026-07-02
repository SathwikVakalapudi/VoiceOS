"""Telephony audio transcoding.

Bridges 8 kHz G.711 (mu-law) telephony audio and the VoiceOS pipeline's
linear-PCM rates (16 kHz in, 24 kHz out). Resampling is *stateful* — the
converter state is threaded across frames so 20 ms chunks don't get clicks
at their boundaries — so use one encoder/decoder per call, per direction.

Codec + resampling use the stdlib ``audioop`` (mu-law via lin2ulaw/ulaw2lin,
sample-rate conversion via ratecv). ``audioop`` was removed in Python 3.13
(PEP 594); install ``audioop-lts`` there, which restores the same module.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    with warnings.catch_warnings():
        # Deprecated on 3.11/3.12, removed on 3.13 — silence the notice.
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop
except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.13+
    raise ModuleNotFoundError(
        "audioop was removed in Python 3.13; run `pip install audioop-lts` "
        "to enable telephony transcoding"
    ) from exc

TELEPHONY_RATE = 8000  # G.711 is always 8 kHz mono
_WIDTH = 2             # 16-bit linear PCM


def ulaw_to_pcm16(ulaw: bytes) -> bytes:
    """Decode G.711 mu-law bytes to 16-bit linear PCM."""
    return audioop.ulaw2lin(ulaw, _WIDTH)


def pcm16_to_ulaw(pcm: bytes) -> bytes:
    """Encode 16-bit linear PCM to G.711 mu-law bytes."""
    return audioop.lin2ulaw(pcm, _WIDTH)


class Resampler:
    """Stateful mono PCM resampler for one direction of one call."""

    def __init__(self, from_rate: int, to_rate: int) -> None:
        self._from = from_rate
        self._to = to_rate
        self._state = None

    def process(self, pcm: bytes) -> bytes:
        if self._from == self._to:
            return pcm
        converted, self._state = audioop.ratecv(
            pcm, _WIDTH, 1, self._from, self._to, self._state
        )
        return converted


class TelephonyDecoder:
    """Inbound: one call's telephony frames -> pipeline PCM (int16 numpy).

    encoding "mulaw" (G.711, the default) or "pcm16" (linear, e.g. some
    mod_audio_stream setups). Output is int16 at ``target_rate``.
    """

    def __init__(
        self, target_rate: int = 16000, encoding: str = "mulaw",
        telephony_rate: int = TELEPHONY_RATE,
    ) -> None:
        self._encoding = encoding
        self._resampler = Resampler(telephony_rate, target_rate)

    def decode(self, payload: bytes) -> np.ndarray:
        pcm = ulaw_to_pcm16(payload) if self._encoding == "mulaw" else payload
        pcm = self._resampler.process(pcm)
        return np.frombuffer(pcm, dtype="<i2")


class TelephonyEncoder:
    """Outbound: pipeline PCM (int16 numpy at ``source_rate``) -> telephony
    frames the media server plays into the call.

    VoiceOS TTS is 24 kHz, so ``source_rate`` defaults to 24000; the frames
    come back as 8 kHz mu-law (or linear PCM when encoding="pcm16").
    """

    def __init__(
        self, source_rate: int = 24000, encoding: str = "mulaw",
        telephony_rate: int = TELEPHONY_RATE,
    ) -> None:
        self._encoding = encoding
        self._resampler = Resampler(source_rate, telephony_rate)

    def encode(self, audio: np.ndarray) -> bytes:
        pcm = np.ascontiguousarray(audio, dtype="<i2").tobytes()
        pcm = self._resampler.process(pcm)
        return pcm16_to_ulaw(pcm) if self._encoding == "mulaw" else pcm
