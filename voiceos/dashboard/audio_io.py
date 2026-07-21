"""Audio codec helpers for the dashboard voice routes.

Pure functions with no state: decode a browser audio blob to PCM, and wrap PCM
back into a base64 WAV for the browser to play.
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np


def decode_audio(data: bytes, rate: int = 16000) -> np.ndarray:
    """Decode a browser audio blob (webm/opus/etc.) to mono int16 PCM at `rate`."""
    import av  # optional dep, present in requirements

    container = av.open(io.BytesIO(data))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    out: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            out.append(rf.to_ndarray().reshape(-1))
    for rf in resampler.resample(None):  # flush
        out.append(rf.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(out).astype(np.int16) if out else np.zeros(0, dtype=np.int16)


def wav_b64(pcm: np.ndarray, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.ascontiguousarray(pcm, dtype="<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")
