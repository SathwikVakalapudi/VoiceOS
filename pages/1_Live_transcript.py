"""Live streaming transcript — partials while you speak, finals on endpoint.

Mic → WebRTC → Silero (hysteresis, real `StreamingEndpointer`) → Whisper.

An honest note on what "streaming" means here. Whisper is an encoder-decoder
trained on fixed 30 s windows with full bidirectional attention; it cannot emit
a token before seeing its whole input, so it is *architecturally* offline. What
this page does is pseudo-streaming: re-decode the growing utterance buffer every
`partial_interval_ms` and show the newest result. That is the same approach the
pipeline's `RollingTranscriber` takes, and it costs CPU quadratic in utterance
length — an 8 s utterance at 600 ms intervals is ~13 decodes, the last over the
full 8 s. Real streaming needs an RNN-T. Use `tiny`/`base` here.

Because partials are re-decodes rather than an extending prefix, they *revise*:
"my open" becomes "my opinion". That flicker is measured live below.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import streamlit as st

from voiceos.config.settings import STTSettings, VADSettings
from voiceos.dashboard.streaming_vad import StreamingEndpointer
from voiceos.utils.audio import int16_to_float32
from voiceos.vad.silero_vad import SileroVAD

SAMPLE_RATE = 16000

st.set_page_config(page_title="Live transcript · VoiceOS", page_icon="🔴", layout="wide")


# ─────────────────────────── models ───────────────────────────


@st.cache_resource(show_spinner="Loading faster-whisper…")
def load_stt(model: str, device: str, language: str | None):
    from voiceos.stt.whisper import FasterWhisperSTT

    stt = FasterWhisperSTT(STTSettings(model=model, device=device, language=language))
    asyncio.run(stt.load())
    # Warm it: the first decode is 10-100x slower (see F-26 / prewarm).
    asyncio.run(stt.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE))
    return stt


@st.cache_resource(show_spinner="Loading Silero VAD…")
def load_vad_factory():
    """Silero carries recurrent state, so each session needs its OWN instance.

    The model weights are shared; `SileroVAD` is a thin stateful wrapper, so we
    cache one loaded instance and hand it out — a single live session at a time
    is the assumption here.
    """
    vad = SileroVAD(VADSettings())
    asyncio.run(vad.load())
    return vad


# ─────────────────────────── transcription worker ───────────────────────────


@dataclass
class LiveState:
    """Shared between the frame loop and the STT worker thread."""

    partial: str = ""
    finals: list[str] = field(default_factory=list)
    final_latencies: list[float] = field(default_factory=list)
    partial_latencies: list[float] = field(default_factory=list)
    # Flicker accounting (F-20). A partial that is not a prefix-extension of the
    # one before it is a revision — the beam reordered under us.
    partials_emitted: int = 0
    revisions: int = 0
    unstable_words: int = 0
    final_words: int = 0
    dropped_partials: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class Transcriber(threading.Thread):
    """Runs STT off the frame loop.

    Two queues, and finals always win: under load a missing partial degrades
    responsiveness, a missing final breaks the transcript (doc §8.2). Partials
    are explicitly droppable — only the newest snapshot is ever kept.
    """

    def __init__(self, stt, state: LiveState, partial_interval_ms: int) -> None:
        super().__init__(daemon=True)
        self._stt = stt
        self._state = state
        self._interval = partial_interval_ms / 1000
        self._finals: queue.Queue[np.ndarray] = queue.Queue()
        self._snapshot: np.ndarray | None = None
        self._snap_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_partial_at = 0.0
        self._prev_partial = ""

    def submit_final(self, audio: np.ndarray) -> None:
        self._finals.put(audio)

    def offer_snapshot(self, audio: np.ndarray) -> None:
        with self._snap_lock:
            if self._snapshot is not None:
                self._state.dropped_partials += 1  # superseded before we got to it
            self._snapshot = audio

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while not self._stop.is_set():
                if self._drain_final(loop):
                    continue
                if not self._maybe_partial(loop):
                    time.sleep(0.02)
        finally:
            loop.close()

    def _drain_final(self, loop) -> bool:
        try:
            audio = self._finals.get_nowait()
        except queue.Empty:
            return False
        t0 = time.perf_counter()
        text = self._transcribe(loop, audio)
        with self._state.lock:
            if text:
                self._state.finals.append(text)
                self._state.final_words += len(text.split())
            self._state.final_latencies.append(time.perf_counter() - t0)
            self._state.partial = ""
        self._prev_partial = ""
        return True

    def _maybe_partial(self, loop) -> bool:
        if time.monotonic() - self._last_partial_at < self._interval:
            return False
        with self._snap_lock:
            audio, self._snapshot = self._snapshot, None
        if audio is None or audio.size < SAMPLE_RATE // 4:
            return False
        self._last_partial_at = time.monotonic()
        t0 = time.perf_counter()
        text = self._transcribe(loop, audio)
        dt = time.perf_counter() - t0
        with self._state.lock:
            self._state.partial = text
            self._state.partial_latencies.append(dt)
            if text:
                self._state.partials_emitted += 1
                if self._prev_partial and not text.startswith(self._prev_partial):
                    self._state.revisions += 1
                    self._state.unstable_words += _diverged_words(self._prev_partial, text)
                self._prev_partial = text
        return True

    def _transcribe(self, loop, audio: np.ndarray) -> str:
        try:
            res = loop.run_until_complete(
                self._stt.transcribe(int16_to_float32(audio), SAMPLE_RATE)
            )
            return res.text.strip()
        except Exception as exc:  # a failed decode must not kill the session
            st.session_state.setdefault("_stt_errors", []).append(str(exc))
            return ""


def _diverged_words(prev: str, new: str) -> int:
    """How many words of `prev` did not survive into `new`."""
    a, b = prev.split(), new.split()
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return len(a) - common


# ─────────────────────────── sidebar ───────────────────────────

st.sidebar.title("🔴 Live transcript")

model = st.sidebar.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=0,
                             help="`small`+ will not keep up with partials on CPU.")
language = st.sidebar.selectbox("Language", ["en", "auto", "hi", "te", "ta", "kn", "ml", "mr", "bn"], index=0)
device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

st.sidebar.subheader("Endpointing")
on_threshold = st.sidebar.slider("on threshold", 0.05, 0.95, 0.5, 0.05)
off_threshold = st.sidebar.slider("off threshold", 0.0, 0.95, 0.35, 0.05)
min_silence_ms = st.sidebar.slider("min_silence_ms", 200, 1500, 600, 50)
min_speech_ms = st.sidebar.slider("min_speech_ms", 50, 600, 200, 10)

st.sidebar.subheader("Partials")
partial_interval_ms = st.sidebar.slider(
    "partial_interval_ms", 200, 2000, 600, 100,
    help="Google's finding: raising 50→200 ms cost 75 ms of delay and removed "
         "~70% of flicker. Longer interval = steadier text, later text.",
)

st.sidebar.caption(
    "Silero needs one instance per session and carries recurrent state — "
    "reload the page between runs if segmentation starts behaving oddly."
)


# ─────────────────────────── page ───────────────────────────

st.title("Live streaming transcript")
st.caption(
    "Partials are re-decodes of the growing buffer, not an extending prefix — so they "
    "revise as the beam reorders. The flicker that causes is measured below."
)

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:
    st.error("`streamlit-webrtc` is not installed. Run: `pip install streamlit-webrtc`")
    st.stop()

ctx = webrtc_streamer(
    key="voiceos-live",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=2048,
    media_stream_constraints={
        # Browser-side AEC/NS/AGC. Leaving these on is closer to how the
        # WebSocket/browser ingress path behaves in production.
        "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True},
        "video": False,
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if not ctx.state.playing:
    st.info("Press **START** above, allow the mic, and start talking.")
    st.stop()

stt = load_stt(model, device, None if language == "auto" else language)
vad = load_vad_factory()
vad.reset()

endpointer = StreamingEndpointer(
    vad,
    on_threshold=on_threshold,
    off_threshold=off_threshold,
    min_speech_ms=min_speech_ms,
    min_silence_ms=min_silence_ms,
)

state = LiveState()
worker = Transcriber(stt, state, partial_interval_ms)
worker.start()

status_box = st.empty()
live_box = st.empty()
final_box = st.container()
metrics_box = st.empty()

import av  # noqa: E402  (only needed once we are actually streaming)

resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
rendered = 0
frames_seen = 0

try:
    while ctx.state.playing:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            continue

        for frame in frames:
            for out in resampler.resample(frame):
                pcm = out.to_ndarray().reshape(-1).astype(np.int16)
                frames_seen += 1
                for utterance in endpointer.push(pcm):
                    worker.submit_final(utterance)
                if endpointer.in_speech:
                    worker.offer_snapshot(endpointer.snapshot())

        with state.lock:
            partial, finals = state.partial, list(state.finals)
            p_lat = list(state.partial_latencies)
            f_lat = list(state.final_latencies)
            emitted, revisions = state.partials_emitted, state.revisions
            unstable, fwords = state.unstable_words, state.final_words
            dropped = state.dropped_partials

        speaking = endpointer.in_speech
        status_box.markdown(
            f"### {'🔴 speaking' if speaking else '⚪ listening'}"
            f" &nbsp;&nbsp;<span style='color:#8A94A6;font-size:0.6em'>"
            f"{frames_seen} chunks</span>",
            unsafe_allow_html=True,
        )
        live_box.markdown(
            f"<div style='min-height:3.5em;padding:0.8em 1em;border-radius:8px;"
            f"background:rgba(76,154,255,0.10);font-size:1.3em;color:#8A94A6;"
            f"font-style:italic'>{partial or '…'}</div>",
            unsafe_allow_html=True,
        )

        for text in finals[rendered:]:
            final_box.markdown(f"**{text}**")
        rendered = len(finals)

        upwr = (unstable / fwords * 100) if fwords else 0.0
        c = metrics_box.columns(5)
        c[0].metric("Finals", len(finals))
        c[1].metric("Partial p50", f"{np.percentile(p_lat, 50):.2f} s" if p_lat else "—")
        c[2].metric("Final p50", f"{np.percentile(f_lat, 50):.2f} s" if f_lat else "—")
        c[3].metric("Revisions", f"{revisions}/{emitted}",
                    help="Partials that were not a prefix-extension of the previous one.")
        c[4].metric("UPWR", f"{upwr:.0f} %",
                    help="Unstable partial word ratio — words shown then changed, "
                         "over words in the final transcript. The metric the doc says "
                         "you are flying blind without.")

        if dropped:
            st.session_state["_dropped"] = dropped
finally:
    worker.stop()

st.success("Stopped.")
if st.session_state.get("_dropped"):
    st.caption(
        f"{st.session_state['_dropped']} partial snapshots were superseded before the "
        "decoder got to them — expected, and the correct thing to shed under load."
    )
