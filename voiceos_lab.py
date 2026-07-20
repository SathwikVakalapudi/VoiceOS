"""VoiceOS Lab — a Streamlit bench for the speech-to-transcript path.

    streamlit run voiceos_lab.py

Records or loads audio, then replays it through the *real* `SpeechDetector`
frame by frame so you can see exactly what the production segmenter does:
VAD probability, the hysteresis latch, the silence timer, where utterances
commit, and what Whisper makes of them.

Nothing here reimplements the pipeline. The detector, the VAD, the recorder
and the settings models are all imported from `voiceos/`, so what you tune
here is what runs on a call. The one deviation is noted in `replay()`.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from voiceos.audio.audio_queue import AudioQueue, Utterance, make_frame
from voiceos.config.settings import STTSettings, VADSettings
from voiceos.interfaces.vad import BaseVAD
from voiceos.pipeline.events import EventBus
from voiceos.pipeline.state import PipelineState, StateMachine
from voiceos.utils.audio import int16_to_float32
from voiceos.vad.detector import SpeechDetector
from voiceos.vad.silero_vad import SileroVAD

SAMPLE_RATE = 16000
FRAME_SIZE = 512
FRAME_MS = FRAME_SIZE / SAMPLE_RATE * 1000  # 32.0

st.set_page_config(page_title="VoiceOS Lab", page_icon="🎙", layout="wide")


# ─────────────────────────── audio decoding ───────────────────────────


def _to_wav(audio: np.ndarray, rate: int) -> bytes:
    """Wrap raw int16 PCM in a WAV container so `st.audio` can play it."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def decode_audio(data: bytes) -> np.ndarray:
    """Any container the browser or disk hands us -> mono int16 @ 16 kHz.

    PyAV rather than soundfile because `st.audio_input` yields webm/opus on
    some browsers and plain wav on others.
    """
    import av

    container = av.open(io.BytesIO(data))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):  # flush — easy to forget, loses the tail
        chunks.append(out.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks).astype(np.int16)


# ─────────────────────────── replay harness ───────────────────────────


class _ProbeVAD(BaseVAD):
    """Wraps the real Silero VAD and records every probability it returns."""

    def __init__(self, inner: BaseVAD) -> None:
        self._inner = inner
        self.probs: list[float] = []

    async def load(self) -> None:
        await self._inner.load()

    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        prob = self._inner.process(frame, sample_rate)
        self.probs.append(prob)
        return prob

    def reset(self) -> None:
        self._inner.reset()


@dataclass
class FrameTrace:
    t_s: float
    prob: float
    rms: float
    recording: bool
    speech_ms: float
    silence_ms: float
    required_ms: float
    predicted: bool


@dataclass
class ReplayResult:
    frames: list[FrameTrace] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    commits: list[float] = field(default_factory=list)   # t_s of each commit
    discarded: int = 0
    wall_s: float = 0.0

    @property
    def vad_ms_per_frame(self) -> float:
        return self.wall_s * 1000 / max(1, len(self.frames))


@st.cache_resource(show_spinner="Loading Silero VAD…")
def _load_silero(use_onnx: bool) -> SileroVAD:
    vad = SileroVAD(VADSettings(use_onnx=use_onnx))
    asyncio.run(vad.load())
    return vad


@st.cache_resource(show_spinner="Loading Smart Turn v3…")
def _load_smart_turn(path: str):
    from voiceos.dashboard.smart_turn import SmartTurn

    turn = SmartTurn(path)
    turn.load()
    return turn


def replay(pcm: np.ndarray, settings: VADSettings, smart_turn_path: str | None) -> ReplayResult:
    """Push `pcm` through the real SpeechDetector, one 512-sample frame at a time.

    Deviation from production: after each commit the detector moves to THINKING
    (which normally gates the mic while the assistant answers). Here we snap it
    straight back to IDLE, i.e. an assistant that replies instantly, so the whole
    recording gets segmented instead of the tail landing in `_pending`.
    """
    # Models load out here, in sync context. The cached loaders each spin their
    # own event loop, and asyncio.run() cannot be called from inside one — so
    # loading them within _replay_async would nest and raise.
    inner = _load_silero(settings.use_onnx)
    smart_turn = _load_smart_turn(smart_turn_path) if smart_turn_path else None
    return asyncio.run(_replay_async(pcm, settings, inner, smart_turn))


async def _replay_async(
    pcm: np.ndarray, settings: VADSettings, inner: BaseVAD, smart_turn
) -> ReplayResult:
    inner.reset()
    vad = _ProbeVAD(inner)

    state = StateMachine()
    bus = EventBus()
    utt_queue: asyncio.Queue = asyncio.Queue()

    turn_predictor = None
    if smart_turn is not None:

        async def turn_predictor(audio: np.ndarray) -> float:  # noqa: F811
            return float(smart_turn.complete_prob(audio))

    detector = SpeechDetector(
        vad=vad,
        audio_queue=AudioQueue(),
        utterance_queue=utt_queue,
        state=state,
        event_bus=bus,
        settings=settings,
        frame_ms=FRAME_MS,
        turn_predictor=turn_predictor,
    )

    result = ReplayResult()
    n_frames = len(pcm) // FRAME_SIZE
    started = time.perf_counter()

    for i in range(n_frames):
        chunk = pcm[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        frame = make_frame(chunk, SAMPLE_RATE)
        before = utt_queue.qsize()

        await detector._process_frame(frame)

        if utt_queue.qsize() > before:
            result.utterances.append(utt_queue.get_nowait())
            result.commits.append((i + 1) * FRAME_MS / 1000)
        if state.state is PipelineState.THINKING:
            state.transition(PipelineState.IDLE)   # instant-assistant, see docstring

        sig = int16_to_float32(chunk)
        result.frames.append(
            FrameTrace(
                t_s=i * FRAME_MS / 1000,
                prob=vad.probs[-1] if vad.probs else 0.0,
                rms=float(np.sqrt(np.mean(np.square(sig)))),
                recording=detector._recorder.recording,
                speech_ms=detector._speech_ms,
                silence_ms=detector._silence_ms,
                required_ms=detector._required_silence_ms(),
                predicted=detector._endpoint_predicted,
            )
        )

    result.wall_s = time.perf_counter() - started

    # Whatever is still latched at end-of-file would have committed on the next
    # silence; count it so the numbers add up.
    if detector._recorder.recording:
        await detector._finish_utterance()
        if not utt_queue.empty():
            result.utterances.append(utt_queue.get_nowait())
            result.commits.append(n_frames * FRAME_MS / 1000)
        else:
            result.discarded += 1

    return result


# ─────────────────────────── transcription ───────────────────────────


@st.cache_resource(show_spinner="Loading faster-whisper…")
def _load_whisper(model: str, device: str, compute_type: str, language: str | None):
    from voiceos.stt.whisper import FasterWhisperSTT

    stt = FasterWhisperSTT(
        STTSettings(model=model, device=device, compute_type=compute_type, language=language)
    )
    asyncio.run(stt.load())
    return stt


def transcribe_all(utterances: list[Utterance], stt) -> list[tuple[str, str, float]]:
    async def run():
        out = []
        for utt in utterances:
            t0 = time.perf_counter()
            res = await stt.transcribe(int16_to_float32(utt.audio), utt.sample_rate)
            out.append((res.text, res.language, time.perf_counter() - t0))
        return out

    return asyncio.run(run())


# ─────────────────────────── plotting ───────────────────────────


def plot_trace(result: ReplayResult, settings: VADSettings) -> go.Figure:
    t = [f.t_s for f in result.frames]
    fig = go.Figure()

    # Shade every latched (recording) stretch.
    start = None
    for f in result.frames:
        if f.recording and start is None:
            start = f.t_s
        elif not f.recording and start is not None:
            fig.add_vrect(x0=start, x1=f.t_s, fillcolor="#4C9AFF", opacity=0.16, line_width=0)
            start = None
    if start is not None:
        fig.add_vrect(x0=start, x1=t[-1], fillcolor="#4C9AFF", opacity=0.16, line_width=0)

    fig.add_trace(go.Scatter(x=t, y=[f.prob for f in result.frames],
                             name="VAD p(speech)", line=dict(color="#4C9AFF", width=2)))

    peak = max((f.rms for f in result.frames), default=0.0) or 1.0
    fig.add_trace(go.Scatter(x=t, y=[f.rms / peak for f in result.frames],
                             name="RMS (normalised)", line=dict(color="#8A94A6", width=1),
                             opacity=0.55))

    fig.add_hline(y=settings.threshold, line_dash="dash", line_color="#36B37E",
                  annotation_text=f"on {settings.threshold}", annotation_position="right")
    fig.add_hline(y=settings.neg_threshold, line_dash="dot", line_color="#FFAB00",
                  annotation_text=f"off {settings.neg_threshold}", annotation_position="right")

    for i, c in enumerate(result.commits):
        fig.add_vline(x=c, line_color="#FF5630", line_width=2,
                      annotation_text=f"#{i + 1}", annotation_position="top")

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(range=[0, 1.02], title="probability"),
        xaxis=dict(title="seconds"),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
    )
    return fig


def plot_timers(result: ReplayResult) -> go.Figure:
    t = [f.t_s for f in result.frames]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=[f.silence_ms for f in result.frames],
                             name="trailing silence", line=dict(color="#FFAB00", width=2)))
    fig.add_trace(go.Scatter(x=t, y=[f.required_ms for f in result.frames],
                             name="required to commit", line=dict(color="#FF5630", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=[f.speech_ms for f in result.frames],
                             name="speech so far", line=dict(color="#36B37E", width=1), opacity=0.5))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="ms"), xaxis=dict(title="seconds"),
        legend=dict(orientation="h", y=1.18, x=0), hovermode="x unified",
    )
    return fig


# ─────────────────────────── audio health ───────────────────────────


def audio_health(pcm: np.ndarray, result: ReplayResult) -> dict[str, float]:
    """The §8.3 metrics the pipeline does not currently emit."""
    if pcm.size == 0:
        return {}
    clipped = int(np.sum(np.abs(pcm.astype(np.int32)) >= 32700))
    speech = [f.rms for f in result.frames if f.recording]
    quiet = [f.rms for f in result.frames if not f.recording]
    speech_rms = float(np.mean(speech)) if speech else 0.0
    noise_rms = float(np.percentile(quiet, 20)) if quiet else 0.0
    snr = 20 * np.log10(speech_rms / noise_rms) if speech_rms > 0 and noise_rms > 0 else float("nan")
    return {
        "duration_s": pcm.size / SAMPLE_RATE,
        "clipping_pct": 100 * clipped / pcm.size,
        "peak_dbfs": 20 * np.log10(max(1, np.abs(pcm).max()) / 32768),
        "snr_db": snr,
        "noise_floor_rms": noise_rms,
        "speech_rms": speech_rms,
    }


# ─────────────────────────── sidebar ───────────────────────────

st.sidebar.title("🎙 VoiceOS Lab")
st.sidebar.caption("Tuning the real `SpeechDetector`, not a mock.")

preset = st.sidebar.selectbox(
    "Preset",
    ["Defaults", "Quiet laptop mic", "Noisy room", "Snappy (Smart Turn)", "No hysteresis (broken)"],
)

_PRESETS = {
    "Defaults":              dict(threshold=0.5, neg=None, min_speech=150, min_sil=700, adaptive_sil=False, adaptive_noise=False, smart=False),
    "Quiet laptop mic":      dict(threshold=0.35, neg=0.22, min_speech=120, min_sil=700, adaptive_sil=True, adaptive_noise=False, smart=False),
    "Noisy room":            dict(threshold=0.6, neg=0.45, min_speech=200, min_sil=800, adaptive_sil=False, adaptive_noise=True, smart=False),
    "Snappy (Smart Turn)":   dict(threshold=0.5, neg=0.35, min_speech=150, min_sil=700, adaptive_sil=True, adaptive_noise=False, smart=True),
    "No hysteresis (broken)": dict(threshold=0.5, neg=0.5, min_speech=250, min_sil=700, adaptive_sil=False, adaptive_noise=False, smart=False),
}
p = _PRESETS[preset]

st.sidebar.subheader("Hysteresis")
threshold = st.sidebar.slider("on threshold", 0.05, 0.95, p["threshold"], 0.05,
                              help="Probability that STARTS speech.")
neg_default = p["neg"] if p["neg"] is not None else round(max(0.0, threshold - 0.15), 3)
neg_threshold = st.sidebar.slider("off threshold (stay-bar)", 0.0, 0.95, neg_default, 0.05,
                                  help="Once latched, stay in speech above this. Lower = fewer splits.")

st.sidebar.subheader("Segmentation")
min_speech_ms = st.sidebar.slider("min_speech_ms", 50, 600, p["min_speech"], 10,
                                  help="Shorter bursts are discarded as noise.")
min_silence_ms = st.sidebar.slider("min_silence_ms", 200, 1500, p["min_sil"], 50,
                                   help="Trailing silence that ends a turn.")
pre_roll_ms = st.sidebar.slider("pre_roll_ms", 0, 600, 300, 50,
                                help="Retroactive capture. >500 ms feeds Whisper leading noise.")

st.sidebar.subheader("Adaptive silence")
adaptive_silence = st.sidebar.checkbox("adaptive_silence", p["adaptive_sil"],
                                       help="Short utterance -> wait LONGER (probably mid-thought).")
min_silence_short_ms = st.sidebar.slider("min_silence_short_ms", 400, 2000, 1100, 50,
                                         disabled=not adaptive_silence)
short_utterance_ms = st.sidebar.slider("short_utterance_ms", 400, 3000, 1200, 100,
                                       disabled=not adaptive_silence)

st.sidebar.subheader("Noise gate")
adaptive_noise = st.sidebar.checkbox("adaptive_noise", p["adaptive_noise"])
noise_margin = st.sidebar.slider("noise_margin", 1.0, 5.0, 2.0, 0.1, disabled=not adaptive_noise)

st.sidebar.subheader("Smart Turn v3")
smart_turn = st.sidebar.checkbox("smart_turn", p["smart"],
                                 help="Semantic end-of-turn on the raw waveform.")
smart_turn_model = st.sidebar.text_input("model path", "models/smart-turn-v3.2-cpu.onnx",
                                         disabled=not smart_turn)
smart_turn_threshold = st.sidebar.slider("smart_turn_threshold", 0.1, 0.9, 0.5, 0.05,
                                         disabled=not smart_turn)
predicted_silence_ms = st.sidebar.slider("predicted_silence_ms", 100, 600, 250, 25,
                                         disabled=not smart_turn,
                                         help="Silence required once the turn looks complete.")

vad_settings = VADSettings(
    threshold=threshold,
    neg_threshold=neg_threshold,
    min_speech_ms=min_speech_ms,
    min_silence_ms=min_silence_ms,
    pre_roll_ms=pre_roll_ms,
    adaptive_silence=adaptive_silence,
    min_silence_short_ms=min_silence_short_ms,
    short_utterance_ms=short_utterance_ms,
    adaptive_noise=adaptive_noise,
    noise_margin=noise_margin,
    smart_turn=smart_turn,
    smart_turn_threshold=smart_turn_threshold,
    predicted_silence_ms=predicted_silence_ms,
)
turn_path = smart_turn_model if smart_turn else None


# ─────────────────────────── input ───────────────────────────

st.title("Utterance → transcript, under a microscope")

src_col, up_col = st.columns([1, 1])
with src_col:
    recorded = st.audio_input("Record from your mic")
with up_col:
    uploaded = st.file_uploader("…or load a file", type=["wav", "mp3", "m4a", "ogg", "webm", "flac"])

blob = recorded or uploaded
# Set VOICEOS_LAB_FIXTURE to a path to pin one clip across reruns — handy when
# sweeping settings (and it lets the replay path be tested headlessly).
_fixture = os.environ.get("VOICEOS_LAB_FIXTURE")
if blob is None and _fixture and Path(_fixture).is_file():
    blob = io.BytesIO(Path(_fixture).read_bytes())
    st.caption(f"Using fixture `{_fixture}` (VOICEOS_LAB_FIXTURE)")

if blob is None:
    st.info(
        "Record a few seconds and hit the tabs below. Try saying something with a "
        "**mid-sentence pause** — *“I'd like to book a flight to… uh… Chennai”* — and watch "
        "whether it commits one utterance or three."
    )
    st.stop()

pcm = decode_audio(blob.getvalue())
if pcm.size < FRAME_SIZE:
    st.error("That clip is too short to frame.")
    st.stop()

result = replay(pcm, vad_settings, turn_path)

tab_seg, tab_ab, tab_health, tab_audit = st.tabs(
    ["Segmentation", "A/B compare", "Audio health", "Doc cross-check"]
)


# ─────────────────────────── tab: segmentation ───────────────────────────

with tab_seg:
    m = st.columns(5)
    m[0].metric("Utterances", len(result.utterances))
    m[1].metric("Duration", f"{pcm.size / SAMPLE_RATE:.1f} s")
    m[2].metric("Frames", len(result.frames))
    m[3].metric("VAD / frame", f"{result.vad_ms_per_frame:.2f} ms")
    m[4].metric("Realtime factor", f"{result.wall_s / (pcm.size / SAMPLE_RATE):.3f}×")

    st.plotly_chart(plot_trace(result, vad_settings), width="stretch")
    st.caption(
        "Blue band = latched (recording). Red line = commit. Between the green and amber "
        "lines is the hysteresis dead-zone: a dip in there does **not** break the latch."
    )

    st.plotly_chart(plot_timers(result), width="stretch")
    st.caption(
        "The amber line climbing to meet the dashed red one is the endpoint decision. "
        "Red dropping mid-pause means a semantic prediction fired and shortened the wait."
    )

    if not result.utterances:
        st.warning(
            "Nothing committed. Either the clip is silent, or `on threshold` is above what "
            "your mic produces — drop it to 0.35 and look at where the blue curve peaks."
        )
    else:
        st.subheader("Utterances")
        for i, utt in enumerate(result.utterances, 1):
            with st.expander(f"#{i} — {utt.duration_s:.2f} s", expanded=i == 1):
                st.audio(_to_wav(utt.audio, utt.sample_rate), format="audio/wav")

        st.divider()
        st.subheader("Transcribe")
        c1, c2, c3 = st.columns([1, 1, 1])
        w_model = c1.selectbox("model", ["tiny", "base", "small", "medium"], index=2)
        w_lang = c2.selectbox("language", ["auto", "en", "hi", "te", "ta", "kn", "ml", "mr", "bn"], index=1)
        w_device = c3.selectbox("device", ["auto", "cpu", "cuda"], index=0)

        if st.button("Run faster-whisper", type="primary"):
            stt = _load_whisper(w_model, w_device, "default", None if w_lang == "auto" else w_lang)
            with st.spinner("Transcribing…"):
                rows = transcribe_all(result.utterances, stt)
            for i, ((text, lang, dt), utt) in enumerate(zip(rows, result.utterances), 1):
                st.markdown(f"**#{i}** · `{lang}` · {dt:.2f} s · {utt.duration_s:.2f} s audio")
                st.write(text or "_(empty — a false trigger; raise min_speech_ms)_")


# ─────────────────────────── tab: A/B ───────────────────────────

with tab_ab:
    st.markdown(
        "Same audio, two configs. This is how you check whether a claim in the research "
        "doc actually holds on **your** mic and **your** voice."
    )
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**A — current sidebar config**")
        st.json({"on": threshold, "off": neg_threshold, "min_silence_ms": min_silence_ms,
                 "adaptive_silence": adaptive_silence, "smart_turn": smart_turn}, expanded=False)
    with a2:
        st.markdown("**B — comparison**")
        b_thresh = st.slider("B: on threshold", 0.05, 0.95, threshold, 0.05, key="bt")
        b_neg = st.slider("B: off threshold", 0.0, 0.95, threshold, 0.05, key="bn",
                          help="Set equal to `on` to reproduce the pre-hysteresis behaviour.")
        b_minsil = st.slider("B: min_silence_ms", 200, 1500, min_silence_ms, 50, key="bs")

    if st.button("Compare", type="primary"):
        b_settings = vad_settings.model_copy(
            update={"threshold": b_thresh, "neg_threshold": b_neg, "min_silence_ms": b_minsil}
        )
        b_result = replay(pcm, b_settings, turn_path)

        c1, c2 = st.columns(2)
        c1.metric("A utterances", len(result.utterances))
        c2.metric("B utterances", len(b_result.utterances),
                  delta=len(b_result.utterances) - len(result.utterances))
        st.plotly_chart(plot_trace(result, vad_settings), width="stretch")
        st.plotly_chart(plot_trace(b_result, b_settings), width="stretch")
        if len(b_result.utterances) > len(result.utterances):
            st.warning(
                f"B split the same speech into {len(b_result.utterances)} turns vs A's "
                f"{len(result.utterances)} — that is endpoint flicker (F-15). Each extra "
                "turn is a wasted STT call and an interruption."
            )


# ─────────────────────────── tab: health ───────────────────────────

with tab_health:
    st.markdown(
        "The research doc §8.3 lists these as the metrics that predict user-visible failure. "
        "**The pipeline does not currently emit any of them** — computed here so you can at "
        "least see them for your own hardware."
    )
    h = audio_health(pcm, result)
    if h:
        c = st.columns(4)
        c[0].metric("Clipping", f"{h['clipping_pct']:.3f} %",
                    delta="over budget" if h["clipping_pct"] > 0.1 else "ok",
                    delta_color="inverse" if h["clipping_pct"] > 0.1 else "normal")
        c[1].metric("Peak", f"{h['peak_dbfs']:.1f} dBFS")
        c[2].metric("Est. SNR", "n/a" if np.isnan(h["snr_db"]) else f"{h['snr_db']:.1f} dB")
        c[3].metric("Noise floor", f"{h['noise_floor_rms']:.4f} RMS")

        if h["clipping_pct"] > 0.1:
            st.error("Sustained clipping (>0.1 %). Input gain is too high — this is not "
                     "recoverable downstream (F-03). Turn the mic gain down at the OS level.")
        if not np.isnan(h["snr_db"]) and h["snr_db"] < 10:
            st.warning("SNR under 10 dB. Silero degrades here (F-07); consider `adaptive_noise` "
                       "or getting closer to the mic.")
        if h["peak_dbfs"] < -30:
            st.warning("Very quiet capture. Lower the `on threshold` — this is the classic "
                       "'laptop mic doesn't detect speech' failure.")

        st.divider()
        st.subheader("Noise-gate window bug")
        st.markdown(
            "`_passes_noise_gate` appends frame RMS to its rolling window, but the call site is "
            "`prob >= threshold and self._passes_noise_gate(signal)` — Python short-circuits, so "
            "**silent frames never reach the function and never enter the window**. The 'noise "
            "floor' is therefore the 20th percentile of *speech*, not of background. Below is "
            "what the floor looks like both ways on this clip."
        )
        speech_rms = [f.rms for f in result.frames if f.prob >= threshold]
        all_rms = [f.rms for f in result.frames]
        if speech_rms:
            cc = st.columns(2)
            cc[0].metric("Floor as implemented (speech only)", f"{np.percentile(speech_rms, 20):.5f}")
            cc[1].metric("Floor as intended (all frames)", f"{np.percentile(all_rms, 20):.5f}")
            ratio = np.percentile(speech_rms, 20) / max(1e-9, np.percentile(all_rms, 20))
            st.caption(f"The gate is running **{ratio:.1f}× stricter** than intended on this clip.")


# ─────────────────────────── tab: audit ───────────────────────────

with tab_audit:
    st.markdown(
        "`docs/RESEARCH-speech-to-transcript.md` §9.3 claims a compliance status for each "
        "technique. Every row was re-checked against the code on disk. Four discrepancies were "
        "found and **all four are now fixed** — see the bottom of this tab. The confirmed-missing "
        "gaps below are still open."
    )

    st.subheader("Doc says ✅ — verified")
    st.dataframe(
        [
            {"Technique": "Hysteresis VAD (on 0.5 / off 0.35)", "Where": "detector.py:122-127", "Verdict": "✅ correct — validator fills threshold−0.15"},
            {"Technique": "Pre-roll, moved-not-copied", "Where": "recorder.py:27", "Verdict": "✅ correct — start() drains the ring"},
            {"Technique": "Hangover (append while waiting)", "Where": "detector.py:138", "Verdict": "✅ correct — push() before the speech test"},
            {"Technique": "Inverted adaptive silence", "Where": "detector.py:217-230", "Verdict": "✅ correct — short ⇒ 1100 ms"},
            {"Technique": "Prediction may only shorten", "Where": "detector.py:229", "Verdict": "✅ correct — min(predicted, base)"},
            {"Technique": "One inference per pause", "Where": "detector.py:265-284", "Verdict": "✅ correct — _turn_checked latch"},
            {"Technique": "VAD reset at every commit", "Where": "detector.py:369", "Verdict": "✅ correct"},
            {"Technique": "Stateful resampling", "Where": "transcode.py:55", "Verdict": "✅ correct — ratecv state threaded"},
            {"Technique": "Ordering as proof of playback", "Where": "events.py:80", "Verdict": "✅ correct — SentenceSpoken marker"},
            {"Technique": "Commit-before-drain on barge-in", "Where": "pipeline.py:281-297", "Verdict": "✅ correct — no await inside"},
            {"Technique": "Prewarm with a real inference", "Where": "dashboard/app.py:297", "Verdict": "✅ correct"},
            {"Technique": "One VAD instance per call", "Where": "app.py:349", "Verdict": "✅ correct"},
            {"Technique": "Never time turns from ASR", "Where": "detector.py:139-141", "Verdict": "✅ correct — frame counter"},
            {"Technique": "Drop-oldest backpressure", "Where": "audio_queue.py:45-53", "Verdict": "✅ correct — drops now surfaced in snapshot()"},
        ],
        width="stretch", hide_index=True,
    )

    st.subheader("Doc says ❌ — confirmed still missing")
    st.dataframe(
        [
            {"Gap": "Streaming ASR (pseudo only, quadratic)", "Impact": "High", "Where": "stt/streaming.py"},
            {"Gap": "Speculative execution", "Impact": "High — doc's #5", "Where": "primitive exists in state.py:27, unused"},
            {"Gap": "Backchannel classifier", "Impact": "High — Vapi table stakes", "Where": "absent"},
            {"Gap": "Echo cross-correlation gate", "Impact": "Medium — doc's #3", "Where": "absent; threshold gate only"},
            {"Gap": "Jitter buffer", "Impact": "Medium (telephony)", "Where": "absent on all paths"},
            {"Gap": "Reconnect DSP reset", "Impact": "Medium (telephony)", "Where": "absent"},
            {"Gap": "Batched VAD", "Impact": "Scaling ceiling ~25-30 calls", "Where": "detector.py:119 on the loop"},
            {"Gap": "UPWR / UPSR metrics", "Impact": "Low today (no partials shipped)", "Where": "absent"},
            {"Gap": "Speaker-embedding gate", "Impact": "Medium (TV problem)", "Where": "absent"},
            {"Gap": "AudioWorklet in browser", "Impact": "Medium — doc's #2", "Where": "live.html:121 uses createScriptProcessor"},
        ],
        width="stretch", hide_index=True,
    )

    st.subheader("Where the doc was wrong — now fixed")
    st.success(
        "**1. §5.4 invariant 3 — fixed.** `detector.py` gated the staleness reset on "
        "`if self._turn_predictor is not None`, so with `predictive_endpointing=True` and "
        "`smart_turn=False` a stale `_endpoint_predicted` latched for the rest of the utterance "
        "and committed after 250 ms at the next pause. The reset is now unconditional. "
        "Regression test: `test_resumed_speech_clears_a_stale_endpoint_prediction_without_smart_turn`."
    )
    st.success(
        "**2. Noise-floor window — fixed.** Short-circuit evaluation meant `_passes_noise_gate` "
        "(which maintains the rolling RMS window as a side effect) was never called on sub-threshold "
        "frames, so the 'noise floor' was the 20th percentile of *speech*. The gate is now evaluated "
        "before the `and`. This was absent from the doc entirely — §5.4 presents the floor as "
        "implemented. The *Audio health* tab still shows the two floors so you can see the size of "
        "the effect on your own clip. Regression test: `test_noise_floor_window_receives_silent_frames`."
    )
    st.success(
        "**3. F-23 mitigations 3 and 4 — fixed.** `whisper.py` passed only `language` and "
        "`beam_size`. It now passes `condition_on_previous_text=False` (the loop-repetition guard), "
        "`no_speech_threshold` and `compression_ratio_threshold`, all tunable via `STTSettings`. "
        "Regression tests in `tests/test_whisper_options.py`."
    )
    st.success(
        "**4. §8.3 'count, not just log' — fixed.** `AudioQueue.dropped` was incremented and never "
        "read by anything. `MetricsCollector` now takes the queue and reports `frames_dropped` in "
        "`snapshot()`; the pipeline wires it in. Regression tests in `tests/test_metrics_collector.py`."
    )
    st.info(
        "**5. F-25 is right and it bites on default settings.** `STTSettings.language` defaults to "
        "`\"en\"` (`settings.py:107`), which force-decodes English. For the Indian-language survey "
        "use case the local Whisper path silently produces garbage unless overridden. The doc flags "
        "this; worth restating because the default is still `\"en\"`."
    )
