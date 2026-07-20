"""VoiceOS STT Lab — speech in, text out. Nothing else.

    streamlit run voiceos_stt.py

No LLM, no TTS. The point is to find out why a transcript does not match what
was said, and the only way to do that is to remove variables:

  Push-to-talk   browser records a whole clip -> STT.        No VAD, no WebRTC
                 streaming, no resampling of ours.
  Live (VAD)     the real pipeline path: WebRTC -> resample ->
                 Silero endpointing -> STT.

If push-to-talk is accurate and live is not, the fault is in the capture or
segmentation path. If both are wrong, it is the microphone or the recogniser —
and running two recognisers over the same bytes tells you which.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from voiceos.config.settings import STTSettings, VADSettings, get_settings
from voiceos.utils.audio import int16_to_float32

SAMPLE_RATE = 16000
CAPTURE_DIR = Path("debug_audio")

st.set_page_config(page_title="VoiceOS STT Lab", page_icon="✎", layout="wide")

# Sarvam's supported set (BCP-47). Autodetect is deliberately last and labelled:
# on short utterances it mis-fires between related Indic scripts, which is how a
# one-word "yes" came back as Bengali.
LANGUAGES = {
    "English (India) — en-IN": ("en-IN", "en"),
    "Telugu — te-IN": ("te-IN", "te"),
    "Hindi — hi-IN": ("hi-IN", "hi"),
    "Tamil — ta-IN": ("ta-IN", "ta"),
    "Kannada — kn-IN": ("kn-IN", "kn"),
    "Malayalam — ml-IN": ("ml-IN", "ml"),
    "Marathi — mr-IN": ("mr-IN", "mr"),
    "Bengali — bn-IN": ("bn-IN", "bn"),
    "Gujarati — gu-IN": ("gu-IN", "gu"),
    "Punjabi — pa-IN": ("pa-IN", "pa"),
    "Odia — od-IN": ("od-IN", "or"),
    "Autodetect (not recommended)": (None, None),
}


# ─────────────────────────── audio helpers ───────────────────────────


def to_wav(audio: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def decode(data: bytes) -> np.ndarray:
    """Whatever the browser recorded -> mono int16 @ 16 kHz."""
    import av

    container = av.open(io.BytesIO(data))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    out: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for r in resampler.resample(frame):
            out.append(r.to_ndarray().reshape(-1))
    for r in resampler.resample(None):
        out.append(r.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(out).astype(np.int16) if out else np.zeros(0, np.int16)


def health(pcm: np.ndarray) -> dict:
    """Can an acoustic model actually read this?

    Level alone is not enough. What matters is energy in 1-3 kHz, where vowel
    formants and consonant cues live: clean close speech puts ~25% there, while
    a far-field or heavily-denoised signal collapses below 10% and every
    recogniser starts hallucinating.
    """
    if pcm.size < 512:
        return {}
    spec = np.abs(np.fft.rfft(pcm.astype(np.float64) * np.hanning(len(pcm))))
    freqs = np.fft.rfftfreq(len(pcm), 1 / SAMPLE_RATE)
    total = spec.sum() or 1.0
    return {
        "dur": pcm.size / SAMPLE_RATE,
        "peak_db": 20 * np.log10(max(1, np.abs(pcm).max()) / 32768),
        "clip_pct": 100 * np.sum(np.abs(pcm.astype(np.int32)) >= 32700) / pcm.size,
        "low_pct": spec[freqs < 300].sum() / total * 100,
        "voice_pct": spec[(freqs >= 1000) & (freqs < 3000)].sum() / total * 100,
    }


def verdict(h: dict) -> tuple[str, str]:
    """Level is the reliable signal; spectral balance is only advisory.

    Calibrated against real results rather than assumption. An earlier version
    called anything under 15% in the 1-3 kHz band "muffled", a threshold taken
    from TTS audio which is synthetically bright. Real speech puts far less
    energy there, and clips flagged muffled at 10-14% transcribed perfectly
    ("I want to book a flight", "To Chennai", "Narendra Modi"). Level was the
    honest predictor: every failure sat below -26 dBFS.
    """
    if not h:
        return "—", "grey"
    if h["clip_pct"] > 0.1:
        return "clipping", "red"
    if h["peak_db"] <= -26:
        return "too quiet", "red"
    if h["peak_db"] <= -20:
        return "quiet", "amber"
    return "good", "green"


# ─────────────────────────── recognisers ───────────────────────────


@st.cache_resource
def _loop() -> asyncio.AbstractEventLoop:
    """One long-lived event loop for every STT call.

    `asyncio.run()` closes its loop when it returns. An httpx.AsyncClient binds
    its connection pool to the loop it was created on, so a client built during
    a cached `load_*()` is attached to an already-dead loop by the time the
    second transcription runs — hence "Event loop is closed" on every call after
    the first. Keeping one loop alive on a background thread makes the cached
    clients valid for the life of the session.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="stt-loop").start()
    return loop


def _run(coro, timeout: float = 120.0):
    return asyncio.run_coroutine_threadsafe(coro, _loop()).result(timeout)


@st.cache_resource(show_spinner="Loading local Whisper…")
def load_whisper(model: str, language: str | None):
    from voiceos.stt.whisper import FasterWhisperSTT

    stt = FasterWhisperSTT(STTSettings(model=model, device="auto", language=language))
    _run(stt.load())
    return stt


@st.cache_resource(show_spinner="Connecting Sarvam…")
def load_sarvam(language: str | None, model: str):
    from voiceos.stt.sarvam import SarvamSTT

    base = get_settings().stt
    stt = SarvamSTT(base.model_copy(update={"sarvam_language": language, "sarvam_model": model}))
    _run(stt.load())
    return stt


def transcribe(stt, pcm: np.ndarray) -> tuple[str, str | None, float]:
    t0 = time.perf_counter()
    result = _run(stt.transcribe(int16_to_float32(pcm), SAMPLE_RATE))
    return result.text.strip(), result.language, time.perf_counter() - t0


# ─────────────────────────── sidebar ───────────────────────────

settings = get_settings()

st.sidebar.title("✎ STT Lab")

lang_label = st.sidebar.selectbox(
    "Language", list(LANGUAGES), index=0,
    help="Force it. The language is known here, so autodetect is a liability — "
         "it mis-fires between related Indic scripts on short utterances.",
)
sarvam_lang, whisper_lang = LANGUAGES[lang_label]

st.sidebar.subheader("Recognisers")
use_sarvam = st.sidebar.checkbox("Sarvam (hosted, costs credits)", True)
sarvam_model = st.sidebar.selectbox(
    "Sarvam model", ["saarika:v2.5", "saarika:v2"], index=0,
    help="saarika = transcription. Never pick a saaras model here — those "
         "translate to English instead of transcribing. saarika:flash was "
         "removed: the API reports it deprecated in favour of v2.5.",
)
use_whisper = st.sidebar.checkbox("Local Whisper (free)", True)
whisper_model = st.sidebar.selectbox(
    "Whisper model", ["base", "small", "medium", "large-v3"], index=1,
    help="`base` is weak on Indian languages. `small`/`medium` are far better "
         "for Telugu — worth the extra second while diagnosing.",
)

st.sidebar.caption(
    "Running both over identical bytes is the whole point: if they agree, the "
    "audio is fine and the transcript is real. If they both produce nonsense, "
    "the microphone is the problem."
)

st.sidebar.divider()
st.sidebar.subheader("Endpointing (Live mode only)")
vad_threshold = st.sidebar.slider("VAD on threshold", 0.05, 0.95, settings.vad.threshold, 0.05)
vad_neg = st.sidebar.slider("VAD off threshold", 0.0, 0.95, settings.vad.neg_threshold, 0.05)
min_silence = st.sidebar.slider("min_silence_ms", 200, 2000, settings.vad.min_silence_ms, 50)
min_speech = st.sidebar.slider("min_speech_ms", 50, 800, settings.vad.min_speech_ms, 10)

use_smart_turn = st.sidebar.checkbox(
    "Smart Turn v3 (semantic end-of-turn)", True,
    help="Instead of always waiting min_silence_ms, ask a model on each short "
         "pause whether the sentence sounds finished. Complete -> commit now; "
         "incomplete -> keep listening. Fixes mid-clause truncation without "
         "paying the full silence timer on every turn.",
)
smart_turn_path = st.sidebar.text_input(
    "Smart Turn model", settings.vad.smart_turn_model, disabled=not use_smart_turn)
smart_pause = st.sidebar.slider(
    "pause_ms (when to ask)", 100, 800, settings.vad.smart_turn_pause_ms, 50,
    disabled=not use_smart_turn)
smart_threshold = st.sidebar.slider(
    "turn_threshold", 0.1, 0.98, 0.95, 0.01,
    disabled=not use_smart_turn,
    help="Higher = more certain before committing = fewer cut-offs, more waiting.")
smart_max_silence = st.sidebar.slider(
    "max_silence_ms (hard stop)", 600, 3000, 1000, 100, disabled=not use_smart_turn,
    help="Guarantees the turn ends even if the model never says 'complete' — "
         "for someone who simply trails off.")

vad_settings = settings.vad.model_copy(update={
    "threshold": vad_threshold, "neg_threshold": vad_neg,
    "min_silence_ms": min_silence, "min_speech_ms": min_speech,
})


# ─────────────────────────── results rendering ───────────────────────────


@dataclass
class Result:
    pcm: np.ndarray
    health: dict
    rows: list[tuple[str, str, str | None, float]] = field(default_factory=list)
    source: str = ""
    saved: str | None = None


if "results" not in st.session_state:
    st.session_state.results = []


def run_recognisers(pcm: np.ndarray, source: str) -> Result:
    h = health(pcm)
    res = Result(pcm=pcm, health=h, source=source)
    try:
        CAPTURE_DIR.mkdir(exist_ok=True)
        path = CAPTURE_DIR / f"stt-{datetime.now():%H%M%S}.wav"
        path.write_bytes(to_wav(pcm))
        res.saved = str(path)
    except OSError:
        pass

    if use_sarvam:
        try:
            text, lang, dt = transcribe(load_sarvam(sarvam_lang, sarvam_model), pcm)
            res.rows.append(("Sarvam", text, lang, dt))
        except Exception as exc:
            res.rows.append(("Sarvam", f"‼ {type(exc).__name__}: {exc}", None, 0.0))
    if use_whisper:
        try:
            text, lang, dt = transcribe(load_whisper(whisper_model, whisper_lang), pcm)
            res.rows.append((f"Whisper {whisper_model}", text, lang, dt))
        except Exception as exc:
            res.rows.append((f"Whisper {whisper_model}", f"‼ {type(exc).__name__}: {exc}", None, 0.0))
    return res


def render(res: Result, index: int) -> None:
    v, colour = verdict(res.health)
    dot = {"green": "🟢", "amber": "🟡", "red": "🔴", "grey": "⚪"}[colour]
    h = res.health
    with st.container(border=True):
        head, meta = st.columns([2, 3])
        head.markdown(f"**#{index}** · {res.source} · {dot} **{v}**")
        if h:
            meta.markdown(
                f"<span style='font-size:0.78rem;color:#8A94A6'>"
                f"{h['dur']:.2f}s · peak <b>{h['peak_db']:.0f} dBFS</b>"
                f"{' · clipping!' if h['clip_pct'] > 0.1 else ''}</span>",
                unsafe_allow_html=True,
            )
        st.audio(to_wav(res.pcm), format="audio/wav")

        for name, text, lang, dt in res.rows:
            st.markdown(
                f"<div style='padding:.35em .6em;margin:.2em 0;border-radius:6px;"
                f"background:rgba(128,128,128,.09)'>"
                f"<span style='font-size:.72rem;color:#8A94A6'>{name}"
                f"{' · ' + lang if lang else ''} · {dt:.2f}s</span><br>"
                f"<span style='font-size:1.05rem'>{text or '<i>(empty)</i>'}</span></div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────── page ───────────────────────────

st.title("Speech → text, in isolation")
st.caption(
    f"Language forced to **{lang_label}** · "
    f"{'Sarvam + ' if use_sarvam else ''}{'local Whisper' if use_whisper else ''}"
)

tab_native, tab_ptt, tab_live = st.tabs(
    ["🎚 Native mic  (start here)", "🎤 Push to talk (browser)", "🔴 Live (VAD)"]
)

with tab_native:
    st.markdown(
        "Records **directly from the sound card** with `sounddevice` — the same "
        "path `main.py` uses. No browser, no WebRTC, no getUserMedia, no AGC or "
        "noise suppression. If this is clean and the browser tabs are not, the "
        "fault is the browser's device choice or its DSP. If this is *also* "
        "quiet and muffled, it is the microphone or its Windows input level."
    )
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        labels = [f"{i}: {d['name']} ({d['hostapi']})" for i, d in inputs]
        default_in = sd.default.device[0]
        pick = st.selectbox(
            "Input device", range(len(inputs)),
            format_func=lambda k: labels[k],
            index=next((k for k, (i, _) in enumerate(inputs) if i == default_in), 0),
            help="Pick the USB PnP entry. Windows/WASAPI variants of the same "
                 "physical mic can behave differently — WASAPI is usually best.",
        )
        seconds = st.slider("Record for", 2, 8, 4)
        if st.button("Record from this device", type="primary"):
            dev_index = inputs[pick][0]
            with st.spinner(f"Recording {seconds}s — speak now…"):
                rec = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                             channels=1, dtype="int16", device=dev_index)
                sd.wait()
            pcm = rec.reshape(-1)
            st.session_state.results.insert(
                0, run_recognisers(pcm, f"native · {inputs[pick][1]['name']}"))
            st.rerun()
    except Exception as exc:
        st.error(f"sounddevice unavailable: {type(exc).__name__}: {exc}")


with tab_ptt:
    st.markdown(
        "Record a whole clip and send it straight to STT. **No VAD, no streaming, "
        "no resampling of ours** — so whatever comes back is purely the microphone "
        "plus the recogniser. Get this accurate before trusting the live path."
    )
    clip = st.audio_input("Press record, say one clear sentence, press stop")
    if clip is not None and st.button("Transcribe", type="primary"):
        pcm = decode(clip.getvalue())
        if pcm.size < SAMPLE_RATE // 4:
            st.error("That clip is too short.")
        else:
            with st.spinner("Transcribing…"):
                st.session_state.results.insert(0, run_recognisers(pcm, "push-to-talk"))

with tab_live:
    st.markdown(
        "The real pipeline path: WebRTC → resample → Silero endpointing → STT. "
        "Each committed utterance is transcribed as it lands."
    )
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("`pip install streamlit-webrtc`")
        st.stop()

    simple = st.checkbox(
        "simplest constraints (audio: true)", True,
        help="OverconstrainedError means getUserMedia could not satisfy the "
             "requested constraints with any device — usually a deviceId Chrome "
             "still remembers for this site that no longer resolves. Plain "
             "`audio: true` asks for nothing specific and always resolves if a "
             "microphone exists.",
    )
    ns = st.checkbox("browser noiseSuppression", False, disabled=simple,
                     help="Off by default: denoisers are tuned for human ears and "
                          "their artifacts are out-of-distribution for ASR. They "
                          "also low-pass hard, which is what wrecked earlier captures.")
    constraints = (
        {"audio": True, "video": False} if simple
        else {"audio": {"echoCancellation": True, "noiseSuppression": ns,
                        "autoGainControl": True}, "video": False}
    )
    st.caption(
        "Still OverconstrainedError? Click the 🔒/🎙 icon in the address bar → "
        "**Reset permission**, then reload. That clears the remembered device. "
        "Or skip the browser entirely and use the **Native mic** tab."
    )
    ctx = webrtc_streamer(
        key="stt-live",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints=constraints,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    if ctx.state.playing:
        import queue as _queue

        import av

        from voiceos.dashboard.streaming_vad import (
            SmartTurnEndpointer,
            StreamingEndpointer,
        )
        from voiceos.vad.silero_vad import SileroVAD

        @st.cache_resource(show_spinner="Loading Silero VAD…")
        def _vad():
            v = SileroVAD(VADSettings())
            _run(v.load())
            return v

        @st.cache_resource(show_spinner="Loading Smart Turn v3…")
        def _smart_turn(path: str):
            from voiceos.dashboard.smart_turn import SmartTurn

            t = SmartTurn(path)
            t.load()
            return t

        vad = _vad()
        vad.reset()

        if use_smart_turn:
            turn = _smart_turn(smart_turn_path)

            async def _predict(audio: np.ndarray) -> float:
                return float(turn.complete_prob(audio))

            endpointer = SmartTurnEndpointer(
                vad, _predict,
                on_threshold=vad_threshold, off_threshold=vad_neg,
                min_speech_ms=min_speech,
                pause_ms=smart_pause, max_silence_ms=smart_max_silence,
                turn_threshold=smart_threshold,
            )
            # SmartTurnEndpointer.push is a coroutine (it awaits the model), so
            # it runs on the shared STT loop rather than blocking this thread.
            push = lambda pcm: _run(endpointer.push(pcm))
        else:
            endpointer = StreamingEndpointer(
                vad, on_threshold=vad_threshold, off_threshold=vad_neg,
                min_speech_ms=min_speech, min_silence_ms=min_silence,
            )
            push = endpointer.push
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        mic_box = st.empty()
        live_results = st.container()
        recent: list[np.ndarray] = []

        while ctx.state.playing:
            try:
                frames = ctx.audio_receiver.get_frames(timeout=1)
            except _queue.Empty:
                continue
            for frame in frames:
                for out in resampler.resample(frame):
                    pcm = out.to_ndarray().reshape(-1).astype(np.int16)
                    recent.append(pcm)
                    for utterance in push(pcm):
                        with live_results:
                            render(run_recognisers(utterance, "live VAD"),
                                   len(st.session_state.results) + 1)
            if recent:
                h = health(np.concatenate(recent[-25:]))
                del recent[:-25]
                v, colour = verdict(h)
                dot = {"green": "🟢", "amber": "🟡", "red": "🔴", "grey": "⚪"}[colour]
                if h:
                    mic_box.markdown(
                        f"### {dot} mic: **{v}** &nbsp;&nbsp;"
                        f"<span style='font-size:.6em;color:#8A94A6'>peak "
                        f"{h['peak_db']:.0f} dBFS — want above -20 while speaking"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

st.divider()
if st.session_state.results:
    head, clear = st.columns([4, 1])
    head.markdown("### Results")
    if clear.button("Clear"):
        st.session_state.results = []
        st.rerun()
    for i, res in enumerate(st.session_state.results, 1):
        render(res, len(st.session_state.results) - i + 1)
else:
    st.info("No recordings yet. Start with **Push to talk** above.")
