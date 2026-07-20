"""VoiceOS Chat — a conversational UI over the real pipeline.

    streamlit run voiceos_chat.py

Talk into the mic; replies appear as chat bubbles and play back automatically.
Every stage is timestamped to the millisecond in the timeline on the right —
mic frames, VAD latch, STT request/response, LLM first token, TTS first chunk,
and the total turn latency, all measured from the moment you stopped speaking.

Uses the real components (`create_stt`/`create_llm`/`create_tts`,
`StreamingEndpointer`, `ConversationManager`, `SentenceChunker`,
`ThinkTagFilter`), so it honours `.env` and behaves like a live call.
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from voiceos.config.settings import VADSettings, get_settings
from voiceos.conversation.manager import ConversationManager
from voiceos.dashboard.streaming_vad import (
    SmartTurnEndpointer,
    StreamingEndpointer,
)
from voiceos.llm.inference import ThinkTagFilter
from voiceos.pipeline.pipeline import create_llm, create_stt, create_tts
from voiceos.tts.streaming import SentenceChunker, clean_for_speech
from voiceos.utils.audio import int16_to_float32
from voiceos.utils.logging import setup_logging
from voiceos.vad.silero_vad import SileroVAD

SAMPLE_RATE = 16000
# Every captured utterance is written here. When a transcript looks wrong the
# only way to tell "the VAD clipped it" from "STT mis-heard it" is to listen to
# the exact bytes that were sent.
CAPTURE_DIR = Path("debug_audio")

st.set_page_config(page_title="VoiceOS Chat", page_icon="🎙", layout="wide")
setup_logging(get_settings().log_level)


# ─────────────────────────── timeline ───────────────────────────


@dataclass
class Entry:
    wall: str          # HH:MM:SS.mmm — real clock
    session_s: float   # seconds since the session began
    turn_ms: float | None   # ms since the user stopped speaking
    stage: str
    detail: str
    kind: str = "info"      # info | timing | error


_KIND_COLOR = {"info": "#8A94A6", "timing": "#36B37E", "error": "#FF5630"}
_STAGE_ICON = {
    "session": "○", "mic": "🎙", "vad": "◍", "stt": "✎",
    "llm": "✦", "tts": "♪", "turn": "▣", "error": "✕",
}


class Timeline:
    """Thread-safe, timestamped event log shared by the engine and the UI."""

    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._turn_t0: float | None = None

    def mark_turn_start(self, backdate_s: float = 0.0) -> None:
        """Anchor turn-relative timing at the moment the user's voice stopped.

        `backdate_s` exists because the endpointer only *tells* us the turn
        ended after it has waited out `min_silence_ms` of trailing silence — by
        which point the user has already been waiting that long. Measuring from
        the emit instant would silently hide the single largest component of
        perceived latency, so t=0 is backdated to mouth-close.
        """
        self._turn_t0 = time.monotonic() - backdate_s

    def end_turn(self) -> None:
        self._turn_t0 = None

    def add(self, stage: str, detail: str, kind: str = "info") -> None:
        now = time.monotonic()
        entry = Entry(
            wall=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            session_s=now - self._t0,
            turn_ms=(now - self._turn_t0) * 1000 if self._turn_t0 is not None else None,
            stage=stage,
            detail=detail,
            kind=kind,
        )
        with self._lock:
            self._entries.append(entry)

    def snapshot(self) -> list[Entry]:
        with self._lock:
            return list(self._entries)

    def as_text(self) -> str:
        lines = ["wall_clock    session_s  turn_ms   stage  detail"]
        for e in self.snapshot():
            turn = f"{e.turn_ms:7.0f}" if e.turn_ms is not None else "      -"
            lines.append(
                f"{e.wall}  {e.session_s:8.2f}  {turn}   {e.stage:<5}  {e.detail}"
            )
        return "\n".join(lines)


# ─────────────────────────── engine ───────────────────────────


@dataclass
class Turn:
    role: str
    text: str
    audio: bytes | None = None          # user: what the VAD captured
                                        # assistant: what TTS produced
    timings: dict = field(default_factory=dict)
    capture_path: str | None = None


class Engine(threading.Thread):
    """Runs the whole pipeline off the Streamlit thread.

    Streamlit's script thread must keep pulling WebRTC frames or audio backs up
    and the VAD sees a bursty, gap-ridden stream. All model work therefore lives
    here, on a dedicated thread with its own event loop.
    """

    def __init__(self, timeline: Timeline, vad_settings: VADSettings,
                 smart_turn: dict | None = None, streaming_stt: bool = False) -> None:
        super().__init__(daemon=True)
        self.timeline = timeline
        self._vad_settings = vad_settings
        self._smart_turn = smart_turn
        self._streaming_stt = streaming_stt
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_in: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.turns: list[Turn] = []
        self.status = "starting"
        self.ready = threading.Event()
        self.error: str | None = None

    # -- called from the Streamlit frame loop --------------------------------
    def feed(self, pcm: np.ndarray) -> None:
        self._audio_in.put(pcm)

    def stop(self) -> None:
        self._stop.set()

    def read(self) -> tuple[list[Turn], str]:
        with self._lock:
            return list(self.turns), self.status

    async def _open_stream(self, preroll: list[np.ndarray]) -> None:
        """Open a streaming session and prime it with the pre-latch audio."""
        from voiceos.stt.sarvam_streaming import SarvamStreamingSTT

        try:
            self._stream = SarvamStreamingSTT(get_settings().stt)
            await self._stream.start()
            for chunk in preroll:
                await self._stream.send(chunk)
            self.timeline.add("stt", "streaming socket open — transcribing as you speak")
        except Exception as exc:
            self.timeline.add("stt", f"streaming unavailable ({type(exc).__name__}); "
                                     f"using batch", "error")
            self._stream = None

    def extract_survey(self, survey, timeout: float = 60.0) -> dict:
        """Run the post-call extraction over the conversation so far.

        The transcript is deliberately extracted in one pass at the end rather
        than slot-filled turn by turn: people answer out of order, correct
        themselves, and answer two questions at once. A single LLM read of the
        whole conversation handles all of that; incremental slot-filling does
        not.
        """
        from voiceos.survey.extractor import SurveyExtractor

        if self._loop is None:
            raise RuntimeError("engine not started")
        extractor = SurveyExtractor(self.llm, survey)
        transcript = self.conversation.history.messages
        future = asyncio.run_coroutine_threadsafe(
            extractor.extract(transcript), self._loop)
        return future.result(timeout)

    async def _greet(self, text: str) -> None:
        """Synthesise and post the campaign's opening line."""
        self._set_status("synthesising")
        self.timeline.add("tts", f"opening line → “{text[:60]}…”")
        t = time.perf_counter()
        chunks: list[np.ndarray] = []
        try:
            for sentence in _split_sentences(text):
                async for chunk in self.tts.synthesize(sentence):
                    chunks.append(chunk)
        except Exception as exc:
            self.timeline.add("tts", f"opening line failed: {type(exc).__name__}", "error")
            return
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        self.timeline.add("tts", f"opening line ready ({time.perf_counter() - t:.2f}s, "
                                 f"{audio.size / max(1, self.tts.sample_rate):.1f}s audio)",
                          "timing")
        self._append(Turn("assistant", text, _to_wav(audio, self.tts.sample_rate)))

    async def _close_stream(self) -> None:
        if self._stream is not None:
            await self._stream.close()
            self._stream = None

    # -- engine thread -------------------------------------------------------
    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.timeline.add("error", self.error, "error")
            self.ready.set()
        finally:
            loop.close()

    async def _main(self) -> None:
        settings = get_settings()
        tl = self.timeline

        tl.add("session", "loading models…")
        t0 = time.perf_counter()
        self.stt, self.llm, self.tts = create_stt(settings), create_llm(settings), create_tts(settings)
        await asyncio.gather(self.stt.load(), self.llm.load(), self.tts.load())
        tl.add("session", f"STT/LLM/TTS ready in {time.perf_counter() - t0:.2f}s", "timing")

        vad = SileroVAD(self._vad_settings)
        await vad.load()
        if self._smart_turn:
            from voiceos.dashboard.smart_turn import SmartTurn

            turn = SmartTurn(self._smart_turn["model"])
            turn.load()

            async def _predict(audio: np.ndarray) -> float:
                return float(turn.complete_prob(audio))

            endpointer = SmartTurnEndpointer(
                vad, _predict,
                on_threshold=self._vad_settings.threshold,
                off_threshold=self._vad_settings.neg_threshold,
                min_speech_ms=self._vad_settings.min_speech_ms,
                pause_ms=self._smart_turn["pause_ms"],
                max_silence_ms=self._smart_turn["max_silence_ms"],
                turn_threshold=self._smart_turn["threshold"],
            )
            tl.add("session", f"Smart Turn v3 @{self._smart_turn['threshold']} "
                              f"(pause {self._smart_turn['pause_ms']} ms, "
                              f"hard stop {self._smart_turn['max_silence_ms']} ms)")
        else:
            endpointer = StreamingEndpointer(
                vad,
                on_threshold=self._vad_settings.threshold,
                off_threshold=self._vad_settings.neg_threshold,
                min_speech_ms=self._vad_settings.min_speech_ms,
                min_silence_ms=self._vad_settings.min_silence_ms,
            )
        self.conversation = ConversationManager(settings.conversation)
        tl.add("session", f"VAD ready · model={settings.llm.model} · stt={settings.stt.provider}"
                          f" · tts={settings.tts.provider}", "info")
        # A campaign call opens by speaking. ConversationManager has already
        # put this in history, so the model knows it greeted; it just has to be
        # heard as well.
        if self.conversation.first_message:
            await self._greet(self.conversation.first_message)

        tl.add("session", "listening")
        self._set_status("listening")
        self.ready.set()

        first_frame = True
        was_speaking = False
        preroll: list[np.ndarray] = []
        self._stream = None
        while not self._stop.is_set():
            try:
                pcm = self._audio_in.get(timeout=0.1)
            except queue.Empty:
                continue
            if first_frame:
                tl.add("mic", f"first audio frame ({pcm.size} samples @ {SAMPLE_RATE} Hz)")
                first_frame = False

            utterances = (await endpointer.push(pcm)
                          if self._smart_turn else endpointer.push(pcm))

            if endpointer.in_speech and not was_speaking:
                tl.add("vad", f"speech started (latch at p≥{self._vad_settings.threshold})")
                self._set_status("speaking")
                if self._streaming_stt:
                    await self._open_stream(preroll)
            was_speaking = endpointer.in_speech

            # Keep sending after our VAD drops out: Sarvam runs its own VAD and
            # needs the trailing silence to decide the utterance ended. Cutting
            # the feed at in_speech meant it never saw a boundary.
            if self._stream is not None:
                try:
                    await self._stream.send(pcm)
                except Exception as exc:
                    tl.add("stt", f"stream send failed ({type(exc).__name__}); "
                                  f"falling back to batch", "error")
                    await self._close_stream()

            # Onset lives before the VAD latches, so keep a little history to
            # prepend when the socket opens — same reason the recorder keeps a
            # pre-roll ring. Without it the first syllable never reaches STT.
            preroll.append(pcm)
            if sum(p.size for p in preroll) > SAMPLE_RATE // 2:
                preroll.pop(0)

            for audio in utterances:
                endpoint_wait = ((self._smart_turn['pause_ms'] if self._smart_turn
                                  else self._vad_settings.min_silence_ms) / 1000)
                tl.mark_turn_start(backdate_s=endpoint_wait)
                tl.add("vad", "◀ USER STOPPED SPEAKING — clock starts here", "timing")
                tl.add("vad", f"waited {endpoint_wait * 1000:.0f} ms to be sure "
                              f"the turn ended")
                tl.add("vad", f"endpoint committed — {audio.size / SAMPLE_RATE:.2f}s captured "
                              f"(incl. pre-roll + trailing pad)", "timing")
                try:
                    await self._turn(audio, endpoint_wait)
                except Exception as exc:
                    tl.add("error", f"{type(exc).__name__}: {exc}", "error")
                    self._set_status("listening")
                tl.end_turn()

        await asyncio.gather(self.stt.close(), self.llm.close(), self.tts.close(),
                             return_exceptions=True)
        tl.add("session", "stopped")

    async def _turn(self, audio: np.ndarray, endpoint_wait_s: float) -> None:
        tl = self.timeline
        # perf_counter origin backdated to mouth-close, so every measurement
        # below is "time since the user stopped talking" — what they feel.
        mouth_close = time.perf_counter() - endpoint_wait_s
        timings: dict[str, float] = {"endpoint_wait_s": endpoint_wait_s}

        # ---- STT ----
        self._set_status("transcribing")
        text, language = "", None
        t = time.perf_counter()
        if self._stream is not None:
            # Most of the work already happened while the user was talking;
            # only the tail remains. Falls through to batch if it comes back
            # empty, since we still hold the audio either way.
            try:
                text, language = await self._stream.finish(quiet_ms=150)
                timings["stt_s"] = time.perf_counter() - t
                tl.add("stt", f"streamed transcript ({timings['stt_s'] * 1000:.0f} ms "
                              f"after speech) lang={language} → “{text}”", "timing")
            except Exception as exc:
                tl.add("stt", f"stream failed ({type(exc).__name__}); retrying batch", "error")
            finally:
                await self._close_stream()

        if not text.strip():
            tl.add("stt", f"batch request → {type(self.stt).__name__} "
                          f"({audio.size / SAMPLE_RATE:.2f}s audio)")
            t = time.perf_counter()
            result = await self.stt.transcribe(int16_to_float32(audio), SAMPLE_RATE)
            timings["stt_s"] = time.perf_counter() - t
            text, language = result.text, result.language
            tl.add("stt", f"batch response ({timings['stt_s']:.2f}s) lang={language} "
                          f"→ “{text}”", "timing")

        result = type("R", (), {"text": text, "language": language})()

        if not result.text.strip():
            tl.add("stt", "empty transcript — discarding turn (false trigger)", "error")
            self._set_status("listening")
            return

        captured = _to_wav(audio, SAMPLE_RATE)
        path = None
        try:
            CAPTURE_DIR.mkdir(exist_ok=True)
            path = CAPTURE_DIR / f"utt-{datetime.now():%H%M%S}-{len(self.turns):02d}.wav"
            path.write_bytes(captured)
            tl.add("stt", f"captured audio saved → {path}")
        except OSError as exc:
            tl.add("stt", f"could not save capture: {exc}", "error")
        self._append(Turn("user", result.text, captured, capture_path=str(path) if path else None))

        # ---- LLM ----
        self._set_status("thinking")
        messages = self.conversation.build_messages(result.text)
        tl.add("llm", f"request → {get_settings().llm.model} ({len(messages)} messages)")
        chunker, think = SentenceChunker(), ThinkTagFilter()
        sentences: list[str] = []
        reply: list[str] = []
        t = time.perf_counter()
        first_token = None

        async for delta in self.llm.generate(messages):
            if first_token is None:
                first_token = time.perf_counter() - t
                timings["llm_ttft_s"] = first_token
                tl.add("llm", f"first token ({first_token * 1000:.0f} ms)", "timing")
            visible = think.feed(delta)
            reply.append(visible)
            for sentence in chunker.feed(visible):
                sentences.append(sentence)
                tl.add("llm", f"sentence {len(sentences)} → “{sentence}”")
        tail = think.flush()
        for sentence in chunker.feed(tail) + chunker.flush():
            sentences.append(sentence)
            tl.add("llm", f"sentence {len(sentences)} → “{sentence}”")

        timings["llm_total_s"] = time.perf_counter() - t
        text = "".join(reply).strip()
        tl.add("llm", f"done ({timings['llm_total_s']:.2f}s, {len(text)} chars, "
                      f"{len(sentences)} sentences)", "timing")

        # ---- TTS ----
        self._set_status("synthesising")
        chunks: list[np.ndarray] = []
        t = time.perf_counter()
        first_chunk = None
        for i, sentence in enumerate(sentences, 1):
            spoken = clean_for_speech(sentence)
            if not spoken:
                continue
            tl.add("tts", f"synthesise {i}/{len(sentences)} → “{spoken[:60]}”")
            async for chunk in self.tts.synthesize(spoken):
                if first_chunk is None:
                    first_chunk = time.perf_counter() - t
                    timings["tts_first_chunk_s"] = first_chunk
                    # THE number: mouth-close → first audio the user could hear.
                    response = time.perf_counter() - mouth_close
                    timings["response_s"] = response
                    tl.add("tts", f"first audio chunk ({first_chunk * 1000:.0f} ms from request)")
                    tl.add("turn", f"▶ AUDIO OUT — {response * 1000:.0f} ms after you stopped "
                                   f"speaking", "timing")
                chunks.append(chunk)

        audio_out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        timings["tts_total_s"] = time.perf_counter() - t
        timings["audio_s"] = audio_out.size / max(1, self.tts.sample_rate)
        timings["turn_total_s"] = time.perf_counter() - mouth_close
        tl.add("tts", f"all sentences done ({timings['tts_total_s']:.2f}s → "
                      f"{timings['audio_s']:.2f}s of audio)")

        r = timings.get("response_s", 0)
        tl.add("turn", "── breakdown of the {:.0f} ms ──".format(r * 1000), "timing")
        tl.add("turn", f"   endpoint wait  {endpoint_wait_s * 1000:6.0f} ms  "
                       f"({endpoint_wait_s / r * 100 if r else 0:.0f}%)  ← deliberate; the VAD "
                       f"waiting to be sure you finished")
        tl.add("turn", f"   STT            {timings['stt_s'] * 1000:6.0f} ms  "
                       f"({timings['stt_s'] / r * 100 if r else 0:.0f}%)")
        tl.add("turn", f"   LLM to 1st tok {timings.get('llm_ttft_s', 0) * 1000:6.0f} ms  "
                       f"({timings.get('llm_ttft_s', 0) / r * 100 if r else 0:.0f}%)")
        tl.add("turn", f"   TTS to 1st aud {timings.get('tts_first_chunk_s', 0) * 1000:6.0f} ms  "
                       f"({timings.get('tts_first_chunk_s', 0) / r * 100 if r else 0:.0f}%)")

        self.conversation.history.add_assistant(text)
        self._append(Turn("assistant", text, _to_wav(audio_out, self.tts.sample_rate), timings))
        self._set_status("listening")

    def _append(self, turn: Turn) -> None:
        with self._lock:
            self.turns.append(turn)

    def _set_status(self, status: str) -> None:
        with self._lock:
            self.status = status


def _split_sentences(text: str) -> list[str]:
    """Chunk a whole message for TTS so playback can start on the first part."""
    chunker = SentenceChunker()
    out = chunker.feed(text) + chunker.flush()
    return [clean_for_speech(s) for s in out if clean_for_speech(s)]


def _to_wav(audio: np.ndarray, rate: int) -> bytes:
    if audio.size == 0:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def mic_health(pcm: np.ndarray) -> dict:
    """Is this signal something an ASR model can actually read?

    Level alone is not enough — a boomy, distant capture can be loud and still
    unreadable. What matters is how much energy sits in 1-3 kHz, where vowel
    formants and consonant cues live. Clean close speech puts ~25% there;
    a far-field or heavily-denoised signal collapses to well under 10%.
    """
    if pcm.size < 512:
        return {}
    x = pcm.astype(np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    total = spec.sum() or 1.0
    return {
        "peak_db": 20 * np.log10(max(1, np.abs(pcm).max()) / 32768),
        "rms": float(np.sqrt(np.mean((pcm / 32768.0) ** 2))),
        "low_pct": spec[freqs < 300].sum() / total * 100,
        "voice_pct": spec[(freqs >= 1000) & (freqs < 3000)].sum() / total * 100,
    }


# ─────────────────────────── sidebar ───────────────────────────

settings = get_settings()

st.sidebar.title("🎙 VoiceOS Chat")
st.sidebar.markdown(
    f"**LLM** `{settings.llm.model}`  \n"
    f"**STT** `{settings.stt.provider}`  \n"
    f"**TTS** `{settings.tts.provider}` @ {settings.tts.sample_rate} Hz"
)
st.sidebar.divider()

st.sidebar.subheader("Endpointing")
threshold = st.sidebar.slider("VAD on threshold", 0.05, 0.95, settings.vad.threshold, 0.05)
neg = st.sidebar.slider("VAD off threshold", 0.0, 0.95, settings.vad.neg_threshold, 0.05)
min_silence = st.sidebar.slider("min_silence_ms", 200, 1500, settings.vad.min_silence_ms, 50,
                                help="Trailing silence that ends your turn. Lower = snappier "
                                     "but cuts you off mid-thought.")
min_speech = st.sidebar.slider("min_speech_ms", 50, 600, settings.vad.min_speech_ms, 10)

st.sidebar.subheader("Smart Turn v3")
use_smart_turn = st.sidebar.checkbox(
    "semantic end-of-turn", True,
    help="Asks a model on each short pause whether the sentence sounds "
         "finished, instead of always waiting min_silence_ms. Tuned in the STT "
         "lab: 0.95 stopped mid-sentence cut-offs; 0.85 made them worse.",
)
smart_threshold = st.sidebar.slider("turn_threshold", 0.10, 0.98, 0.95, 0.01,
                                    disabled=not use_smart_turn)
smart_pause = st.sidebar.slider("pause_ms", 100, 800, 300, 50,
                                disabled=not use_smart_turn)
smart_max_silence = st.sidebar.slider(
    "max_silence_ms (hard stop)", 600, 3000, 1000, 100, disabled=not use_smart_turn,
    help="Ends the turn even if the model never says 'complete'. At 1600 a "
         "13-second turn merged two separate sentences.")

st.sidebar.subheader("Streaming STT")
use_streaming_stt = st.sidebar.checkbox(
    "stream audio while speaking", False,
    help="Uploads audio over a WebSocket as you talk. Helps only when the "
         "utterance contains a pause — Sarvam's server VAD emits then, so the "
         "text is ready before you stop. On long continuous turns nothing is "
         "emitted until flush and it measured ~190 ms SLOWER than batch in "
         "live use. Off by default; worth trying if your turns are short.",
)

smart_cfg = {
    "model": settings.vad.smart_turn_model,
    "threshold": smart_threshold,
    "pause_ms": smart_pause,
    "max_silence_ms": smart_max_silence,
} if use_smart_turn else None

vad_settings = settings.vad.model_copy(update={
    "threshold": threshold, "neg_threshold": neg,
    "min_silence_ms": min_silence, "min_speech_ms": min_speech,
})

st.sidebar.divider()
st.sidebar.caption(
    "Each turn costs ~3 API calls (STT + LLM + TTS). Barge-in is off, so the "
    "mic is gated while the assistant speaks — otherwise it hears itself."
)


# ─────────────────────────── page ───────────────────────────

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:
    st.error("`streamlit-webrtc` is not installed. Run: `pip install streamlit-webrtc`")
    st.stop()

st.sidebar.subheader("Microphone")
st.sidebar.caption(
    "Browsers pick their own default mic — often the laptop's far-field array "
    "rather than a USB mic. Click the 🎙 icon in the address bar to choose."
)
# Noise suppression defaults OFF: denoisers are tuned for human ears, and their
# artifacts are out-of-distribution for an ASR encoder. It also aggressively
# low-passes, which is exactly the damage seen in the captured utterances.
noise_suppression = st.sidebar.checkbox("browser noiseSuppression", False)
auto_gain = st.sidebar.checkbox("browser autoGainControl", True)
echo_cancel = st.sidebar.checkbox("browser echoCancellation", True)

ctx = webrtc_streamer(
    key="voiceos-chat",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=2048,
    media_stream_constraints={
        "audio": {
            "echoCancellation": echo_cancel,
            "noiseSuppression": noise_suppression,
            "autoGainControl": auto_gain,
        },
        "video": False,
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

chat_col, log_col = st.columns([3, 2], gap="medium")
with chat_col:
    st.markdown("### Conversation")
    chat_area = st.container()
SURVEY = None
if settings.conversation.campaign_file:
    from voiceos.survey.definition import SurveyDefinition

    SURVEY = SurveyDefinition.from_campaign_file(settings.conversation.campaign_file)


def render_survey(engine) -> None:
    """Extract structured answers from the conversation, on demand."""
    st.markdown(f"### Survey — {SURVEY.name}")
    st.caption(f"{len(SURVEY.questions)} questions · extracted from the whole "
               f"transcript in one pass")

    if st.button("Extract answers", type="primary", key="extract"):
        with st.spinner("Reading the transcript…"):
            try:
                st.session_state.answers = engine.extract_survey(SURVEY)
            except Exception as exc:
                st.error(f"Extraction failed: {type(exc).__name__}: {exc}")

    answers = st.session_state.get("answers")
    if answers:
        st.dataframe(
            [{"question": q.prompt, "answer": answers.get(q.id) if answers.get(q.id)
              is not None else "—"} for q in SURVEY.questions],
            width="stretch", hide_index=True,
        )
        filled = sum(1 for v in answers.values() if v is not None)
        st.caption(f"{filled}/{len(SURVEY.field_ids)} answered")

        if st.button("Save to results", key="save"):
            from datetime import datetime as _dt

            from voiceos.survey.store import ResultStore

            name = Path(settings.conversation.campaign_file).stem
            store = ResultStore(f"results/{name}.jsonl")
            store.add({
                "call_id": f"ui-{_dt.now():%Y%m%d-%H%M%S}",
                "number": "ui-session",
                "timestamp": _dt.now().isoformat(timespec="seconds"),
                "status": "completed",
                "survey": SURVEY.name,
                "answers": answers,
            })
            st.success(f"Appended to results/{name}.jsonl "
                       f"({len(store.records())} record(s))")


with log_col:
    st.markdown("### Mic check")
    st.caption("fix this before trusting any transcript")
    mic_area = st.empty()
    st.markdown("### Response latency")
    st.caption("you stop speaking → first audio out")
    latency_area = st.empty()
    st.markdown("### Timeline")
    st.caption("wall clock · **+ms since you stopped speaking** · stage")
    log_area = st.empty()
    download_area = st.empty()


def render_mic(area, h: dict) -> None:
    if not h:
        area.caption("_waiting for audio…_")
        return
    # Level is the honest predictor. The 1-3 kHz share was measured against
    # real transcripts and had no relationship to accuracy: 9% produced a
    # near-perfect result, 25% and 32% produced empty ones.
    verdict = ("🔴 too quiet" if h["peak_db"] <= -26 else
               "🟡 quiet" if h["peak_db"] <= -20 else "🟢 good")
    c = area.columns(4)
    c[0].metric("Input", verdict)
    c[1].metric("Peak", f"{h['peak_db']:.0f} dBFS")
    c[2].metric("RMS", f"{h['rms']:.3f}")
    c[3].metric("", "")


def render_latency(turns: list[Turn]) -> None:
    samples = [t.timings["response_s"] for t in turns
               if t.role == "assistant" and t.timings.get("response_s")]
    if not samples:
        latency_area.caption("_no turns yet_")
        return
    last = samples[-1]
    cols = latency_area.columns(3)
    cols[0].metric("Last", f"{last * 1000:.0f} ms")
    cols[1].metric("Median", f"{np.percentile(samples, 50) * 1000:.0f} ms")
    cols[2].metric("Worst", f"{max(samples) * 1000:.0f} ms",
                   help="P99 is what users remember — one 3 s stall reads as "
                        "'broken' far more than a consistently mediocre median.")

if not ctx.state.playing:
    with chat_area:
        st.info("Press **START** above, allow the mic, then just talk.")
    st.stop()


def render_log(timeline: Timeline, limit: int = 60) -> None:
    entries = timeline.snapshot()[-limit:]
    rows = []
    for e in entries:
        turn = f"+{e.turn_ms:5.0f}ms" if e.turn_ms is not None else "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&mdash;&nbsp;&nbsp;"
        icon = _STAGE_ICON.get(e.stage, "·")
        color = _KIND_COLOR[e.kind]
        weight = "600" if e.kind == "timing" else "400"
        rows.append(
            f"<div style='font-family:ui-monospace,monospace;font-size:0.74rem;"
            f"padding:1px 0;color:{color};font-weight:{weight}'>"
            f"<span style='opacity:.55'>{e.wall}</span> "
            f"<span style='opacity:.8'>{turn}</span> {icon} {e.detail}</div>"
        )
    log_area.markdown(
        "<div style='max-height:62vh;overflow-y:auto;border-radius:8px;"
        "background:rgba(128,128,128,0.07);padding:0.6em 0.8em'>"
        + "".join(reversed(rows)) + "</div>",
        unsafe_allow_html=True,
    )


# The engine outlives Streamlit reruns; the frame loop below feeds it.
if "engine" not in st.session_state:
    st.session_state.timeline = Timeline()
    st.session_state.timeline.add("session", "session started")
    st.session_state.engine = Engine(st.session_state.timeline, vad_settings, smart_cfg, use_streaming_stt)
    st.session_state.engine.start()

engine: Engine = st.session_state.engine
timeline: Timeline = st.session_state.timeline

# Rendered before the frame loop below, which blocks: a widget created after it
# would never be reachable. Clicking the button reruns the script, ending the
# loop; the engine lives in session_state so the conversation survives.
if SURVEY is not None:
    with log_col:
        render_survey(engine)

import av  # noqa: E402

resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
level_buf: list[np.ndarray] = []
rendered = 0

with chat_area:
    status_line = st.empty()

try:
    while ctx.state.playing:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            frames = []

        for frame in frames:
            for out in resampler.resample(frame):
                pcm = out.to_ndarray().reshape(-1).astype(np.int16)
                engine.feed(pcm)
                level_buf.append(pcm)

        turns, status = engine.read()

        for turn in turns[rendered:]:
            with chat_area:
                with st.chat_message("user" if turn.role == "user" else "assistant"):
                    st.markdown(turn.text)
                    if turn.role == "user" and turn.audio:
                        # Autoplay OFF here — this is for inspection, and playing
                        # your own voice back into the room would feed the mic.
                        st.audio(turn.audio, format="audio/wav")
                        st.caption(
                            f"↑ exactly what was sent to STT"
                            + (f" · `{turn.capture_path}`" if turn.capture_path else "")
                        )
                    elif turn.audio:
                        st.audio(turn.audio, format="audio/wav", autoplay=True)
                    if turn.timings:
                        t = turn.timings
                        r = t.get("response_s", 0)
                        st.markdown(
                            f"<span style='font-size:0.78rem;color:#8A94A6'>"
                            f"<b style='color:#36B37E;font-size:1.05rem'>{r * 1000:.0f} ms</b>"
                            f" from you stopping to audio out &nbsp;·&nbsp; "
                            f"endpoint {t.get('endpoint_wait_s', 0) * 1000:.0f} + "
                            f"stt {t.get('stt_s', 0) * 1000:.0f} + "
                            f"llm {t.get('llm_ttft_s', 0) * 1000:.0f} + "
                            f"tts {t.get('tts_first_chunk_s', 0) * 1000:.0f} ms</span>",
                            unsafe_allow_html=True,
                        )
        rendered = len(turns)

        _LABEL = {
            "starting": "⏳ loading models…", "listening": "🟢 listening",
            "speaking": "🔴 hearing you", "transcribing": "✎ transcribing…",
            "thinking": "✦ thinking…", "synthesising": "♪ speaking…",
            "stopped": "⏹ stopped",
        }
        status_line.markdown(f"**{_LABEL.get(status, status)}**")
        if level_buf:
            recent = np.concatenate(level_buf[-25:])   # ~0.5 s window
            render_mic(mic_area, mic_health(recent))
            del level_buf[:-25]
        render_latency(turns)
        render_log(timeline)

        if engine.error:
            st.error(engine.error)
            break
finally:
    download_area.download_button(
        "Download full timeline", timeline.as_text(),
        file_name="voiceos-timeline.txt", mime="text/plain",
    )
