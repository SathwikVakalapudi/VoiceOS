"""Campaign dashboard API + static UI (FastAPI).

`create_app(...)` is dependency-injected so tests drive it with a fake LLM and
temp dirs. Routes:

    GET    /                                  -> dashboard UI
    GET    /api/campaigns                      -> list campaigns
    GET    /api/campaigns/{name}               -> one campaign (persona + survey)
    PUT    /api/campaigns/{name}               -> create/update (validated)
    DELETE /api/campaigns/{name}               -> delete
    POST   /api/campaigns/{name}/test/start    -> begin a text test, get greeting
    POST   /api/test/message                   -> send a turn, get the reply
    POST   /api/campaigns/{name}/dryrun        -> consent-gate a contact list
    GET    /api/campaigns/{name}/results        -> extracted survey results (JSON)
    GET    /api/campaigns/{name}/results.csv    -> results as CSV
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Spoken when the respondent never answers. Deliberately not the
# campaign error_message, which is about a technical fault.
_NO_RESPONSE_FAREWELL = "समय देने के लिए धन्यवाद, नमस्ते जी।"


class _EndCall(Exception):
    """Raised to unwind out of the nested call loop when the assistant hangs up."""

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel

# UI BCP-47 code -> Cartesia language code (Sarvam STT autodetects, no map needed).
_TTS_LANG = {
    "hi-IN": "hi", "hi": "hi", "te-IN": "te", "te": "te", "en-IN": "en",
    "en-US": "en", "en": "en", "mr-IN": "hi", "ta-IN": "ta", "kn-IN": "kn",
    "bn-IN": "bn", "gu-IN": "gu",
}


async def _retry(make_coro: Callable, tries: int = 3, delay: float = 0.6):
    """Retry a coroutine factory. Backs off on 429/5xx but stays snappy for a
    live call — fail in a few seconds and let the caller re-capture rather than
    freezing the turn."""
    import httpx

    last: Exception | None = None
    for i in range(tries):
        try:
            return await make_coro()
        except httpx.HTTPStatusError as exc:
            last = exc
            code = exc.response.status_code
            if code == 429 or code >= 500:  # rate limit or transient server error
                ra = exc.response.headers.get("retry-after", "")
                wait = float(ra) if ra.replace(".", "", 1).isdigit() else delay * (2 ** i) + 0.5
                await asyncio.sleep(min(wait, 4))
            else:
                raise  # 4xx (bad request / auth) won't improve on retry
        except Exception as exc:  # noqa: BLE001 - varied network errors
            last = exc
            await asyncio.sleep(delay)
    raise last  # type: ignore[misc]


def _decode_audio(data: bytes, rate: int = 16000) -> np.ndarray:
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


def _wav_b64(pcm: np.ndarray, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.ascontiguousarray(pcm, dtype="<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")

from voiceos.dashboard.sandbox import TestSandbox
from voiceos.dashboard.store import CampaignError, CampaignStore
from voiceos.monitoring.calls import CallRecorder, CallStore
from voiceos.monitoring.pricing import estimate_call_cost, load_pricing
from voiceos.pipeline.events import EventBus
from voiceos.interfaces.llm import BaseLLM
from voiceos.llm.tools import END_CALL_SCHEMA, wants_end_call
from voiceos.survey.definition import SurveyDefinition
from voiceos.survey.store import ResultStore

_STATIC = Path(__file__).parent / "static"


class TestMessage(BaseModel):
    session_id: str
    message: str


class ContactIn(BaseModel):
    number: str
    consented: bool = False
    name: str = ""


class DryRunBody(BaseModel):
    contacts: list[ContactIn]
    require_consent: bool = True


class LiveTurn(BaseModel):
    system_prompt: str
    history: list[dict] = []           # [{role: user|assistant, content: str}]
    message: str
    first_message: str | None = None


class VoiceOpen(BaseModel):
    system_prompt: str
    first_message: str | None = None
    language: str = "hi-IN"
    stream: bool = False               # if true, return reply text only (audio streamed)


class VoiceTurn(BaseModel):
    audio_b64: str                     # browser mic recording (webm/opus)
    system_prompt: str
    history: list[dict] = []
    language: str = "hi-IN"
    stream: bool = False               # if true, return transcript+reply only


class TtsStream(BaseModel):
    text: str
    language: str = "hi-IN"


class VoiceText(BaseModel):
    message: str
    system_prompt: str
    history: list[dict] = []
    language: str = "hi-IN"


def create_app(
    *,
    campaigns_dir: str = "campaigns",
    results_dir: str = "results",
    llm_factory: Callable[[], BaseLLM] | None = None,
    settings: Any | None = None,
) -> FastAPI:
    # Settings are needed regardless of who supplies the LLM: the call loop,
    # the config route and the call store all read them. Previously they were
    # only loaded when no llm_factory was injected, so any caller passing one
    # left `settings` as None and blew up later.
    if settings is None:
        from voiceos.config.settings import get_settings

        settings = get_settings()
    if llm_factory is None:
        from voiceos.pipeline.pipeline import create_llm

        llm_factory = lambda: create_llm(settings)  # noqa: E731

    store = CampaignStore(campaigns_dir)
    call_store = CallStore(settings.monitoring.calls_file)
    pricing = load_pricing(settings.monitoring.pricing_file)
    sandbox = TestSandbox(store, llm_factory=llm_factory)
    results_root = Path(results_dir)

    app = FastAPI(title="VoiceOS Campaign Dashboard")
    app.state.store = store
    app.state.sandbox = sandbox

    def _results_store(name: str) -> ResultStore:
        return ResultStore(str(results_root / f"{name}.jsonl"))

    def _field_ids(name: str) -> list[str]:
        survey = SurveyDefinition.from_campaign_file(store.path_for(name))
        return survey.field_ids if survey else []

    _llm_holder: dict = {}

    async def _shared_llm():
        if "llm" not in _llm_holder:
            inst = llm_factory()
            await inst.load()
            _llm_holder["llm"] = inst
        return _llm_holder["llm"]

    _st_holder: dict = {}

    def _shared_smart_turn():
        """Load Smart Turn v3 once if the model + deps are present, else None
        (graceful fallback to plain silence endpointing)."""
        if "loaded" in _st_holder:
            return _st_holder.get("st")
        _st_holder["loaded"] = True
        model = Path("models/smart-turn-v3.2-cpu.onnx")
        if not model.exists():
            logger.info("Smart Turn model not found; using silence endpointing")
            return None
        try:
            from voiceos.dashboard.smart_turn import SmartTurn

            st = SmartTurn(str(model))
            st.load()
            _st_holder["st"] = st
            logger.info("Smart Turn v3 enabled (semantic end-of-turn)")
            return st
        except Exception as exc:  # noqa: BLE001
            logger.warning("Smart Turn unavailable (%s); using silence endpointing", exc)
            return None

    async def _shared_stt(language: str):
        if settings is None:
            raise HTTPException(503, "voice STT not configured (no settings)")
        key = f"stt:{language}"
        if key not in _llm_holder:
            from voiceos.pipeline.pipeline import create_stt

            s2 = settings.model_copy(deep=True)
            # Force the language instead of autodetect — Sarvam otherwise
            # mis-detects Hindi as Kannada/Bengali/Gujarati/etc.
            s2.stt.sarvam_language = language
            inst = create_stt(s2)
            await inst.load()
            _llm_holder[key] = inst
        return _llm_holder[key]

    async def _shared_tts(language: str):
        if settings is None:
            raise HTTPException(503, "voice TTS not configured (no settings)")
        code = _TTS_LANG.get(language, "en")
        key = f"tts:{code}"
        if key not in _llm_holder:
            from voiceos.pipeline.pipeline import create_tts

            s2 = settings.model_copy(deep=True)
            s2.tts.cartesia_language = code
            inst = create_tts(s2)
            await inst.load()
            _llm_holder[key] = inst
        return _llm_holder[key]

    async def _synth_chunks(tts, text: str) -> list:
        """Collect TTS chunks, retrying if it fails before emitting any audio
        (Cartesia occasionally returns a transient 4xx/5xx that clears)."""
        last: Exception | None = None
        for _ in range(3):
            try:
                out = []
                async for c in tts.synthesize(text):
                    out.append(c)
                return out
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(0.4)
        raise last  # type: ignore[misc]

    async def _synthesize(text: str, language: str) -> str:
        tts = await _shared_tts(language)
        clean = (text or "").replace("end_call_tool", "").strip()
        chunks = await _synth_chunks(tts, clean) if clean else []
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return _wav_b64(audio, tts.sample_rate)

    async def _chat(system_prompt: str, history: list, user_message: str,
                    allow_end_call: bool = False) -> str:
        text, _ = await _chat_ex(system_prompt, history, user_message, allow_end_call)
        return text

    async def _chat_ex(system_prompt: str, history: list, user_message: str,
                       allow_end_call: bool = False) -> tuple[str, str | None]:
        """One LLM turn: system + trimmed recent history + user. Trimming caps
        tokens-per-request so long prompts don't blow the provider rate limit."""
        import httpx

        turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        msgs = [{"role": "system", "content": system_prompt}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in turns[-16:]]
        msgs.append({"role": "user", "content": user_message})
        llm = await _shared_llm()
        tools = [END_CALL_SCHEMA] if allow_end_call else None
        try:
            r = await _retry(lambda: llm.complete(msgs, tools=tools))
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429:
                raise HTTPException(429, "LLM is rate-limited — wait a moment and try again.")
            if code >= 500:
                raise HTTPException(503, "LLM service is busy right now — please tap and "
                                    "speak again.")
            raise HTTPException(502, f"LLM error: {exc}")
        end, reason = wants_end_call(r) if allow_end_call else (False, "")
        text = (r.get("content", "") if r else "") or ""
        # The model is told to speak a farewell and call the tool in the same
        # turn. If it only called the tool, there is nothing to say — hanging up
        # in silence is worse than a short goodbye.
        if end and not text.strip():
            text = "धन्यवाद, नमस्ते जी।"
        return text, (reason if end else None)

    @app.on_event("startup")
    async def _prewarm() -> None:
        # Warm the LLM/STT/TTS *connections* with a real tiny call so the first
        # conversation turn isn't a cold start (the first live LLM call otherwise
        # spikes ~4-5s on TLS/connection setup).
        try:
            llm = await _shared_llm()
            await _retry(lambda: llm.complete([{"role": "user", "content": "hi"}]))
            if settings is not None:
                await _shared_tts("hi-IN")
                stt = await _shared_stt("hi-IN")
                import numpy as _np

                await stt.transcribe(_np.zeros(16000, dtype=_np.int16), 16000)
        except Exception:  # never block startup on a provider hiccup
            logger.info("prewarm incomplete (will warm on first request)")
        # Smart Turn's transformers import is slow (~30s) — load it in the
        # background so startup and early calls aren't blocked (they use Silero-
        # only endpointing until it's ready).
        asyncio.get_event_loop().run_in_executor(None, _shared_smart_turn)

    # ---- UI ---- (no-store so the browser never serves a stale tester page)
    _NOCACHE = {"Cache-Control": "no-store, max-age=0"}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html", headers=_NOCACHE)

    @app.get("/live", response_class=HTMLResponse)
    async def live_page() -> FileResponse:
        return FileResponse(_STATIC / "live.html", headers=_NOCACHE)

    # ---- Streaming call: continuous mic in → Silero VAD → STT → LLM → TTS out ----
    @app.websocket("/ws/call")
    async def call_ws(ws: WebSocket) -> None:
        await ws.accept()
        try:
            cfg = await ws.receive_json()
        except Exception:
            await ws.close()
            return
        lang = cfg.get("language", "hi-IN")
        prompt = cfg.get("system_prompt", "")
        history: list[dict] = []

        # This path predates the EventBus and runs its own loop, so the record
        # is filled directly rather than by subscribing. Same shape either way.
        recorder = CallRecorder(
            EventBus(), store=None, direction="web",
            campaign=cfg.get("campaign"), assistant=cfg.get("assistant"),
        )
        _stt_lat: list[float] = []
        _llm_lat: list[float] = []
        reason = "completed"

        import asyncio as _asyncio

        from voiceos.dashboard.streaming_vad import (
            SmartTurnEndpointer,
            StreamingEndpointer,
        )
        from voiceos.vad.silero_vad import SileroVAD

        vad = SileroVAD(settings.vad)      # per-connection state
        await vad.load()
        loop = _asyncio.get_event_loop()
        smart_turn = await loop.run_in_executor(None, _shared_smart_turn)
        if smart_turn is not None:
            async def _predict(audio):
                return await loop.run_in_executor(None, smart_turn.complete_prob, audio)

            endpointer = SmartTurnEndpointer(vad, _predict)

            async def next_utts(pcm):
                return await endpointer.push(pcm)
        else:
            endpointer = StreamingEndpointer(vad, min_silence_ms=550)

            async def next_utts(pcm):
                return endpointer.push(pcm)

        stt = await _shared_stt(lang)
        speaking = False
        say_task: asyncio.Task | None = None

        from voiceos.vad.echo import EchoGate

        echo_gate = EchoGate(sample_rate=16000,
                             window_ms=settings.vad.echo_window_ms,
                             threshold=settings.vad.echo_gate_threshold)
        barge_vad = SileroVAD(settings.vad)     # separate: Silero carries state
        await barge_vad.load()
        _barge_ms = 0.0

        # Live partials. Sarvam's streaming socket transcribes while the caller
        # is still talking, so text appears as they speak instead of only after
        # they stop. The committed utterance is still transcribed in batch if
        # the stream returns nothing, so a socket failure costs partials, not
        # the turn.
        from voiceos.stt.sarvam_streaming import SarvamStreamingSTT

        _live_stt = (settings.stt.provider == "sarvam"
                     and settings.stt.sarvam_streaming)
        _stream = None
        _shown = ""
        _was_speech = False

        def nonlocal_idle_reset(at: float | None = None) -> None:
            """Start the silence clock — by default now, or when playback ends."""
            nonlocal _idle_since
            _idle_since = at if at is not None else time.monotonic()

        async def say(text: str) -> None:
            """Stream a reply. Runs as a task so the mic keeps being read —
            blocking here made barge-in impossible, since no frames were
            received at all while the assistant was talking."""
            nonlocal speaking
            speaking = True
            speak_until = time.monotonic()
            try:
                tts = await _shared_tts(lang)
                await ws.send_json({"type": "audio_start", "rate": tts.sample_rate})
                clean = (text or "").replace("end_call_tool", "").strip()
                if not clean:
                    # An empty reply is a real failure, not something to hand to
                    # TTS: the provider rejects it and the caller hears silence
                    # with no indication anything went wrong.
                    logger.error("live call: LLM returned no text — nothing to speak")
                    await ws.send_json({"type": "error",
                                        "text": "LLM returned an empty reply"})
                    await ws.send_json({"type": "listening"})
                    return
                _sent = 0
                _t_first = None
                async for chunk in tts.synthesize(clean):
                    pcm = np.ascontiguousarray(chunk, dtype="<i2")
                    if _t_first is None:
                        _t_first = time.monotonic()
                    _sent += pcm.size
                    # Reference for the echo gate: this is exactly what the
                    # caller is about to hear, so anything correlating with it
                    # in the mic is our own voice, not an interruption.
                    echo_gate.push_reference(pcm, tts.sample_rate)
                    await ws.send_bytes(pcm.tobytes())
                await ws.send_json({"type": "audio_end"})
                logger.info("live call: sent %.2fs of audio (%d samples @ %d Hz)",
                            _sent / max(1, tts.sample_rate), _sent, tts.sample_rate)
                # Sending finishes far sooner than playback: eight seconds of
                # greeting streams out in about one and a half. Charging the
                # idle timer from here fired the silence nudge while the caller
                # was still listening to the greeting.
                audio_s = _sent / max(1, tts.sample_rate)
                elapsed = time.monotonic() - (_t_first or time.monotonic())
                speak_until = time.monotonic() + max(0.0, audio_s - elapsed)
            except asyncio.CancelledError:
                await ws.send_json({"type": "interrupt"})   # flush browser buffer
                raise
            except Exception:
                logger.exception("live call: TTS failed")
            finally:
                endpointer.reset()         # drop any echo captured during playback
                echo_gate.reset()
                speaking = False
                nonlocal_idle_reset(speak_until)

        async def speak(text: str) -> None:
            nonlocal say_task
            say_task = asyncio.create_task(say(text))
            try:
                await asyncio.shield(say_task)
            except asyncio.CancelledError:
                pass                        # barge-in cancelled it; keep listening
            finally:
                say_task = None
            await ws.send_json({"type": "listening"})
            await ws.send_json({"type": "listening"})

        # opening line
        try:
            reply = cfg.get("first_message") or await _chat(
                prompt, [], "[The call has just connected. Speak your opening greeting "
                "and consent line now.]")
            history.append({"role": "assistant", "content": reply})
            await ws.send_json({"type": "reply", "text": reply})
            await speak(reply)
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "text": str(exc)})
            except Exception:
                return  # client already gone

        # Without these the loop fails silently: audio can flow for a minute
        # with the endpointer never committing, and the log looks identical to
        # a browser that is sending nothing at all.
        _frames = _samples = _ignored = _committed = 0
        _peak = 0.0
        _logged_first = False
        # Audio keeps arriving whether or not anyone speaks, so the frame
        # stream doubles as the clock: no committed utterance for this long
        # means the respondent has gone quiet.
        _idle_since = time.monotonic()
        _prompts = 0
        _no_input_s = settings.conversation.no_input_timeout_s
        _max_prompts = settings.conversation.no_input_max_prompts

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None:
                    continue
                if speaking:
                    _ignored += 1
                    # Rule J: stop immediately and listen, never talk over them.
                    # The echo gate is what makes this safe on a loudspeaker —
                    # without it the assistant hears itself and interrupts
                    # itself in a loop.
                    _sig = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
                    _p = barge_vad.process(_sig[:512], 16000)
                    _corr = echo_gate.correlation(_sig[:512])
                    if (_p >= settings.vad.barge_in_threshold
                            and _corr < settings.vad.echo_gate_threshold):
                        _barge_ms += len(data) / 2 / 16
                    else:
                        _barge_ms = 0.0
                    if _barge_ms >= settings.vad.barge_in_speech_ms and say_task:
                        logger.info("live call: BARGE-IN fired after %.0f ms "
                                    "(vad=%.2f, echo_corr=%.2f) — cancelling playback",
                                    _barge_ms, _p, _corr)
                        _barge_ms = 0.0
                        say_task.cancel()
                    continue
                _frames += 1
                _samples += len(data) // 2
                _pcm = np.frombuffer(data, dtype="<i2")
                # Level matters as much as arrival: 64 s of frames with the VAD
                # never latching means silence, and a frame count alone cannot
                # tell a muted device from a quiet one.
                _peak = max(_peak, float(np.abs(_pcm).max()) / 32768 if _pcm.size else 0.0)
                if not _logged_first:
                    _logged_first = True
                    logger.info("live call: first audio frame (%d samples)", _pcm.size)
                if _frames % 100 == 0:
                    _db = 20 * np.log10(_peak) if _peak > 0 else -99
                    logger.info("live call: %d frames / %.1fs in · peak %.0f dBFS%s · "
                                "in_speech=%s · %d committed",
                                _frames, _samples / 16000, _db,
                                "  <-- SILENT, check the mic device" if _db < -45 else "",
                                getattr(endpointer, "in_speech", "?"), _committed)
                    _peak = 0.0
                if (not speaking and _no_input_s > 0
                        and time.monotonic() - _idle_since > _no_input_s):
                    _prompts += 1
                    _idle_since = time.monotonic()
                    if _prompts > _max_prompts:
                        logger.info("live call: silent after %d prompts — hanging up",
                                    _max_prompts)
                        await speak(_NO_RESPONSE_FAREWELL)
                        await ws.send_json({"type": "ended", "reason": "no-response"})
                        reason = "no-response"
                        raise _EndCall
                    logger.info("live call: no input for %.0fs — nudge %d/%d",
                                _no_input_s, _prompts, _max_prompts)
                    # The campaign prompt already defines the escalation
                    # ("क्या आप लाइन पर हैं?" -> re-ask -> farewell), so tell the
                    # model what happened rather than hardcoding a line here.
                    try:
                        nudge, end_reason = await _chat_ex(
                            prompt, history,
                            f"[SYSTEM: The respondent has been silent for "
                            f"{_no_input_s:.0f} seconds. This is silence prompt "
                            f"{_prompts} of {_max_prompts}. Follow your no-response "
                            f"procedure. Speak only the line, nothing else.]",
                            allow_end_call=True)
                    except Exception:
                        logger.exception("live call: silence nudge failed")
                        nudge, end_reason = "", None
                    if nudge.strip():
                        history.append({"role": "assistant", "content": nudge})
                        recorder.record.transcript.append(
                            {"role": "assistant", "text": nudge, "nudge": True})
                        await ws.send_json({"type": "reply", "text": nudge})
                        await speak(nudge)
                        _idle_since = time.monotonic()
                    if end_reason:
                        await ws.send_json({"type": "ended", "reason": end_reason})
                        reason = end_reason
                        raise _EndCall

                # Open the streaming socket the moment speech latches, and
                # keep feeding it so the transcript is nearly done by the time
                # the endpointer commits.
                _now_speech = getattr(endpointer, "in_speech", False)
                if _live_stt and _now_speech and _stream is None:
                    try:
                        _stream = SarvamStreamingSTT(settings.stt)
                        await _stream.start()
                        _shown = ""
                    except Exception:
                        logger.exception("live call: could not open streaming STT")
                        _stream = None
                if _stream is not None:
                    try:
                        await _stream.send(_pcm)
                        if _stream.partial and _stream.partial != _shown:
                            _shown = _stream.partial
                            await ws.send_json({"type": "partial", "text": _shown})
                    except Exception:
                        logger.exception("live call: streaming STT dropped")
                        _stream = None
                if _now_speech != _was_speech:
                    await ws.send_json({"type": "speech", "on": bool(_now_speech)})
                _was_speech = _now_speech

                for utt in await next_utts(_pcm):
                    _committed += 1
                    _idle_since = time.monotonic()   # they spoke; reset the clock
                    _prompts = 0
                    logger.info("live call: utterance committed (%.2fs)", utt.size / 16000)
                    await ws.send_json({"type": "thinking"})
                    _t0 = time.perf_counter()
                    transcript = ""
                    if _stream is not None:
                        try:
                            transcript, _ = await _stream.finish(quiet_ms=150)
                            transcript = (transcript or "").strip()
                        except Exception:
                            logger.exception("live call: streaming finish failed")
                        finally:
                            _stream = None
                            _shown = ""
                    try:
                        if not transcript:
                            res = await _retry(lambda: stt.transcribe(utt, 16000))
                            transcript = (res.text or "").strip()
                    except Exception as exc:
                        logger.exception("live call: STT failed")
                        await ws.send_json({"type": "error",
                                            "text": f"STT: {type(exc).__name__}"})
                        transcript = ""
                    _stt_lat.append(time.perf_counter() - _t0)
                    if not transcript:
                        await ws.send_json({"type": "listening"})
                        continue
                    await ws.send_json({"type": "transcript", "text": transcript})
                    recorder.record.transcript.append(
                        {"role": "user", "text": transcript, "language": lang})
                    _t1 = time.perf_counter()
                    try:
                        reply, end_reason = await _chat_ex(
                            prompt, history, transcript, allow_end_call=True)
                    except HTTPException as he:
                        logger.warning("live call: LLM rejected (%s)", he.detail)
                        await ws.send_json({"type": "error", "text": he.detail})
                        await ws.send_json({"type": "listening"})
                        continue
                    except Exception as exc:
                        # Previously any non-HTTP error escaped to the outer
                        # handler and silently ended the call; the browser just
                        # stopped responding with no reason shown anywhere.
                        logger.exception("live call: LLM failed")
                        await ws.send_json({"type": "error",
                                            "text": f"LLM: {type(exc).__name__}: {exc}"})
                        await ws.send_json({"type": "listening"})
                        continue
                    _llm_lat.append(time.perf_counter() - _t1)
                    recorder.record.transcript.append({"role": "assistant", "text": reply})
                    recorder.record.turns += 1
                    history.append({"role": "user", "content": transcript})
                    history.append({"role": "assistant", "content": reply})
                    await ws.send_json({"type": "reply", "text": reply})
                    await speak(reply)
                    if end_reason:
                        logger.info("live call: assistant ended the call (%s)", end_reason)
                        await ws.send_json({"type": "ended", "reason": end_reason})
                        reason = end_reason
                        raise _EndCall
            reason = "customer-ended-call"
        except _EndCall:
            pass                       # assistant hung up; record it and close
        except WebSocketDisconnect:
            reason = "customer-ended-call"
        except Exception:
            logger.exception("live call ended on an unhandled error")
            reason = "pipeline-error"
        finally:
            recorder._samples["stt_s"] = _stt_lat
            recorder._samples["llm_total_s"] = _llm_lat
            rec = recorder.finish(ended_reason=reason)
            rec.cost = estimate_call_cost(
                rec.duration_s, rec.turns,
                stt_provider=settings.stt.provider,
                tts_provider=settings.tts.provider,
                llm_model=settings.llm.model,
                pricing=pricing,
            )
            call_store.add(rec)

    # ---- Live voice-conversation tester (any prompt, any language) ----
    @app.post("/api/live/reply")
    async def live_reply(body: LiveTurn) -> dict:
        hist = list(body.history)
        if body.first_message:
            hist = [{"role": "assistant", "content": body.first_message}, *hist]
        return {"reply": await _chat(body.system_prompt, hist, body.message)}

    # ---- Server-side voice (Sarvam STT + Cartesia TTS) — no browser speech ----
    @app.post("/api/live/voice/open")
    async def voice_open(body: VoiceOpen) -> dict:
        if body.first_message:
            reply = body.first_message
        else:
            reply = await _chat(body.system_prompt, [],
                                "[The call has just connected. Speak your opening greeting "
                                "and consent line now.]")
        if body.stream:
            return {"reply": reply, "audio_b64": ""}
        return {"reply": reply, "audio_b64": await _synthesize(reply, body.language)}

    @app.post("/api/live/voice/tts-stream")
    async def tts_stream(body: TtsStream) -> StreamingResponse:
        tts = await _shared_tts(body.language)
        clean = (body.text or "").replace("end_call_tool", "").strip()

        async def gen():
            if not clean:
                return
            # Retry only while nothing has been streamed yet (a mid-stream drop
            # can't be restarted without repeating audio).
            for attempt in range(3):
                emitted = False
                try:
                    async for chunk in tts.synthesize(clean):
                        emitted = True
                        yield np.ascontiguousarray(chunk, dtype="<i2").tobytes()
                    return
                except Exception:  # noqa: BLE001
                    if emitted or attempt == 2:
                        return
                    await asyncio.sleep(0.4)

        return StreamingResponse(
            gen(),
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(tts.sample_rate)},
        )

    @app.post("/api/live/voice/turn")
    async def voice_turn(body: VoiceTurn) -> dict:
        t = {}
        t0 = time.perf_counter()
        try:
            pcm = _decode_audio(base64.b64decode(body.audio_b64))
        except Exception as exc:
            raise HTTPException(400, f"could not decode audio: {exc}")
        t["decode"] = time.perf_counter() - t0

        transcript = ""
        try:
            mark = time.perf_counter()
            stt = await _shared_stt(body.language)
            result = await _retry(lambda: stt.transcribe(pcm, 16000))
            t["stt"] = time.perf_counter() - mark
            transcript = (result.text or "").strip()
            if not transcript:
                return {"transcript": "", "reply": "", "audio_b64": "",
                        "timings": {k: round(v, 2) for k, v in t.items()}}

            mark = time.perf_counter()
            reply = await _chat(body.system_prompt, body.history, transcript)
            t["llm"] = time.perf_counter() - mark

            if body.stream:  # audio fetched separately via /tts-stream
                audio = ""
            else:
                mark = time.perf_counter()
                audio = await _synthesize(reply, body.language)
                t["tts"] = time.perf_counter() - mark
        except HTTPException:
            raise  # already a clear message (e.g. rate limit)
        except Exception as exc:  # network blip to a provider — don't break the call
            logger.warning("voice turn failed at a provider: %s: %s", type(exc).__name__, exc)
            return {"transcript": transcript, "reply": "", "audio_b64": "",
                    "error": "network hiccup reaching the voice service — please tap and speak again",
                    "timings": {k: round(v, 2) for k, v in t.items()}}
        t["total"] = time.perf_counter() - t0
        logger.info("turn timings (s): %s", {k: round(v, 2) for k, v in t.items()})
        return {
            "transcript": transcript,
            "reply": reply,
            "audio_b64": audio,
            "timings": {k: round(v, 2) for k, v in t.items()},
        }

    @app.post("/api/live/voice/text")
    async def voice_text(body: VoiceText) -> dict:
        reply = await _chat(body.system_prompt, body.history, body.message)
        return {"reply": reply, "audio_b64": await _synthesize(reply, body.language)}

    # ---- Campaign CRUD ----
    @app.get("/api/live/defaults")
    def live_defaults() -> dict:
        """Prompt and greeting for the live tester.

        The page refuses to start with an empty prompt box, which looked
        identical to a broken microphone: no socket, no audio, no error. It now
        loads the active campaign so the box is never empty by accident.
        """
        from voiceos.conversation.manager import ConversationManager

        manager = ConversationManager(settings.conversation)
        return {
            "system_prompt": manager._system_prompt,
            "first_message": manager.first_message or "",
            "language": settings.stt.sarvam_language or "hi-IN",
            "campaign": settings.conversation.campaign_file,
            "stack": (f"{settings.stt.provider} + "
                      f"{settings.llm.model.split('/')[-1]} + {settings.tts.provider}"),
        }

    # ---- Logs: per-call records ----
    @app.get("/api/calls")
    def list_calls(limit: int = 100) -> list[dict]:
        """Newest first, without transcripts — the table only needs a summary."""
        return [{k: v for k, v in r.items() if k != "transcript"}
                for r in call_store.records(limit=limit)]

    @app.get("/api/calls/{call_id}")
    def get_call(call_id: str) -> dict:
        record = call_store.get(call_id)
        if record is None:
            raise HTTPException(status_code=404, detail="no such call")
        return record

    # ---- Assistant: what this deployment is actually configured with ----
    @app.get("/api/config")
    def get_config() -> dict:
        """Live provider configuration plus an order-of-magnitude cost model.

        Latency figures are measured medians from this project's own runs, not
        vendor claims. Cost is estimated from a price snapshot — see
        monitoring/pricing.py for why that is only ever an estimate.
        """
        one_minute = estimate_call_cost(
            60, 6, stt_provider=settings.stt.provider,
            tts_provider=settings.tts.provider, llm_model=settings.llm.model,
            pricing=pricing,
        )
        return {
            "transcriber": {
                "provider": settings.stt.provider,
                "model": (settings.stt.sarvam_model if settings.stt.provider == "sarvam"
                          else settings.stt.model),
                "language": settings.stt.sarvam_language or settings.stt.language,
                "typical_latency_ms": 400,
            },
            "model": {
                "provider": settings.llm.base_url,
                "model": settings.llm.model,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens,
                "typical_latency_ms": 160,
            },
            "voice": {
                "provider": settings.tts.provider,
                "voice_id": settings.tts.cartesia_voice_id,
                "language": settings.tts.cartesia_language,
                "sample_rate": settings.tts.sample_rate,
                "typical_latency_ms": 220,
            },
            "endpointing": {
                "smart_turn": settings.vad.smart_turn,
                "threshold": settings.vad.smart_turn_threshold,
                "min_silence_ms": settings.vad.min_silence_ms,
                "barge_in": settings.vad.barge_in,
                "echo_gate": settings.vad.echo_gate,
            },
            "campaign": settings.conversation.campaign_file,
            "cost_per_minute_usd": one_minute["per_minute_usd"],
            "estimated_response_ms": 1050,
            "pricing_captured": pricing.get("_captured", "unknown"),
        }

    # ---- Tools ----
    @app.get("/api/tools")
    def list_tools() -> list[dict]:
        if not settings.llm.tools_enabled:
            return []
        from voiceos.llm.tools import ToolRegistry, register_builtin_tools

        registry = ToolRegistry()
        register_builtin_tools(registry)
        return [{"name": t["function"]["name"],
                 "description": t["function"]["description"],
                 "parameters": t["function"].get("parameters", {})}
                for t in registry.schemas()]

    @app.get("/api/campaigns")
    async def list_campaigns() -> list[dict]:
        return store.list()

    @app.get("/api/campaigns/{name}")
    async def get_campaign(name: str) -> dict:
        try:
            return store.get(name)
        except KeyError:
            raise HTTPException(404, f"no campaign {name!r}")
        except CampaignError as exc:
            raise HTTPException(400, str(exc))

    @app.put("/api/campaigns/{name}")
    async def put_campaign(name: str, data: dict = Body(...)) -> dict:
        try:
            store.save(name, data)
        except CampaignError as exc:
            raise HTTPException(400, str(exc))
        return {"status": "saved", "name": name}

    @app.delete("/api/campaigns/{name}")
    async def delete_campaign(name: str) -> dict:
        try:
            store.delete(name)
        except KeyError:
            raise HTTPException(404, f"no campaign {name!r}")
        except CampaignError as exc:
            raise HTTPException(400, str(exc))
        return {"status": "deleted", "name": name}

    # ---- Test sandbox ----
    @app.post("/api/campaigns/{name}/test/start")
    async def test_start(name: str) -> dict:
        try:
            return sandbox.start(name)
        except KeyError:
            raise HTTPException(404, f"no campaign {name!r}")

    @app.post("/api/test/message")
    async def test_message(body: TestMessage) -> dict:
        try:
            reply = await sandbox.message(body.session_id, body.message)
        except KeyError:
            raise HTTPException(404, "unknown or expired test session")
        return {"reply": reply}

    # ---- Dry-run (consent gate, no calls placed) ----
    @app.post("/api/campaigns/{name}/dryrun")
    async def dryrun(name: str, body: DryRunBody) -> dict:
        try:
            store.get(name)
        except KeyError:
            raise HTTPException(404, f"no campaign {name!r}")
        from voiceos.telephony.campaign import CampaignRunner, Contact

        async def _never(number, caller_id):  # pragma: no cover - never called
            raise AssertionError("dry-run must not originate")

        runner = CampaignRunner(
            _never, caller_id="preview", dry_run=True,
            require_consent=body.require_consent,
        )
        results = await runner.run(
            [Contact(c.number, consented=c.consented, name=c.name) for c in body.contacts]
        )
        rows = [
            {"number": r.contact.number, "name": r.contact.name, "status": r.status}
            for r in results
        ]
        summary: dict[str, int] = {}
        for r in rows:
            summary[r["status"]] = summary.get(r["status"], 0) + 1
        return {"results": rows, "summary": summary}

    # ---- Results ----
    @app.get("/api/campaigns/{name}/results")
    async def results(name: str) -> dict:
        records = _results_store(name).records()
        return {"fields": _field_ids(name), "records": records}

    @app.get("/api/campaigns/{name}/results.csv", response_class=PlainTextResponse)
    async def results_csv(name: str) -> PlainTextResponse:
        import io

        store_ = _results_store(name)
        buf = io.StringIO()
        # export_csv writes to a path; reuse its logic via a temp in-memory list.
        fields = _field_ids(name)
        records = store_.records()
        import csv as _csv

        writer = _csv.writer(buf)
        meta = ["call_id", "number", "timestamp", "status"]
        writer.writerow(meta + fields)
        for rec in records:
            ans = rec.get("answers", {})
            writer.writerow([rec.get(c, "") for c in meta] + [ans.get(f, "") for f in fields])
        return PlainTextResponse(
            buf.getvalue(),
            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
            media_type="text/csv",
        )

    return app
