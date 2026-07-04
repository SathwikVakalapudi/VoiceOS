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
from voiceos.interfaces.llm import BaseLLM
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
    if llm_factory is None:
        if settings is None:
            from voiceos.config.settings import get_settings

            settings = get_settings()
        from voiceos.pipeline.pipeline import create_llm

        llm_factory = lambda: create_llm(settings)  # noqa: E731

    store = CampaignStore(campaigns_dir)
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

    async def _chat(system_prompt: str, history: list, user_message: str) -> str:
        """One LLM turn: system + trimmed recent history + user. Trimming caps
        tokens-per-request so long prompts don't blow the provider rate limit."""
        import httpx

        turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        msgs = [{"role": "system", "content": system_prompt}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in turns[-16:]]
        msgs.append({"role": "user", "content": user_message})
        llm = await _shared_llm()
        try:
            r = await _retry(lambda: llm.complete(msgs))
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429:
                raise HTTPException(429, "LLM is rate-limited — wait a moment and try again.")
            if code >= 500:
                raise HTTPException(503, "LLM service is busy right now — please tap and "
                                    "speak again.")
            raise HTTPException(502, f"LLM error: {exc}")
        return r.get("content", "") if r else ""

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

        async def say(text: str) -> None:
            nonlocal speaking
            speaking = True
            tts = await _shared_tts(lang)
            await ws.send_json({"type": "audio_start", "rate": tts.sample_rate})
            clean = (text or "").replace("end_call_tool", "").strip()
            try:
                async for chunk in tts.synthesize(clean):
                    await ws.send_bytes(np.ascontiguousarray(chunk, dtype="<i2").tobytes())
            except Exception:
                pass
            await ws.send_json({"type": "audio_end"})
            endpointer.reset()             # discard any echo captured during playback
            speaking = False
            await ws.send_json({"type": "listening"})

        # opening line
        try:
            reply = cfg.get("first_message") or await _chat(
                prompt, [], "[The call has just connected. Speak your opening greeting "
                "and consent line now.]")
            history.append({"role": "assistant", "content": reply})
            await ws.send_json({"type": "reply", "text": reply})
            await say(reply)
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "text": str(exc)})
            except Exception:
                return  # client already gone

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None or speaking:      # ignore mic while the agent talks
                    continue
                for utt in await next_utts(np.frombuffer(data, dtype="<i2")):
                    await ws.send_json({"type": "thinking"})
                    try:
                        res = await _retry(lambda: stt.transcribe(utt, 16000))
                        transcript = (res.text or "").strip()
                    except Exception:
                        transcript = ""
                    if not transcript:
                        await ws.send_json({"type": "listening"})
                        continue
                    await ws.send_json({"type": "transcript", "text": transcript})
                    try:
                        reply = await _chat(prompt, history, transcript)
                    except HTTPException as he:
                        await ws.send_json({"type": "error", "text": he.detail})
                        await ws.send_json({"type": "listening"})
                        continue
                    history.append({"role": "user", "content": transcript})
                    history.append({"role": "assistant", "content": reply})
                    await ws.send_json({"type": "reply", "text": reply})
                    await say(reply)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

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
