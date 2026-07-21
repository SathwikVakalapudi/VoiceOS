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
    POST   /api/campaigns/{name}/dryrun         -> consent-gate a contact list
    GET    /api/campaigns/{name}/results        -> extracted survey results (JSON)
    GET    /api/campaigns/{name}/results.csv    -> results as CSV

The live-call WebSocket loop lives in live_call.py; the shared LLM/STT/TTS
provider layer and chat helper live in voice_services.py.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel

from voiceos.dashboard.audio_io import decode_audio
from voiceos.dashboard.live_call import run_call
from voiceos.dashboard.sandbox import TestSandbox
from voiceos.dashboard.store import CampaignError, CampaignStore
from voiceos.dashboard.voice_services import VoiceServices, retry
from voiceos.interfaces.llm import BaseLLM
from voiceos.monitoring.calls import CallStore
from voiceos.monitoring.pricing import estimate_call_cost, load_pricing
from voiceos.survey.definition import SurveyDefinition
from voiceos.survey.store import ResultStore

logger = logging.getLogger(__name__)

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
    services = VoiceServices(settings, llm_factory)
    results_root = Path(results_dir)

    app = FastAPI(title="VoiceOS Campaign Dashboard")
    app.state.store = store
    app.state.sandbox = sandbox

    def _results_store(name: str) -> ResultStore:
        return ResultStore(str(results_root / f"{name}.jsonl"))

    def _field_ids(name: str) -> list[str]:
        survey = SurveyDefinition.from_campaign_file(store.path_for(name))
        return survey.field_ids if survey else []

    @app.on_event("startup")
    async def _prewarm() -> None:
        # Warm the LLM/STT/TTS *connections* with a real tiny call so the first
        # conversation turn isn't a cold start (the first live LLM call otherwise
        # spikes ~4-5s on TLS/connection setup).
        try:
            llm = await services.llm()
            await retry(lambda: llm.complete([{"role": "user", "content": "hi"}]))
            await services.tts("hi-IN")
            stt = await services.stt("hi-IN")
            await stt.transcribe(np.zeros(16000, dtype=np.int16), 16000)
        except Exception:  # never block startup on a provider hiccup
            logger.info("prewarm incomplete (will warm on first request)")
        # Smart Turn's transformers import is slow (~30s) — load it in the
        # background so startup and early calls aren't blocked (they use Silero-
        # only endpointing until it's ready).
        asyncio.get_event_loop().run_in_executor(None, services.smart_turn)

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
        await run_call(ws, services=services, settings=settings,
                       call_store=call_store, pricing=pricing)

    # ---- Live voice-conversation tester (any prompt, any language) ----
    @app.post("/api/live/reply")
    async def live_reply(body: LiveTurn) -> dict:
        hist = list(body.history)
        if body.first_message:
            hist = [{"role": "assistant", "content": body.first_message}, *hist]
        return {"reply": await services.chat(body.system_prompt, hist, body.message)}

    # ---- Server-side voice (Sarvam STT + Cartesia TTS) — no browser speech ----
    @app.post("/api/live/voice/open")
    async def voice_open(body: VoiceOpen) -> dict:
        if body.first_message:
            reply = body.first_message
        else:
            reply = await services.chat(body.system_prompt, [],
                                        "[The call has just connected. Speak your opening greeting "
                                        "and consent line now.]")
        if body.stream:
            return {"reply": reply, "audio_b64": ""}
        return {"reply": reply, "audio_b64": await services.synthesize(reply, body.language)}

    @app.post("/api/live/voice/tts-stream")
    async def tts_stream(body: TtsStream) -> StreamingResponse:
        tts = await services.tts(body.language)
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
            pcm = decode_audio(base64.b64decode(body.audio_b64))
        except Exception as exc:
            raise HTTPException(400, f"could not decode audio: {exc}")
        t["decode"] = time.perf_counter() - t0

        transcript = ""
        try:
            mark = time.perf_counter()
            stt = await services.stt(body.language)
            result = await retry(lambda: stt.transcribe(pcm, 16000))
            t["stt"] = time.perf_counter() - mark
            transcript = (result.text or "").strip()
            if not transcript:
                return {"transcript": "", "reply": "", "audio_b64": "",
                        "timings": {k: round(v, 2) for k, v in t.items()}}

            mark = time.perf_counter()
            reply = await services.chat(body.system_prompt, body.history, transcript)
            t["llm"] = time.perf_counter() - mark

            if body.stream:  # audio fetched separately via /tts-stream
                audio = ""
            else:
                mark = time.perf_counter()
                audio = await services.synthesize(reply, body.language)
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
        reply = await services.chat(body.system_prompt, body.history, body.message)
        return {"reply": reply, "audio_b64": await services.synthesize(reply, body.language)}

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
        import csv as _csv
        import io

        store_ = _results_store(name)
        buf = io.StringIO()
        # export_csv writes to a path; reuse its logic via a temp in-memory list.
        fields = _field_ids(name)
        records = store_.records()

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
