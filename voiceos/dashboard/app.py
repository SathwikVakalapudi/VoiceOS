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

from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

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

    # ---- UI ----
    @app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

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
