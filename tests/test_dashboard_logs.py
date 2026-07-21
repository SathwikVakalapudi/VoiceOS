"""Dashboard routes backing the Logs, Assistant and Tools tabs."""

import pytest
from fastapi.testclient import TestClient

from voiceos.dashboard.app import create_app
from voiceos.monitoring.calls import CallRecorder, CallStore
from voiceos.pipeline.events import EventBus, EventType


class _FakeLLM:
    async def load(self): ...
    async def close(self): ...
    async def complete(self, messages, tools=None):
        return {"role": "assistant", "content": "{}"}

    async def generate(self, messages):
        yield "ok"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from voiceos.config.settings import Settings

    settings = Settings()
    settings.monitoring.calls_file = str(tmp_path / "calls.jsonl")
    app = create_app(
        campaigns_dir=str(tmp_path / "campaigns"),
        results_dir=str(tmp_path),
        llm_factory=lambda: _FakeLLM(),
        settings=settings,
    )
    return TestClient(app), CallStore(settings.monitoring.calls_file)


async def _seed(store, call_id, text="नमस्ते"):
    bus = EventBus()
    rec = CallRecorder(bus, store=None, direction="outbound",
                       number="+919848000000", campaign="rajasthan_hindi",
                       call_id=call_id)
    await bus.emit(EventType.SPEECH_ENDED, {"accepted": True})
    await bus.emit(EventType.TRANSCRIPT_READY,
                   {"text": text, "language": "hi-IN", "stt_latency_s": 0.4})
    await bus.emit(EventType.LLM_FINISHED,
                   {"text": "जी बताइए", "latency_first_token_s": 0.15,
                    "latency_total_s": 0.8})
    await bus.emit(EventType.PLAYBACK_FINISHED, {"turn_id": 1, "sentences_spoken": 1})
    store.add(rec.finish())


async def test_calls_list_is_summary_only(client):
    api, store = client
    await _seed(store, "call-one")
    await _seed(store, "call-two")

    rows = api.get("/api/calls").json()
    assert [r["call_id"] for r in rows] == ["call-two", "call-one"]   # newest first
    # The table shows a summary; transcripts are fetched per call so a long
    # campaign does not ship every word to the browser at once.
    assert "transcript" not in rows[0]
    assert rows[0]["turns"] == 1
    assert rows[0]["direction"] == "outbound"
    assert rows[0]["latency_s"]["stt_s"]["p50"] == 0.4


async def test_single_call_includes_the_transcript(client):
    api, store = client
    await _seed(store, "call-one", text="मैं ठीक हूँ")

    body = api.get("/api/calls/call-one").json()
    assert [t["role"] for t in body["transcript"]] == ["user", "assistant"]
    assert body["transcript"][0]["text"] == "मैं ठीक हूँ"


def test_unknown_call_is_404(client):
    api, _ = client
    assert api.get("/api/calls/nope").status_code == 404


def test_empty_log_is_an_empty_list_not_an_error(client):
    api, _ = client
    assert api.get("/api/calls").json() == []


def test_config_reports_the_live_providers(client):
    api, _ = client
    cfg = api.get("/api/config").json()

    for section in ("transcriber", "model", "voice", "endpointing"):
        assert section in cfg
    assert cfg["cost_per_minute_usd"] >= 0
    # The captured date must travel with the estimate, so nobody mistakes a
    # stale price list for current billing.
    assert cfg["pricing_captured"]


def test_tools_are_empty_unless_enabled(client):
    api, _ = client
    assert api.get("/api/tools").json() == []
