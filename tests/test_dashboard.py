"""Campaign dashboard API tests (FastAPI TestClient, fake LLM, temp dirs)."""

import json

import pytest
from fastapi.testclient import TestClient

from voiceos.dashboard.app import create_app
from voiceos.survey.store import ResultStore


class FakeLLM:
    def __init__(self, reply="Thank you, noted."):
        self._reply = reply

    async def load(self):
        pass

    async def close(self):
        pass

    async def complete(self, messages, tools=None):
        return {"role": "assistant", "content": self._reply}

    def generate(self, messages):
        raise NotImplementedError


@pytest.fixture
def client(tmp_path):
    campaigns = tmp_path / "campaigns"
    results = tmp_path / "results"
    app = create_app(
        campaigns_dir=str(campaigns),
        results_dir=str(results),
        llm_factory=lambda: FakeLLM(),
    )
    app.state._campaigns_dir = campaigns
    app.state._results_dir = results
    return TestClient(app)


CAMPAIGN = {
    "system_prompt": "You are a survey caller.",
    "first_message": "Hello, may I ask a few questions?",
    "error_message": "Sorry, say that again?",
    "survey": {"name": "demo", "questions": [
        {"id": "age", "type": "number", "prompt": "age"},
    ]},
}


# ---- CRUD ----

def test_create_get_list_delete_campaign(client):
    assert client.get("/api/campaigns").json() == []

    r = client.put("/api/campaigns/demo-camp", json=CAMPAIGN)
    assert r.status_code == 200 and r.json()["status"] == "saved"

    listing = client.get("/api/campaigns").json()
    assert listing[0]["name"] == "demo-camp"
    assert listing[0]["has_survey"] and listing[0]["question_count"] == 1

    got = client.get("/api/campaigns/demo-camp").json()
    assert got["system_prompt"] == CAMPAIGN["system_prompt"]

    assert client.delete("/api/campaigns/demo-camp").status_code == 200
    assert client.get("/api/campaigns/demo-camp").status_code == 404


def test_list_skips_non_campaign_json(client):
    # A contacts list (or any non-campaign JSON) in the dir must not break listing.
    (client.app.state._campaigns_dir).mkdir(parents=True, exist_ok=True)
    (client.app.state._campaigns_dir / "contacts.json").write_text(
        json.dumps([{"number": "+1"}]), encoding="utf-8"
    )
    client.put("/api/campaigns/real", json=CAMPAIGN)
    names = [c["name"] for c in client.get("/api/campaigns").json()]
    assert names == ["real"]  # contacts.json skipped


def test_invalid_campaign_rejected(client):
    # No system_prompt and no first_message.
    r = client.put("/api/campaigns/bad", json={"error_message": "x"})
    assert r.status_code == 400 and "system_prompt" in r.json()["detail"]


def test_bad_name_rejected(client):
    r = client.put("/api/campaigns/..%2fetc", json=CAMPAIGN)
    assert r.status_code in (400, 404)  # never writes outside the dir


def test_invalid_survey_block_rejected(client):
    bad = dict(CAMPAIGN, survey={"questions": [{"id": "a"}, {"id": "a"}]})  # dup ids, no prompt
    r = client.put("/api/campaigns/dupe", json=bad)
    assert r.status_code == 400


# ---- test sandbox ----

def test_sandbox_start_and_message(client):
    client.put("/api/campaigns/s", json=CAMPAIGN)
    start = client.post("/api/campaigns/s/test/start").json()
    assert start["first_message"] == CAMPAIGN["first_message"]
    assert start["session_id"]

    reply = client.post(
        "/api/test/message",
        json={"session_id": start["session_id"], "message": "yes go ahead"},
    ).json()
    assert reply["reply"] == "Thank you, noted."


def test_sandbox_unknown_session_404(client):
    r = client.post("/api/test/message", json={"session_id": "nope", "message": "hi"})
    assert r.status_code == 404


def test_sandbox_start_missing_campaign_404(client):
    assert client.post("/api/campaigns/ghost/test/start").status_code == 404


# ---- dry-run ----

def test_dryrun_applies_consent_gate(client):
    client.put("/api/campaigns/d", json=CAMPAIGN)
    r = client.post("/api/campaigns/d/dryrun", json={"contacts": [
        {"number": "+1", "consented": True, "name": "A"},
        {"number": "+2", "consented": False, "name": "B"},
    ]}).json()
    status = {row["number"]: row["status"] for row in r["results"]}
    assert status == {"+1": "dry_run", "+2": "skipped_no_consent"}
    assert r["summary"]["dry_run"] == 1


# ---- results ----

def test_results_json_and_csv(client, tmp_path):
    client.put("/api/campaigns/r", json=CAMPAIGN)
    # Simulate a completed call by writing to the results file the API reads.
    store = ResultStore(str(client.app.state._results_dir / "r.jsonl"))
    store.add({"call_id": "c1", "number": "+15551112222", "timestamp": "T",
               "status": "completed", "answers": {"age": 42}})

    res = client.get("/api/campaigns/r/results").json()
    assert res["fields"] == ["age"]
    assert res["records"][0]["answers"]["age"] == 42

    csv = client.get("/api/campaigns/r/results.csv")
    assert csv.status_code == 200
    assert "call_id,number,timestamp,status,age" in csv.text
    assert "c1,+15551112222,T,completed,42" in csv.text


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Campaign Dashboard" in r.text


def test_live_page_served(client):
    r = client.get("/live")
    assert r.status_code == 200 and "Live Call" in r.text


def test_live_reply_uses_custom_prompt_and_history(client):
    r = client.post("/api/live/reply", json={
        "system_prompt": "You are a Hindi survey caller.",
        "history": [{"role": "assistant", "content": "नमस्ते!"},
                    {"role": "user", "content": "जी बोलिए"}],
        "message": "मैं तैयार हूँ",
    })
    assert r.status_code == 200
    assert r.json()["reply"] == "Thank you, noted."   # FakeLLM canned reply
