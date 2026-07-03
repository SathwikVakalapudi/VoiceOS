"""Survey extraction, storage, and session-wrapper tests (no real LLM)."""

import json

from voiceos.survey.definition import SurveyDefinition
from voiceos.survey.extractor import SurveyExtractor, parse_json_object
from voiceos.survey.store import ResultStore

SURVEY = SurveyDefinition.from_dict(
    {
        "name": "demo",
        "questions": [
            {"id": "like_modi", "type": "choice",
             "options": ["a lot", "not at all"], "prompt": "likes Modi"},
            {"id": "age", "type": "number", "prompt": "age"},
            {"id": "religion", "type": "text", "prompt": "religion"},
        ],
    }
)


class FakeLLM:
    """Returns a canned assistant message; records the prompt it received."""

    def __init__(self, content):
        self._content = content
        self.seen = None
        self.loaded = self.closed = False

    async def load(self):
        self.loaded = True

    async def close(self):
        self.closed = True

    async def complete(self, messages, tools=None):
        self.seen = messages
        return {"role": "assistant", "content": self._content}

    def generate(self, messages):  # unused here
        raise NotImplementedError


# --- parse_json_object -----------------------------------------------------

def test_parse_json_tolerates_code_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 1, "b": null} — done') == {"a": 1, "b": None}
    assert parse_json_object("not json at all") == {}
    assert parse_json_object("") == {}


# --- extractor -------------------------------------------------------------

async def test_extract_maps_answers_and_fills_missing_with_null():
    llm = FakeLLM('{"like_modi": "a lot", "age": 42}')  # religion omitted
    result = await SurveyExtractor(llm, SURVEY).extract(
        [{"role": "assistant", "content": "Do you like Modi?"},
         {"role": "user", "content": "chala ishtam"}]
    )
    assert result == {"like_modi": "a lot", "age": 42, "religion": None}


async def test_extract_returns_all_null_on_unparseable_reply():
    llm = FakeLLM("I could not determine the answers.")
    result = await SurveyExtractor(llm, SURVEY).extract([])
    assert result == {"like_modi": None, "age": None, "religion": None}


def test_extractor_prompt_lists_options_and_transcript():
    llm = FakeLLM("{}")
    msgs = SurveyExtractor(llm, SURVEY).build_messages(
        [{"role": "user", "content": "I am forty"}]
    )
    user = msgs[-1]["content"]
    assert "a lot | not at all" in user       # choice options offered
    assert "Respondent: I am forty" in user   # transcript rendered


# --- store -----------------------------------------------------------------

def test_store_appends_jsonl_and_exports_csv(tmp_path):
    store = ResultStore(str(tmp_path / "r.jsonl"))
    store.add({"call_id": "c1", "number": "+1", "timestamp": "t1",
               "status": "completed", "answers": {"age": 42, "religion": "hindu"}})
    store.add({"call_id": "c2", "number": "+2", "timestamp": "t2",
               "status": "completed", "answers": {"age": None, "religion": "muslim"}})

    recs = store.records()
    assert len(recs) == 2 and recs[0]["answers"]["age"] == 42

    csv_path = tmp_path / "out.csv"
    n = store.export_csv(str(csv_path), field_ids=["age", "religion"])
    assert n == 2
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "call_id,number,timestamp,status,age,religion"
    assert lines[1].startswith("c1,+1,t1,completed,42,hindu")


# --- collector + session wrapper -------------------------------------------

async def test_collector_builds_record_with_injected_llm(tmp_path):
    from voiceos.survey.collector import SurveyCollector

    llm = FakeLLM('{"like_modi": "not at all", "age": 30, "religion": "hindu"}')
    store = ResultStore(str(tmp_path / "r.jsonl"))
    collector = SurveyCollector(
        SURVEY, store, llm_factory=lambda: llm,
        clock=lambda: "T", id_factory=lambda: "CALLID",
    )

    record = await collector.collect([], number="+15551112222")
    assert llm.loaded and llm.closed                 # lifecycle honored
    assert record["call_id"] == "CALLID" and record["number"] == "+15551112222"
    assert record["answers"]["age"] == 30
    assert store.records()[0]["answers"]["like_modi"] == "not at all"


async def test_survey_session_extracts_after_pipeline_stops(tmp_path):
    from voiceos.survey.collector import SurveyCollector
    from voiceos.survey.session import SurveySession

    class FakeHistory:
        messages = [{"role": "user", "content": "forty years"}]

    class FakeConversation:
        history = FakeHistory()

    class FakePipeline:
        def __init__(self):
            self.conversation = FakeConversation()
            self.events = []

        async def start(self):
            self.events.append("start")

        async def stop(self):
            self.events.append("stop")

    llm = FakeLLM('{"like_modi": null, "age": 40, "religion": null}')
    collector = SurveyCollector(
        SURVEY, ResultStore(str(tmp_path / "r.jsonl")),
        llm_factory=lambda: llm, clock=lambda: "T", id_factory=lambda: "ID",
    )
    pipeline = FakePipeline()
    session = SurveySession(pipeline, collector, number="+199")

    await session.start()
    await session.stop()

    assert pipeline.events == ["start", "stop"]      # pipeline stopped first
    recs = ResultStore(str(tmp_path / "r.jsonl")).records()
    assert recs[0]["answers"]["age"] == 40 and recs[0]["number"] == "+199"
