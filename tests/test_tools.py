"""Tool registry + LLM-worker tool-resolution loop tests."""

import asyncio

from voiceos.config.settings import ConversationSettings
from voiceos.conversation.manager import ConversationManager
from voiceos.llm.inference import LLMWorker
from voiceos.llm.tools import Tool, ToolRegistry, register_builtin_tools
from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.state import InterruptController, StateMachine

OBJ = {"type": "object", "properties": {}}


async def test_registry_executes_and_passes_args():
    reg = ToolRegistry()
    seen = {}

    async def handler(args):
        seen.update(args)
        return "sunny, 25C"

    reg.register(Tool("weather", "get weather", OBJ, handler))
    assert len(reg) == 1
    assert await reg.execute("weather", {"city": "Paris"}) == "sunny, 25C"
    assert seen == {"city": "Paris"}


async def test_registry_unknown_and_crashing_tools_are_caught():
    reg = ToolRegistry()

    async def bad(args):
        raise ValueError("boom")

    reg.register(Tool("bad", "d", OBJ, bad))
    assert "no tool" in await reg.execute("missing", {})
    assert "Error running bad" in await reg.execute("bad", {})


async def test_builtin_time_tool_and_schema():
    reg = ToolRegistry()
    register_builtin_tools(reg)
    schemas = reg.schemas()
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "get_current_time"
    assert isinstance(await reg.execute("get_current_time", {}), str)


class ToolThenAnswerLLM:
    """complete() asks for a tool once, then returns a plain answer."""

    def __init__(self):
        self.calls = 0

    async def generate(self, messages):  # pragma: no cover - unused here
        if False:
            yield ""

    async def complete(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "get_current_time", "arguments": "{}"}}
                ],
            }
        return {"role": "assistant", "content": "It is noon."}


def make_worker(llm, tools):
    return LLMWorker(
        llm, ConversationManager(ConversationSettings()),
        asyncio.Queue(), asyncio.Queue(), EventBus(), StateMachine(),
        interrupts=InterruptController(), tools=tools, max_tool_iterations=4,
    )


async def test_worker_runs_tool_then_stops():
    reg = ToolRegistry()
    register_builtin_tools(reg)
    llm = ToolThenAnswerLLM()
    worker = make_worker(llm, reg)

    called = []
    worker._bus.subscribe(EventType.TOOL_CALLED, lambda e: called.append(e.data["name"]))

    messages = [{"role": "user", "content": "what time is it?"}]
    out = await worker._resolve_tools(messages)

    assert llm.calls == 2                       # tool round, then answer round
    assert called == ["get_current_time"]       # tool was invoked
    assert any(m.get("role") == "tool" for m in out)  # result fed back
    assert out[0] == {"role": "user", "content": "what time is it?"}  # original kept


# ---- call control ---------------------------------------------------------

from voiceos.llm.tools import END_CALL, END_CALL_SCHEMA, wants_end_call  # noqa: E402


def test_end_call_schema_is_openai_shaped():
    fn = END_CALL_SCHEMA["function"]
    assert END_CALL_SCHEMA["type"] == "function"
    assert fn["name"] == END_CALL
    # `reason` must be optional: a model that hangs up without explaining
    # itself should still hang up rather than fail the call.
    assert fn["parameters"]["required"] == []
    assert "reason" in fn["parameters"]["properties"]


def test_a_tool_call_ends_the_call_with_its_reason():
    reply = {"tool_calls": [{"function": {"name": END_CALL,
                                          "arguments": '{"reason": "survey-complete"}'}}]}
    assert wants_end_call(reply) == (True, "survey-complete")


def test_malformed_arguments_still_end_the_call():
    reply = {"tool_calls": [{"function": {"name": END_CALL, "arguments": "not json"}}]}
    end, reason = wants_end_call(reply)
    assert end and reason == "assistant-ended-call"


def test_the_tool_name_spoken_aloud_counts_as_an_end_signal():
    # Prompts in this repo explicitly forbid speaking the tool name, which is
    # only written down because models do it. Ending the call beats reading
    # "end_call_tool" to a respondent and then continuing forever.
    assert wants_end_call({"content": "धन्यवाद! end_call_tool"})[0] is True


def test_an_ordinary_reply_does_not_end_the_call():
    assert wants_end_call({"content": "आप मोदी को कितना पसंद करते हैं?"}) == (False, "")
    assert wants_end_call({"tool_calls": [{"function": {"name": "get_current_time"}}]}) \
        == (False, "")
    assert wants_end_call(None) == (False, "")
    assert wants_end_call({}) == (False, "")
