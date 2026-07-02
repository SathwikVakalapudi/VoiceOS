"""FallbackLLM tests: a failing primary brain rolls over to a backup."""

import pytest

from voiceos.llm.fallback import FallbackLLM


class FakeLLM:
    def __init__(self, text=None, fail=False, emit_then_fail=False, tool_msg=None):
        self._text = text
        self._fail = fail
        self._emit_then_fail = emit_then_fail
        self._tool_msg = tool_msg

    async def load(self):
        pass

    async def generate(self, messages):
        if self._fail:
            raise RuntimeError("brain down")
        for word in (self._text or "").split():
            yield word + " "
            if self._emit_then_fail:
                raise RuntimeError("dropped mid-stream")

    async def complete(self, messages, tools=None):
        if self._fail:
            raise RuntimeError("brain down")
        return self._tool_msg or {"role": "assistant", "content": self._text or ""}

    async def close(self):
        pass


async def _stream(llm):
    return "".join([d async for d in llm.generate([])]).strip()


async def test_generate_falls_back_when_primary_fails():
    llm = FallbackLLM([FakeLLM(fail=True), FakeLLM(text="hello from backup")])
    assert await _stream(llm) == "hello from backup"


async def test_generate_does_not_fall_back_after_tokens_emitted():
    llm = FallbackLLM([FakeLLM(text="partial", emit_then_fail=True), FakeLLM(text="backup")])
    with pytest.raises(RuntimeError):
        await _stream(llm)


async def test_complete_falls_back():
    primary = FakeLLM(fail=True)
    backup = FakeLLM(tool_msg={"role": "assistant", "content": "ok"})
    llm = FallbackLLM([primary, backup])
    result = await llm.complete([], tools=[])
    assert result["content"] == "ok"


async def test_all_fail_raises():
    llm = FallbackLLM([FakeLLM(fail=True), FakeLLM(fail=True)])
    with pytest.raises(RuntimeError):
        await _stream(llm)
