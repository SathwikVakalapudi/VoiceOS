"""LLM interface. Input: messages. Output: streamed response text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Sequence

# OpenAI-style chat message: {"role": "system|user|assistant", "content": "..."}
# Tool interactions carry non-string fields (tool_calls, tool_call_id), so the
# value type is loosened to Any.
Message = dict[str, Any]


class BaseLLM(ABC):
    async def load(self) -> None:
        """Open connections / warm the model. Called once at startup."""

    @abstractmethod
    def generate(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Stream response text deltas for the given conversation."""

    async def complete(
        self, messages: Sequence[Message], tools: list[dict] | None = None
    ) -> Message:
        """Non-streaming completion, used for the tool-calling loop. Returns
        the assistant message (which may contain ``tool_calls``). Optional —
        implement only in backends that support tool calling."""
        raise NotImplementedError("this LLM backend does not support tool calling")

    async def close(self) -> None:
        """Release connections."""
