"""Tool calling.

A registry of functions the model may invoke mid-turn (look something up,
hit an API, take an action) before it speaks its answer. The LLM worker
runs a resolution loop — ask the model, run any tools it requests, feed
results back — until the model is ready to answer, then streams that
answer to TTS as usual.

Tools are described with JSON Schema in OpenAI's function-calling format,
so any OpenAI-compatible endpoint that supports tools (Ollama qwen3,
vLLM, Groq, ...) can drive them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict], Awaitable[str]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema for the arguments object
    handler: ToolHandler      # async (args) -> result string


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        """OpenAI-format tool definitions for the chat completions request."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: no tool named {name!r}"
        try:
            return await tool.handler(args)
        except Exception as exc:  # a tool crash must not kill the turn
            logger.exception("tool %s failed", name)
            return f"Error running {name}: {exc}"

    def __len__(self) -> int:
        return len(self._tools)


async def _get_current_time(args: dict) -> str:
    # Local import so the module has no import-time clock dependency.
    from datetime import datetime

    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def register_builtin_tools(registry: ToolRegistry) -> None:
    """A couple of harmless example tools so tool-calling works out of the
    box; real deployments register their own (search, CRM, booking, ...)."""
    registry.register(
        Tool(
            name="get_current_time",
            description="Get the current local date and time.",
            parameters={"type": "object", "properties": {}},
            handler=_get_current_time,
        )
    )
