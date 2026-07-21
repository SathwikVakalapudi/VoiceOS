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


# ---------------------------------------------------------------------------
# Call control
# ---------------------------------------------------------------------------

END_CALL = "end_call_tool"

# Unlike a lookup tool, this one returns nothing useful to the model — it is a
# signal to the transport. The caller checks `wants_end_call()` on the reply
# and hangs up after the farewell has finished playing.
END_CALL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": END_CALL,
        "description": (
            "End the phone call. Call this immediately after speaking a farewell "
            "line, once the conversation is complete or the respondent has "
            "declined. Produces no speech of its own."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the call ended, e.g. survey-complete, "
                                   "declined, wrong-number, hostile.",
                }
            },
            "required": [],
        },
    },
}


def wants_end_call(reply: dict | None) -> tuple[bool, str]:
    """Did the model ask to hang up? Returns (should_end, reason).

    Checks the tool call properly, and also the literal name appearing in the
    spoken text. The second path is not paranoia: prompts in this repo
    explicitly warn the model never to "read, speak, pronounce, or spell out"
    the tool name, which is only written down because models do it. Treating
    that as an end signal is better than speaking "end_call_tool" at a
    respondent and then continuing the call forever.
    """
    if not reply:
        return False, ""
    for call in reply.get("tool_calls") or []:
        if (call.get("function") or {}).get("name") == END_CALL:
            import json as _json

            try:
                args = _json.loads((call["function"] or {}).get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            return True, str(args.get("reason") or "assistant-ended-call")
    if END_CALL in (reply.get("content") or ""):
        return True, "assistant-ended-call"
    return False, ""
