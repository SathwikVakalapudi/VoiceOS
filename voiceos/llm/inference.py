"""LLM worker (response processor).

Consumes transcripts, streams a response from the LLM, strips Qwen 3's
<think>...</think> reasoning blocks, and forwards speakable sentence
chunks to the TTS queue as soon as they complete — the assistant starts
talking before the full response is generated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from voiceos.conversation.manager import ConversationManager
from voiceos.interfaces.llm import BaseLLM, Message
from voiceos.llm.tools import ToolRegistry
from voiceos.pipeline.events import EndOfTurn, EventBus, EventType, SpeakSentence
from voiceos.pipeline.state import InterruptController, PipelineState, StateMachine
from voiceos.tts.streaming import SentenceChunker

logger = logging.getLogger(__name__)


class ThinkTagFilter:
    """Stateful filter that drops <think>...</think> spans from a token
    stream, even when tags are split across deltas."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            if self._in_think:
                idx = self._buffer.find(self.CLOSE)
                if idx == -1:
                    # Keep just enough tail to detect a close tag split
                    # across deltas; the rest is discarded reasoning.
                    self._buffer = self._buffer[-(len(self.CLOSE) - 1) :]
                    break
                self._buffer = self._buffer[idx + len(self.CLOSE) :]
                self._in_think = False
            else:
                idx = self._buffer.find(self.OPEN)
                if idx == -1:
                    keep = self._partial_tag_len(self._buffer, self.OPEN)
                    if keep:
                        out.append(self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    else:
                        out.append(self._buffer)
                        self._buffer = ""
                    break
                out.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self.OPEN) :]
                self._in_think = True
        return "".join(out)

    @staticmethod
    def _partial_tag_len(text: str, tag: str) -> int:
        for length in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def flush(self) -> str:
        remainder = "" if self._in_think else self._buffer
        self._buffer = ""
        self._in_think = False
        return remainder


class LLMWorker:
    def __init__(
        self,
        llm: BaseLLM,
        conversation: ConversationManager,
        transcript_queue: asyncio.Queue[str],
        tts_queue: asyncio.Queue,
        event_bus: EventBus,
        state: StateMachine,
        interrupts: InterruptController | None = None,
        sentence_min_chars: int = 24,
        tools: ToolRegistry | None = None,
        max_tool_iterations: int = 4,
    ) -> None:
        self._llm = llm
        self._conversation = conversation
        self._transcript_queue = transcript_queue
        self._tts_queue = tts_queue
        self._bus = event_bus
        self._state = state
        self._interrupts = interrupts or InterruptController()
        self._sentence_min_chars = sentence_min_chars
        self._tools = tools
        self._max_tool_iterations = max_tool_iterations

    async def run(self) -> None:
        while True:
            transcript = await self._transcript_queue.get()
            try:
                await self._handle(transcript)
            except Exception:
                logger.exception("LLM generation failed")
                await self._bus.emit(EventType.ERROR, {"stage": "llm"})
                self._state.transition(PipelineState.IDLE)
            finally:
                self._transcript_queue.task_done()

    async def _resolve_tools(self, messages: list[Message]) -> list[Message]:
        """Ask the model, run any tools it requests, feed results back, and
        loop until it stops requesting tools. Returns the augmented message
        list to stream the final spoken answer from. Tool messages are
        ephemeral to this turn — they never enter conversation history."""
        messages = list(messages)
        for _ in range(self._max_tool_iterations):
            reply = await self._llm.complete(messages, tools=self._tools.schemas())
            tool_calls = reply.get("tool_calls") or []
            if not tool_calls:
                break
            messages.append(reply)
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                await self._bus.emit(EventType.TOOL_CALLED, {"name": name, "arguments": args})
                result = await self._tools.execute(name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id", ""), "content": result}
                )
        return messages

    async def _handle(self, transcript: str) -> None:
        turn_id = self._conversation.context.new_turn()
        messages = self._conversation.build_messages(transcript)
        self._conversation.begin_assistant(turn_id)
        await self._bus.emit(EventType.LLM_STARTED, {"turn_id": turn_id})

        # Tool phase: let the model call functions before it speaks. The final
        # answer is streamed below without tools, so it can't loop forever.
        if self._tools is not None and len(self._tools) > 0:
            try:
                messages = await self._resolve_tools(messages)
            except Exception:
                logger.exception("tool resolution failed; answering without tools")
                await self._bus.emit(EventType.ERROR, {"stage": "tools"})

        started = time.monotonic()
        first_token_at: float | None = None
        think_filter = ThinkTagFilter()
        chunker = SentenceChunker(min_chars=self._sentence_min_chars)
        spoken_parts: list[str] = []

        async def emit_sentence(sentence: str) -> None:
            # Stage the text and tag the audio with its position so playback
            # can report exactly which sentences the user heard.
            self._conversation.add_pending_segment(turn_id, sentence)
            index = len(spoken_parts)
            spoken_parts.append(sentence)
            await self._tts_queue.put(SpeakSentence(turn_id, index, sentence))

        generation = self._interrupts.generation
        cancelled = False
        try:
            async for delta in self._llm.generate(messages):
                if self._interrupts.generation != generation:
                    cancelled = True  # user barged in; stop feeding TTS
                    break
                if first_token_at is None:
                    first_token_at = time.monotonic()
                visible = think_filter.feed(delta)
                if not visible:
                    continue
                for sentence in chunker.feed(visible):
                    await emit_sentence(sentence)
            if not cancelled:
                tail = think_filter.flush()
                for sentence in [*chunker.feed(tail), *chunker.flush()]:
                    await emit_sentence(sentence)
        except Exception:
            # Speak the failure instead of going silent — a voice product
            # must degrade audibly, not into dead air.
            logger.exception("LLM generation failed")
            await self._bus.emit(EventType.ERROR, {"stage": "llm"})
            if not spoken_parts and self._interrupts.generation == generation:
                await emit_sentence(self._conversation.error_message)
        finally:
            # Close the turn so playback returns the pipeline to IDLE — but
            # not for a barged-in turn, where the user is already speaking.
            if self._interrupts.generation == generation:
                await self._tts_queue.put(EndOfTurn(turn_id=turn_id))

        # History is committed by the playback side (on end-of-turn) or by
        # the barge-in handler — recording only what was actually spoken,
        # never the sentences cut off mid-turn.
        reply = " ".join(spoken_parts).strip()
        await self._bus.emit(
            EventType.LLM_FINISHED,
            {
                "turn_id": turn_id,
                "text": reply,
                "latency_first_token_s": (
                    (first_token_at - started) if first_token_at else None
                ),
                "latency_total_s": time.monotonic() - started,
            },
        )
