"""Round-robin key rotation.

Spreads requests across several API keys (e.g. multiple free-tier Groq
accounts) so no single key's per-minute rate limit is hit as fast. Each
request starts on the next key in rotation; if a key fails (429 / network),
the remaining keys are tried before giving up. With N keys the effective
throughput is ~N× a single key — a free way to sustain a real conversation.

Behaves like any other `BaseLLM`; drop-in at the `create_llm` factory.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from voiceos.llm.base import BaseLLM, Message

logger = logging.getLogger(__name__)


class RotatingLLM(BaseLLM):
    def __init__(self, providers: list[BaseLLM]) -> None:
        if not providers:
            raise ValueError("RotatingLLM needs at least one provider")
        self._providers = providers
        self._next = 0

    def _rotation(self) -> list[BaseLLM]:
        """Providers ordered from the next start index (round-robin), wrapping."""
        n = len(self._providers)
        start = self._next
        self._next = (self._next + 1) % n
        return [self._providers[(start + k) % n] for k in range(n)]

    async def load(self) -> None:
        for provider in self._providers:
            try:
                await provider.load()
            except Exception:
                logger.exception("LLM key failed to load: %s", type(provider).__name__)

    async def generate(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        last_error: Exception | None = None
        for provider in self._rotation():
            emitted = False
            try:
                async for delta in provider.generate(messages):
                    emitted = True
                    yield delta
                return
            except Exception as exc:
                last_error = exc
                if emitted:  # tokens already spoken — another key would repeat them
                    raise
                logger.warning("LLM key failed before any token — rotating to next key")
        if last_error is not None:
            raise last_error

    async def complete(self, messages, tools=None) -> Message:
        last_error: Exception | None = None
        for provider in self._rotation():
            try:
                return await provider.complete(messages, tools=tools)
            except Exception as exc:
                last_error = exc
                logger.warning("LLM key complete() failed — rotating to next key")
        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        for provider in self._providers:
            try:
                await provider.close()
            except Exception:
                logger.exception("LLM key failed to close")
