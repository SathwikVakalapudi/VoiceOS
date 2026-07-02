"""Fallback LLM.

Wraps an ordered list of LLM brains and behaves like any other
`BaseLLM`. If a provider raises *before streaming any token*, the next
one is tried; once tokens are flowing a mid-stream failure ends the turn
gracefully with the partial reply (the underlying QwenLLM already does
this), because re-answering on another model would talk over itself.
Drops in at the `create_llm` factory — the LLM worker never knows there
is more than one brain.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from voiceos.llm.base import BaseLLM, Message

logger = logging.getLogger(__name__)


class FallbackLLM(BaseLLM):
    def __init__(self, providers: list[BaseLLM]) -> None:
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        self._providers = providers

    async def load(self) -> None:
        for provider in self._providers:
            try:
                await provider.load()
            except Exception:
                logger.exception(
                    "LLM provider failed to load: %s", type(provider).__name__
                )

    async def generate(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        last_error: Exception | None = None
        for provider in self._providers:
            emitted = False
            try:
                async for delta in provider.generate(messages):
                    emitted = True
                    yield delta
                return
            except Exception as exc:
                last_error = exc
                if emitted:
                    # Tokens already spoken — another brain would repeat them.
                    raise
                logger.warning(
                    "LLM %s failed before any token — falling back",
                    type(provider).__name__,
                )
        if last_error is not None:
            raise last_error

    async def complete(self, messages, tools=None):
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                return await provider.complete(messages, tools=tools)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM %s complete() failed — falling back", type(provider).__name__
                )
        assert last_error is not None  # providers is non-empty
        raise last_error

    async def close(self) -> None:
        for provider in self._providers:
            try:
                await provider.close()
            except Exception:
                logger.exception(
                    "LLM provider failed to close: %s", type(provider).__name__
                )
