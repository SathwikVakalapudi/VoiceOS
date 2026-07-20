"""Qwen 3 via any OpenAI-compatible chat completions endpoint.

Defaults to Ollama (http://localhost:11434/v1) but works unchanged
against vLLM, llama.cpp server, LM Studio, or a hosted API — only the
base_url/model settings differ. Nothing here is actually Qwen-specific
except the default model name.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Sequence

import httpx

from voiceos.config.settings import LLMSettings
from voiceos.llm.base import BaseLLM, Message
from voiceos.utils.http import RETRYABLE_STATUS, error_detail

logger = logging.getLogger(__name__)


class QwenLLM(BaseLLM):
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def load(self) -> None:
        # Short connect timeout: fail fast and retry beats stalling a turn.
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            headers={"Authorization": f"Bearer {self._settings.api_key}"},
            timeout=httpx.Timeout(self._settings.timeout_s, connect=3.0),
        )
        # Health check with retries: transient connection resets are common
        # on flaky networks and don't mean the endpoint is down.
        for attempt in range(3):
            try:
                response = await self._client.get("/models")
                if response.status_code >= 400:
                    # Report what the provider said, not just the status code —
                    # "API key expired" is actionable, "400 Bad Request" is not.
                    logger.warning(
                        "LLM endpoint %s rejected the health check: %s — %s",
                        self._settings.base_url,
                        response.status_code,
                        await error_detail(response),
                    )
                    return
                logger.info("LLM endpoint reachable at %s", self._settings.base_url)
                return
            except httpx.HTTPError as exc:
                if attempt < 2:
                    await asyncio.sleep(1.0)
                    continue
                logger.warning(
                    "LLM endpoint %s unreachable (%s: %s) — "
                    "generation may still work; otherwise check the server/key",
                    self._settings.base_url,
                    type(exc).__name__,
                    exc,
                )

    async def generate(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError("QwenLLM.load() must be called first")
        payload = {
            "model": self._settings.model,
            "messages": list(messages),
            "stream": True,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        if self._settings.reasoning_effort is not None:
            payload["reasoning_effort"] = self._settings.reasoning_effort

        # Flaky networks drop pooled connections: retry as long as nothing
        # has been yielded yet. Once tokens are flowing, a dropped stream
        # ends the turn gracefully with the partial reply instead of erroring.
        for attempt in range(3):
            emitted = False
            backoff = None
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        detail = await error_detail(response)
                        if response.status_code in RETRYABLE_STATUS and attempt < 2:
                            logger.warning(
                                "LLM %s from %s (%s); retrying",
                                response.status_code, self._settings.model, detail,
                            )
                            backoff = 0.4 * 2**attempt
                        else:
                            logger.error(
                                "LLM request failed: %s — %s",
                                response.status_code, detail,
                            )
                            response.raise_for_status()
                    if backoff is None:
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[len("data:") :].strip()
                            if data == "[DONE]":
                                return
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                logger.warning("unparseable stream chunk: %.120s", data)
                                continue
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            content = (choices[0].get("delta") or {}).get("content")
                            if content:
                                emitted = True
                                yield content
                        return
            except httpx.HTTPStatusError:
                raise  # non-retryable: bad key, unknown model, malformed request
            except httpx.HTTPError as exc:
                if emitted:
                    logger.warning(
                        "LLM stream dropped mid-response (%s); "
                        "finishing turn with partial reply",
                        type(exc).__name__,
                    )
                    return
                if attempt < 2:
                    logger.warning(
                        "LLM connect attempt %d/3 failed (%s); retrying",
                        attempt + 1,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(0.4 * 2**attempt)  # 0.4, 0.8s
                    continue
                raise

            # A retryable status (429/5xx) was seen — back off and try again.
            await asyncio.sleep(backoff)

    async def complete(self, messages, tools=None):
        """Single non-streaming completion for the tool-calling loop. Returns
        the raw assistant message dict (possibly carrying tool_calls)."""
        if self._client is None:
            raise RuntimeError("QwenLLM.load() must be called first")
        payload = {
            "model": self._settings.model,
            "messages": list(messages),
            "stream": False,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self._settings.reasoning_effort is not None:
            payload["reasoning_effort"] = self._settings.reasoning_effort

        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            logger.error(
                "LLM request failed: %s — %s",
                response.status_code, await error_detail(response),
            )
            response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or [{}]
        return choices[0].get("message") or {"role": "assistant", "content": ""}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
