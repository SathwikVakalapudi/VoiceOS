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
import re
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
        # OpenAI-compatible is a family, not a standard: gpt-5 renamed
        # max_tokens, refuses any temperature but 1, and rejects
        # reasoning_effort values Groq requires. Rather than encode a matrix of
        # provider quirks, learn them — a rejected parameter is dropped or
        # renamed on the provider's own say-so and remembered for the session.
        self._max_tokens_key = "max_tokens"
        self._unsupported: set[str] = set()

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
        payload = self._payload(messages, stream=True)

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
                        if self._adapt(detail, payload) and attempt < 2:
                            logger.info("retrying with %s", self._max_tokens_key)
                            backoff = 0.0
                        elif response.status_code in RETRYABLE_STATUS and attempt < 2:
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
        payload = self._payload(messages, stream=False)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            detail = await error_detail(response)
            # Several parameters can be rejected in turn (max_tokens, then
            # temperature, ...), so keep adapting while the endpoint keeps
            # telling us what it will not accept.
            for _ in range(3):
                if not self._adapt(detail, payload):
                    break
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code < 400:
                    break
                detail = await error_detail(response)
            if response.status_code >= 400:
                logger.error("LLM request failed: %s — %s",
                             response.status_code, await error_detail(response))
                response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or [{}]
        return choices[0].get("message") or {"role": "assistant", "content": ""}

    def _payload(self, messages: Sequence[Message], *, stream: bool) -> dict:
        """Request body, minus anything this endpoint has already rejected."""
        body = {
            "model": self._settings.model,
            "messages": list(messages),
            "stream": stream,
            "temperature": self._settings.temperature,
            self._max_tokens_key: self._settings.max_tokens,
        }
        if self._settings.reasoning_effort is not None:
            body["reasoning_effort"] = self._settings.reasoning_effort
        for name in self._unsupported:
            body.pop(name, None)
        return body

    def _adapt(self, detail: str, payload: dict) -> bool:
        """Drop or rename whatever the provider just refused.

        OpenAI names the offending field in quotes, either
        "Unsupported parameter: 'max_tokens' ... Use 'max_completion_tokens'
        instead" or "Unsupported value: 'temperature' does not support 0.7".
        Renames are honoured; anything else is dropped so the request can
        proceed on the provider's default. Returns True if the payload changed.
        """
        text = detail or ""
        rename = re.search(r"'([\w.]+)'[^']*?Use '([\w.]+)' instead", text)
        if rename:
            old, new = rename.group(1), rename.group(2)
            if payload.pop(old, None) is not None or old == self._max_tokens_key:
                if old == self._max_tokens_key:
                    self._max_tokens_key = new
                payload[new] = self._settings.max_tokens
                logger.info("endpoint renamed %r -> %r", old, new)
                return True
        bad = re.search(r"Unsupported (?:value|parameter): '([\w.]+)'", text)
        if bad and bad.group(1) in payload:
            name = bad.group(1)
            payload.pop(name)
            self._unsupported.add(name)
            logger.info("endpoint rejects %r; dropping it and using its default", name)
            return True
        return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
