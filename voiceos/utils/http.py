"""Shared HTTP behaviour for hosted providers (LLM, STT, TTS).

Two things every provider adapter needs and gets wrong the same way:

1. **Which failures are worth retrying.** Treating every 4xx as permanent is
   nearly right but throws away 429, which is exactly the case backoff exists
   for; treating every failure as transient burns three attempts on an expired
   key.
2. **Reporting why a request failed.** On a *streamed* response the body has
   not been read when the status arrives, so `raise_for_status()` produces
   "400 Bad Request" and discards the part that tells you what to do about it
   ("API key expired", "model not found"). Read the body first.
"""

from __future__ import annotations

import json

import httpx

# Worth another attempt: the endpoint is up but momentarily unable to serve.
# Everything else in 4xx (bad key, expired key, unknown model, malformed
# request) fails identically on retry, so it is raised immediately.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


async def error_detail(response: httpx.Response) -> str:
    """The provider's own explanation for a failed request."""
    try:
        await response.aread()
    except Exception:  # body already consumed, or the connection died
        return "<no body>"
    text = response.text.strip().replace("\n", " ")
    try:  # OpenAI-compatible shape: {"error": {"message": ...}}
        payload = json.loads(text)
        if isinstance(payload, list) and payload:  # Google wraps it in a list
            payload = payload[0]
        message = payload.get("error", {}).get("message")
        if message:
            return message
    except (json.JSONDecodeError, AttributeError):
        pass
    return text[:300] or "<empty body>"


def retryable(response: httpx.Response) -> bool:
    return response.status_code in RETRYABLE_STATUS
