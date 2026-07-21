"""Provider errors must be legible, and retryable ones must actually retry."""

import json

import httpx
import pytest

from voiceos.config.settings import LLMSettings
from voiceos.llm.qwen import QwenLLM
from voiceos.utils.http import RETRYABLE_STATUS, error_detail


def _llm(handler) -> QwenLLM:
    llm = QwenLLM(LLMSettings(model="test-model"))
    llm._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    return llm


def _sse(*chunks: str) -> bytes:
    body = "".join(
        'data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % c for c in chunks
    )
    return (body + "data: [DONE]\n\n").encode()


async def test_error_detail_extracts_the_provider_message():
    # A streamed response has not read its body when the status arrives, so the
    # bare HTTPStatusError says only "400 Bad Request". The reason lives here.
    response = httpx.Response(
        400, json={"error": {"message": "API key expired. Please renew the API key."}}
    )
    assert await error_detail(response) == "API key expired. Please renew the API key."


async def test_error_detail_handles_a_list_wrapped_payload():
    # Google's OpenAI-compat layer wraps the error object in a list.
    response = httpx.Response(400, json=[{"error": {"message": "boom"}}])
    assert await error_detail(response) == "boom"


async def test_error_detail_falls_back_to_raw_text():
    assert await error_detail(httpx.Response(500, text="upstream exploded")) == (
        "upstream exploded"
    )


async def test_non_retryable_status_raises_immediately():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in _llm(handler).generate([{"role": "user", "content": "hi"}]):
            pass

    assert len(calls) == 1  # an expired key will not fix itself


async def test_retryable_status_is_retried_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, content=_sse("hello"))

    out = [c async for c in _llm(handler).generate([{"role": "user", "content": "hi"}])]

    assert out == ["hello"]
    assert len(calls) == 3  # 429, 429, then the real answer


async def test_retryable_status_gives_up_after_three_attempts():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in _llm(handler).generate([{"role": "user", "content": "hi"}]):
            pass

    assert len(calls) == 3


def test_retryable_set_excludes_client_errors_that_will_not_clear():
    assert 429 in RETRYABLE_STATUS and 503 in RETRYABLE_STATUS
    for status in (400, 401, 403, 404, 422):
        assert status not in RETRYABLE_STATUS


# ── the same contract, applied to STT and TTS ────────────────────────────────
# All three adapters previously treated every HTTPStatusError as permanent, so
# a single 429 or 503 killed the turn. They now share one policy.


def _sarvam(handler):
    import httpx as _httpx

    from voiceos.config.settings import STTSettings
    from voiceos.stt.sarvam import SarvamSTT

    stt = SarvamSTT(STTSettings(sarvam_api_key="k"))
    stt._client = _httpx.AsyncClient(
        base_url="http://test", transport=_httpx.MockTransport(handler)
    )
    return stt


def _cartesia(handler):
    import httpx as _httpx

    from voiceos.config.settings import TTSSettings
    from voiceos.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(TTSSettings(cartesia_api_key="k"))
    tts._client = _httpx.AsyncClient(
        base_url="http://test", transport=_httpx.MockTransport(handler)
    )
    return tts


async def test_sarvam_retries_429_then_succeeds():
    import numpy as np

    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json={"transcript": "hello", "language_code": "en-IN"})

    result = await _sarvam(handler).transcribe(np.zeros(16000, dtype=np.float32), 16000)

    assert result.text == "hello"
    assert len(calls) == 3


async def test_sarvam_does_not_retry_a_bad_key():
    import numpy as np

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"error": {"message": "invalid key"}})

    with pytest.raises(httpx.HTTPStatusError):
        await _sarvam(handler).transcribe(np.zeros(16000, dtype=np.float32), 16000)

    assert len(calls) == 1


async def test_cartesia_retries_503_then_streams_audio():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 2:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(200, content=b"\x01\x00\x02\x00")

    chunks = [c async for c in _cartesia(handler).synthesize("hi")]

    assert sum(c.size for c in chunks) == 2
    assert len(calls) == 2


async def test_cartesia_does_not_retry_a_bad_voice_id():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, json={"error": {"message": "voice not found"}})

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in _cartesia(handler).synthesize("hi"):
            pass

    assert len(calls) == 1


# ---- provider parameter drift ---------------------------------------------
# OpenAI's newer models renamed max_tokens; Groq/Ollama/vLLM did not. The
# client adapts on the provider's own error rather than requiring the operator
# to remember which name goes with which base_url.


def _sse_ok() -> bytes:
    return (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            b'data: [DONE]\n\n')


async def test_streaming_retries_once_with_max_completion_tokens():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append("max_completion_tokens" if "max_completion_tokens" in body
                    else "max_tokens")
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message":
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."}})
        return httpx.Response(200, content=_sse_ok())

    llm = _llm(handler)
    out = [c async for c in llm.generate([{"role": "user", "content": "hi"}])]

    assert out == ["hi"]
    assert seen == ["max_tokens", "max_completion_tokens"]
    # The swap is remembered, so only the first call of a session pays for it.
    assert llm._max_tokens_key == "max_completion_tokens"


async def test_complete_retries_once_with_max_completion_tokens():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append("max_completion_tokens" if "max_completion_tokens" in body
                    else "max_tokens")
        if "max_tokens" in body:
            # Verbatim from OpenAI — the old name is quoted first, which is
            # what makes the rename unambiguous.
            return httpx.Response(400, json={"error": {"message":
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."}})
        return httpx.Response(200, json={"choices": [{"message":
            {"role": "assistant", "content": "ok"}}]})

    llm = _llm(handler)
    reply = await llm.complete([{"role": "user", "content": "hi"}])

    assert reply["content"] == "ok"
    assert seen == ["max_tokens", "max_completion_tokens"]


async def test_an_unrelated_400_is_not_retried_as_a_parameter_swap():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "model not found"}})

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in _llm(handler).generate([{"role": "user", "content": "hi"}]):
            pass

    assert len(calls) == 1     # no pointless retry on a different problem


def test_groq_style_providers_keep_the_original_parameter():
    from voiceos.config.settings import LLMSettings
    from voiceos.llm.qwen import QwenLLM

    assert QwenLLM(LLMSettings())._max_tokens_key == "max_tokens"


async def test_a_rejected_temperature_is_dropped_and_the_call_proceeds():
    """gpt-5 accepts only its default temperature. Dropping the parameter is
    better than failing the turn — the caller gets an answer either way."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(sorted(k for k in body if k in ("temperature", "max_tokens",
                                                    "max_completion_tokens")))
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message":
                "Unsupported value: 'temperature' does not support 0.7 with this "
                "model. Only the default (1) value is supported."}})
        return httpx.Response(200, content=_sse_ok())

    llm = _llm(handler)
    out = [c async for c in llm.generate([{"role": "user", "content": "hi"}])]

    assert out == ["hi"]
    assert "temperature" in llm._unsupported
    assert seen[-1] == ["max_tokens"]          # temperature gone, rest intact


async def test_several_rejected_parameters_are_learned_in_turn():
    """gpt-5-nano rejects max_tokens, then temperature. One call must survive
    both rather than needing a human between them."""
    def handler(request):
        body = json.loads(request.content)
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message":
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."}})
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message":
                "Unsupported value: 'temperature' does not support 0.7 with this model."}})
        return httpx.Response(200, json={"choices": [{"message":
            {"role": "assistant", "content": "ok"}}]})

    llm = _llm(handler)
    reply = await llm.complete([{"role": "user", "content": "hi"}])

    assert reply["content"] == "ok"
    assert llm._max_tokens_key == "max_completion_tokens"
    assert "temperature" in llm._unsupported


async def test_learned_quirks_persist_so_later_turns_pay_nothing():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(1)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message":
                "Unsupported value: 'temperature' does not support 0.7."}})
        return httpx.Response(200, content=_sse_ok())

    llm = _llm(handler)
    for _ in range(3):
        [c async for c in llm.generate([{"role": "user", "content": "hi"}])]

    # First turn costs one extra request; the next two go straight through.
    assert len(calls) == 4
