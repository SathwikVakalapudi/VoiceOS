"""Cost estimation per call.

WARNING: these are published list prices captured on 2026-07-20, not billing
data. Providers change them without notice — `saarika:flash` was deprecated out
from under this project the same week. Treat the output as an order-of-magnitude
estimate for comparing configurations, never as an invoice. Reconcile against
the provider's own billing before charging anyone.

Override without editing code:

    VOICEOS_MONITORING__PRICING_FILE=pricing.json

with the same shape as `DEFAULT_PRICING`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# USD. Audio priced per minute of audio processed; LLM per million tokens.
DEFAULT_PRICING: dict = {
    "_captured": "2026-07-20",
    "stt": {
        "sarvam": {"per_minute": 0.005},
        "whisper": {"per_minute": 0.0},          # local
    },
    "tts": {
        "cartesia": {"per_minute": 0.022},
        "edge": {"per_minute": 0.0},             # free
        "piper": {"per_minute": 0.0},            # local
        "svara": {"per_minute": 0.0},            # self-hosted
    },
    "llm": {
        # Rough blended rate; input and output differ, but a voice turn is
        # small and dominated by the system prompt, so one number is honest
        # enough for comparing configurations.
        "default": {"per_million_tokens": 0.30},
        "qwen/qwen3.6-27b": {"per_million_tokens": 0.30},
        "llama-3.1-8b-instant": {"per_million_tokens": 0.10},
        "llama-3.3-70b-versatile": {"per_million_tokens": 0.79},
    },
}


def load_pricing(path: str | None = None) -> dict:
    if not path:
        return DEFAULT_PRICING
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("pricing file %s unusable (%s); using defaults", path, exc)
        return DEFAULT_PRICING


def estimate_call_cost(
    duration_s: float,
    turns: int,
    *,
    stt_provider: str,
    tts_provider: str,
    llm_model: str,
    speech_ratio: float = 0.45,
    tokens_per_turn: int = 700,
    pricing: dict | None = None,
) -> dict:
    """Break a call's cost down by stage.

    `speech_ratio` is the share of wall-clock that is actually audio through a
    model: nobody talks continuously, and silence is not billed by either
    provider. 0.45 is a plausible default for a two-way conversation — adjust
    it once you have real invoices to compare against.
    """
    p = pricing or DEFAULT_PRICING
    audio_minutes = max(0.0, duration_s) / 60 * speech_ratio

    stt_rate = p["stt"].get(stt_provider, {}).get("per_minute", 0.0)
    tts_rate = p["tts"].get(tts_provider, {}).get("per_minute", 0.0)
    llm_rate = p["llm"].get(llm_model, p["llm"]["default"])["per_million_tokens"]

    stt = audio_minutes * stt_rate
    tts = audio_minutes * tts_rate
    llm = max(0, turns) * tokens_per_turn / 1_000_000 * llm_rate
    total = stt + tts + llm
    return {
        "stt_usd": round(stt, 5),
        "tts_usd": round(tts, 5),
        "llm_usd": round(llm, 5),
        "total_usd": round(total, 5),
        "per_minute_usd": round(total / max(duration_s / 60, 1e-6), 4),
        "estimated": True,          # never let this be mistaken for billing
    }
