"""Call records assembled from the event stream, and cost estimation."""

import json

import pytest

from voiceos.monitoring.calls import CallRecorder, CallStore
from voiceos.monitoring.pricing import estimate_call_cost
from voiceos.pipeline.events import EventBus, EventType


async def _turn(bus, user, reply, *, stt=0.4, ttft=0.15, total=0.9, gap=0.9):
    await bus.emit(EventType.SPEECH_ENDED, {"accepted": True, "duration_s": 2.0})
    await bus.emit(EventType.TRANSCRIPT_READY,
                   {"text": user, "language": "hi-IN", "stt_latency_s": stt})
    await bus.emit(EventType.LLM_FINISHED,
                   {"text": reply, "latency_first_token_s": ttft, "latency_total_s": total})
    # PLAYBACK_STARTED carries no latency field; response time is derived from
    # its timestamp minus SPEECH_ENDED, so it must be emitted to be measured.
    await bus.emit(EventType.PLAYBACK_STARTED, {})
    await bus.emit(EventType.PLAYBACK_FINISHED, {"turn_id": 1, "sentences_spoken": 2})


async def test_transcript_is_assembled_in_order():
    bus = EventBus()
    rec = CallRecorder(bus, direction="outbound", number="+919848000000",
                       campaign="rajasthan_hindi")
    await _turn(bus, "नमस्ते", "नमस्ते जी, कैसे हैं आप?")
    await _turn(bus, "ठीक हूँ", "बहुत बढ़िया।")
    record = rec.finish()

    assert [t["role"] for t in record.transcript] == [
        "user", "assistant", "user", "assistant"]
    assert record.transcript[0]["text"] == "नमस्ते"
    assert record.transcript[0]["language"] == "hi-IN"
    assert record.turns == 2
    assert record.direction == "outbound"
    assert record.campaign == "rajasthan_hindi"
    assert record.status == "completed"


async def test_latency_series_are_summarised():
    bus = EventBus()
    rec = CallRecorder(bus)
    await _turn(bus, "a", "b", stt=0.2, ttft=0.10, total=0.5)
    await _turn(bus, "c", "d", stt=0.6, ttft=0.30, total=1.5)
    record = rec.finish()

    assert record.latency_s["stt_s"]["count"] == 2
    assert record.latency_s["stt_s"]["max"] == 0.6
    assert record.latency_s["llm_ttft_s"]["p50"] in (0.10, 0.30)
    # response_s is derived from event timestamps, so it exists but its value
    # depends on wall-clock; only its presence is asserted.
    assert record.latency_s["response_s"]["count"] == 2


async def test_barge_ins_and_errors_are_counted():
    bus = EventBus()
    rec = CallRecorder(bus)
    await bus.emit(EventType.BARGE_IN, {})
    await bus.emit(EventType.ERROR, {"stage": "tts"})
    await bus.emit(EventType.ERROR, {"stage": "tts"})
    record = rec.finish(ended_reason="customer-ended-call")

    assert record.barge_ins == 1
    assert record.errors == {"tts": 2}
    assert record.ended_reason == "customer-ended-call"


async def test_store_round_trips_newest_first(tmp_path):
    store = CallStore(str(tmp_path / "calls.jsonl"))
    assert store.records() == []          # missing file is not an error

    bus = EventBus()
    first = CallRecorder(bus, store, call_id="aaa").finish()
    second = CallRecorder(EventBus(), store, call_id="bbb").finish()

    rows = store.records()
    assert [r["call_id"] for r in rows] == ["bbb", "aaa"]   # newest first
    assert store.get("aaa")["call_id"] == first.call_id
    assert store.get("missing") is None
    assert len(store.records(limit=1)) == 1

    # One JSON object per line, so a crash cannot corrupt earlier calls.
    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2 and all(json.loads(ln) for ln in lines)
    assert second.call_id == "bbb"


def test_cost_scales_with_duration_and_turns():
    cheap = estimate_call_cost(60, 4, stt_provider="sarvam",
                               tts_provider="cartesia", llm_model="qwen/qwen3.6-27b")
    longer = estimate_call_cost(120, 8, stt_provider="sarvam",
                                tts_provider="cartesia", llm_model="qwen/qwen3.6-27b")

    assert longer["total_usd"] > cheap["total_usd"]
    assert cheap["stt_usd"] > 0 and cheap["tts_usd"] > 0 and cheap["llm_usd"] > 0
    assert cheap["estimated"] is True


def test_local_providers_are_free():
    cost = estimate_call_cost(60, 4, stt_provider="whisper",
                              tts_provider="piper", llm_model="qwen/qwen3.6-27b")
    assert cost["stt_usd"] == 0.0
    assert cost["tts_usd"] == 0.0
    assert cost["llm_usd"] > 0          # the LLM is still hosted


def test_unknown_model_falls_back_to_the_default_rate():
    known = estimate_call_cost(60, 4, stt_provider="sarvam", tts_provider="cartesia",
                               llm_model="something-nobody-has-heard-of")
    assert known["llm_usd"] > 0
