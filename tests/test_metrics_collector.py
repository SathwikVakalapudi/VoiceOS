"""MetricsCollector tests: aggregate events into a dashboard snapshot."""

from voiceos.monitoring.collector import MetricsCollector, _percentile
from voiceos.pipeline.events import EventBus, EventType


def test_percentile_nearest_rank():
    assert _percentile([], 50) == 0.0
    assert _percentile([1.0], 95) == 1.0
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([1, 2, 3, 4, 5], 100) == 5


async def emit_turn(bus, *, stt, ttft, total, response_gap):
    # A turn's worth of events, timestamps faked via a tiny event sequence.
    await bus.emit(EventType.SPEECH_ENDED, {"accepted": True})
    await bus.emit(EventType.TRANSCRIPT_READY, {"stt_latency_s": stt})
    await bus.emit(EventType.LLM_FINISHED, {"latency_first_token_s": ttft, "latency_total_s": total})
    await bus.emit(EventType.PLAYBACK_STARTED, {})
    await bus.emit(EventType.PLAYBACK_FINISHED, {"turn_id": 1})


async def test_counts_turns_barge_ins_tools_and_errors():
    bus = EventBus()
    collector = MetricsCollector(bus)

    await emit_turn(bus, stt=0.3, ttft=0.2, total=0.9, response_gap=0.5)
    await emit_turn(bus, stt=0.5, ttft=0.4, total=1.1, response_gap=0.7)
    await bus.emit(EventType.BARGE_IN, {})
    await bus.emit(EventType.TOOL_CALLED, {"name": "get_current_time"})
    await bus.emit(EventType.ERROR, {"stage": "tts"})
    await bus.emit(EventType.ERROR, {"stage": "tts"})

    snap = collector.snapshot()
    assert snap["turns"] == 2
    assert snap["barge_ins"] == 1
    assert snap["tool_calls"] == 1
    assert snap["errors"] == {"tts": 2}


async def test_latency_percentiles_present():
    bus = EventBus()
    collector = MetricsCollector(bus)
    await emit_turn(bus, stt=0.3, ttft=0.2, total=0.9, response_gap=0.5)

    snap = collector.snapshot()
    assert snap["latency_s"]["stt_s"]["count"] == 1
    assert snap["latency_s"]["stt_s"]["p50"] == 0.3
    assert snap["latency_s"]["llm_total_s"]["max"] == 0.9
