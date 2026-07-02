import logging

from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.metrics import LatencyMonitor


async def test_full_turn_produces_timing_log(caplog):
    bus = EventBus()
    LatencyMonitor(bus)

    with caplog.at_level(logging.INFO, logger="voiceos.pipeline.metrics"):
        await bus.emit(EventType.SPEECH_ENDED, {"accepted": True, "duration_s": 2.0})
        await bus.emit(EventType.TRANSCRIPT_READY, {"text": "hi"})
        await bus.emit(
            EventType.LLM_FINISHED,
            {"turn_id": 1, "latency_first_token_s": 0.25, "latency_total_s": 1.1},
        )
        await bus.emit(EventType.TTS_STARTED, {"text": "hello"})
        await bus.emit(EventType.PLAYBACK_STARTED, {})
        await bus.emit(EventType.PLAYBACK_FINISHED, {"turn_id": 1})

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "turn 1 timings" in message
    assert "listen 2.00s" in message
    assert "llm first token 0.25s" in message
    assert "response" in message and "total" in message


async def test_rejected_utterance_does_not_log(caplog):
    bus = EventBus()
    LatencyMonitor(bus)

    with caplog.at_level(logging.INFO, logger="voiceos.pipeline.metrics"):
        await bus.emit(EventType.SPEECH_ENDED, {"accepted": False, "duration_s": 0.1})
        await bus.emit(EventType.PLAYBACK_FINISHED, {"turn_id": 0})

    assert not caplog.records


async def test_incomplete_turn_logs_dashes_not_crash(caplog):
    bus = EventBus()
    LatencyMonitor(bus)

    with caplog.at_level(logging.INFO, logger="voiceos.pipeline.metrics"):
        # STT produced nothing; an error path skipped LLM/TTS entirely,
        # but playback still closed the turn via EndOfTurn.
        await bus.emit(EventType.SPEECH_ENDED, {"accepted": True, "duration_s": 1.0})
        await bus.emit(EventType.PLAYBACK_FINISHED, {"turn_id": 2})

    assert len(caplog.records) == 1
    assert "llm first token -" in caplog.records[0].getMessage()
