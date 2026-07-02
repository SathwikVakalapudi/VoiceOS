from voiceos.pipeline.events import EventBus, EventType


async def test_sync_and_async_handlers_both_fire():
    bus = EventBus()
    seen: list[str] = []

    def sync_handler(event):
        seen.append(f"sync:{event.type.value}")

    async def async_handler(event):
        seen.append(f"async:{event.type.value}")

    bus.subscribe(EventType.TRANSCRIPT_READY, sync_handler)
    bus.subscribe(EventType.TRANSCRIPT_READY, async_handler)
    await bus.emit(EventType.TRANSCRIPT_READY, {"text": "hi"})

    assert seen == ["sync:transcript_ready", "async:transcript_ready"]


async def test_wildcard_subscription_sees_everything():
    bus = EventBus()
    seen: list[EventType] = []
    bus.subscribe(None, lambda e: seen.append(e.type))

    await bus.emit(EventType.SPEECH_STARTED)
    await bus.emit(EventType.SPEECH_ENDED)

    assert seen == [EventType.SPEECH_STARTED, EventType.SPEECH_ENDED]


async def test_failing_handler_does_not_break_others():
    bus = EventBus()
    seen = []

    def bad_handler(event):
        raise RuntimeError("boom")

    bus.subscribe(EventType.ERROR, bad_handler)
    bus.subscribe(EventType.ERROR, lambda e: seen.append(e))
    await bus.emit(EventType.ERROR)

    assert len(seen) == 1
