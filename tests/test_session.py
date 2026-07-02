"""SessionManager tests using a fake session (no pipeline/audio needed)."""

import pytest

from voiceos.pipeline.session import SessionManager


class FakeSession:
    def __init__(self):
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


async def test_create_starts_and_tracks_session():
    mgr = SessionManager(FakeSession)
    session = await mgr.create("call-1")
    assert session.started is True
    assert mgr.get("call-1") is session
    assert mgr.list_sessions() == ["call-1"]
    assert len(mgr) == 1


async def test_duplicate_session_id_rejected():
    mgr = SessionManager(FakeSession)
    await mgr.create("call-1")
    with pytest.raises(ValueError):
        await mgr.create("call-1")


async def test_stop_removes_and_stops_session():
    mgr = SessionManager(FakeSession)
    session = await mgr.create("call-1")
    await mgr.stop("call-1")
    assert session.stopped is True
    assert mgr.get("call-1") is None
    assert len(mgr) == 0


async def test_stop_all():
    mgr = SessionManager(FakeSession)
    await mgr.create("a")
    await mgr.create("b")
    await mgr.stop_all()
    assert len(mgr) == 0


async def test_concurrent_sessions_are_independent():
    mgr = SessionManager(FakeSession)
    a = await mgr.create("a")
    b = await mgr.create("b")
    assert a is not b
    assert set(mgr.list_sessions()) == {"a", "b"}
