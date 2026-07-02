"""Session manager — run and track multiple voice pipelines at once.

Phase 1 runs a single local session (`main.py`). Multi-user service means
many concurrent sessions, each its own `VoicePipeline`. This manager owns
their lifecycle; it is factory-based so the pipeline construction (and its
transport) is chosen by the caller and so it can be tested without audio
hardware.

Note: concurrent *local* sessions all contend for the one microphone —
real multi-tenancy pairs each session with its own transport (e.g. a
telephony call), which is exactly why audio I/O was moved behind
`AudioTransport`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

# A session is anything with async start()/stop() — VoicePipeline in practice,
# a fake in tests. Kept generic so this module needs no pipeline import.
Session = TypeVar("Session")
SessionFactory = Callable[[], "Session"]


class SessionManager(Generic[Session]):
    def __init__(self, factory: SessionFactory) -> None:
        self._factory = factory
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, session_id: str) -> Session:
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"session already exists: {session_id}")
            session = self._factory()
            self._sessions[session_id] = session
        # start() outside the lock so a slow model load doesn't block others.
        await session.start()  # type: ignore[attr-defined]
        logger.info("session started: %s (%d active)", session_id, len(self._sessions))
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        return list(self._sessions)

    async def stop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            await session.stop()  # type: ignore[attr-defined]
        finally:
            logger.info("session stopped: %s (%d active)", session_id, len(self._sessions))

    async def stop_all(self) -> None:
        for session_id in list(self._sessions):
            await self.stop(session_id)

    def __len__(self) -> int:
        return len(self._sessions)
