"""Campaign test sandbox.

Lets you *test* a campaign as a text chat — the same persona (system prompt +
first message) and LLM the phone calls use, without any audio/telephony. Each
browser session gets its own ConversationManager so multi-turn state (including
the spoken-first greeting) behaves exactly like a real call's brain.

The LLM is built once via an injected factory and shared across sessions (this
is a low-traffic testing tool), so tests can pass a fake.
"""

from __future__ import annotations

import uuid
from typing import Callable

from voiceos.config.settings import ConversationSettings
from voiceos.conversation.manager import ConversationManager
from voiceos.dashboard.store import CampaignStore
from voiceos.interfaces.llm import BaseLLM


class TestSandbox:
    def __init__(self, store: CampaignStore, *, llm_factory: Callable[[], BaseLLM]) -> None:
        self._store = store
        self._llm_factory = llm_factory
        self._llm: BaseLLM | None = None
        self._sessions: dict[str, ConversationManager] = {}

    async def _llm_instance(self) -> BaseLLM:
        if self._llm is None:
            self._llm = self._llm_factory()
            await self._llm.load()
        return self._llm

    def start(self, name: str) -> dict:
        """Begin a test conversation for a campaign. Returns session id + greeting."""
        self._store.get(name)  # raises KeyError if the campaign doesn't exist
        settings = ConversationSettings(campaign_file=self._store.path_for(name))
        manager = ConversationManager(settings)
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = manager
        return {"session_id": session_id, "first_message": manager.first_message}

    async def message(self, session_id: str, text: str) -> str:
        """Send a user turn, get the assistant's reply (and record it)."""
        manager = self._sessions[session_id]  # KeyError -> unknown session
        messages = manager.build_messages(text)
        reply = await (await self._llm_instance()).complete(messages)
        content = reply.get("content", "") if reply else ""
        manager.history.add_assistant(content)
        return content

    def end(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def close(self) -> None:
        if self._llm is not None:
            await self._llm.close()
            self._llm = None
