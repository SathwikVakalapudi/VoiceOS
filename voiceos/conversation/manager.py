"""Conversation manager.

Owns history + context and builds the message list for the LLM.
The LLM worker never touches history internals directly.

Supports campaigns: a JSON file with a custom system prompt and an
opening line the assistant speaks first (outbound-call style, like
Vapi's firstMessage).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from voiceos.config.settings import ConversationSettings
from voiceos.conversation.context import ConversationContext
from voiceos.conversation.history import ConversationHistory
from voiceos.interfaces.llm import Message
from voiceos.llm.prompts import DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, settings: ConversationSettings) -> None:
        system_prompt = settings.system_prompt
        first_message = settings.first_message
        error_message = settings.error_message
        if settings.campaign_file:
            campaign = json.loads(
                Path(settings.campaign_file).read_text(encoding="utf-8")
            )
            system_prompt = campaign.get("system_prompt") or system_prompt
            first_message = campaign.get("first_message") or first_message
            error_message = campaign.get("error_message") or error_message
            logger.info(
                "campaign loaded: %s%s",
                settings.campaign_file,
                " (with first message)" if first_message else "",
            )
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.first_message = first_message
        self.error_message = error_message
        self.history = ConversationHistory(max_turns=settings.max_turns)
        self.context = ConversationContext()
        # {"turn_id": int, "segments": [str, ...]} for the reply in flight.
        self._pending: dict | None = None
        if self.first_message:
            # The greeting is part of the conversation the LLM must see.
            self.history.add_assistant(self.first_message)

    def build_messages(self, user_text: str) -> list[Message]:
        """Record the user turn and return the full prompt for the LLM."""
        self.history.add_user(user_text)
        return [
            {"role": "system", "content": self._system_prompt},
            *self.history.messages,
        ]

    def begin_assistant(self, turn_id: int) -> None:
        """Open a reply. Sentences are staged, not recorded, until the
        turn is committed with the count actually spoken."""
        self._pending = {"turn_id": turn_id, "segments": []}

    def add_pending_segment(self, turn_id: int, text: str) -> None:
        """Stage one sentence that was sent to TTS for this turn."""
        if self._pending and self._pending["turn_id"] == turn_id and text:
            self._pending["segments"].append(text)

    def commit_assistant(self, turn_id: int, spoken_segments: int | None = None) -> None:
        """Record only the sentences the user actually heard.

        ``spoken_segments`` caps how many staged sentences are written —
        on barge-in it is the number fully played; ``None`` records all.
        Idempotent: the first call for a turn clears the pending buffer,
        so a later duplicate (barge-in racing end-of-turn) is a no-op.
        """
        pending = self._pending
        if not pending or pending["turn_id"] != turn_id:
            return
        segments = pending["segments"]
        if spoken_segments is not None:
            segments = segments[: max(0, spoken_segments)]
        text = " ".join(segments).strip()
        self._pending = None
        if text:
            self.history.add_assistant(text)

    def reset(self) -> None:
        self.history.clear()
        self._pending = None
        self.context = ConversationContext()
