"""Conversation history with turn-based trimming."""

from __future__ import annotations

from voiceos.interfaces.llm import Message


class ConversationHistory:
    def __init__(self, max_turns: int = 20) -> None:
        self._max_messages = max_turns * 2  # one turn = user + assistant
        self._messages: list[Message] = []

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self) -> None:
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages = []
