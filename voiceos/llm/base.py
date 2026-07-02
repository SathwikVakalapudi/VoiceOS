"""Re-export of the LLM contract for implementations in this package."""

from voiceos.interfaces.llm import BaseLLM, Message

__all__ = ["BaseLLM", "Message"]
