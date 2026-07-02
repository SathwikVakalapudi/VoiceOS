"""Abstract interfaces for every pluggable stage.

The pipeline only ever talks to these — never to a concrete model.
Swap Whisper for Parakeet, Qwen for Llama, Svara for anything else:
the pipeline never changes.
"""

from voiceos.interfaces.llm import BaseLLM, Message
from voiceos.interfaces.stt import BaseSTT, TranscriptionResult
from voiceos.interfaces.tts import BaseTTS
from voiceos.interfaces.vad import BaseVAD

__all__ = [
    "BaseLLM",
    "BaseSTT",
    "BaseTTS",
    "BaseVAD",
    "Message",
    "TranscriptionResult",
]
