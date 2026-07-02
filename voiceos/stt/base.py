"""Re-export of the STT contract for implementations in this package."""

from voiceos.interfaces.stt import BaseSTT, TranscriptionResult

__all__ = ["BaseSTT", "TranscriptionResult"]
