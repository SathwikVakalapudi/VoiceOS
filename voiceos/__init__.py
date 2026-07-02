"""VoiceOS — an open-source, self-hosted, modular voice AI engine.

Phase 1 pipeline:

    Mic -> VAD -> STT -> Conversation Manager -> LLM -> TTS -> Speaker

Every stage is an independent async worker connected by queues and
coordinated through an event bus. No module performs another module's job.
"""

__version__ = "0.1.0"
