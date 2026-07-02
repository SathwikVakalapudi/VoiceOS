"""Local audio transport — system microphone in, system speaker out.

This is the default (Phase 1) transport: one mic + one speaker on the
machine running VoiceOS.
"""

from __future__ import annotations

from voiceos.audio.microphone import Microphone
from voiceos.audio.speaker import Speaker
from voiceos.config.settings import AudioSettings
from voiceos.transport.base import AudioSink, AudioSource, AudioTransport


class LocalAudioTransport(AudioTransport):
    def __init__(self, settings: AudioSettings) -> None:
        self._microphone = Microphone(settings)
        self._speaker = Speaker(settings)

    @property
    def source(self) -> AudioSource:
        return self._microphone

    @property
    def sink(self) -> AudioSink:
        return self._speaker
