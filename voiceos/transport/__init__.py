"""Audio transports.

A transport is where audio comes from and goes to. Phase 1 is a local
microphone + speaker; a telephony/WebRTC transport plugs in here without
the pipeline changing, because the pipeline only ever talks to the
`AudioSource` / `AudioSink` contracts.
"""

from voiceos.transport.base import AudioSink, AudioSource, AudioTransport
from voiceos.transport.local import LocalAudioTransport

__all__ = ["AudioSink", "AudioSource", "AudioTransport", "LocalAudioTransport"]
