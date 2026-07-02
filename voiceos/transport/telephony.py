"""Telephony transport — DESIGN STUB (not yet implemented).

This is the seam where phone/WebRTC calls plug in. It is intentionally
not implemented: a working telephony layer needs an external media
provider (Twilio Media Streams, a SIP trunk via Telnyx/Plivo, or a
WebRTC SFU such as LiveKit) plus credentials and a public endpoint —
none of which can run from a local process alone.

To implement one, satisfy the same contracts the local transport does:

  AudioSource.start(loop, queue):
      Subscribe to the call's inbound media (e.g. Twilio's WebSocket media
      frames or an RTP stream). Decode to mono int16 at the pipeline input
      rate, wrap each frame with `make_frame`, and hand it to the loop with
      `loop.call_soon_threadsafe(queue.put_drop_oldest, frame)` — exactly
      as Microphone does. Telephony is usually 8 kHz µ-law, so resample to
      the configured input_sample_rate here.

  AudioSink.open / play / interrupt / resume / close:
      Encode outbound int16 audio to the call codec and write it to the
      media channel. `interrupt()` must stop the outbound stream promptly
      for barge-in, mirroring Speaker.

Because the pipeline only depends on AudioSource/AudioSink, wiring a real
implementation in requires no pipeline changes — pass it as the
`transport` argument to VoicePipeline. Per-call transports are also what
make real multi-tenancy work: each concurrent session gets its own
telephony transport instead of contending for one local device.
"""

from __future__ import annotations

from voiceos.transport.base import AudioTransport


class TelephonyTransport(AudioTransport):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Telephony transport is a design stub — see voiceos/transport/telephony.py. "
            "Implement AudioSource/AudioSink against a media provider "
            "(Twilio Media Streams, SIP via Telnyx/Plivo, or a WebRTC SFU)."
        )

    @property
    def source(self):  # pragma: no cover - stub
        raise NotImplementedError

    @property
    def sink(self):  # pragma: no cover - stub
        raise NotImplementedError
