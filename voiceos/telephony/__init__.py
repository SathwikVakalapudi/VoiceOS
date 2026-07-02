"""Telephony bridge — connect VoiceOS to the phone network via a media
server (FreeSWITCH/Asterisk) that streams call audio over a socket.

Telephony audio is 8 kHz G.711 (mu-law/PCMU); the VoiceOS pipeline runs
at 16 kHz in / 24 kHz out. This package owns the transcoding and the
per-call transport that bridges the two.
"""

from voiceos.telephony.transcode import TelephonyDecoder, TelephonyEncoder

__all__ = ["TelephonyDecoder", "TelephonyEncoder"]
