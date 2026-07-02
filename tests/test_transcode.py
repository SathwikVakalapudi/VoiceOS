"""Telephony transcoding tests: G.711 mu-law <-> PCM and 8k<->16k/24k."""

import numpy as np

from voiceos.telephony.transcode import (
    Resampler,
    TelephonyDecoder,
    TelephonyEncoder,
    pcm16_to_ulaw,
    ulaw_to_pcm16,
)


def test_mulaw_encode_is_stable_under_decode_reencode():
    # mu-law is lossy (and has a +-0 code ambiguity), but once a signal is
    # encoded, decoding and re-encoding is a fixed point.
    pcm = (np.sin(np.linspace(0, 40, 800)) * 10000).astype("<i2").tobytes()
    once = pcm16_to_ulaw(pcm)
    twice = pcm16_to_ulaw(ulaw_to_pcm16(once))
    assert once == twice


def test_ulaw_decode_doubles_byte_width():
    # 160 mu-law bytes (20 ms @ 8 kHz) -> 160 int16 samples = 320 bytes.
    assert len(ulaw_to_pcm16(b"\x00" * 160)) == 320


def test_resampler_8k_to_16k_roughly_doubles_samples():
    pcm8 = np.zeros(160, dtype="<i2").tobytes()  # 20 ms @ 8k = 160 samples
    out = Resampler(8000, 16000).process(pcm8)
    assert 300 <= len(out) // 2 <= 340  # ~320 samples @ 16k


def test_decoder_inbound_mulaw_8k_to_int16_16k():
    dec = TelephonyDecoder(target_rate=16000, encoding="mulaw")
    frame = dec.decode(b"\x00" * 160)  # one 20 ms telephony frame
    assert frame.dtype == np.dtype("<i2")
    assert 300 <= len(frame) <= 340


def test_encoder_outbound_24k_int16_to_mulaw_8k():
    enc = TelephonyEncoder(source_rate=24000, encoding="mulaw")
    audio = np.zeros(480, dtype="<i2")  # 20 ms @ 24k = 480 samples
    out = enc.encode(audio)
    assert 150 <= len(out) <= 170  # ~160 mu-law bytes @ 8k


def test_stateful_resampling_is_continuous_across_chunks():
    # Two half-frames through one resampler ~= the whole frame at once.
    r1 = Resampler(8000, 16000)
    whole = r1.process(np.zeros(160, dtype="<i2").tobytes())
    r2 = Resampler(8000, 16000)
    part = r2.process(np.zeros(80, dtype="<i2").tobytes())
    part += r2.process(np.zeros(80, dtype="<i2").tobytes())
    assert abs(len(whole) - len(part)) <= 4  # within a couple samples


def test_pcm16_passthrough_encoding():
    dec = TelephonyDecoder(target_rate=8000, encoding="pcm16")
    pcm = np.arange(160, dtype="<i2").tobytes()
    frame = dec.decode(pcm)
    assert np.array_equal(frame, np.arange(160, dtype="<i2"))
