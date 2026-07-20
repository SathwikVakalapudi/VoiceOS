"""Does the microphone produce decodable speech? Nothing else.

    python mic_test.py            # default input device
    python mic_test.py --device 7 # pick one; --list to see them

Records from the sound card, reports level, transcribes with local Whisper.
No browser, no network, no API calls.
"""

import argparse
import sys

import numpy as np
import sounddevice as sd

SR = 16000

p = argparse.ArgumentParser()
p.add_argument("--device", type=int, default=None)
p.add_argument("--seconds", type=float, default=4.0)
p.add_argument("--model", default="small")
p.add_argument("--language", default="en")
p.add_argument("--list", action="store_true")
args = p.parse_args()

if args.list:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i}: {d['name']}")
    sys.exit()

print(f"Recording {args.seconds:.0f}s — speak now…")
pcm = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1,
             dtype="int16", device=args.device)
sd.wait()
pcm = pcm.reshape(-1)

peak_db = 20 * np.log10(max(1, np.abs(pcm).max()) / 32768)
spec = np.abs(np.fft.rfft(pcm.astype(float) * np.hanning(len(pcm))))
freqs = np.fft.rfftfreq(len(pcm), 1 / SR)
voice_pct = spec[(freqs >= 1000) & (freqs < 3000)].sum() / spec.sum() * 100

print(f"peak {peak_db:.0f} dBFS (want > -30) | 1-3 kHz {voice_pct:.0f}% (want > 15)")

from faster_whisper import WhisperModel  # noqa: E402  slow import, only if recording worked

segments, _ = WhisperModel(args.model, device="auto").transcribe(
    pcm.astype(np.float32) / 32768.0, language=args.language,
    condition_on_previous_text=False,
)
print("transcript:", " ".join(s.text.strip() for s in segments).strip() or "(nothing)")
