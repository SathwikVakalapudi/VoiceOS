"""Live microphone level meter — watch RMS/peak dBFS while you adjust gain.

    python mic_meter.py            # default input device
    python mic_meter.py --device 1 # pick one; --list to see them

Prints an updating RMS reading (with a bar) plus peak, ~7x/second, so you can
drag the Windows "Microphone Boost" slider and watch the level climb into the
healthy -15..-10 dBFS band in real time. Ctrl+C to stop.

Local only — no browser, no network, no API calls.
"""

import argparse
import sys

import numpy as np
import sounddevice as sd

SR = 16000
BLOCK = 2400              # ~150 ms windows
WIDTH, LO, HI = 40, -60, 0   # bar covers -60..0 dBFS

p = argparse.ArgumentParser()
p.add_argument("--device", type=int, default=None)
p.add_argument("--list", action="store_true")
args = p.parse_args()

if args.list:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i}: {sd.query_hostapis(d['hostapi'])['name']} | {d['name']}")
    sys.exit()


def dbfs(x: np.ndarray, reducer) -> float:
    val = reducer(x)
    return 20 * np.log10(val) if val > 1e-9 else -99.0


def col(db: float) -> int:
    return max(0, min(WIDTH, int((db - LO) / (HI - LO) * WIDTH)))


def verdict(rms_db: float, peak_db: float) -> str:
    if peak_db >= -1:
        return "CLIPPING — lower the gain"
    if rms_db <= -45:
        return "silent? — check you picked the right device"
    if rms_db >= -15:
        return "HEALTHY"
    if rms_db >= -20:
        return "a little low"
    return "TOO QUIET — raise the mic boost"


# Header: mark where the healthy band sits on the same scale as the live bar.
marker = [" "] * (WIDTH + 1)
for c in range(col(-15), col(-10) + 1):
    marker[c] = "v"
print("Speak normally and adjust the mic boost. Aim RMS into the 'v' band. Ctrl+C to stop.\n")
print(" " * 11 + "".join(marker) + "  (-15..-10 dBFS)")

try:
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        device=args.device, blocksize=BLOCK) as stream:
        while True:
            data, _overflow = stream.read(BLOCK)
            x = data.reshape(-1).astype(np.float32) / 32768.0
            rms_db = dbfs(x, lambda a: float(np.sqrt(np.mean(a ** 2))))
            peak_db = dbfs(x, lambda a: float(np.abs(a).max()))
            filled = col(rms_db)
            bar = "#" * filled + " " * (WIDTH - filled)
            line = f"RMS {rms_db:5.0f} [{bar}] peak {peak_db:4.0f}  {verdict(rms_db, peak_db)}"
            print("\r" + line.ljust(90), end="", flush=True)
except KeyboardInterrupt:
    print("\nstopped.")
