"""Offline mic-health analysis over debug_audio/*.wav (no API key, no mic).

These are committed STT utterances — audio that already passed the VAD gate —
so this measures the level of speech that got through, not the whole mic stream.
"""
from __future__ import annotations

import glob
import sys
import wave

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

FILES = sorted(glob.glob("debug_audio/*.wav"))
FLOOR_DB = -99.0


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x / 32768.0, sr


def dbfs(rms):
    return 20 * np.log10(rms) if rms > 1e-9 else FLOOR_DB


all_frame_db = []          # per-frame RMS dBFS, pooled across every file
file_peak_db = []          # per-file peak dBFS
file_med_frame_db = []     # per-file median frame dBFS
durations = []

for path in FILES:
    x, sr = read_wav(path)
    if x.size == 0:
        continue
    durations.append(x.size / sr)
    fsize = int(0.032 * sr)          # ~32 ms frames, matches the pipeline
    frames = [x[i:i + fsize] for i in range(0, len(x) - fsize + 1, fsize)]
    if not frames:
        frames = [x]
    fdb = [dbfs(float(np.sqrt(np.mean(f ** 2)))) for f in frames]
    all_frame_db.extend(fdb)
    file_peak_db.append(dbfs(float(np.abs(x).max())))
    file_med_frame_db.append(float(np.median(fdb)))

fd = np.array(all_frame_db)
print(f"files analyzed : {len(durations)}")
print(f"total audio    : {sum(durations):.1f} s")
print(f"frames pooled  : {fd.size}  (~32 ms each)")

print("\n=== per-FRAME RMS dBFS (pooled across all files) ===")
for label, p in [("min", 0), ("p10", 10), ("p25", 25), ("median", 50),
                 ("p75", 75), ("p90", 90), ("max", 100)]:
    print(f"  {label:>6}: {np.percentile(fd, p):6.1f} dBFS")
print(f"  frames < -30 dBFS : {100 * np.mean(fd < -30):.0f}%")
print(f"  frames < -20 dBFS : {100 * np.mean(fd < -20):.0f}%")
print(f"  frames in -15..-10: {100 * np.mean((fd >= -15) & (fd <= -10)):.0f}%")

pk = np.array(file_peak_db)
md = np.array(file_med_frame_db)
print("\n=== per-FILE peak dBFS ===")
print(f"  median peak: {np.median(pk):.1f} | p10 {np.percentile(pk,10):.1f} | "
      f"p90 {np.percentile(pk,90):.1f} dBFS")
print("=== per-FILE median-frame dBFS (typical speaking level in each clip) ===")
print(f"  median: {np.median(md):.1f} | p10 {np.percentile(md,10):.1f} | "
      f"p90 {np.percentile(md,90):.1f} dBFS")
print(f"  files whose median frame is healthy (-15..-10): {100*np.mean((md>=-15)&(md<=-10)):.0f}%")
print(f"  files whose median frame is weak (< -20)      : {100*np.mean(md < -20):.0f}%")

# Optional: Silero VAD speech-frame fraction (local model, no key).
try:
    from voiceos.config.settings import get_settings
    from voiceos.vad.silero_vad import SileroVAD
    import asyncio

    settings = get_settings()
    vad = SileroVAD(settings.vad)
    asyncio.get_event_loop().run_until_complete(vad.load())
    speech = total = 0
    speech_db = []          # dBFS of frames the VAD calls speech
    for path in FILES:
        x, sr = read_wav(path)
        vad.reset()
        for i in range(0, len(x) - 512 + 1, 512):
            frame = x[i:i + 512].astype(np.float32)
            p = vad.process(frame, 16000)
            total += 1
            if p >= settings.vad.threshold:
                speech += 1
                speech_db.append(dbfs(float(np.sqrt(np.mean(frame ** 2)))))
    print("\n=== Silero VAD over the same frames (threshold %.2f) ===" % settings.vad.threshold)
    print(f"  frames the VAD calls speech: {100*speech/max(1,total):.0f}%  ({speech}/{total})")
    sdb = np.array(speech_db)
    print("  dBFS of SPEECH-only frames (the real speaking level):")
    for label, pp in [("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90)]:
        print(f"    {label:>6}: {np.percentile(sdb, pp):6.1f} dBFS")
    print(f"    speech frames in healthy -15..-10: {100*np.mean((sdb>=-15)&(sdb<=-10)):.0f}%")
    print(f"    speech frames below -25 dBFS     : {100*np.mean(sdb < -25):.0f}%")
except Exception as exc:
    print(f"\n(VAD pass skipped: {type(exc).__name__}: {exc})")
