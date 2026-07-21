"""Per-call instrumentation for the live /ws/call loop.

Two sinks, one object:
  - console (always on): per-turn events, decisions, timings, summaries;
  - JSONL frame trace (only when enabled): one row per frame + per event,
    written to debug_traces/<call_id>.jsonl for offline analysis.

Enable the frame trace with `?debug=1` on /live (carried in the WS config) or
VOICEOS_TRACE=1. A final CALL END line is always emitted, trace on or off.

No new dependencies — stdlib only. Level/RMS numbers are computed by the caller
(which already holds the audio) and passed in.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class CallTrace:
    def __init__(self, call_id: str, *, enabled: bool, trace_dir: str = "debug_traces") -> None:
        self.call_id = call_id
        self.turn = 0
        self._t0 = time.monotonic()
        self._fh = None
        self._errors = 0
        self._resp_ms: list[float] = []
        # per-utterance RMS accumulator (console summary, always on)
        self._utt_rms: list[float] = []
        # per-playback barge-in accumulators
        self._barge_vads: list[float] = []
        self._barge_corrs: list[float] = []
        self._barge_max_ms = 0.0
        # first-audio time of the reply in flight (ms since call start)
        self.tts_first_ms: float | None = None
        if enabled:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            self._fh = open(Path(trace_dir) / f"{call_id}.jsonl", "w", encoding="utf-8")

    def now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000

    def _write(self, obj: dict) -> None:
        if self._fh:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fh.flush()

    # -- events (console + JSONL) --------------------------------------------
    def event(self, stage: str, msg: str = "", **fields) -> None:
        t = self.now_ms()
        line = f"[{self.call_id[:6]} +{t:.0f}ms] t{self.turn} {stage}"
        if msg:
            line += f" {msg}"
        if fields:
            line += " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
        logger.info(line)
        self._write({"call": self.call_id, "turn": self.turn, "t": round(t),
                     "stage": stage, "msg": msg, **fields})

    def error(self, stage: str, msg: str) -> None:
        self._errors += 1
        self.event("ERROR", f"{stage}: {msg}")

    # -- frames (JSONL only; no-op when trace is off) ------------------------
    def frame(self, stage: str, **fields) -> None:
        if not self._fh:
            return
        self._write({"call": self.call_id, "turn": self.turn,
                     "t": round(self.now_ms()), "stage": stage, **fields})

    # -- per-utterance level summary -----------------------------------------
    def utt_reset(self) -> None:
        self._utt_rms = []

    def utt_add_rms(self, rms_db: float) -> None:
        self._utt_rms.append(rms_db)

    def utt_summary(self) -> dict:
        if not self._utt_rms:
            return {}
        return {"frames": len(self._utt_rms),
                "rms_med": round(statistics.median(self._utt_rms), 1),
                "rms_max": round(max(self._utt_rms), 1)}

    # -- barge-in diagnostics -------------------------------------------------
    def barge_reset(self) -> None:
        self._barge_vads = []
        self._barge_corrs = []
        self._barge_max_ms = 0.0

    def barge_observe(self, vad: float, corr: float, barge_ms: float) -> None:
        self._barge_vads.append(vad)
        self._barge_corrs.append(corr)
        self._barge_max_ms = max(self._barge_max_ms, barge_ms)
        self.frame("barge", vad=round(vad, 3), corr=round(corr, 3), barge_ms=round(barge_ms))

    def barge_fired(self, vad: float, corr: float, barge_ms: float) -> None:
        self.event("BARGE-IN", "FIRED", vad=round(vad, 2), corr=round(corr, 2),
                   barge_ms=round(barge_ms))

    def barge_summary(self, need_ms: float) -> None:
        if not self._barge_vads:
            return
        fired = self._barge_max_ms >= need_ms
        self.event("barge-eval",
                   "fired" if fired else f"NEVER FIRED (closest {self._barge_max_ms:.0f}ms, need {need_ms:.0f})",
                   vad_med=round(statistics.median(self._barge_vads), 2),
                   corr_med=round(statistics.median(self._barge_corrs), 2),
                   windows=len(self._barge_vads))

    # -- response latency + final summary ------------------------------------
    def response(self, latency_ms: float) -> None:
        self._resp_ms.append(latency_ms)
        self.event("RESPONSE", ms=round(latency_ms))

    def call_end(self, *, duration_s: float, turns: int, end_reason: str) -> None:
        avg = round(statistics.mean(self._resp_ms)) if self._resp_ms else None
        logger.info("[%s] CALL END | %.1fs | turns=%d | avg_response=%sms | errors=%d | end=%s",
                    self.call_id[:6], duration_s, turns, avg, self._errors, end_reason)
        self._write({"call": self.call_id, "stage": "call_end", "duration_s": duration_s,
                     "turns": turns, "avg_response_ms": avg, "errors": self._errors,
                     "end_reason": end_reason})
        if self._fh:
            self._fh.close()
            self._fh = None
