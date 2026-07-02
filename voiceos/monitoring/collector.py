"""Metrics collector.

Subscribes to the EventBus and accumulates per-turn latency samples and
event counts across a session, exposing a JSON-able snapshot (turn count,
errors by stage, barge-ins, and p50/p95/max for the key latencies). This
is the aggregation the dashboard reads — the analytics module the
`metrics.py` docstring anticipated.
"""

from __future__ import annotations

from voiceos.pipeline.events import Event, EventBus, EventType


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of a non-empty list (pct in [0, 100])."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


class MetricsCollector:
    # Latencies accumulated across turns, keyed by snapshot name -> event field.
    _SAMPLES = {
        "response_s": None,       # derived: playback_start - speech_ended
        "stt_s": "stt_latency_s",
        "llm_ttft_s": "latency_first_token_s",
        "llm_total_s": "latency_total_s",
    }

    def __init__(self, event_bus: EventBus) -> None:
        self._samples: dict[str, list[float]] = {k: [] for k in self._SAMPLES}
        self._turns = 0
        self._barge_ins = 0
        self._tool_calls = 0
        self._errors: dict[str, int] = {}
        self._turn: dict[str, float] = {}
        event_bus.subscribe(None, self._on_event)

    def _on_event(self, event: Event) -> None:
        if event.type is EventType.SPEECH_ENDED and event.data.get("accepted"):
            self._turn = {"speech_ended": event.timestamp}
        elif event.type is EventType.TRANSCRIPT_READY:
            self._record("stt_s", event.data.get("stt_latency_s"))
        elif event.type is EventType.LLM_FINISHED:
            self._record("llm_ttft_s", event.data.get("latency_first_token_s"))
            self._record("llm_total_s", event.data.get("latency_total_s"))
        elif event.type is EventType.PLAYBACK_STARTED and "speech_ended" in self._turn:
            self._record("response_s", event.timestamp - self._turn["speech_ended"])
        elif event.type is EventType.BARGE_IN:
            self._barge_ins += 1
        elif event.type is EventType.TOOL_CALLED:
            self._tool_calls += 1
        elif event.type is EventType.ERROR:
            stage = event.data.get("stage", "unknown")
            self._errors[stage] = self._errors.get(stage, 0) + 1
        elif event.type is EventType.PLAYBACK_FINISHED and "speech_ended" in self._turn:
            self._turns += 1
            self._turn = {}

    def _record(self, name: str, value: float | None) -> None:
        if value is not None:
            self._samples[name].append(float(value))

    def snapshot(self) -> dict:
        latency = {
            name: {
                "count": len(vals),
                "p50": round(_percentile(vals, 50), 3),
                "p95": round(_percentile(vals, 95), 3),
                "max": round(max(vals), 3) if vals else 0.0,
            }
            for name, vals in self._samples.items()
        }
        return {
            "turns": self._turns,
            "barge_ins": self._barge_ins,
            "tool_calls": self._tool_calls,
            "errors": dict(self._errors),
            "latency_s": latency,
        }
