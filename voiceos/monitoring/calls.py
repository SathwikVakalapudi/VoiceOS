"""Per-call records — the thing a Logs tab reads.

`ResultStore` saves survey answers; `MetricsCollector` aggregates latency across
a session. Neither keeps a per-call row, so there has been nothing to list: no
duration, no ended reason, no transcript, no cost.

`CallRecorder` subscribes to the EventBus and assembles one from events the
pipeline already emits, the same way `LatencyMonitor` and `MetricsCollector` do.
No worker knows it exists.

Storage is append-only JSONL, matching `survey/store.py`: one line per call, so
a crash mid-campaign cannot corrupt earlier records.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from voiceos.pipeline.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[rank], 3)


@dataclass
class CallRecord:
    call_id: str
    started_at: str
    ended_at: str | None = None
    duration_s: float = 0.0
    direction: str = "web"            # web | inbound | outbound
    number: str | None = None
    campaign: str | None = None
    assistant: str | None = None
    status: str = "in-progress"       # in-progress | completed | failed
    ended_reason: str | None = None
    turns: int = 0
    barge_ins: int = 0
    transcript: list[dict] = field(default_factory=list)
    latency_s: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class CallStore:
    """Append-only JSONL of finished calls."""

    def __init__(self, path: str = "results/calls.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: CallRecord | dict) -> None:
        payload = record.to_dict() if isinstance(record, CallRecord) else record
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def records(self, limit: int | None = None) -> list[dict]:
        """Newest first — which is how a log table wants them."""
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        rows.reverse()
        return rows[:limit] if limit else rows

    def get(self, call_id: str) -> dict | None:
        return next((r for r in self.records() if r.get("call_id") == call_id), None)


class CallRecorder:
    """Builds a CallRecord from the event stream of one call."""

    # Event field -> latency series. `response_s` is derived, not reported.
    _SERIES = {
        "stt_s": "stt_latency_s",
        "llm_ttft_s": "latency_first_token_s",
        "llm_total_s": "latency_total_s",
    }

    def __init__(
        self,
        event_bus: EventBus,
        store: CallStore | None = None,
        *,
        direction: str = "web",
        number: str | None = None,
        campaign: str | None = None,
        assistant: str | None = None,
        call_id: str | None = None,
    ) -> None:
        self._store = store
        self._samples: dict[str, list[float]] = {k: [] for k in self._SERIES}
        self._samples["response_s"] = []
        self._speech_ended: float | None = None
        self._t0 = time.monotonic()
        self.record = CallRecord(
            call_id=call_id or uuid.uuid4().hex[:16],
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            direction=direction,
            number=number,
            campaign=campaign,
            assistant=assistant,
        )
        event_bus.subscribe(None, self._on_event)

    def _on_event(self, event: Event) -> None:
        rec = self.record
        if event.type is EventType.SPEECH_ENDED and event.data.get("accepted"):
            self._speech_ended = event.timestamp
        elif event.type is EventType.TRANSCRIPT_READY:
            rec.transcript.append({
                "role": "user",
                "text": event.data.get("text", ""),
                "language": event.data.get("language"),
                "at_s": round(time.monotonic() - self._t0, 2),
            })
            self._record("stt_s", event.data.get("stt_latency_s"))
        elif event.type is EventType.LLM_FINISHED:
            rec.transcript.append({
                "role": "assistant",
                "text": event.data.get("text", ""),
                "at_s": round(time.monotonic() - self._t0, 2),
            })
            self._record("llm_ttft_s", event.data.get("latency_first_token_s"))
            self._record("llm_total_s", event.data.get("latency_total_s"))
        elif event.type is EventType.PLAYBACK_STARTED and self._speech_ended is not None:
            # What the caller actually waits: mouth-close to first audio.
            self._record("response_s", event.timestamp - self._speech_ended)
            self._speech_ended = None
        elif event.type is EventType.PLAYBACK_FINISHED:
            rec.turns += 1
        elif event.type is EventType.BARGE_IN:
            rec.barge_ins += 1
        elif event.type is EventType.ERROR:
            stage = event.data.get("stage", "unknown")
            rec.errors[stage] = rec.errors.get(stage, 0) + 1

    def _record(self, name: str, value: float | None) -> None:
        if value is not None:
            self._samples[name].append(float(value))

    def finish(self, ended_reason: str = "completed", status: str = "completed") -> CallRecord:
        """Close the record, compute stats, and persist it."""
        rec = self.record
        rec.ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rec.duration_s = round(time.monotonic() - self._t0, 2)
        rec.ended_reason = ended_reason
        rec.status = status
        rec.latency_s = {
            name: {
                "count": len(vals),
                "p50": _percentile(vals, 50),
                "p95": _percentile(vals, 95),
                "max": round(max(vals), 3) if vals else 0.0,
            }
            for name, vals in self._samples.items()
        }
        if self._store is not None:
            try:
                self._store.add(rec)
            except OSError:
                logger.exception("could not write call record %s", rec.call_id)
        return rec
