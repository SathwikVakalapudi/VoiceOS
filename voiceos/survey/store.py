"""Result store — one JSONL record per call, with a CSV export.

JSONL (append-only) is crash-safe under concurrent calls: each completed call
appends one line, so a mid-campaign crash never corrupts earlier results.

⚠️ These records are call data (possibly PII / opinion data). Keep the output
under an ignored path (see .gitignore `results/`) and handle per your consent
notice and local data-protection rules.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict) -> None:
        """Append one call's result as a JSON line."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> list[dict]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def export_csv(self, path: str, field_ids: list[str]) -> int:
        """Flatten records to CSV: metadata columns + one column per survey field.

        Returns the number of rows written.
        """
        rows = self.records()
        meta_cols = ["call_id", "number", "timestamp", "status"]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(meta_cols + field_ids)
            for rec in rows:
                answers = rec.get("answers", {})
                writer.writerow(
                    [rec.get(c, "") for c in meta_cols]
                    + [answers.get(fid, "") for fid in field_ids]
                )
        return len(rows)
