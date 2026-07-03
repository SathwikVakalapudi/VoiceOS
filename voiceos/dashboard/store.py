"""Campaign store — CRUD over the campaign JSON files on disk.

A campaign file bundles the persona ({system_prompt, first_message,
error_message}) and an optional machine-readable `survey` block. This store
lists, reads, validates, writes, and deletes them, guarding against path
traversal (names are restricted to a safe charset).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from voiceos.survey.definition import SurveyDefinition

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class CampaignError(ValueError):
    """Raised for an invalid campaign name or payload."""


class CampaignStore:
    def __init__(self, campaigns_dir: str) -> None:
        self._dir = Path(campaigns_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        if not _SAFE_NAME.match(name):
            raise CampaignError(
                f"invalid campaign name {name!r}: use letters, digits, - and _ only"
            )
        return self._dir / f"{name}.json"

    @staticmethod
    def validate(data: dict) -> None:
        if not isinstance(data, dict):
            raise CampaignError("campaign must be a JSON object")
        if not (data.get("system_prompt") or data.get("first_message")):
            raise CampaignError("campaign needs a system_prompt or a first_message")
        if "survey" in data and data["survey"]:
            try:
                survey = SurveyDefinition.from_dict(data["survey"])
            except (KeyError, TypeError) as exc:
                raise CampaignError(f"invalid survey block: {exc}") from exc
            if not survey.questions:
                raise CampaignError("survey block has no questions")
            ids = survey.field_ids
            if len(ids) != len(set(ids)):
                raise CampaignError("survey question ids must be unique")

    def list(self) -> list[dict]:
        out = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue  # skip unreadable files
            # Skip non-campaign JSON in the dir (e.g. a contacts list).
            if not isinstance(data, dict) or not (
                data.get("system_prompt") or data.get("first_message")
            ):
                continue
            survey = data.get("survey") or {}
            out.append(
                {
                    "name": path.stem,
                    "first_message": data.get("first_message", ""),
                    "has_survey": bool(survey.get("questions")),
                    "question_count": len(survey.get("questions", [])),
                }
            )
        return out

    def get(self, name: str) -> dict:
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, name: str, data: dict) -> None:
        self.validate(data)
        path = self._path(name)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def delete(self, name: str) -> None:
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        path.unlink()

    def path_for(self, name: str) -> str:
        """Filesystem path of a campaign (for ConversationManager/collector)."""
        return str(self._path(name))
