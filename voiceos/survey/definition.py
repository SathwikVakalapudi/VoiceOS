"""Survey definition — the machine-readable questions to extract from a call.

Lives alongside the persona in the campaign JSON under a ``survey`` key, e.g.:

    {
      "system_prompt": "...", "first_message": "...",
      "survey": {
        "name": "rajasthan-political-survey",
        "questions": [
          {"id": "q1", "prompt": "How much they like Modi",
           "type": "choice", "options": ["a lot", "somewhat", "not much", "not at all"]},
          {"id": "q6", "prompt": "Respondent's age", "type": "number"},
          {"id": "q7", "prompt": "Religion", "type": "text"}
        ]
      }
    }

The AI still asks the questions in the campaign's language; `prompt` here is a
short English description that tells the extractor which answer to pull out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SurveyQuestion:
    id: str
    prompt: str                       # short description of what to extract
    type: str = "text"                # "choice" | "text" | "number"
    options: list[str] | None = None  # allowed values for type == "choice"

    @classmethod
    def from_dict(cls, data: dict) -> "SurveyQuestion":
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            type=data.get("type", "text"),
            options=data.get("options"),
        )


@dataclass
class SurveyDefinition:
    name: str
    questions: list[SurveyQuestion]

    @property
    def field_ids(self) -> list[str]:
        return [q.id for q in self.questions]

    @classmethod
    def from_dict(cls, data: dict) -> "SurveyDefinition":
        return cls(
            name=data.get("name", "survey"),
            questions=[SurveyQuestion.from_dict(q) for q in data.get("questions", [])],
        )

    @classmethod
    def from_campaign_file(cls, path: str) -> "SurveyDefinition | None":
        """Load the ``survey`` block from a campaign JSON, or None if absent."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        survey = data.get("survey")
        if not survey or not survey.get("questions"):
            return None
        return cls.from_dict(survey)
