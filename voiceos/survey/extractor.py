"""Post-call structured extraction.

Feeds the finished call transcript to the LLM and asks for a strict JSON object
mapping each survey question id to the respondent's answer — normalized to the
allowed options, a number, or short text, or null if the question was never
answered or the respondent refused. Transcripts may be in any language (e.g.
Telugu); the extractor maps answers to the canonical option labels.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

from voiceos.interfaces.llm import BaseLLM, Message
from voiceos.survey.definition import SurveyDefinition

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a precise data-extraction engine for phone survey transcripts. "
    "You output ONLY a single JSON object, no prose, no code fences. The "
    "transcript may be in another language; still return the canonical labels "
    "specified. Use null when a question was not answered or the respondent "
    "declined. Never invent answers."
)


def _field_spec(survey: SurveyDefinition) -> str:
    lines = []
    for q in survey.questions:
        if q.type == "choice" and q.options:
            allowed = " | ".join(q.options)
            lines.append(f'- "{q.id}": {q.prompt}. One of [{allowed}] or null.')
        elif q.type == "number":
            lines.append(f'- "{q.id}": {q.prompt}. A number or null.')
        else:
            lines.append(f'- "{q.id}": {q.prompt}. Short text or null.')
    return "\n".join(lines)


def _render_transcript(transcript: Sequence[Message]) -> str:
    out = []
    for m in transcript:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            speaker = "Respondent" if role == "user" else "Interviewer"
            out.append(f"{speaker}: {content}")
    return "\n".join(out)


def parse_json_object(text: str) -> dict:
    """Best-effort parse of a JSON object from an LLM reply.

    Tolerates code fences and surrounding prose by taking the substring from the
    first ``{`` to the last ``}``. Returns {} if nothing parseable is found.
    """
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        result = json.loads(text[start : end + 1])
        return result if isinstance(result, dict) else {}
    except ValueError:
        return {}


class SurveyExtractor:
    def __init__(self, llm: BaseLLM, survey: SurveyDefinition) -> None:
        self._llm = llm
        self._survey = survey

    def build_messages(self, transcript: Sequence[Message]) -> list[Message]:
        user = (
            f"Survey: {self._survey.name}\n\n"
            f"Extract these fields as a JSON object with exactly these keys:\n"
            f"{_field_spec(self._survey)}\n\n"
            f"Transcript:\n{_render_transcript(transcript)}\n\n"
            f"Return only the JSON object."
        )
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]

    async def extract(self, transcript: Sequence[Message]) -> dict:
        """Return {question_id: answer|null} for every survey field."""
        message = await self._llm.complete(self.build_messages(transcript))
        raw = parse_json_object(message.get("content", "") if message else "")
        # Guarantee every field is present (null when the model omitted it).
        return {qid: raw.get(qid) for qid in self._survey.field_ids}
