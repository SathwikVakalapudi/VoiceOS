"""Survey collector — orchestrates extraction + storage for one finished call.

Uses a dedicated short-lived LLM (built from settings) rather than the call's
pipeline LLM, because `VoicePipeline.stop()` closes that one. The LLM is loaded,
used for a single extraction, and closed per call.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Sequence

from voiceos.config.settings import Settings
from voiceos.interfaces.llm import BaseLLM, Message
from voiceos.survey.definition import SurveyDefinition
from voiceos.survey.extractor import SurveyExtractor
from voiceos.survey.store import ResultStore

logger = logging.getLogger(__name__)


class SurveyCollector:
    def __init__(
        self,
        survey: SurveyDefinition,
        store: ResultStore,
        *,
        settings: Settings | None = None,
        llm_factory: Callable[[], BaseLLM] | None = None,
        clock: Callable[[], str] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if llm_factory is None:
            if settings is None:
                raise ValueError("SurveyCollector needs settings or llm_factory")
            from voiceos.pipeline.pipeline import create_llm

            llm_factory = lambda: create_llm(settings)  # noqa: E731
        self._survey = survey
        self._store = store
        self._llm_factory = llm_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)

    async def collect(
        self,
        transcript: Sequence[Message],
        *,
        call_id: str | None = None,
        number: str | None = None,
        status: str = "completed",
    ) -> dict:
        """Extract answers from a transcript and append one result record."""
        llm = self._llm_factory()
        await llm.load()
        try:
            answers = await SurveyExtractor(llm, self._survey).extract(transcript)
        finally:
            await llm.close()

        record = {
            "call_id": call_id or self._id_factory(),
            "number": number,
            "timestamp": self._clock(),
            "status": status,
            "survey": self._survey.name,
            "answers": answers,
        }
        self._store.add(record)
        logger.info("survey result stored for call %s", record["call_id"])
        return record
