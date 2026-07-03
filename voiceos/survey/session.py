"""Survey session wrapper.

Wraps a VoicePipeline so the telephony servers (AudioSocket / WebSocket) need
no changes: it has the same async start()/stop() contract, and on stop() it
snapshots the transcript, stops the pipeline, then runs post-call extraction.

Extraction failure never propagates — a survey miss must not break call
teardown.
"""

from __future__ import annotations

import logging

from voiceos.survey.collector import SurveyCollector

logger = logging.getLogger(__name__)


class SurveySession:
    def __init__(
        self,
        pipeline,
        collector: SurveyCollector,
        *,
        call_id: str | None = None,
        number: str | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._collector = collector
        self._call_id = call_id
        self._number = number

    async def start(self) -> None:
        await self._pipeline.start()

    async def stop(self) -> None:
        # Snapshot the transcript before stop() tears the pipeline down.
        transcript = self._pipeline.conversation.history.messages
        await self._pipeline.stop()
        try:
            await self._collector.collect(
                transcript, call_id=self._call_id, number=self._number
            )
        except Exception:
            logger.exception("survey extraction failed; call already ended")
