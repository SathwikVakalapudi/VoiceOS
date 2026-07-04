"""VoicePipeline — wires independent workers together with queues.

    Microphone -> AudioQueue -> SpeechDetector -> utterance queue
    -> STTWorker -> transcript queue -> LLMWorker -> tts queue
    -> TTSWorker -> playback queue -> PlaybackWorker -> Speaker

The pipeline performs no inference. It only builds components from
settings, connects queues, starts/stops worker tasks, and owns the
shared EventBus and StateMachine.
"""

from __future__ import annotations

import asyncio
import logging

from voiceos.audio.audio_queue import AudioQueue, Utterance
from voiceos.audio.player import PlaybackWorker
from voiceos.config.settings import Settings
from voiceos.conversation.manager import ConversationManager
from voiceos.interfaces import BaseLLM, BaseSTT, BaseTTS, BaseVAD
from voiceos.llm.inference import LLMWorker
from voiceos.llm.qwen import QwenLLM
from voiceos.pipeline.backchannel import BackchannelWorker
from voiceos.pipeline.events import EventBus, EventType
from voiceos.pipeline.metrics import LatencyMonitor
from voiceos.pipeline.state import InterruptController, StateMachine
from voiceos.stt.processor import STTWorker
from voiceos.stt.whisper import FasterWhisperSTT
from voiceos.transport.base import AudioTransport
from voiceos.transport.local import LocalAudioTransport
from voiceos.tts.player import TTSWorker
from voiceos.tts.svara import SvaraTTS
from voiceos.vad.detector import SpeechDetector
from voiceos.vad.silero_vad import SileroVAD

logger = logging.getLogger(__name__)


def _drain(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            return


def create_vad(settings: Settings) -> BaseVAD:
    return SileroVAD(settings.vad)


def _build_stt(provider: str, settings: Settings) -> BaseSTT:
    if provider == "sarvam":
        from voiceos.stt.sarvam import SarvamSTT

        return SarvamSTT(settings.stt)
    return FasterWhisperSTT(settings.stt)


def create_stt(settings: Settings) -> BaseSTT:
    primary = _build_stt(settings.stt.provider, settings)
    if not settings.stt.fallback:
        return primary
    from voiceos.stt.fallback import FallbackSTT

    backups = [_build_stt(name, settings) for name in settings.stt.fallback]
    return FallbackSTT([primary, *backups])


def create_llm(settings: Settings) -> BaseLLM:
    keys = settings.llm.api_keys or [settings.llm.api_key]
    if len(keys) > 1:
        from voiceos.llm.rotating import RotatingLLM

        primary: BaseLLM = RotatingLLM(
            [QwenLLM(settings.llm.model_copy(update={"api_key": k})) for k in keys]
        )
    else:
        primary = QwenLLM(settings.llm)
    if not settings.llm.fallbacks:
        return primary
    from voiceos.llm.fallback import FallbackLLM

    backups = [
        QwenLLM(
            settings.llm.model_copy(
                update={
                    "base_url": ep.base_url,
                    "model": ep.model,
                    "api_key": ep.api_key,
                    "reasoning_effort": ep.reasoning_effort,
                }
            )
        )
        for ep in settings.llm.fallbacks
    ]
    return FallbackLLM([primary, *backups])


def _build_tts(provider: str, settings: Settings) -> BaseTTS:
    if provider == "edge":
        from voiceos.tts.edge import EdgeTTS

        return EdgeTTS(settings.tts)
    if provider == "cartesia":
        from voiceos.tts.cartesia import CartesiaTTS

        return CartesiaTTS(settings.tts)
    if provider == "piper":
        from voiceos.tts.piper import PiperTTS

        return PiperTTS(settings.tts)
    return SvaraTTS(settings.tts)


def create_tts(settings: Settings) -> BaseTTS:
    primary = _build_tts(settings.tts.provider, settings)
    if not settings.tts.fallback:
        return primary
    from voiceos.tts.fallback import FallbackTTS

    backups = [_build_tts(name, settings) for name in settings.tts.fallback]
    return FallbackTTS([primary, *backups])


class VoicePipeline:
    def __init__(
        self,
        settings: Settings,
        event_bus: EventBus | None = None,
        vad: BaseVAD | None = None,
        stt: BaseSTT | None = None,
        llm: BaseLLM | None = None,
        tts: BaseTTS | None = None,
        transport: AudioTransport | None = None,
    ) -> None:
        self.settings = settings
        self.event_bus = event_bus or EventBus()
        self.state = StateMachine()
        self.metrics = LatencyMonitor(self.event_bus)
        # Aggregated metrics for the optional dashboard (started in start()).
        self.metrics_collector = None
        self._metrics_server = None
        if settings.monitoring.enabled:
            from voiceos.monitoring.collector import MetricsCollector

            self.metrics_collector = MetricsCollector(self.event_bus)
        self.interrupts = InterruptController()

        # Pluggable stages — pass alternatives in, or rely on defaults.
        self.vad = vad or create_vad(settings)
        self.stt = stt or create_stt(settings)
        self.llm = llm or create_llm(settings)
        self.tts = tts or create_tts(settings)

        self.conversation = ConversationManager(settings.conversation)
        # Audio I/O comes from a transport (local mic+speaker by default);
        # a telephony/WebRTC transport can be injected without other changes.
        self.transport = transport or LocalAudioTransport(settings.audio)
        self.microphone = self.transport.source
        self.speaker = self.transport.sink

        # Tool calling: an empty registry unless enabled. Register your own
        # tools on self.tools before start() to extend the assistant.
        self.tools = None
        if settings.llm.tools_enabled:
            from voiceos.llm.tools import ToolRegistry, register_builtin_tools

            self.tools = ToolRegistry()
            register_builtin_tools(self.tools)

        # Queues between workers.
        self.audio_queue = AudioQueue(maxsize=settings.audio.queue_max_frames)
        self.utterance_queue: asyncio.Queue[Utterance] = asyncio.Queue()
        self.transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.playback_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        # Predictive endpointing needs partial transcripts. Give it a
        # dedicated STT instance so rolling partials never run on the same
        # model as the STT worker's final transcription.
        self.partial_transcriber = None
        self.endpoint_predictor = None
        if settings.vad.predictive_endpointing:
            from voiceos.stt.streaming import RollingTranscriber
            from voiceos.vad.endpoint import EndpointPredictor

            self.partial_transcriber = RollingTranscriber(create_stt(settings))
            self.endpoint_predictor = EndpointPredictor(
                min_chars=settings.vad.min_partial_chars
            )

        frame_ms = settings.audio.frame_size / settings.audio.input_sample_rate * 1000
        self.detector = SpeechDetector(
            self.vad, settings.vad, self.audio_queue, self.utterance_queue,
            self.event_bus, self.state, frame_ms=frame_ms,
            on_barge_in=self._handle_barge_in,
            partial_transcriber=self.partial_transcriber,
            endpoint_predictor=self.endpoint_predictor,
        )
        self.stt_worker = STTWorker(
            self.stt, self.utterance_queue, self.transcript_queue,
            self.event_bus, self.state,
        )
        self.llm_worker = LLMWorker(
            self.llm, self.conversation, self.transcript_queue, self.tts_queue,
            self.event_bus, self.state, interrupts=self.interrupts,
            sentence_min_chars=settings.pipeline.sentence_min_chars,
            tools=self.tools, max_tool_iterations=settings.llm.max_tool_iterations,
        )
        self.tts_worker = TTSWorker(
            self.tts, self.tts_queue, self.playback_queue,
            self.event_bus, self.state, interrupts=self.interrupts,
        )
        self.playback_worker = PlaybackWorker(
            self.speaker, self.playback_queue, self.event_bus, self.state,
        )
        self.backchannel = (
            BackchannelWorker(
                self.tts, self.playback_queue, self.event_bus, self.state,
                settings.pipeline,
            )
            if settings.pipeline.backchannel
            else None
        )
        self._workers = [
            self.detector, self.stt_worker, self.llm_worker,
            self.tts_worker, self.playback_worker,
        ]
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

        # A cleanly-finished turn records the whole reply; the count carried
        # on the event equals every staged sentence in that case.
        self.event_bus.subscribe(EventType.PLAYBACK_FINISHED, self._on_playback_finished)

    def _on_playback_finished(self, event) -> None:
        self.conversation.commit_assistant(
            event.data["turn_id"], event.data.get("sentences_spoken")
        )

    async def _handle_barge_in(self) -> None:
        """Kill everything the assistant was about to say, right now."""
        # Record only the sentences fully played before the interrupt, so
        # history reflects what the user actually heard. Read the count and
        # commit before draining — this runs to completion without awaiting,
        # so the playback worker cannot advance the count underneath us.
        turn_id = self.playback_worker.current_turn_id
        if turn_id is not None:
            self.conversation.commit_assistant(
                turn_id, self.playback_worker.sentences_spoken
            )
        self.interrupts.bump()          # in-flight LLM/TTS work stops pushing
        self.speaker.interrupt()        # abort the audio being written
        _drain(self.tts_queue)          # unspoken sentences
        _drain(self.playback_queue)     # unplayed audio
        self.playback_worker.notify_interrupted()
        self.speaker.resume()

    async def start(self) -> None:
        logger.info("loading models...")
        loaders = [self.vad.load(), self.stt.load(), self.llm.load(), self.tts.load()]
        if self.partial_transcriber is not None:
            loaders.append(self.partial_transcriber.load())
        await asyncio.gather(*loaders)

        self.speaker.open(self.tts.sample_rate)
        loop = asyncio.get_running_loop()
        self.microphone.start(loop, self.audio_queue)

        if self.backchannel is not None:
            await self.backchannel.load()  # pre-render fillers on the TTS voice
            self.backchannel.start()

        self._tasks = [
            asyncio.create_task(worker.run(), name=type(worker).__name__)
            for worker in self._workers
        ]
        if self.metrics_collector is not None:
            from voiceos.monitoring.dashboard import serve_metrics

            self._metrics_server = serve_metrics(
                self.metrics_collector,
                self.settings.monitoring.host,
                self.settings.monitoring.port,
            )

        await self.event_bus.emit(EventType.PIPELINE_STARTED, {})

        # Outbound-call style: the assistant speaks first if configured.
        if self.conversation.first_message:
            from voiceos.pipeline.events import EndOfTurn

            await self.tts_queue.put(self.conversation.first_message)
            await self.tts_queue.put(EndOfTurn(turn_id=0))
        logger.info("pipeline running")

    async def wait(self) -> None:
        """Block until stop() is called."""
        await self._stopped.wait()

    async def stop(self) -> None:
        self.microphone.stop()
        if self._metrics_server is not None:
            self._metrics_server.shutdown()
            self._metrics_server = None
        if self.backchannel is not None:
            await self.backchannel.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        closers = [self.stt.close(), self.llm.close(), self.tts.close()]
        if self.partial_transcriber is not None:
            closers.append(self.partial_transcriber.close())
        await asyncio.gather(*closers, return_exceptions=True)
        self.speaker.close()
        await self.event_bus.emit(EventType.PIPELINE_STOPPED, {})
        self._stopped.set()
        logger.info("pipeline stopped")
