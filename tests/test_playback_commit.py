"""End-to-end wiring: TTS worker -> playback worker -> history commit.

Uses a fake TTS and speaker (no models, no audio hardware) to prove that
sentence markers flow through the queues and history records exactly the
sentences whose audio was fully played.
"""

import asyncio

import numpy as np

from voiceos.audio.player import PlaybackWorker
from voiceos.config.settings import ConversationSettings
from voiceos.conversation.manager import ConversationManager
from voiceos.pipeline.events import EndOfTurn, EventBus, EventType, SpeakSentence
from voiceos.pipeline.state import InterruptController, StateMachine
from voiceos.tts.player import TTSWorker


class FakeTTS:
    @property
    def sample_rate(self):
        return 24000

    async def synthesize(self, text):
        # Two chunks per sentence, so playback spans multiple frames.
        yield np.ones(4, dtype="<i2")
        yield np.ones(4, dtype="<i2")


class FakeSpeaker:
    def __init__(self):
        self.frames = 0

    async def play(self, data):
        self.frames += 1


def build_stack():
    bus = EventBus()
    state = StateMachine()
    interrupts = InterruptController()
    conversation = ConversationManager(ConversationSettings())
    tts_queue: asyncio.Queue = asyncio.Queue()
    playback_queue: asyncio.Queue = asyncio.Queue()

    tts_worker = TTSWorker(FakeTTS(), tts_queue, playback_queue, bus, state, interrupts)
    playback_worker = PlaybackWorker(FakeSpeaker(), playback_queue, bus, state)

    bus.subscribe(
        EventType.PLAYBACK_FINISHED,
        lambda e: conversation.commit_assistant(
            e.data["turn_id"], e.data.get("sentences_spoken")
        ),
    )
    return conversation, tts_queue, playback_queue, tts_worker, playback_worker


async def test_full_turn_commits_every_spoken_sentence():
    conv, tts_q, play_q, tts_worker, playback_worker = build_stack()
    conv.build_messages("hi")
    conv.begin_assistant(1)

    tasks = [asyncio.create_task(tts_worker.run()), asyncio.create_task(playback_worker.run())]
    for i, text in enumerate(["Hello there.", "How can I help?"]):
        conv.add_pending_segment(1, text)
        await tts_q.put(SpeakSentence(1, i, text))
    await tts_q.put(EndOfTurn(turn_id=1))

    await tts_q.join()
    await play_q.join()
    for t in tasks:
        t.cancel()

    assert playback_worker.sentences_spoken == 0  # reset after end-of-turn
    assert conv.history.messages[-1] == {
        "role": "assistant",
        "content": "Hello there. How can I help?",
    }


async def test_barge_in_commits_only_the_played_prefix():
    conv, tts_q, play_q, tts_worker, playback_worker = build_stack()
    conv.build_messages("tell me more")
    conv.begin_assistant(1)

    tasks = [asyncio.create_task(tts_worker.run()), asyncio.create_task(playback_worker.run())]
    # Only the first sentence is synthesized and played (no EndOfTurn yet).
    conv.add_pending_segment(1, "First point.")
    await tts_q.put(SpeakSentence(1, 0, "First point."))
    await tts_q.join()
    await play_q.join()

    # User barges in: commit what was played, as the pipeline handler does.
    conv.commit_assistant(1, playback_worker.sentences_spoken)
    for t in tasks:
        t.cancel()

    assert playback_worker.sentences_spoken == 1
    assert conv.history.messages[-1] == {"role": "assistant", "content": "First point."}
