"""Staged-commit tests: history records only what was actually spoken."""

from voiceos.config.settings import ConversationSettings
from voiceos.conversation.manager import ConversationManager


def make_manager() -> ConversationManager:
    return ConversationManager(ConversationSettings())


def test_full_turn_is_recorded():
    manager = make_manager()
    manager.build_messages("hi")
    manager.begin_assistant(1)
    manager.add_pending_segment(1, "Hello there.")
    manager.add_pending_segment(1, "How can I help?")
    manager.commit_assistant(1)  # None -> all spoken

    assert manager.history.messages[-1] == {
        "role": "assistant",
        "content": "Hello there. How can I help?",
    }


def test_barge_in_records_only_spoken_sentences():
    manager = make_manager()
    manager.build_messages("tell me a story")
    manager.begin_assistant(1)
    manager.add_pending_segment(1, "Once upon a time.")
    manager.add_pending_segment(1, "There was a dragon.")
    manager.add_pending_segment(1, "It breathed fire.")
    # User barged in after only the first sentence finished playing.
    manager.commit_assistant(1, spoken_segments=1)

    assert manager.history.messages[-1] == {
        "role": "assistant",
        "content": "Once upon a time.",
    }


def test_nothing_spoken_records_nothing():
    manager = make_manager()
    manager.build_messages("hey")
    manager.begin_assistant(1)
    manager.add_pending_segment(1, "cut off immediately")
    manager.commit_assistant(1, spoken_segments=0)

    # Only the user turn is in history; no empty assistant message.
    assert manager.history.messages == [{"role": "user", "content": "hey"}]


def test_commit_is_idempotent():
    manager = make_manager()
    manager.build_messages("x")
    manager.begin_assistant(1)
    manager.add_pending_segment(1, "one.")
    manager.add_pending_segment(1, "two.")
    manager.commit_assistant(1, spoken_segments=1)  # barge-in path
    manager.commit_assistant(1)                     # racing end-of-turn: no-op

    assistant_msgs = [m for m in manager.history.messages if m["role"] == "assistant"]
    assert assistant_msgs == [{"role": "assistant", "content": "one."}]


def test_stale_turn_id_does_not_commit():
    manager = make_manager()
    manager.begin_assistant(2)
    manager.add_pending_segment(2, "current turn")
    manager.commit_assistant(1)  # wrong turn -> ignored, pending untouched

    manager.commit_assistant(2)
    assert manager.history.messages[-1] == {
        "role": "assistant",
        "content": "current turn",
    }
