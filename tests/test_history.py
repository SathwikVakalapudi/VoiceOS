from voiceos.conversation.history import ConversationHistory


def test_messages_alternate_and_accumulate():
    history = ConversationHistory(max_turns=5)
    history.add_user("hello")
    history.add_assistant("hi there")

    assert history.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_trims_oldest_turns():
    history = ConversationHistory(max_turns=2)
    for i in range(5):
        history.add_user(f"u{i}")
        history.add_assistant(f"a{i}")

    assert len(history) == 4
    assert history.messages[0] == {"role": "user", "content": "u3"}
    assert history.messages[-1] == {"role": "assistant", "content": "a4"}


def test_clear():
    history = ConversationHistory()
    history.add_user("x")
    history.clear()
    assert len(history) == 0
