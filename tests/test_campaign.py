import json

from voiceos.config.settings import ConversationSettings
from voiceos.conversation.manager import ConversationManager


def test_campaign_file_overrides_prompt_and_first_message(tmp_path):
    campaign = {
        "system_prompt": "You are a survey caller.",
        "first_message": "Hello, may I ask you a few questions?",
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")

    manager = ConversationManager(
        ConversationSettings(campaign_file=str(path))
    )

    assert manager.first_message == campaign["first_message"]
    messages = manager.build_messages("yes, go ahead")
    assert messages[0] == {"role": "system", "content": "You are a survey caller."}
    # The greeting must be in history so the LLM knows it was already spoken.
    assert messages[1] == {"role": "assistant", "content": campaign["first_message"]}
    assert messages[2] == {"role": "user", "content": "yes, go ahead"}


def test_no_campaign_keeps_default_persona():
    manager = ConversationManager(ConversationSettings())
    assert manager.first_message is None
    messages = manager.build_messages("hi")
    assert messages[0]["role"] == "system"
    assert len(messages) == 2  # system + user, no greeting


def test_shipped_rajasthan_campaign_parses():
    manager = ConversationManager(
        ConversationSettings(campaign_file=r"C:\Users\sathw\voice os\campaigns\rajasthan_survey.json")
    )
    assert manager.first_message.startswith("నమస్తే")
    assert "ఎస్సీ" in manager.build_messages("సరే")[0]["content"]
