"""Campaign dashboard — a web UI + REST API to create, test, dry-run, and
review VoiceOS calling campaigns.

The API and its pieces (`CampaignStore`, `TestSandbox`) are dependency-injected
so they run headless in tests without a real LLM or telephony stack; `app.py`
wires the production defaults and serves the single-file frontend.
"""

from voiceos.dashboard.app import create_app
from voiceos.dashboard.sandbox import TestSandbox
from voiceos.dashboard.store import CampaignStore, CampaignError

__all__ = ["create_app", "TestSandbox", "CampaignStore", "CampaignError"]
