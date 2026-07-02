"""Per-session context: identity, timing, turn counter.

Grows into user/session metadata, memory hooks, and analytics keys.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationContext:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def new_turn(self) -> int:
        self.turn_count += 1
        return self.turn_count
