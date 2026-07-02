"""Pipeline state machine.

Phase 1 uses the state to gate the microphone: while the assistant is
THINKING or SPEAKING, incoming audio is ignored so the assistant does not
transcribe its own voice. Barge-in later relaxes exactly this gate.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class PipelineState(str, Enum):
    IDLE = "idle"            # waiting for the user
    LISTENING = "listening"  # user is speaking
    THINKING = "thinking"    # STT / LLM in flight
    SPEAKING = "speaking"    # assistant audio is playing


StateListener = Callable[[PipelineState, PipelineState], None]


class InterruptController:
    """Monotonic generation counter for turn cancellation.

    Workers capture the generation when a turn starts; a barge-in bumps
    it, and anything still carrying the old generation stops pushing
    work downstream. Cheap, lock-free, race-tolerant.
    """

    def __init__(self) -> None:
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def bump(self) -> int:
        self._generation += 1
        return self._generation


class StateMachine:
    def __init__(self, initial: PipelineState = PipelineState.IDLE) -> None:
        self._state = initial
        self._listeners: list[StateListener] = []

    @property
    def state(self) -> PipelineState:
        return self._state

    def on_change(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    def transition(self, new_state: PipelineState) -> None:
        if new_state is self._state:
            return
        old_state, self._state = self._state, new_state
        logger.debug("state: %s -> %s", old_state.value, new_state.value)
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception:
                logger.exception("state listener failed")
