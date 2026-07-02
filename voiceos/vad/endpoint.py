"""Predictive endpointing.

Given a partial transcript of what the user has said so far, guess whether
the turn is complete so the detector can close early instead of waiting out
the full trailing silence. This is the rule-based tier (a fusion audio+text
ML model is the later upgrade).

The bias is deliberately conservative: a wrong "complete" guess cuts the
user off mid-thought, which is worse than waiting. So it only predicts
completion on strong terminal signals and explicitly holds back when the
text trails off into a filler, conjunction, or bare number.
"""

from __future__ import annotations

# Words that, when they end an utterance, signal the user is not done.
_TRAILING_INCOMPLETE = {
    "and", "or", "but", "so", "because", "if", "then", "the", "a", "an",
    "to", "of", "for", "with", "um", "uh", "er", "like", "well", "i",
    "my", "your", "that", "this", "is", "are", "was", "were", "he", "she",
    "they", "we", "it",
}

_TERMINAL = "?!।"  # question / exclamation / Devanagari danda: near-certain end


class EndpointPredictor:
    def __init__(self, min_chars: int = 12) -> None:
        self._min_chars = min_chars

    def looks_complete(self, partial: str) -> bool:
        text = (partial or "").strip()
        if len(text) < self._min_chars:
            return False

        last = text[-1]
        if last in _TERMINAL:
            return True
        if last == ".":
            prev = text[-2] if len(text) >= 2 else " "
            # A period after a digit is likely a decimal ("3.") mid-number.
            return not prev.isdigit()

        # No terminal punctuation: only unsafe signals remain, so hold on if
        # the last word is a filler/conjunction/dangling function word.
        last_word = text.split()[-1].lower().strip(",;:")
        if last_word in _TRAILING_INCOMPLETE:
            return False
        return False  # no punctuation and nothing conclusive: keep waiting
