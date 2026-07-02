"""Predictive-endpointing rule tests (the rule-based 'early guess')."""

from voiceos.vad.endpoint import EndpointPredictor

P = EndpointPredictor(min_chars=12)


def test_question_is_complete():
    assert P.looks_complete("what time is it right now?") is True


def test_statement_with_period_is_complete():
    assert P.looks_complete("I want to book a table.") is True


def test_trailing_conjunction_is_incomplete():
    assert P.looks_complete("I would like to book a table and") is False


def test_no_punctuation_is_incomplete():
    # Whisper usually adds punctuation; without it we wait rather than cut off.
    assert P.looks_complete("I would like to book a table") is False


def test_too_short_is_incomplete():
    assert P.looks_complete("yes.") is False


def test_decimal_period_is_not_a_sentence_end():
    assert P.looks_complete("the total comes to 3.") is False


def test_danda_is_complete():
    assert P.looks_complete("नमस्ते, आप कैसे हैं।") is True
