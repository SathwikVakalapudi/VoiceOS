from voiceos.llm.inference import ThinkTagFilter
from voiceos.tts.streaming import SentenceChunker, clean_for_speech


def collect(chunker: SentenceChunker, deltas: list[str]) -> list[str]:
    out: list[str] = []
    for delta in deltas:
        out.extend(chunker.feed(delta))
    out.extend(chunker.flush())
    return out


def test_chunker_splits_on_sentence_boundaries():
    chunker = SentenceChunker(min_chars=10)
    chunks = collect(chunker, ["The weather is sunny today. ", "Tomorrow looks rainy."])
    assert chunks == ["The weather is sunny today.", "Tomorrow looks rainy."]


def test_chunker_does_not_split_decimals():
    chunker = SentenceChunker(min_chars=5)
    chunks = collect(chunker, ["Pi is about 3.14159 and that is neat."])
    assert chunks == ["Pi is about 3.14159 and that is neat."]


def test_chunker_respects_min_chars():
    chunker = SentenceChunker(min_chars=24)
    # "Hi." alone is too short to emit; it merges with the next sentence.
    chunks = collect(chunker, ["Hi. ", "It is a lovely day outside."])
    assert chunks == ["Hi. It is a lovely day outside."]


def test_chunker_handles_devanagari_danda():
    chunker = SentenceChunker(min_chars=4)
    chunks = collect(chunker, ["नमस्ते। आप कैसे हैं?"])
    assert chunks == ["नमस्ते।", "आप कैसे हैं?"]


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("**Bold** and `code` #tag") == "Bold and code tag"


def test_clean_for_speech_strips_emoji():
    assert clean_for_speech("చాలా బాగుంది! \U0001f60a ❤️") == "చాలా బాగుంది!"


def test_think_filter_drops_reasoning_block():
    f = ThinkTagFilter()
    out = f.feed("<think>secret reasoning</think>Hello there!") + f.flush()
    assert out == "Hello there!"


def test_think_filter_handles_tags_split_across_deltas():
    f = ThinkTagFilter()
    out = ""
    for delta in ["<th", "ink>hidden", " stuff</thi", "nk>The answer", " is 42."]:
        out += f.feed(delta)
    out += f.flush()
    assert out == "The answer is 42."


def test_think_filter_passes_plain_text_through():
    f = ThinkTagFilter()
    out = f.feed("Just a normal reply.") + f.flush()
    assert out == "Just a normal reply."
