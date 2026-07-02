"""Streaming text preparation for TTS.

SentenceChunker turns an LLM token stream into speakable sentence
chunks so synthesis can start before the full response exists.
"""

from __future__ import annotations

import re

_BOUNDARY_CHARS = ".!?।"  # includes the Devanagari danda
_TRAILING_OK = " \t\n\"'”’)"

# Characters that read badly aloud; the system prompt discourages
# markdown and emoji, but models slip.
_MARKDOWN_RE = re.compile(r"[*_`#>]+")
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff☀-➿⬀-⯿️‍]"
)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    text = _MARKDOWN_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


class SentenceChunker:
    """Accumulates streamed text and emits complete sentences.

    A boundary character only splits when followed by whitespace (so
    "3.14" survives) and when the chunk has reached min_chars (so TTS
    isn't fed confetti like "Hi.").
    """

    def __init__(self, min_chars: int = 24) -> None:
        self._min_chars = min_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        chunks: list[str] = []
        while True:
            split_at = self._find_boundary()
            if split_at is None:
                break
            chunk = clean_for_speech(self._buffer[:split_at])
            self._buffer = self._buffer[split_at:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _find_boundary(self) -> int | None:
        # Stop before the final char: a trailing boundary can't be
        # confirmed until we see what follows it (feed more or flush).
        for i in range(len(self._buffer) - 1):
            char = self._buffer[i]
            if char == "\n" and i + 1 >= self._min_chars:
                return i + 1
            if (
                char in _BOUNDARY_CHARS
                and self._buffer[i + 1] in _TRAILING_OK
                and i + 1 >= self._min_chars
            ):
                return i + 1
        return None

    def flush(self) -> list[str]:
        chunk = clean_for_speech(self._buffer)
        self._buffer = ""
        return [chunk] if chunk else []
