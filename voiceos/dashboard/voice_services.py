"""Shared voice-provider layer for the dashboard.

Lazily loads and caches the LLM, per-language STT, and per-language TTS, and
owns the one-turn chat helper. Both the REST voice routes and the live-call
WebSocket loop go through this, so provider construction and caching live in
exactly one place instead of being duplicated between them.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import HTTPException

from voiceos.dashboard.audio_io import wav_b64
from voiceos.interfaces.llm import BaseLLM
from voiceos.llm.tools import END_CALL_SCHEMA, wants_end_call

logger = logging.getLogger(__name__)

# UI BCP-47 code -> Cartesia language code (Sarvam STT autodetects, no map needed).
TTS_LANG = {
    "hi-IN": "hi", "hi": "hi", "te-IN": "te", "te": "te", "en-IN": "en",
    "en-US": "en", "en": "en", "mr-IN": "hi", "ta-IN": "ta", "kn-IN": "kn",
    "bn-IN": "bn", "gu-IN": "gu",
}


async def retry(make_coro: Callable, tries: int = 3, delay: float = 0.6):
    """Retry a coroutine factory. Backs off on 429/5xx but stays snappy for a
    live call — fail in a few seconds and let the caller re-capture rather than
    freezing the turn."""
    import httpx

    last: Exception | None = None
    for i in range(tries):
        try:
            return await make_coro()
        except httpx.HTTPStatusError as exc:
            last = exc
            code = exc.response.status_code
            if code == 429 or code >= 500:  # rate limit or transient server error
                ra = exc.response.headers.get("retry-after", "")
                wait = float(ra) if ra.replace(".", "", 1).isdigit() else delay * (2 ** i) + 0.5
                await asyncio.sleep(min(wait, 4))
            else:
                raise  # 4xx (bad request / auth) won't improve on retry
        except Exception as exc:  # noqa: BLE001 - varied network errors
            last = exc
            await asyncio.sleep(delay)
    raise last  # type: ignore[misc]


class VoiceServices:
    """Loaded-once LLM/STT/TTS providers plus the chat helper."""

    def __init__(self, settings, llm_factory: Callable[[], BaseLLM]) -> None:
        self._settings = settings
        self._llm_factory = llm_factory
        self._llm: BaseLLM | None = None
        self._stt: dict[str, object] = {}       # keyed by BCP-47 language
        self._tts: dict[str, object] = {}       # keyed by Cartesia language code
        self._smart_turn = None
        self._smart_turn_loaded = False

    async def llm(self) -> BaseLLM:
        if self._llm is None:
            inst = self._llm_factory()
            await inst.load()
            self._llm = inst
        return self._llm

    async def stt(self, language: str):
        if language not in self._stt:
            from voiceos.pipeline.pipeline import create_stt

            s2 = self._settings.model_copy(deep=True)
            # Force the language instead of autodetect — Sarvam otherwise
            # mis-detects Hindi as Kannada/Bengali/Gujarati/etc.
            s2.stt.sarvam_language = language
            inst = create_stt(s2)
            await inst.load()
            self._stt[language] = inst
        return self._stt[language]

    async def tts(self, language: str):
        code = TTS_LANG.get(language, "en")
        if code not in self._tts:
            from voiceos.pipeline.pipeline import create_tts

            s2 = self._settings.model_copy(deep=True)
            s2.tts.cartesia_language = code
            inst = create_tts(s2)
            await inst.load()
            self._tts[code] = inst
        return self._tts[code]

    def smart_turn(self):
        """Load Smart Turn v3 once if the model + deps are present, else None
        (graceful fallback to plain silence endpointing)."""
        if self._smart_turn_loaded:
            return self._smart_turn
        self._smart_turn_loaded = True
        model = Path("models/smart-turn-v3.2-cpu.onnx")
        if not model.exists():
            logger.info("Smart Turn model not found; using silence endpointing")
            return None
        try:
            from voiceos.dashboard.smart_turn import SmartTurn

            st = SmartTurn(str(model))
            st.load()
            self._smart_turn = st
            logger.info("Smart Turn v3 enabled (semantic end-of-turn)")
            return st
        except Exception as exc:  # noqa: BLE001
            logger.warning("Smart Turn unavailable (%s); using silence endpointing", exc)
            return None

    async def _synth_chunks(self, tts, text: str) -> list:
        """Collect TTS chunks, retrying if it fails before emitting any audio
        (Cartesia occasionally returns a transient 4xx/5xx that clears)."""
        last: Exception | None = None
        for _ in range(3):
            try:
                out = []
                async for c in tts.synthesize(text):
                    out.append(c)
                return out
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(0.4)
        raise last  # type: ignore[misc]

    async def synthesize(self, text: str, language: str) -> str:
        tts = await self.tts(language)
        clean = (text or "").replace("end_call_tool", "").strip()
        chunks = await self._synth_chunks(tts, clean) if clean else []
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return wav_b64(audio, tts.sample_rate)

    async def chat(self, system_prompt: str, history: list, user_message: str,
                   allow_end_call: bool = False) -> str:
        text, _ = await self.chat_ex(system_prompt, history, user_message, allow_end_call)
        return text

    async def chat_ex(self, system_prompt: str, history: list, user_message: str,
                      allow_end_call: bool = False) -> tuple[str, str | None]:
        """One LLM turn: system + trimmed recent history + user. Trimming caps
        tokens-per-request so long prompts don't blow the provider rate limit."""
        import httpx

        turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        msgs = [{"role": "system", "content": system_prompt}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in turns[-16:]]
        msgs.append({"role": "user", "content": user_message})
        llm = await self.llm()
        tools = [END_CALL_SCHEMA] if allow_end_call else None
        try:
            r = await retry(lambda: llm.complete(msgs, tools=tools))
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429:
                raise HTTPException(429, "LLM is rate-limited — wait a moment and try again.")
            if code >= 500:
                raise HTTPException(503, "LLM service is busy right now — please tap and "
                                    "speak again.")
            raise HTTPException(502, f"LLM error: {exc}")
        end, reason = wants_end_call(r) if allow_end_call else (False, "")
        text = (r.get("content", "") if r else "") or ""
        # The model is told to speak a farewell and call the tool in the same
        # turn. If it only called the tool, there is nothing to say — hanging up
        # in silence is worse than a short goodbye.
        if end and not text.strip():
            text = "धन्यवाद, नमस्ते जी।"
        return text, (reason if end else None)
