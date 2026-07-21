"""The live-call WebSocket loop: continuous mic in -> Silero VAD -> STT -> LLM
-> TTS out, with echo-gated barge-in and graduated silence handling.

This path predates the EventBus and runs its own imperative loop rather than the
worker/queue pipeline. It is extracted here so the route handler in app.py stays
a one-liner; all provider access goes through the injected VoiceServices.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from voiceos.dashboard.voice_services import VoiceServices, retry
from voiceos.monitoring.calls import CallRecorder, CallStore
from voiceos.monitoring.pricing import estimate_call_cost
from voiceos.pipeline.events import EventBus

logger = logging.getLogger(__name__)

# Spoken when the respondent never answers. Deliberately not the campaign
# error_message, which is about a technical fault.
NO_RESPONSE_FAREWELL = "समय देने के लिए धन्यवाद, नमस्ते जी।"


class EndCall(Exception):
    """Raised to unwind out of the nested call loop when the assistant hangs up."""


async def run_call(
    ws: WebSocket,
    *,
    services: VoiceServices,
    settings,
    call_store: CallStore,
    pricing: dict,
) -> None:
    await ws.accept()
    try:
        cfg = await ws.receive_json()
    except Exception:
        await ws.close()
        return
    lang = cfg.get("language", "hi-IN")
    prompt = cfg.get("system_prompt", "")
    history: list[dict] = []

    # This path predates the EventBus and runs its own loop, so the record
    # is filled directly rather than by subscribing. Same shape either way.
    recorder = CallRecorder(
        EventBus(), store=None, direction="web",
        campaign=cfg.get("campaign"), assistant=cfg.get("assistant"),
    )
    _stt_lat: list[float] = []
    _llm_lat: list[float] = []
    reason = "completed"

    from voiceos.dashboard.streaming_vad import (
        SmartTurnEndpointer,
        StreamingEndpointer,
    )
    from voiceos.vad.silero_vad import SileroVAD

    vad = SileroVAD(settings.vad)      # per-connection state
    await vad.load()
    loop = asyncio.get_event_loop()
    smart_turn = await loop.run_in_executor(None, services.smart_turn)
    if smart_turn is not None:
        async def _predict(audio):
            return await loop.run_in_executor(None, smart_turn.complete_prob, audio)

        endpointer = SmartTurnEndpointer(vad, _predict)

        async def next_utts(pcm):
            return await endpointer.push(pcm)
    else:
        endpointer = StreamingEndpointer(vad, min_silence_ms=550)

        async def next_utts(pcm):
            return endpointer.push(pcm)

    stt = await services.stt(lang)
    speaking = False
    say_task: asyncio.Task | None = None

    from voiceos.vad.echo import EchoGate

    echo_gate = EchoGate(sample_rate=16000,
                         window_ms=settings.vad.echo_window_ms,
                         threshold=settings.vad.echo_gate_threshold)
    barge_vad = SileroVAD(settings.vad)     # separate: Silero carries state
    await barge_vad.load()
    _barge_ms = 0.0

    # Live partials. Sarvam's streaming socket transcribes while the caller
    # is still talking, so text appears as they speak instead of only after
    # they stop. The committed utterance is still transcribed in batch if
    # the stream returns nothing, so a socket failure costs partials, not
    # the turn.
    from voiceos.stt.sarvam_streaming import SarvamStreamingSTT

    _live_stt = (settings.stt.provider == "sarvam"
                 and settings.stt.sarvam_streaming)
    _stream = None
    _shown = ""
    _was_speech = False

    def nonlocal_idle_reset(at: float | None = None) -> None:
        """Start the silence clock — by default now, or when playback ends."""
        nonlocal _idle_since
        _idle_since = at if at is not None else time.monotonic()

    async def say(text: str) -> None:
        """Stream a reply. Runs as a task so the mic keeps being read —
        blocking here made barge-in impossible, since no frames were
        received at all while the assistant was talking."""
        nonlocal speaking
        speaking = True
        speak_until = time.monotonic()
        try:
            tts = await services.tts(lang)
            await ws.send_json({"type": "audio_start", "rate": tts.sample_rate})
            clean = (text or "").replace("end_call_tool", "").strip()
            if not clean:
                # An empty reply is a real failure, not something to hand to
                # TTS: the provider rejects it and the caller hears silence
                # with no indication anything went wrong.
                logger.error("live call: LLM returned no text — nothing to speak")
                await ws.send_json({"type": "error",
                                    "text": "LLM returned an empty reply"})
                await ws.send_json({"type": "listening"})
                return
            _sent = 0
            _t_first = None
            async for chunk in tts.synthesize(clean):
                pcm = np.ascontiguousarray(chunk, dtype="<i2")
                if _t_first is None:
                    _t_first = time.monotonic()
                _sent += pcm.size
                # Reference for the echo gate: this is exactly what the
                # caller is about to hear, so anything correlating with it
                # in the mic is our own voice, not an interruption.
                echo_gate.push_reference(pcm, tts.sample_rate)
                await ws.send_bytes(pcm.tobytes())
            await ws.send_json({"type": "audio_end"})
            logger.info("live call: sent %.2fs of audio (%d samples @ %d Hz)",
                        _sent / max(1, tts.sample_rate), _sent, tts.sample_rate)
            # Sending finishes far sooner than playback: eight seconds of
            # greeting streams out in about one and a half. Charging the
            # idle timer from here fired the silence nudge while the caller
            # was still listening to the greeting.
            audio_s = _sent / max(1, tts.sample_rate)
            elapsed = time.monotonic() - (_t_first or time.monotonic())
            speak_until = time.monotonic() + max(0.0, audio_s - elapsed)
        except asyncio.CancelledError:
            await ws.send_json({"type": "interrupt"})   # flush browser buffer
            raise
        except Exception:
            logger.exception("live call: TTS failed")
        finally:
            endpointer.reset()         # drop any echo captured during playback
            echo_gate.reset()
            speaking = False
            nonlocal_idle_reset(speak_until)

    async def speak(text: str) -> None:
        nonlocal say_task
        say_task = asyncio.create_task(say(text))
        try:
            await asyncio.shield(say_task)
        except asyncio.CancelledError:
            pass                        # barge-in cancelled it; keep listening
        finally:
            say_task = None
        await ws.send_json({"type": "listening"})
        await ws.send_json({"type": "listening"})

    # opening line
    try:
        reply = cfg.get("first_message") or await services.chat(
            prompt, [], "[The call has just connected. Speak your opening greeting "
            "and consent line now.]")
        history.append({"role": "assistant", "content": reply})
        await ws.send_json({"type": "reply", "text": reply})
        await speak(reply)
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "text": str(exc)})
        except Exception:
            return  # client already gone

    # Without these the loop fails silently: audio can flow for a minute
    # with the endpointer never committing, and the log looks identical to
    # a browser that is sending nothing at all.
    _frames = _samples = _ignored = _committed = 0
    _peak = 0.0
    _logged_first = False
    # Audio keeps arriving whether or not anyone speaks, so the frame
    # stream doubles as the clock: no committed utterance for this long
    # means the respondent has gone quiet.
    _idle_since = time.monotonic()
    _prompts = 0
    _no_input_s = settings.conversation.no_input_timeout_s
    _max_prompts = settings.conversation.no_input_max_prompts

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                continue
            if speaking:
                _ignored += 1
                # Rule J: stop immediately and listen, never talk over them.
                # The echo gate is what makes this safe on a loudspeaker —
                # without it the assistant hears itself and interrupts
                # itself in a loop.
                _sig = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
                _p = barge_vad.process(_sig[:512], 16000)
                _corr = echo_gate.correlation(_sig[:512])
                if (_p >= settings.vad.barge_in_threshold
                        and _corr < settings.vad.echo_gate_threshold):
                    _barge_ms += len(data) / 2 / 16
                else:
                    _barge_ms = 0.0
                if _barge_ms >= settings.vad.barge_in_speech_ms and say_task:
                    logger.info("live call: BARGE-IN fired after %.0f ms "
                                "(vad=%.2f, echo_corr=%.2f) — cancelling playback",
                                _barge_ms, _p, _corr)
                    _barge_ms = 0.0
                    say_task.cancel()
                continue
            _frames += 1
            _samples += len(data) // 2
            _pcm = np.frombuffer(data, dtype="<i2")
            # Level matters as much as arrival: 64 s of frames with the VAD
            # never latching means silence, and a frame count alone cannot
            # tell a muted device from a quiet one.
            _peak = max(_peak, float(np.abs(_pcm).max()) / 32768 if _pcm.size else 0.0)
            if not _logged_first:
                _logged_first = True
                logger.info("live call: first audio frame (%d samples)", _pcm.size)
            if _frames % 100 == 0:
                _db = 20 * np.log10(_peak) if _peak > 0 else -99
                logger.info("live call: %d frames / %.1fs in · peak %.0f dBFS%s · "
                            "in_speech=%s · %d committed",
                            _frames, _samples / 16000, _db,
                            "  <-- SILENT, check the mic device" if _db < -45 else "",
                            getattr(endpointer, "in_speech", "?"), _committed)
                _peak = 0.0
            if (not speaking and _no_input_s > 0
                    and time.monotonic() - _idle_since > _no_input_s):
                _prompts += 1
                _idle_since = time.monotonic()
                if _prompts > _max_prompts:
                    logger.info("live call: silent after %d prompts — hanging up",
                                _max_prompts)
                    await speak(NO_RESPONSE_FAREWELL)
                    await ws.send_json({"type": "ended", "reason": "no-response"})
                    reason = "no-response"
                    raise EndCall
                logger.info("live call: no input for %.0fs — nudge %d/%d",
                            _no_input_s, _prompts, _max_prompts)
                # The campaign prompt already defines the escalation
                # ("क्या आप लाइन पर हैं?" -> re-ask -> farewell), so tell the
                # model what happened rather than hardcoding a line here.
                try:
                    nudge, end_reason = await services.chat_ex(
                        prompt, history,
                        f"[SYSTEM: The respondent has been silent for "
                        f"{_no_input_s:.0f} seconds. This is silence prompt "
                        f"{_prompts} of {_max_prompts}. Follow your no-response "
                        f"procedure. Speak only the line, nothing else.]",
                        allow_end_call=True)
                except Exception:
                    logger.exception("live call: silence nudge failed")
                    nudge, end_reason = "", None
                if nudge.strip():
                    history.append({"role": "assistant", "content": nudge})
                    recorder.record.transcript.append(
                        {"role": "assistant", "text": nudge, "nudge": True})
                    await ws.send_json({"type": "reply", "text": nudge})
                    await speak(nudge)
                    _idle_since = time.monotonic()
                if end_reason:
                    await ws.send_json({"type": "ended", "reason": end_reason})
                    reason = end_reason
                    raise EndCall

            # Open the streaming socket the moment speech latches, and
            # keep feeding it so the transcript is nearly done by the time
            # the endpointer commits.
            _now_speech = getattr(endpointer, "in_speech", False)
            if _live_stt and _now_speech and _stream is None:
                try:
                    _stream = SarvamStreamingSTT(settings.stt)
                    await _stream.start()
                    _shown = ""
                except Exception:
                    logger.exception("live call: could not open streaming STT")
                    _stream = None
            if _stream is not None:
                try:
                    await _stream.send(_pcm)
                    if _stream.partial and _stream.partial != _shown:
                        _shown = _stream.partial
                        await ws.send_json({"type": "partial", "text": _shown})
                except Exception:
                    logger.exception("live call: streaming STT dropped")
                    _stream = None
            if _now_speech != _was_speech:
                await ws.send_json({"type": "speech", "on": bool(_now_speech)})
            _was_speech = _now_speech

            for utt in await next_utts(_pcm):
                _committed += 1
                _idle_since = time.monotonic()   # they spoke; reset the clock
                _prompts = 0
                logger.info("live call: utterance committed (%.2fs)", utt.size / 16000)
                await ws.send_json({"type": "thinking"})
                _t0 = time.perf_counter()
                transcript = ""
                if _stream is not None:
                    try:
                        transcript, _ = await _stream.finish(quiet_ms=150)
                        transcript = (transcript or "").strip()
                    except Exception:
                        logger.exception("live call: streaming finish failed")
                    finally:
                        _stream = None
                        _shown = ""
                try:
                    if not transcript:
                        res = await retry(lambda: stt.transcribe(utt, 16000))
                        transcript = (res.text or "").strip()
                except Exception as exc:
                    logger.exception("live call: STT failed")
                    await ws.send_json({"type": "error",
                                        "text": f"STT: {type(exc).__name__}"})
                    transcript = ""
                _stt_lat.append(time.perf_counter() - _t0)
                if not transcript:
                    await ws.send_json({"type": "listening"})
                    continue
                await ws.send_json({"type": "transcript", "text": transcript})
                recorder.record.transcript.append(
                    {"role": "user", "text": transcript, "language": lang})
                _t1 = time.perf_counter()
                try:
                    reply, end_reason = await services.chat_ex(
                        prompt, history, transcript, allow_end_call=True)
                except HTTPException as he:
                    logger.warning("live call: LLM rejected (%s)", he.detail)
                    await ws.send_json({"type": "error", "text": he.detail})
                    await ws.send_json({"type": "listening"})
                    continue
                except Exception as exc:
                    # Previously any non-HTTP error escaped to the outer
                    # handler and silently ended the call; the browser just
                    # stopped responding with no reason shown anywhere.
                    logger.exception("live call: LLM failed")
                    await ws.send_json({"type": "error",
                                        "text": f"LLM: {type(exc).__name__}: {exc}"})
                    await ws.send_json({"type": "listening"})
                    continue
                _llm_lat.append(time.perf_counter() - _t1)
                recorder.record.transcript.append({"role": "assistant", "text": reply})
                recorder.record.turns += 1
                history.append({"role": "user", "content": transcript})
                history.append({"role": "assistant", "content": reply})
                await ws.send_json({"type": "reply", "text": reply})
                await speak(reply)
                if end_reason:
                    logger.info("live call: assistant ended the call (%s)", end_reason)
                    await ws.send_json({"type": "ended", "reason": end_reason})
                    reason = end_reason
                    raise EndCall
        reason = "customer-ended-call"
    except EndCall:
        pass                       # assistant hung up; record it and close
    except WebSocketDisconnect:
        reason = "customer-ended-call"
    except Exception:
        logger.exception("live call ended on an unhandled error")
        reason = "pipeline-error"
    finally:
        recorder._samples["stt_s"] = _stt_lat
        recorder._samples["llm_total_s"] = _llm_lat
        rec = recorder.finish(ended_reason=reason)
        rec.cost = estimate_call_cost(
            rec.duration_s, rec.turns,
            stt_provider=settings.stt.provider,
            tts_provider=settings.tts.provider,
            llm_model=settings.llm.model,
            pricing=pricing,
        )
        call_store.add(rec)
