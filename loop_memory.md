# Loop Memory — Listening Pipeline → Production Quality

Mission: transform the local (`main.py`) listening pipeline into production quality
comparable to Vapi. Baseline captured 2026-07-09.

## Baseline
- Test suite: **131 passed** (clean) before any change.
- Reported symptom: on laptop, speech often not detected; assistant "waits again"
  after the user already spoke.

## Definition of Done — status
| # | Condition | Status |
|---|---|---|
| 1 | Laptop mic detects speech reliably | ◑ code-fixed (iter 1), needs live mic |
| 2 | Short utterances recognized | ◑ code-fixed (iter 1), needs live mic |
| 3 | No speech lost during THINKING | ☑ done (iter 2) |
| 4 | Proper barge-in | ☐ |
| 5 | Streaming STT | ☐ |
| 6 | Hysteresis VAD | ☑ done (iter 1) |
| 7 | Adaptive endpointing | ☑ done (iter 3, Smart Turn v3; opt-in) |
| 8 | Latency < 600 ms | ☐ (needs live mic + provider timing) |
| 9 | No dropped utterances | ◑ THINKING window covered (iter 2); SPEAKING = barge-in (iter 5) |
| 10 | 5 successful end-to-end conversations | ☐ (needs live mic) |

## Root-cause map (from code + research)
- `SpeechDetector` (main.py path) uses a **single 0.5 threshold** to both start and
  continue speech → mid-word probability dips read as silence → flicker, early cutoff,
  short answers rejected. (Conditions 1, 2, 6)
- `min_speech_ms=250` counts only above-threshold frames → short answers ("అవును")
  discarded as noise. (Condition 2)
- Mic frames **dropped while THINKING/SPEAKING** → fast follow-ups lost. (Conditions 3, 9)
- Batch STT only; no partials feeding the turn → latency + no streaming. (Conditions 5, 8)
- The working fixes already exist in `dashboard/streaming_vad.py`
  (`StreamingEndpointer` hysteresis, `SmartTurnEndpointer`) — not wired into `main.py`.

## Planned iterations (largest bottleneck first)
1. Hysteresis VAD + relaxed min_speech in `SpeechDetector`. (1, 2, 6)
2. Adaptive endpointing / semantic end-of-turn in local path. (7)
3. Buffer audio during THINKING instead of dropping. (3, 9)
4. Streaming STT + latency instrumentation. (5, 8)
5. Barge-in robustness / AEC guidance. (4)
6. End-to-end verification pass. (10)

## Iteration log

### Iteration 1 — Hysteresis VAD + short-utterance fix ✅
- **Files:** `config/settings.py` (added `neg_threshold`, validator defaulting to
  `threshold-0.15`; `min_speech_ms` 250→150), `vad/detector.py` (start bar
  `threshold`, stay bar `neg_threshold` while recording), `tests/test_detector.py`
  (new dip test).
- **Verify:** 132 passed (was 131). neg_threshold defaults confirmed 0.5→0.35,
  0.6→0.45, explicit respected.
- **Result:** DoD 6 done; 1 & 2 code-fixed (await live-mic confirmation).
- **Bottleneck fixed:** single-threshold flicker on quiet laptop mic.

### Iteration 2 — Capture speech during THINKING ✅
- **Files:** `vad/detector.py` (split SPEAKING/THINKING handling; new
  `_capture_during_thinking` + `_flush_pending`; `_pending` field; cleared in
  `_reset`), `tests/test_detector.py` (2 new tests).
- **Design:** capture restricted to THINKING (assistant silent). Completed
  utterance held in `_pending`, flushed on first IDLE frame (→THINKING + enqueue).
  SPEAKING still drops/hands to barge-in — never captures (echo safety).
- **Verify:** 134 passed (was 132).
- **Result:** DoD 3 done; 9 partially (SPEAKING echo path deferred to barge-in iter).

### Iteration 3 — Adaptive endpointing via Smart Turn v3 ✅
- **Files:** `config/settings.py` (smart_turn flags), `vad/detector.py`
  (`turn_predictor` param, `_maybe_predict_turn`/`_predict_turn`/`_cancel_turn`,
  reuses `_endpoint_predicted`→`predicted_silence_ms`), `pipeline/pipeline.py`
  (background model load, predict closure), `tests/test_detector.py` (2 tests).
- **Design:** on a short pause, run local Smart Turn on raw waveform; "complete"
  shortens required silence, "incomplete" keeps the full timer. No STT calls.
  Off by default; background-loads so startup isn't blocked.
- **Gotcha found:** a pre-filled queue never yields, so the async predict task
  starved in the test — fixed the test (two-phase feed). Production mic yields
  between 32 ms frames, so this is test-only.
- **Verify:** 136 passed. Real model load 24.6 s (justifies background load);
  `complete_prob` returns valid probability. Enable with `VOICEOS_VAD__SMART_TURN=true`.
- **Result:** DoD 7 done.

## Iteration log
(next iterations recorded below as they complete)
