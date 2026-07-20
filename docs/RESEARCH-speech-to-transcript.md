# From Utterance to Transcript

**An internal architecture document for VoiceOS**
Scope: the instant a user begins speaking → a stable final transcript. No LLM, no TTS, except where required to explain endpointing.

---

## 0. Epistemic preface — read this first

This document mixes three kinds of statement. They are labeled inline and you should treat them very differently when making engineering decisions.

| Tag | Means | How to treat it |
|---|---|---|
| **[C]** Confirmed | Direct evidence: source code, official docs, RFC, paper, or a claim that survived 3-vote adversarial verification | Build on it |
| **[L]** Likely | Strong inference from open-source behavior, API surface, benchmarks, or standard practice | Verify before betting the architecture on it |
| **[S]** Speculative | Reasoned guess, no direct evidence | Treat as a hypothesis to test |
| **[R]** Refuted | A plausible-sounding claim that **failed** verification | Do not repeat this, even though you will see it elsewhere |

### Claims that failed verification

These are here because they are widely repeated and you will encounter them. A verification pass could not substantiate them from primary sources:

- **[R]** "Vapi performs end-of-turn detection using a custom *fusion audio-text* model." Vapi's docs confirm a suite of real-time models exists **[C]**, but the specific fusion-audio-text architecture claim was not substantiated.
- **[R]** "Vapi applies a proprietary noise-filtering model plus a separate speaker-isolation model that suppresses background voices (TV, other people)." Not substantiated from primary sources.
- **[R]** "Deepgram Flux cuts agent response latency by 200–600 ms, with ~70% of end-of-turn detections within 500 ms and p90/p95 EoT latency of 1 s / 1.5 s." The *mechanism* claims about Flux verified cleanly; these specific **performance numbers did not**. Benchmark them yourself before designing to them.

### Known gaps in this document

Verification was interrupted mid-run. The following are **[L]** rather than **[C]** and are flagged again where they appear:

- whisper_streaming's LocalAgreement-*n* policy details and its reported 3.3 s long-form latency
- Google's prefix-reranking deflicker cost/benefit numbers (α = 0.2, +1–2 ms)
- Chunk-size/WER degradation curves for streaming Conformer
- Buffer-trimming strategies in whisper_streaming

Sources are listed in §12.

---

## 1. PHASE 1 — Complete architecture

### 1.1 The four ingress paths

A production voice platform does not have *one* audio pipeline. It has four, and they differ in sample rate, codec, framing, loss characteristics, and which DSP has already been applied before you see a sample. Conflating them is the single most common architectural mistake.

```
┌─ PATH A: BROWSER / WebRTC ────────────────────────────────────────────┐
│ mic → OS stack → getUserMedia → WebRTC APM (AEC3/NS/AGC2)             │
│   → Opus 48k @20ms → SRTP/UDP → SFU or direct → NetEQ → 48k PCM       │
│ Loss: recoverable (FEC/PLC). AEC: done client-side. Rate: 48k.        │
└───────────────────────────────────────────────────────────────────────┘

┌─ PATH B: BROWSER / RAW WEBSOCKET  ← VoiceOS today ────────────────────┐
│ mic → getUserMedia(ec/ns/agc) → AudioContext(16k)                     │
│   → ScriptProcessor/AudioWorklet → Int16 → WS binary → server         │
│ Loss: none (TCP) but HOL blocking. No jitter buffer. Rate: 16k.       │
└───────────────────────────────────────────────────────────────────────┘

┌─ PATH C: SIP / PSTN ──────────────────────────────────────────────────┐
│ handset → carrier → SBC → SIP/RTP G.711 8k @20ms → Asterisk/FS        │
│   → AudioSocket(TCP,SLIN16) or mod_audio_stream(WS) → server          │
│ Loss: real, unrecoverable. AEC: carrier-side, unreliable. Rate: 8k.   │
└───────────────────────────────────────────────────────────────────────┘

┌─ PATH D: TWILIO MEDIA STREAMS ────────────────────────────────────────┐
│ PSTN → Twilio edge → WebSocket JSON, base64 μ-law 8k @20ms (160 B)    │
│ Events: connected|start|media|stop|mark|clear. Rate: 8k.              │
└───────────────────────────────────────────────────────────────────────┘
```

**The unification point.** Every path must converge on one internal representation before VAD. For VoiceOS that is **int16 mono @ 16 kHz in exactly 512-sample frames** — because Silero v5 requires exactly 512 samples at 16 kHz **[C]**. Everything upstream of that is transport-specific; everything downstream is transport-agnostic. Your `MediaStreamTransport` + `_Rechunker` is precisely this seam, and it is architecturally correct.

### 1.2 Full server-side pipeline

```
                        INGRESS (per-call, N concurrent)
   ┌──────────────────────────────────────────────────────────────┐
   │  transport adapter                                            │
   │    ├─ decode  (μ-law→PCM16 | Opus→PCM | SLIN passthrough)     │
   │    ├─ jitter buffer  (RTP paths only — NetEQ or equivalent)   │
   │    ├─ PLC / FEC      (RTP paths only)                         │
   │    └─ resample       (8k|48k → 16k, stateful per direction)   │
   └───────────────────────────┬──────────────────────────────────┘
                               │  int16 @16k, variable chunk
                    ┌──────────▼──────────┐
                    │   RECHUNKER         │  20ms in → 32ms out
                    │   (ring buffer)     │  512-sample frames
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐   bounded, drop-oldest
                    │   AUDIO QUEUE       │   backpressure boundary #1
                    └──────────┬──────────┘
                               │
   ┌───────────────────────────▼──────────────────────────────────┐
   │  SEGMENTATION WORKER  (the state machine)                     │
   │    ├─ noise-floor estimator   (rolling percentile)            │
   │    ├─ VAD                     (Silero, 512@16k, stateful)     │
   │    ├─ hysteresis latch        (on 0.5 / off 0.35)             │
   │    ├─ pre-roll ring           (~300 ms)                       │
   │    ├─ endpoint policy         (silence timer + semantic)      │
   │    └─ barge-in watcher        (during SPEAKING only)          │
   └───────────────────────────┬──────────────────────────────────┘
                    utterance   │   partial-audio taps
              ┌────────────────┴────────────────┐
              │                                  │
   ┌──────────▼─────────┐          ┌────────────▼────────────┐
   │  ASR (final)       │          │  ASR (partial/rolling)  │
   │  batch or stream   │          │  separate model inst.   │
   └──────────┬─────────┘          └────────────┬────────────┘
              │                                  │
              │                     ┌────────────▼────────────┐
              │                     │  TURN PREDICTOR         │
              │                     │  Smart Turn (audio) or  │
              │                     │  LM classifier (text)   │
              │                     └────────────┬────────────┘
              │                                  │ shortens silence timer
              └──────────────┬───────────────────┘
                             ▼
                     FINAL TRANSCRIPT
```

### 1.3 Thread / concurrency model

There are exactly **three** places a real-time audio pipeline crosses a thread boundary, and every one is a latency and correctness hazard.

```
┌─ HARD-REALTIME (OS audio callback thread) ────────────────────┐
│  Driver → capture callback. Deadline: one buffer period.      │
│  NEVER: allocate, lock, log, do I/O, run inference.           │
│  ONLY: copy the buffer, hand off lock-free, return.           │
└───────────────────────────────────────────────────────────────┘
                    │ lock-free SPSC ring / call_soon_threadsafe
┌─ SOFT-REALTIME (event loop) ──────────────────────────────────┐
│  Decode, resample, rechunk, VAD, state machine, endpointing.  │
│  Budget: << frame period (32 ms). Any blocking call here      │
│  stalls EVERY call on this process.                           │
└───────────────────────────────────────────────────────────────┘
                    │ executor / IPC / RPC
┌─ THROUGHPUT (worker pool, GPU) ───────────────────────────────┐
│  ASR inference, turn prediction, model loading.               │
│  Batched, queued, backpressured. Tail latency lives here.     │
└───────────────────────────────────────────────────────────────┘
```

**[C] VoiceOS finding:** `SileroVAD.process()` is synchronous and called directly on the event loop from `SpeechDetector._process_frame`. At ~1 ms/frame for the ONNX build this is survivable single-call, but it is a hard scaling ceiling — at N concurrent calls you serialize N VAD inferences into every 32 ms window. Above roughly 25–30 concurrent calls per process **[S]** you will start missing frame deadlines. The fix is not "make it async" (that adds hop latency) but **batched VAD**: accumulate frames from all active calls into one batched ONNX invocation per tick.

### 1.4 Sequence diagram — one turn, telephony path

```
Caller     Carrier    Asterisk    Transport   Rechunk   VAD/SM    ASR
  │           │          │            │          │        │        │
  │─speech───▶│          │            │          │        │        │
  │           │─RTP 20ms▶│            │          │        │        │
  │           │          │─SLIN 8k───▶│          │        │        │
  │           │          │            │─decode──▶│        │        │
  │           │          │            │ resample │        │        │
  │           │          │            │  8k→16k  │        │        │
  │           │          │            │          │─512f──▶│        │
  │           │          │            │          │        │ p=0.72 │
  │           │          │            │          │        │ LATCH  │
  │           │          │            │          │        │ +preroll
  │           │          │            │          │        │        │
  │           │          │            │       ...speech...│        │
  │           │          │            │          │        │        │
  │  (pause)  │          │            │          │        │ 300ms  │
  │           │          │            │          │        │ ─turn?─▶│ SmartTurn
  │           │          │            │          │        │ ◀0.87── │ "complete"
  │           │          │            │          │        │ shorten │
  │           │          │            │          │        │ timer   │
  │           │          │            │          │        │ →250ms  │
  │           │          │            │          │        │ COMMIT  │
  │           │          │            │          │        │─utt────▶│
  │           │          │            │          │        │        │ decode
  │           │          │            │          │        │◀─text──│
  │           │          │            │          │        │        │
   t=0        +5ms       +25ms        +26ms      +58ms    +~1.4s   +1.7s
```

The critical observation: **of the ~1.7 s from mouth-close to final transcript, only ~300 ms is compute.** The rest is deliberately waiting to be sure the user stopped. That is why endpointing — not ASR speed — is the dominant lever on perceived latency.

### 1.5 State machine

```
                    ┌──────────────────────────────────┐
                    │             IDLE                 │
                    │  mic live, pre-roll ring filling │
                    └───────┬──────────────────────────┘
                            │ p ≥ on_threshold ∧ energy > floor×margin
                            ▼
                    ┌──────────────────────────────────┐
        ┌───────────│          LISTENING               │
        │           │  recording; stay-bar = off_thresh│
        │           └───────┬──────────────────────────┘
        │  p≥on             │ silence ≥ required_silence_ms
        │  (resume)         ▼
        │           ┌──────────────────────────────────┐
        │           │       ENDPOINT PENDING           │
        └───────────│  (soft state) turn predictor in  │
                    │  flight; timer may be shortened  │
                    └───────┬──────────────────────────┘
                            │ commit
                            ▼
                    ┌──────────────────────────────────┐
                    │          THINKING                │
                    │  ASR/LLM in flight.              │
                    │  MIC STAYS LIVE → capture to     │
                    │  pending slot (assistant silent) │
                    └───────┬──────────────────────────┘
                            │ first audio chunk
                            ▼
                    ┌──────────────────────────────────┐
                    │          SPEAKING                │
                    │  barge-in watcher ONLY.          │
                    │  NEVER capture to pending here   │
                    │  (that is the echo path).        │
                    └───────┬──────────────────────────┘
                            │ barge-in: bump generation, drain, resume
                            ▼
                         LISTENING
```

**Why `THINKING` and `SPEAKING` must be handled differently:** during THINKING the assistant is silent, so the microphone signal is pure user. During SPEAKING the microphone contains the assistant's own voice unless AEC is perfect. Capturing to a pending slot during SPEAKING is not a feature — it is a self-transcription bug. VoiceOS gets this right and documents it inline.

---

## 2. PHASE 2 + 3 — Failure catalog with solutions

Phases 2 and 3 are merged: a failure mode and its solution belong on the same page.

### 2.1 Capture-stage failures

---

#### F-01 · Lost first syllable (clipped onset)

**Why.** VAD is a *detector*, not an oracle. It needs energy to have already arrived before it can fire. Silero at 512 samples needs at least one frame (32 ms) and typically 2–3 frames to cross threshold. Plosives and unvoiced onsets ("s", "f", "th", "k") carry little low-frequency energy and are systematically late-detected. If you start recording *at* the detection instant, you have already thrown away 60–100 ms of speech — usually the entire first phoneme.

**Detection.** Transcripts systematically missing leading consonants. "top" → "op". "Sixty" → "ixty". High WER concentrated in word-initial position.

**Naive fix and why it fails.** Lower the VAD threshold. This does reduce onset latency, but it trades a clipping problem for a false-trigger problem — you now latch on breath, HVAC, and line noise, and every false latch costs a full ASR call plus a spurious turn.

**Production solution: pre-roll ring buffer.** Continuously push frames into a fixed-size ring while idle. On latch, *prepend* the ring contents to the utterance. This is retroactive capture: the audio was already recorded, you just decided later that you wanted it.

```
        detection instant
              │
  ...idle.....▼....speech.....
     └──────┘ pre-roll ring (300 ms) — prepended on latch
```

**Sizing.** 200–500 ms. Below 200 ms you still clip slow onsets; above 500 ms you prepend meaningful background noise, which hurts Whisper specifically (it hallucinates on leading noise — see F-31).

**Cost.** Memory only: `preroll_ms / frame_ms × frame_bytes`. At 300 ms/16 k/int16 ≈ 9.6 KB per call. Zero latency, zero CPU.

**Failure case.** If the ring is not *cleared* on latch, a subsequent utterance re-prepends stale audio from the previous turn. **[C] VoiceOS handles this correctly** — `UtteranceRecorder.start()` moves the ring into the utterance and clears it, rather than copying.

---

#### F-02 · Lost last syllable (premature endpoint)

**Why.** Trailing unvoiced consonants and low-energy vowel decay fall below threshold before the speaker is actually done. If your silence timer starts the instant probability drops, you truncate.

**Production solution: hangover.** Frames below threshold during an active utterance are *still appended* to the recording; only the timer runs. When you finally commit, the trailing pad is already in the buffer.

This is subtle and often missed: hangover is not "wait longer before committing" — it is "keep recording while you wait." The distinction matters because the two behaviors have identical timers but different buffers.

**[C] VoiceOS implements this** in both `SpeechDetector` (frames pushed regardless of speech state while recording) and `StreamingEndpointer` (explicit trailing-pad append).

---

#### F-03 · Microphone clipping / saturation

**Why.** Input gain too high; sample values pin at ±32767. Clipping introduces broadband harmonic distortion that destroys the spectral envelope ASR relies on. It is *not* recoverable downstream — the information is gone.

**Detection.** Count samples at |x| ≥ 32700 per frame. Sustained >0.1% is clipping. Emit as a per-call metric; it is one of the few audio-quality signals that is cheap and unambiguous.

**Naive fix.** Normalize after capture. Useless — you are scaling already-destroyed data.

**Production solution.** Analog gain control at the source. WebRTC's AGC1 has an *analog* mode that drives the OS mixer gain down when clipping is detected **[C]**; AGC2 adds a saturation protector driven by an RNN VAD **[C]**. In the browser, `autoGainControl: true` enables this. On telephony you cannot control it — the carrier's AGC already ran — so detect and report only.

**Tradeoff.** AGC fights VAD: it raises gain during silence, lifting the noise floor and causing false triggers. This is why adaptive noise-floor estimation (F-11) must be *relative* (percentile-tracking) rather than absolute-threshold.

---

#### F-04 · Sample-rate mismatch

**Why.** Silero v5 requires exactly 512 samples @ 16 kHz **[C]**. faster-whisper requires 16 kHz. Telephony delivers 8 kHz. WebRTC delivers 48 kHz. A mismatch does not error — it silently produces garbage, because the model interprets the array as if it were at the expected rate. 8 kHz audio fed to a 16 kHz model sounds pitch-doubled and time-halved to the model.

**Production solution.** Resample at the transport boundary, once, statefully. **Stateful matters**: `audioop.ratecv` (and any polyphase resampler) carries filter state across chunks. Resampling each 20 ms chunk independently produces a discontinuity at every boundary — an audible click at 50 Hz, and a broadband artifact that degrades WER.

**[C] VoiceOS gets this right** — `Resampler` rebinds `ratecv` state per chunk, per direction, per call.

**Quality note.** 8 k → 16 k upsampling does not create information. Telephony speech is band-limited to ~3.4 kHz; the 4–8 kHz band after upsampling is empty. ASR models trained on wideband audio underperform on upsampled narrowband. **[L]** The production answer is a telephony-specific acoustic model (Deepgram, Speechmatics, and Riva all expose 8 kHz "phonecall" model variants) rather than upsampling into a wideband model.

---

#### F-05 · Buffer-size / callback-thread violations

**Why.** The OS audio callback has a hard deadline. Anything that can block — malloc, mutex, logging, GC — can overrun it, producing an xrun (dropped buffer). Dropped input buffers are silent data loss: no error, just missing audio.

**[C] VoiceOS finding:** `live.html` uses `createScriptProcessor(1024, 1, 1)`. `ScriptProcessorNode` is **deprecated and runs its callback on the main thread**, competing with layout, GC, and every other JS task. Under load it drops buffers. The modern replacement is `AudioWorkletNode`, which runs on a dedicated audio rendering thread with a fixed 128-sample render quantum **[C]**.

This is a real, actionable bug in the browser client. The migration is roughly 30 lines and eliminates a whole class of "the transcript randomly missed a word" reports.

**Detection.** In `sounddevice`/PortAudio, the `status` flag in the callback carries `input_overflow`. **[C] VoiceOS logs this** at WARNING in `Microphone._callback` — good. Count it as a metric, don't just log it.

---

### 2.2 Preprocessing failures

---

#### F-06 · Acoustic echo → self-interruption

**Why.** The assistant's TTS output leaves the speaker, traverses the room, and re-enters the microphone. Your VAD sees energy. If barge-in is enabled, the assistant interrupts *itself*, then transcribes its own voice, then responds to itself. This is the single most catastrophic failure mode in speakerphone deployments because it is self-sustaining.

**Why it's hard.** The echo path is a time-varying linear system with 100–500 ms of delay, plus nonlinearity from speaker/amplifier saturation. Linear filtering alone cannot remove it.

**Production solution: WebRTC AEC3** **[C]** (`modules/audio_processing/aec3/`). Architecture:

```
  far-end (TTS out) ──┬────────────────────────▶ speaker
                      │
                 ┌────▼─────────────┐
                 │ delay estimator  │  cross-correlation on
                 │                  │  downsampled envelopes
                 └────┬─────────────┘
                      │ aligned reference
                 ┌────▼─────────────┐
                 │ linear AEC       │  partitioned-block
                 │ (adaptive FIR)   │  frequency-domain NLMS
                 └────┬─────────────┘
                      │ residual (nonlinear echo remains)
                 ┌────▼─────────────┐
                 │ residual echo    │  spectral suppression gain
                 │ suppressor       │  driven by ERLE estimate
                 └────┬─────────────┘
                      │
                 ┌────▼─────────────┐
                 │ comfort noise    │  fills spectral holes
                 └────┬─────────────┘
                      ▼  to VAD/ASR
```

AEC3 processes in 10 ms frames and splits into bands via a QMF filterbank so the adaptive filter runs on the band that matters **[C]**. AECM is the fixed-point mobile variant.

**The architectural consequence for you:** AEC needs the **far-end reference signal** — the exact samples sent to the speaker, time-aligned. In a browser, `echoCancellation: true` gets this for free because the browser owns both ends. **In a server-side pipeline you do not have it** unless you explicitly plumb the TTS output back into the AEC as the reference. This is why server-side AEC is rare and why most platforms rely on client/carrier AEC.

**What VoiceOS does instead:** gates the microphone during SPEAKING and requires a stricter barge-in threshold (0.7 vs 0.5) plus 250 ms of *contiguous* speech. **[C]** This is the correct mitigation when you lack a reference signal — it trades barge-in sensitivity for echo immunity. The documented caveat ("works best with headphones") is honest and correct.

**Tradeoffs table:**

| Approach | Echo immunity | Barge-in latency | Needs reference | Cost |
|---|---|---|---|---|
| Full AEC3 | Excellent | ~0 | **Yes** | ~5–10% of a core |
| Half-duplex gate | Perfect | ∞ (no barge-in) | No | 0 |
| Threshold + duration gate (VoiceOS) | Good | 250 ms | No | 0 |
| Echo-aware VAD (correlate mic vs TTS) | Good | ~50 ms | Yes | Low |

**[S]** The cheapest large improvement available to VoiceOS: since you already have the TTS PCM in `PlaybackWorker`, compute a running cross-correlation between the mic frame and the recent TTS output. If correlation exceeds a threshold, suppress the barge-in trigger regardless of VAD probability. This is ~40 lines, needs no adaptive filter, and kills the dominant false-barge-in mode on speakers.

---

#### F-07 · Stationary background noise (fan, HVAC, road, hum)

**Why.** Broadband stationary noise raises the noise floor uniformly. Energy VADs false-trigger continuously. Neural VADs are more robust but still degrade below ~5 dB SNR **[L]**.

**Production solutions, in order of sophistication:**

1. **Spectral subtraction / Wiener filtering** (WebRTC NS) **[C]**. Estimate the noise spectrum during non-speech via minimum statistics, subtract it. Cheap, well-understood, introduces "musical noise" artifacts at aggressive settings.
2. **RNNoise** **[C]** — hybrid DSP + GRU. Operates on 22 Bark-scale bands plus pitch features (~42 inputs), ~85 k parameters, roughly 40 MFLOPS, 10 ms frames at 48 kHz. It predicts per-band gains rather than a full spectral mask, which is why it is so cheap. **[C]** Pipecat ships an RNNoise filter as a built-in audio utility.
3. **Deep-learning full-band suppressors** (DeepFilterNet, Krisp, NVIDIA Maxine/RTX Voice) **[L]**. Substantially better on non-stationary noise; 10–100× the compute.

**The counterintuitive tradeoff.** Aggressive noise suppression *hurts* ASR. Denoisers are optimized for perceptual quality (human listeners), not for the feature distribution the acoustic model was trained on. Suppression artifacts are out-of-distribution for the ASR encoder. **[L]** The production pattern used by several vendors is to **run denoising on the VAD path but feed the ASR the unsuppressed (or lightly suppressed) signal** — you want clean speech/non-speech decisions and untouched acoustics.

This is a genuinely non-obvious architectural point and worth implementing as a fork in the pipeline:

```
              ┌──▶ denoise ──▶ VAD ──▶ speech/silence decisions
  raw audio ──┤
              └──▶ (light HPF only) ──▶ ASR
```

---

#### F-08 · Non-stationary noise: keyboard, dog, door slam, cough

**Why.** Transient, broadband, speech-like in energy envelope. Minimum-statistics noise estimators cannot track them (they are, by construction, tracking the *minimum*).

**Production solution.** WebRTC ships a dedicated **transient suppressor** for keyboard clicks **[C]**. Beyond that, this is where neural VAD decisively beats energy VAD — Silero is trained on noise-augmented data and assigns low speech probability to impulsive transients **[L]**.

**Residual failure.** Coughs and laughter are *produced by the vocal tract* and therefore have speech-like spectral structure. Every VAD fires on them. The mitigation is not at the VAD layer but at the **ASR + semantic layer**: Whisper transcribes a cough as "" or "(coughs)" or hallucinates; a semantic turn detector correctly classifies the resulting fragment as not-a-complete-turn.

---

#### F-09 · Room reverberation

**Why.** Late reflections smear the spectral envelope over 200–800 ms (RT60). Consonant discrimination collapses. This is the dominant failure mode in conference rooms and hard-surfaced spaces.

**Production solution.** Weighted Prediction Error (WPE) dereverberation **[L]** — estimates and subtracts the late-reverberation component via multichannel linear prediction. Available in `nara_wpe`. Single-channel WPE works but multichannel is much stronger.

**Cost/benefit for a voice agent.** WPE has meaningful latency (it needs a look-ahead window) and CPU cost. **[S]** For most phone/headset deployments it is not worth it; for far-field speakerphone it is the difference between usable and unusable. Treat it as a per-deployment option, not a default.

---

#### F-10 · Multiple speakers / TV / background conversation

**Why.** Another human voice is, by every acoustic measure, speech. VAD fires. ASR transcribes. Your agent responds to a television.

**Why this is the hardest unsolved problem in the list.** There is no acoustic property that distinguishes "the person on the call" from "a person on TV." The distinguishing information is *spatial* (direction of arrival), *speaker-identity* (voice print), or *contextual* (turn structure).

**Production solutions:**

| Technique | Mechanism | Requires | Real-time viable |
|---|---|---|---|
| Beamforming (MVDR/GSC) | Spatial filtering toward DOA | **Mic array** | Yes |
| Speaker verification gate | Enroll target voice, gate on embedding similarity | Enrollment or first-utterance bootstrap | Yes, ~10 ms/frame |
| Target-speaker VAD (TS-VAD) | VAD conditioned on a speaker embedding | Speaker embedding | Yes |
| Speech separation (Conv-TasNet, SepFormer) | Separate mixture into streams | — | Marginal; high latency |
| Diarization + filtering | Cluster by speaker, keep dominant | — | Adds latency, needs buffering |

**[L] The practical production answer for single-channel telephony** is a **speaker-embedding gate bootstrapped from the first confident utterance**: extract an ECAPA-TDNN or ResNet embedding from the first clean user utterance, then gate all subsequent utterances on cosine similarity to it. This costs ~5–10 ms per utterance and eliminates the TV problem almost entirely, at the risk of rejecting the legitimate user if they move away from the mic or a second legitimate speaker joins.

**[R] Reminder:** the claim that Vapi ships a proprietary speaker-isolation model was **not substantiated**. Do not assume competitors have solved this.

---

### 2.3 Transport failures

---

#### F-11 · Jitter

**Why.** Packets are emitted every 20 ms but arrive with variable delay. Playing them out as they arrive produces gaps and overlaps. On the *ingress* side (what we care about), jitter means your frames arrive in bursts — three at once, then nothing for 60 ms.

**Naive fix.** Fixed 100 ms buffer. Fails both ways: too small on bad networks (starvation), needlessly laggy on good ones.

**Production solution: adaptive jitter buffer.** The canonical implementation is **WebRTC's NetEQ** **[C]** (`modules/audio_coding/neteq/`). Its design is worth internalizing because it solves jitter, clock drift, and packet loss *in one integrated mechanism*:

```
   NetEQ decision logic — per 10 ms output request

   ┌─────────────────────────────────────────────────────┐
   │ DelayManager: tracks inter-arrival time histogram,   │
   │ computes target delay as a high percentile of it     │
   └────────────────────┬────────────────────────────────┘
                        │ target vs actual buffer level
        ┌───────────────┼───────────────┬─────────────┐
        ▼               ▼               ▼             ▼
    ┌───────┐      ┌─────────┐    ┌──────────┐  ┌──────────┐
    │NORMAL │      │ EXPAND  │    │ACCELERATE│  │PREEMPTIVE│
    │decode │      │ PLC:    │    │time-     │  │ EXPAND   │
    │& play │      │ synth   │    │compress  │  │time-     │
    │       │      │ from    │    │(buffer   │  │stretch   │
    │       │      │ LPC of  │    │too full) │  │(buffer   │
    │       │      │ history │    │          │  │too empty)│
    └───────┘      └─────────┘    └──────────┘  └──────────┘
                                        │             │
                                   WSOLA-style time scaling:
                                   changes duration without pitch
```

The elegance: **Accelerate and PreemptiveExpand simultaneously absorb jitter *and* clock drift**, because both manifest as the buffer level drifting away from target. You do not need a separate drift-compensation mechanism.

**[C]** NetEQ's target delay is derived from a percentile of the observed inter-arrival distribution rather than a fixed constant — that is what makes it adaptive.

**For VoiceOS:** the AudioSocket and Twilio paths are **TCP/WebSocket**, so packets cannot reorder or drop — but TCP head-of-line blocking converts what would have been a dropped packet into a *latency spike*, and you have **no jitter buffer at all**. Under network stress your 20 ms frames arrive in bursts, the rechunker absorbs it, and VAD timing becomes irregular — silence timers measured in frames drift relative to wall-clock. **[S]** Worth measuring: log inter-arrival time percentiles per call; if p99 exceeds ~60 ms you have a real problem that a small adaptive buffer would fix.

---

#### F-12 · Packet loss

**Why.** UDP drops. On the PSTN, 1–3% loss is normal; 5%+ happens on congested mobile links.

**Solution stack, in order of application:**

1. **Opus in-band FEC (LBRR)** **[C]** — each packet carries a low-bitrate redundant copy of the *previous* frame. Recovers isolated loss at the cost of ~20–30% bitrate. Requires the decoder to be told a packet was lost so it can request the FEC path.
2. **RED (RFC 2198)** — explicit redundant encoding. Higher overhead, works for any codec.
3. **PLC (packet loss concealment)** — synthesize the missing frame. NetEQ's `Expand` does this via LPC extrapolation of recent history **[C]**. Good for <40 ms; beyond that it becomes an obvious robotic buzz.
4. **Neural PLC** — Google's WaveNetEQ **[L]** and similar generative models produce far more natural fill for longer gaps.

**The ASR-specific consideration nobody mentions:** PLC is optimized for *human perceptual continuity*, not for ASR. A PLC-synthesized 40 ms segment is acoustically plausible but linguistically meaningless — it can cause the ASR to hallucinate a phoneme. **[S]** If you have a loss flag, the better strategy for the ASR path may be to insert *silence* (which the model handles gracefully) rather than PLC output (which is out-of-distribution). Test this; it is cheap and could be a measurable WER win on lossy calls.

---

#### F-13 · Clock drift

**Why.** The sender's 8000 Hz crystal and the receiver's are not the same 8000 Hz. A ±50 ppm mismatch — well within spec for consumer hardware — accumulates to 180 ms of divergence per hour. On the receive side, either your buffer grows without bound or it starves.

**Detection.** Track `(packets_received × frame_duration)` against wall-clock elapsed. The slope of the divergence *is* the drift rate.

**Production solution.** NetEQ's Accelerate/PreemptiveExpand absorb it implicitly **[C]**. Standalone systems use an **asynchronous sample-rate converter (ASRC)** whose ratio is continuously adjusted by a control loop tracking buffer fullness — effectively a PLL in the sample domain.

**[S] For VoiceOS specifically:** on the WebSocket paths the browser's `AudioContext` clock and your server's clock drift independently. Over a 10-minute call at 50 ppm this is ~30 ms — below the threshold where it affects endpointing decisions. **Low priority**, but worth a metric so you'd know if a device were badly out of spec.

---

#### F-14 · Reconnection / mid-call transport failure

**Why.** WebSockets drop. Mobile networks hand off between towers. TCP connections reset.

**What breaks.** Every piece of stateful audio machinery: Silero's LSTM state, the resampler's filter state, the rechunker's partial frame, the VAD hysteresis latch, the pre-roll ring, the in-flight utterance.

**Production solution.** Treat reconnection as a **session-level** event, not a transport-level one. On reconnect:
- Reset all DSP state explicitly (`vad.reset()`, new `Resampler`, clear rechunker)
- Discard the in-flight partial utterance — it is unrecoverable and committing it produces a truncated transcript
- Preserve *conversation* state (history, turn counter) — that lives above the transport

**[C] VoiceOS has the right seam for this** (`SessionManager` + `AudioTransport` are separate), but does not currently implement reconnect-time DSP reset. That is a real gap for production telephony.

---

### 2.4 Segmentation failures

---

#### F-15 · Endpoint oscillation / turn flicker

**Why.** VAD probability hovers near threshold. The latch opens and closes repeatedly, splitting one utterance into three, each producing its own ASR call and its own turn.

**Naive fix.** Raise the threshold. Now you miss quiet speakers.

**Production solution: hysteresis** — two thresholds. Enter speech at `on_threshold`, remain in speech until probability drops below a *lower* `off_threshold`. This is a Schmitt trigger; it is the same fix used in every noisy comparator circuit for a century.

```
  p
 1.0│      ╭──╮    ╭─────╮
    │     ╱    ╲__╱       ╲
 0.5├────╱ ◀── on_threshold ╲──────
    │   ╱                     ╲
0.35├──╱  ◀── off_threshold    ╲───
    │ ╱                          ╲
 0.0╰──────────────────────────────
      │◀────── ONE utterance ─────▶│
         (dip to 0.4 does not break the latch)
```

**[C] VoiceOS implements this** with `neg_threshold` defaulting to `threshold − 0.15` — which matches the Silero project's own convention. Both the pipeline detector and the dashboard endpointer use on = 0.5 / off = 0.35.

---

#### F-16 · False endpoint on mid-utterance pause

**Why.** Humans pause mid-thought. "I'd like to book a flight to… uh… Chennai." A fixed 700 ms silence timer commits after "to", produces a garbage turn, and the agent interrupts.

**This is the single highest-impact failure mode in conversational voice AI.** It is what makes agents feel rude.

**Solution ladder:**

| Tier | Technique | Signal used | Latency cost |
|---|---|---|---|
| 0 | Fixed silence timeout | duration only | 500–800 ms |
| 1 | Adaptive silence | utterance duration | variable |
| 2 | Prosody (pitch contour) | F0 trajectory | ~0 |
| 3 | Semantic — text | partial transcript | needs partial ASR |
| 4 | Semantic — audio | raw waveform | ~12–50 ms |
| 5 | Fused ASR+turn model | joint | ~0 (integrated) |

**Tier 1 — adaptive silence.** The insight: a *short* utterance is more likely to be a fragment ("I'd like to…"), so wait *longer*. A *long* utterance is more likely complete, so commit faster. **[C] VoiceOS implements exactly this** — `min_silence_short_ms = 1100` for utterances under 1200 ms, versus a 700 ms baseline. Note this is the *opposite* of the naive intuition and it is correct.

**Tier 3 — text-based semantic endpointing.** **[C] LiveKit's open turn-detector** is a text classifier fine-tuned from **Qwen2.5-0.5B-Instruct**. Mechanism: feed up to 6 conversation turns (truncated to 128 tokens) through the Qwen chat template and read the **probability of the `<|im_end|>` token** as the end-of-turn signal. Elegant — it reuses the base model's native turn-boundary representation rather than training a new head.

**[C] Critically, LiveKit's design does not replace VAD with the LM.** VAD retains responsibility for speech-presence detection and barge-in triggering; the LM supplies only the semantic signal for *committing* a turn. This layering is the correct architecture and it is what VoiceOS also does.

**Tier 4 — audio-native semantic endpointing.** **[C] Smart Turn v3** (Pipecat/Daily) classifies directly on the waveform via a Whisper-tiny encoder plus a classification head, ~12 ms on CPU, 8-second context window. Advantages over text-based: no dependency on partial ASR (saves the entire partial-transcription cost), works before any transcript exists, and captures prosody that text discards.

**[C] VoiceOS uses Smart Turn v3** — `models/smart-turn-v3.2-cpu.onnx`, invoked once per pause via a `_turn_checked` latch, gated at 300 ms of silence, shortening the required silence to 250 ms on a positive prediction. The one-inference-per-pause latch is the right call; running it every 32 ms frame would 10× the cost for no benefit.

**Tier 5 — fused model.** **[C] Deepgram Flux** ("Conversational Speech Recognition") fuses transcription and turn detection into a *single* model — the same model that emits transcripts also models conversational flow and decides end-of-turn. **[C]** Its turn detection is semantic rather than silence-based: it distinguishes incomplete utterances (trailing "because…", filler "uh, sorry…") from complete ones ("Thanks so much."). Deepgram claims ~30% fewer false interruptions than pipeline approaches **[C — claim documented; magnitude not independently verified]**.

---

#### F-17 · Late endpoint (agent feels sluggish)

**Why.** Conservative silence timers. Every millisecond of the timer is dead air the user experiences as latency.

**Production solution: speculative execution.** Do not wait for certainty — start work at medium confidence and cancel if wrong.

**[C] Deepgram Flux exposes this as first-class API events:**

```
  StartOfTurn ─────────▶ trigger barge-in / stop TTS
       │
  EagerEndOfTurn ──────▶ START SPECULATIVE LLM GENERATION
       │                  (fires 150–250 ms before EndOfTurn)
       ├─ TurnResumed ──▶ CANCEL the speculative response
       │                  (user was just pausing)
       └─ EndOfTurn ────▶ COMMIT — high confidence
```

**[C] The documented cost: 50–70% more LLM calls** in exchange for 150–250 ms earlier response. That is an explicit, quantified latency-for-money trade, and it is the most interesting design idea in the current generation of voice APIs.

**[S] VoiceOS could implement this today** without any vendor: `Smart Turn` already produces a continuous probability. Fire a speculative LLM turn at p ≥ 0.5 and commit at p ≥ 0.8 or on timer expiry, using the existing `InterruptController` generation counter to cancel the speculative branch. The generation-counter machinery you already have is exactly the cancellation primitive this needs.

---

#### F-18 · Backchannel misread as interruption

**Why.** "mm-hmm", "yeah", "right" are *not* turn claims. Treating them as barge-in makes the agent stop talking every time the user is being polite.

**Production solution.** **[C] Vapi runs a custom model that distinguishes true interruptions from backchannel affirmations**, and additionally tracks *where* the assistant's speech was cut off so the LLM knows what it was unable to say. That second half is underappreciated: after a barge-in, the conversation history must reflect what the user actually *heard*, not what was generated.

**[C] VoiceOS implements the second half correctly** — the staged-commit mechanism (`add_pending_segment` / `commit_assistant(turn_id, n)`) records exactly the sentences that finished playing, with `SentenceSpoken` markers providing ordering-based proof of playback. That is a genuinely well-designed piece of machinery and matches what Vapi documents.

**[S] VoiceOS does not implement the first half.** A backchannel classifier is a small, high-value addition: a short word-list plus duration gate (< 500 ms, matches a backchannel lexicon, no pitch rise) suppresses the barge-in without touching the VAD.

---

#### F-19 · Barge-in false positives from echo

Covered in F-06. The VoiceOS mitigation (0.7 threshold + 250 ms contiguous) is sound; the cross-correlation gate is the cheap upgrade.

---

### 2.5 ASR failures

---

#### F-20 · Partial transcript flickering

**Why.** **[C]** Flickering is fundamentally caused by *repeatedly emitting the current top beam-search hypothesis mid-decoding*, with **no guarantee that a previously emitted partial is a prefix of the next one**. The beam reorders as more audio arrives; "my open" becomes "my opinion".

**Measurement.** **[C]** Two metrics from the Google literature:
- **UPWR** (unstable partial word ratio) = unstable words ÷ words in final hypothesis
- **UPSR** (unstable partial segment ratio) = revised segments ÷ utterances

Both are measurable live, on-device, with no added latency. **If you are building a streaming ASR product and not tracking UPWR, you are flying blind on the metric users actually perceive.**

**Surprising empirical finding.** **[C]** In Google's on-device multi-domain streaming RNN-T, roughly **half of all partial revisions are text-normalization artifacts, not acoustic ones**: 47.6% normalization (24.7% capitalization, 21.2% punctuation/spacing, 1.7% numerals) versus 52.4% genuine streaming instability. Meaning: *half your flicker problem may be a formatting problem, fixable without touching the decoder.*

**Fix 1 — delay partial emission.** **[C]** Raising the partial emission interval from 50 ms to 200 ms added only **75 ms** mean partial delay yet cut **UPWR by 71.6%** and **UPSR by 68.8%**. Instability decreases logarithmically with interval length — but only up to the point where the delay becomes perceptible. This is the highest return-on-effort fix available.

**Fix 2 — normalize training transcripts + domain-ID feature.** **[C]** Multi-domain training improved WER (4.3 → 4.0) but *increased* UPWR by 76.2% and UPSR by 45%. Applying text normalization to training transcripts plus adding a domain-id input feature recovered stability **with zero added emission delay**. The lesson generalizes: **a change that improves WER can badly damage perceived stability, and WER alone will not tell you.**

**Fix 3 — prefix-constrained reranking.** **[L, verification incomplete]** At partial-emission time only, add a fixed cost penalty α to any beam hypothesis that does not have the previously emitted partial as a prefix. Beam search itself is untouched. Reported to roughly halve flicker at negligible latency cost. This is architecturally attractive because it is a decode-time-only change.

---

#### F-21 · Whisper is not a streaming model

**Why.** Whisper is an encoder-decoder (AED) trained on fixed 30-second windows with full bidirectional attention over the encoder. There is no causal masking; it cannot emit a token before seeing the whole window. It is *architecturally* offline.

**What "streaming Whisper" actually means — three distinct strategies:**

**(a) Pseudo-streaming via repeated re-decode.** Re-run Whisper on a growing buffer every N ms and diff the outputs. **[L, verification incomplete]** `whisper_streaming` (UFAL) uses a **LocalAgreement-*n*** policy: commit a prefix only when *n* consecutive updates agree on it. Buffer growth is bounded by trimming at sentence punctuation or at Whisper's own segment boundaries.

This is what **[C] VoiceOS's `RollingTranscriber` does** — and the code is refreshingly honest about it, describing itself as "honest pseudo-streaming" that trades CPU for partials because faster-whisper has no incremental decoder. **The cost is quadratic**: transcribing a growing buffer every 400 ms means an 8-second utterance is transcribed ~20 times, with the last pass processing 8 seconds of audio. This is why the setting is documented as CPU-heavy and defaults off.

**(b) Add a CTC head.** **[L]** Attach a CTC decoder to Whisper's encoder and fine-tune with hybrid CTC-attention loss in a U2-style two-pass architecture: the CTC branch emits streaming partials, the attention decoder rescores for the final. This makes Whisper genuinely streaming at the cost of a fine-tuning run.

**(c) Don't use Whisper.** Use a natively streaming architecture (below).

---

#### F-22 · Choosing a streaming ASR architecture

| | **CTC** | **RNN-T (Transducer)** | **AED (Whisper)** |
|---|---|---|---|
| Streaming native | Yes | **Yes** | No |
| Output dependency | Conditionally independent per frame | Autoregressive via predictor | Fully autoregressive |
| Needs external LM | Usually | No (predictor is an internal LM) | No |
| Alignment | Monotonic, spiky | Monotonic | Unconstrained (can reorder/hallucinate) |
| Latency control | Frame-level | Frame-level | Window-level |
| Typical use | Fast partials, rescored later | **Production streaming default** | Batch / highest accuracy |

**[L] RNN-T is the production default** for streaming ASR (Google, NVIDIA Riva, most on-device systems) because it is streaming-native, monotonic (cannot hallucinate out-of-order text), and needs no external LM.

**Encoder choices.** Conformer (convolution + self-attention) is the standard modern encoder. For streaming you need bounded context, which means one of:
- **Chunked / block-wise attention** — attend only within a chunk plus limited history
- **Dynamic chunk training (U2/U2++, WeNet)** — train with randomly sampled chunk sizes so one model serves all latency settings at inference
- **Emformer** — efficient memory transformer with a memory bank carrying compressed history across blocks

**[L, verification incomplete]** Streaming accuracy degrades sharply as chunk size shrinks — one reported curve shows WER 25.5% at 100 ms chunks versus 17.3% at 1000 ms and 16.7% at 1500 ms. **The shape of this curve is the fundamental latency/accuracy tradeoff of streaming ASR** and you should measure it on your own data before choosing a chunk size.

---

#### F-23 · Whisper hallucination

**Why.** Whisper was trained on 680 k hours of weakly-labeled web audio including subtitle tracks. On silence or noise it emits high-likelihood training artifacts: "Thank you for watching", "Subtitles by…", "♪♪♪", or loops the previous phrase.

**Triggers.** Leading/trailing silence, pure noise segments, very short inputs, low-SNR audio.

**Production mitigations:**
1. **Never feed silence.** Gate on VAD — only send segments the VAD confirmed as speech. **[C] VoiceOS does this** (utterances come only from `SpeechDetector`).
2. **Bound the pre-roll.** Excessive leading silence is a hallucination trigger; this is a direct argument against setting `pre_roll_ms` too high.
3. **`condition_on_previous_text=False`** in faster-whisper — breaks the loop-repetition failure mode at a small cost in coherence.
4. **`no_speech_threshold` / `compression_ratio_threshold`** — reject segments whose token distribution indicates degenerate repetition.
5. **Minimum duration gate.** **[C] VoiceOS's `min_speech_ms = 150`** does this implicitly.

---

#### F-24 · Timestamp drift

**Why.** Whisper's timestamps come from decoded timestamp *tokens*, not from an alignment. They drift, especially on long segments. Word-level timestamps require a forced-alignment pass (`whisperX`, or faster-whisper's `word_timestamps=True` which uses cross-attention-derived alignment).

**Consequence for a voice agent.** If you use ASR timestamps to decide *when* the user stopped speaking, drift corrupts your endpointing. **The fix is architectural: never derive turn timing from ASR output.** Derive it from the VAD's frame counter, which is exact by construction. **[C] VoiceOS does this correctly** — the silence timer accumulates `frame.duration_ms`, computed from the actual array length, independent of ASR.

---

#### F-25 · Code-switching and multilingual

**Why.** Indian-language conversation is natively code-mixed: "phone number చెప్పండి", "meeting కి వెళ్తున్నా". Models with a hard language-ID switch mis-handle this: they either force everything into one script or oscillate.

**Failure modes specific to your stack:**

- **[C] Sarvam `saaras:*` is a *translation* model; `saarika:*` is transcription.** Selecting `saaras` silently produces English output instead of same-language transcription. Your `settings.py` documents this inline — good, it is a genuine trap.
- **[C] Language autodetect mis-fires.** Your dashboard code force-sets `sarvam_language` with the comment that autodetect otherwise misdetects Hindi as Kannada/Bengali/Gujarati. Related-script Indic languages are genuinely hard to distinguish from short utterances.
- **[C] faster-whisper defaults to `language="en"`** in your settings, which forces English decoding — wrong for Telugu/Hindi and a silent accuracy killer if the local provider is ever used.

**Production approaches:**

| Approach | Mechanism | Tradeoff |
|---|---|---|
| Force language per call | Config from campaign metadata | Best accuracy, no code-switch handling |
| Utterance-level LID | Classify then route | Adds latency, fails on intra-utterance switching |
| Multilingual joint model | Single model, shared vocab | Handles intra-utterance switching natively |
| Romanized output | Transcribe to Latin script | Sidesteps script confusion; hurts downstream NLU |

**[L] For an outbound survey campaign in a known language — your actual use case — forcing the language is correct and you should not try to be clever.** The language is known from the campaign definition. Autodetect is a liability, not a feature, here.

---

#### F-26 · GPU queueing and tail latency

**Why.** Under concurrency, inference requests queue. Mean latency looks fine; p99 explodes. A single voice call is ruined by one p99 event — the user hears three seconds of silence.

**Mechanisms.**
- **Head-of-line blocking**: one long utterance (30 s) blocks short ones behind it.
- **Batching latency**: dynamic batching waits to accumulate a batch, trading latency for throughput.
- **Cold start**: first inference after model load, or after CUDA context creation, is 10–100× slower.
- **Memory pressure**: allocator thrash under variable sequence lengths.

**Production solutions:**
1. **Separate queues by expected cost.** Short utterances must not queue behind long ones.
2. **Bounded dynamic batching** with a hard timeout (e.g. "batch up to 8 or wait 10 ms, whichever first").
3. **Aggressive prewarm.** **[C] VoiceOS does this** — `_prewarm()` makes a real tiny LLM call, warms TTS, and runs STT over `np.zeros(16000)` at startup, with the comment that the first live call otherwise spikes 4–5 s on TLS/connection setup. This is exactly right and under-practiced.
4. **Admission control.** Reject new calls when queue depth exceeds a threshold, rather than degrading every in-flight call. Voice is a domain where refusing one call is far better than ruining ten.
5. **Measure p99 per stage, not end-to-end.** End-to-end p99 tells you something is wrong; per-stage p99 tells you what.

---

## 3. PHASE 4 — Platform reverse engineering

**Methodology note.** Vendors disclose mechanism far more readily than performance. Where a vendor publishes a number, I mark it as a *claim* rather than a *fact* unless independently verified. Three widely-repeated claims failed verification (§0) — treat vendor benchmark numbers as marketing until you reproduce them.

### 3.1 Vapi

| Aspect | Finding | Confidence |
|---|---|---|
| Architecture | Orchestration layer running "a suite of real-time models" **on top of** the STT→LLM→TTS core — explicitly not just a 3-stage pipeline | **[C]** docs |
| Barge-in | Custom model distinguishing true interruptions from backchannel affirmations ("yeah", "uh-huh") | **[C]** docs |
| Post-interruption state | Tracks where the assistant's speech was cut off so the LLM knows what it failed to say | **[C]** docs |
| Endpointing | "Fusion audio-text model" claim **could not be substantiated** | **[R]** |
| Noise/speaker isolation | Proprietary noise-filter + speaker-isolation model claim **could not be substantiated** | **[R]** |
| Transport | Twilio/Vonage/Telnyx SIP trunking, WebRTC for web | **[L]** |

**Architectural lesson:** Vapi's differentiator, per its own docs, is the *orchestration models around* the pipeline, not the pipeline. Backchannel classification and cut-off tracking are the two named ones. If you are competing with Vapi, those are the table stakes.

### 3.2 LiveKit Agents

| Aspect | Finding | Confidence |
|---|---|---|
| Turn detector | Text-based EOU classifier fine-tuned from **Qwen2.5-0.5B-Instruct** | **[C]** model card |
| Mechanism | Up to 6 turns, truncated to 128 tokens, through the Qwen chat template; reads **P(`<\|im_end\|>`)** as the EOU signal | **[C]** |
| Layering | LM does **not** replace VAD — VAD keeps speech-presence + barge-in; LM supplies commit signal | **[C]** |
| Implication | Semantic endpointing operates on **STT transcripts**, so it inherits STT latency and errors | **[C]** derived |
| Transport | WebRTC-native (LiveKit is an SFU company); SIP via LiveKit SIP | **[C]** docs |
| Scaling | Worker pool, agents dispatched per room | **[L]** |

**The key architectural tradeoff versus Smart Turn:** LiveKit's text-based detector requires a partial transcript to exist before it can judge. That means its endpointing latency is *bounded below* by STT partial latency. Smart Turn's audio-native approach has no such dependency. In exchange, text has access to semantics that prosody alone cannot capture ("...and then I said" is clearly incomplete; the audio may not reveal that).

**[S]** The strongest system probably runs both and fuses them — audio-native for speed, text for the ambiguous cases. Neither vendor appears to do this publicly, which makes it an interesting place to compete.

### 3.3 Deepgram (Nova / Flux)

| Aspect | Finding | Confidence |
|---|---|---|
| Flux architecture | **Fused** transcription + turn detection in one model ("Conversational Speech Recognition") | **[C]** |
| Turn detection | Semantic, not silence-based: distinguishes "because…" / "uh, sorry…" from "Thanks so much." | **[C]** |
| API events | `StartOfTurn`, `EagerEndOfTurn`, `TurnResumed`, `EndOfTurn` | **[C]** |
| Speculative execution | `EagerEndOfTurn` fires **150–250 ms earlier** than `EndOfTurn`, costing **50–70% more LLM calls** | **[C]** |
| False-interruption claim | ~30% fewer than pipeline approaches | **[C]** as a documented claim |
| Latency numbers | 200–600 ms improvement, 70% within 500 ms, p90/p95 1 s/1.5 s | **[R]** not substantiated |
| Nova encoder | Conformer-family, optimized for inference | **[L]** |

**This is the most architecturally significant development in the space.** Fusing ASR and turn detection eliminates an entire layer — and more importantly, the turn decision gets access to the ASR's *internal* representations rather than just its text output. A separate turn detector sees `"I'd like to book a flight to"`; a fused model sees the encoder states, the decoder's uncertainty, and the acoustic prosody simultaneously.

**Exposing speculation as an API contract is the second big idea.** It moves a latency/cost tradeoff that was previously buried in the vendor's implementation into the application developer's control.

### 3.4 Pipecat (open source — highest-fidelity reference)

| Aspect | Finding | Confidence |
|---|---|---|
| Model | Frame-based pipeline; processors consume/emit typed frames | **[C]** source |
| Turn detection | **Smart Turn v3** — audio-native, Whisper-tiny encoder, ~12 ms CPU, 8 s window | **[C]** |
| Noise suppression | RNNoise filter shipped as a built-in audio utility | **[C]** docs |
| VAD | Silero | **[C]** |
| Value to you | It is open source — **read it rather than guess** | — |

Pipecat is the single most useful reference implementation available, because everything above is verifiable by reading code rather than inferring from behavior. VoiceOS's Smart Turn integration already draws from this lineage.

### 3.5 OpenAI Realtime / Gemini Live (speech-native)

**[L]** These are architecturally different from everything above: audio tokens go directly into a multimodal model; there is no discrete "transcript" stage in the critical path. Transcription, when exposed, is a *side output*, not a pipeline stage.

**Consequences [L]:**
- Endpointing is internal to the model and not separately controllable (both expose only coarse VAD settings)
- Latency floor is much lower — no STT→LLM serialization
- You lose the ability to inspect, log, correct, or route on the transcript
- Language/voice control is weaker

**[S] For your use case — outbound Indian-language survey calls with structured post-call extraction — a speech-native model is the wrong choice.** You need the transcript as a first-class artifact for survey extraction, auditing, and compliance. The cascaded architecture you have is correct for this product.

### 3.6 Others (briefer, lower confidence)

- **Retell / Bland [L]:** cascaded pipelines with proprietary endpointing; both market low latency; neither publishes mechanism at Vapi's or Deepgram's level.
- **AssemblyAI [L]:** Universal-Streaming targets low-latency partials; publishes immutable-vs-mutable transcript semantics, which is the same token-stabilization problem as F-20 exposed at the API layer.
- **Speechmatics [L]:** strong multilingual/accent robustness reputation; exposes explicit `max_delay` and partial/final semantics, i.e. the latency/stability trade is a documented API knob.
- **NVIDIA Riva [C, from docs]:** Conformer/RNN-T on Triton, explicit dynamic batching and GPU scheduling — the best publicly documented model for *how to run streaming ASR at scale on GPUs*, independent of model quality.
- **Gladia [L]:** Whisper-derived hosted streaming; inherits Whisper's hallucination profile.

---

## 4. PHASE 5 — Streaming pipeline engineering

### 4.1 The buffer hierarchy

Six distinct buffers, each solving a different problem. Conflating them is a common source of bugs.

| Buffer | Purpose | Typical size | Failure if wrong |
|---|---|---|---|
| **Driver/OS ring** | Decouple hardware clock from app | 10–20 ms | Xruns, dropped input |
| **Jitter buffer** | Absorb network variance + drift | adaptive 20–200 ms | Gaps or excess latency |
| **Rechunk buffer** | Convert transport framing → model framing | < 1 frame | Model contract violation |
| **Pre-roll ring** | Retroactive onset capture | 200–500 ms | Clipped first syllable |
| **Utterance buffer** | Accumulate current turn | up to `max_utterance_s` | Unbounded growth |
| **Rolling ASR context** | Streaming decoder history | model-dependent | Context loss at boundaries |

### 4.2 Backpressure — the load-shedding question

Real-time audio arrives whether or not you are ready. When the consumer is slower than the producer, you must choose what to lose. There is no option that loses nothing.

| Policy | Behavior | Correct for |
|---|---|---|
| **Block producer** | Backpressure upstream | ❌ Never — the mic cannot be paused |
| **Unbounded queue** | Grow forever | ❌ Never — OOM plus unbounded latency |
| **Drop newest** | Discard arriving frames | Rarely — loses the most recent speech |
| **Drop oldest** | Discard queued frames | ✅ Usually — keeps latency bounded |
| **Degrade** | Skip optional stages | ✅ Best — drop partials before finals |

**[C] VoiceOS uses drop-oldest** (`AudioQueue.put_drop_oldest`, 256-frame cap ≈ 8.2 s) and **counts drops** — the counter is essential, because silent loss is indistinguishable from silence.

**The subtlety:** dropping the *oldest* frame when a backlog exists means you lose the **beginning** of an utterance — precisely the part pre-roll worked to preserve. **[S]** A smarter policy for a voice pipeline is *state-aware*: drop oldest while IDLE (nothing valuable is queued), but if recording, prefer to degrade — skip partial transcription, skip the noise gate — before discarding audio.

### 4.3 The zero-copy / lock-free question

Standard advice for real-time audio: lock-free SPSC ring buffers, memory pools, pinned memory for GPU transfers, zero-copy handoff.

**[S] Honest assessment for a Python asyncio pipeline: most of this does not apply, and pursuing it is misdirected effort.**

- Python's GIL makes true lock-free structures moot at the application layer
- `numpy` slicing is already zero-copy where it matters
- Per-frame allocation of a 512-sample int16 array is ~1 KB — trivial versus the 1 ms VAD inference in the same tick
- `call_soon_threadsafe` is the correct and idiomatic thread handoff

**Where it *does* apply:** if you rewrite the hot path in Rust/C++ (as LiveKit does for media), or if you batch GPU inference — there, pinned memory and pre-allocated device buffers are genuine multi-millisecond wins.

**The real Python-specific hazards, in priority order:**
1. **Blocking the event loop.** Any synchronous inference on the loop serializes every call in the process.
2. **GC pauses.** Large object churn causes multi-millisecond stalls. Pre-allocate long-lived buffers.
3. **`asyncio` scheduling latency** under many tasks. With N calls × M workers you may have thousands of tasks; scheduling is not free.

### 4.4 Chunk and frame sizing — the actual numbers

```
  8 ms   ── minimum meaningful analysis window
 10 ms   ── WebRTC APM internal frame; WebRTC VAD option
 20 ms   ── RTP/Opus/telephony packet standard  ◀── transport granularity
 32 ms   ── Silero v5 @16k (512 samples)        ◀── VAD granularity
 64 ms   ── VoiceOS browser WS chunk (1024 @16k)
100 ms   ── typical streaming-ASR chunk (low latency, higher WER)
250 ms   ── common partial-emission interval
500 ms   ── typical minimum endpoint silence
1000 ms  ── streaming-ASR chunk (good WER)
8000 ms  ── Smart Turn v3 context window
30000 ms ── Whisper window
```

**The impedance mismatch that defines the ingress design:** transport speaks 20 ms, Silero speaks 32 ms. 20 and 32 share no common factor below 160 ms, so you *must* buffer. That is the entire justification for the rechunker, and it is why it cannot be optimized away.

---

## 5. PHASE 6 — Voice activity detection, in depth

### 5.1 WebRTC VAD (GMM-based)

**[C]** Architecture: split into **6 sub-bands** (80–250, 250–500, 500–1k, 1k–2k, 2k–3k, 3k–4k Hz), compute log-energy features per band, evaluate under **two Gaussian mixture models** (speech and noise), apply a likelihood-ratio test. Four aggressiveness modes (0 = permissive → 3 = aggressive). Accepts 10/20/30 ms frames at 8/16/32/48 kHz.

**Properties.** Extremely fast (microseconds), tiny, deterministic. Degrades badly below ~10 dB SNR and fires readily on non-stationary noise. Still appropriate as a cheap **pre-gate** ahead of a neural VAD when CPU-bound at high concurrency.

### 5.2 Silero VAD

**[C]** v5 requires **exactly 512 samples at 16 kHz** (or 256 at 8 kHz). Carries internal recurrent state across frames — which is precisely why `reset_states()` must be called between utterances. Returns a speech probability in [0, 1]. Recommended operating point: threshold 0.5 with a negative threshold 0.15 lower.

**Why the state matters more than people realize.** Because Silero is stateful, **frame order and continuity are part of the contract**. If you drop frames (backpressure!), reorder them, or interleave frames from two different calls into one model instance, the recurrent state is corrupted and probabilities become meaningless. This creates a hard requirement: **one VAD instance per call**, and dropped frames should be counted as a correctness signal, not just a capacity signal.

**[C] VoiceOS creates a fresh `SileroVAD` per WebSocket connection** in the dashboard path — correct. The pipeline path creates one per `VoicePipeline`, also correct.

### 5.3 FSMN-VAD

**[L]** Feedforward Sequential Memory Network, from the FunASR/Alibaba stack. Uses memory blocks rather than recurrence, making it naturally streaming and parallelizable. Reported strong on Mandarin. Relevant to you as a comparison point for Indic-language robustness — worth benchmarking against Silero on Telugu/Hindi noise conditions, since Silero's training distribution for Indic languages is undocumented.

### 5.4 The complete endpointing state machine

This is the synthesized best-practice design, generalizing what VoiceOS implements:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ per frame (32 ms):                                                │
 │                                                                   │
 │   rms      = sqrt(mean(x²))                                       │
 │   floor    = percentile(rms_window[3000ms], 20)                   │
 │   gated    = rms ≥ floor × noise_margin                           │
 │   p        = vad.process(x)                                       │
 │   bar      = off_threshold if recording else on_threshold         │
 │   speech   = (p ≥ bar) and gated                                  │
 │                                                                   │
 │   ── NOT RECORDING ────────────────────────────────────────       │
 │   preroll.push(frame)                                             │
 │   if speech:                                                      │
 │       start(); frames = preroll.drain()   ← retroactive capture   │
 │       speech_ms = frame_ms; silence_ms = 0                        │
 │                                                                   │
 │   ── RECORDING ────────────────────────────────────────────       │
 │   frames.push(frame)              ← ALWAYS: this is hangover      │
 │   if speech:                                                      │
 │       speech_ms += frame_ms; silence_ms = 0                       │
 │       turn_checked = False        ← user resumed; re-evaluate     │
 │       endpoint_predicted = False  ← stale prediction is invalid   │
 │   else:                                                           │
 │       silence_ms += frame_ms                                      │
 │                                                                   │
 │   ── SEMANTIC TIER (once per pause) ───────────────────────       │
 │   if (not turn_checked and speech_ms ≥ min_speech                 │
 │        and silence_ms ≥ pause_ms and no task in flight):          │
 │       turn_checked = True                                         │
 │       spawn: p_complete = turn_model(audio_so_far)                │
 │              if p_complete ≥ τ: endpoint_predicted = True         │
 │                                                                   │
 │   ── COMMIT DECISION ──────────────────────────────────────       │
 │   base = min_silence_short if (adaptive and                       │
 │            speech_ms < short_utterance_ms) else min_silence       │
 │   required = min(predicted_silence, base) if endpoint_predicted   │
 │              else base                    ← prediction can only   │
 │                                              SHORTEN, never       │
 │                                              lengthen             │
 │   if silence_ms ≥ required or duration ≥ max_utterance:           │
 │       commit(frames) if speech_ms ≥ min_speech else discard       │
 │       vad.reset()                                                 │
 └──────────────────────────────────────────────────────────────────┘
```

**Five invariants worth stating explicitly, because each is a bug if violated:**

1. **Frames are appended regardless of speech state while recording.** Hangover is a buffering property, not a timer property.
2. **A semantic prediction may only shorten the timer.** The `min()` is a safety property — a mispredicting model can make you fast, never rude-and-slow.
3. **Resumed speech invalidates a prior "complete" prediction.** Without this, one early positive prediction commits the turn the moment the user takes a breath later.
4. **The semantic check runs once per pause, not once per frame.** A latch, cleared by resumed speech.
5. **VAD state is reset at every commit.** Stateful model; stale state corrupts the next utterance.

**[C] VoiceOS implements 1, 2, 4, and 5 correctly.** For (3), the reset of `endpoint_predicted` on resumed speech is gated on `turn_predictor is not None` — so with predictive (text) endpointing enabled but Smart Turn disabled, a stale prediction persists for the rest of the utterance. That is a real bug, and it manifests as occasional premature commits in exactly the configuration you would use if you disabled Smart Turn for CPU reasons.

---

## 6. PHASE 8 — Latency budget

### 6.1 Where the time actually goes

Mouth-close → final transcript, cascaded pipeline. **These are engineering estimates [S] except where marked**; the point is the *shape*, not the digits.

| Stage | P50 | P99 | Notes |
|---|---|---|---|
| Capture + OS buffer | 10–20 ms | 40 ms | Fixed by buffer size |
| Client DSP (APM) | ~10 ms | 10 ms | 10 ms algorithmic frame |
| Network (regional) | 15–40 ms | 150 ms+ | Mobile tail is brutal |
| Jitter buffer | 20–60 ms | 200 ms | Adaptive — trades against loss |
| Decode + resample | < 1 ms | 2 ms | Cheap |
| Rechunk | ≤ 32 ms | 32 ms | Bounded by frame size |
| VAD | ~1 ms/frame | 5 ms | ONNX CPU |
| **Endpoint wait** | **250–800 ms** | **1100 ms** | **← DOMINATES** |
| Turn model | 12–50 ms | 100 ms | **[C]** ~12 ms for Smart Turn v3 |
| ASR (batch, 3 s utt) | 200–600 ms | 2000 ms+ | GPU queueing is the tail |
| **Total** | **~0.6–1.5 s** | **3 s+** | |

### 6.2 The one conclusion that matters

**The endpoint wait is 40–60% of the P50 budget and it is pure deliberate delay.** Every other optimization competes for the remaining half.

This reframes the entire engineering priority list:

1. **Better endpointing** — semantic detection converting an 800 ms wait into 250 ms saves more than every other optimization combined.
2. **Speculative execution** — overlapping the wait with work makes the remaining wait free. This is why Flux's `EagerEndOfTurn` is significant: it does not make endpointing faster, it makes the *uncertainty window productive*.
3. **Streaming ASR** — decoding *during* speech means the final transcript is ready ~immediately at commit, removing the ASR term entirely.
4. Everything else — worth doing, but second-order.

### 6.3 Tail latency

P99 is dominated by GPU queueing and network. The disproportionate impact in voice: one 3-second stall in a 20-turn call is remembered as "the agent is broken," while consistently mediocre P50 is merely "a bit slow." **Optimize P99 before P50** — the opposite of the usual web-service instinct.

---

## 7. PHASE 9 — Techniques that are real but under-documented

Consolidated from the above; these are the ones rarely written down.

1. **Retroactive pre-roll capture** — record before you decide to record.
2. **Hysteresis with an asymmetric stay-bar** — universal in production, rarely explained.
3. **Inverted adaptive silence** — wait *longer* after *short* utterances. Counterintuitive; correct.
4. **Prediction may only shorten, never lengthen** — the `min()` as a safety property.
5. **One-inference-per-pause latch** — makes semantic endpointing affordable.
6. **Denoise the VAD path, not the ASR path** — suppression artifacts are out-of-distribution for acoustic models.
7. **Delay partial emission** — **[C]** 50 → 200 ms costs 75 ms and removes ~70% of flicker.
8. **Half your flicker is text normalization** — **[C]** fixable without touching the decoder.
9. **Prefix-constrained beam reranking at emission time only** — **[L]**.
10. **Speculative turn commit with a cancel token** — **[C]** as an API in Flux; implementable locally with a generation counter.
11. **Ordering as proof of playback** — a marker after a sentence's frames in a FIFO *is* the acknowledgment. No callbacks needed. **[C] VoiceOS's `SentenceSpoken`.**
12. **Commit-before-drain on barge-in** — read the spoken count and commit history *before* draining queues, without awaiting, so the count cannot advance underneath you. **[C] VoiceOS.**
13. **Never derive turn timing from ASR timestamps** — use the VAD frame counter.
14. **Cross-correlate mic against recent TTS output** to suppress echo-triggered barge-in without a full AEC. **[S]**
15. **Aggressive prewarm including a real inference call** — cold TLS + CUDA context is a 4–5 second first-turn penalty. **[C] VoiceOS.**
16. **Speaker-embedding gate bootstrapped from the first utterance** — the practical single-channel answer to the TV problem. **[L]**
17. **State-aware load shedding** — drop-oldest while idle, degrade-optional-stages while recording. **[S]**
18. **Insert silence rather than PLC output on the ASR path** — PLC is tuned for ears, not acoustic models. **[S]**

---

## 8. PHASE 10 — Reference design

### 8.1 Structure

```
voiceos/
├── transport/           # ingress adapters — the ONLY rate/codec-aware layer
│   ├── webrtc/          #   Opus 48k, needs jitter buffer
│   ├── websocket/       #   Twilio μ-law 8k | binary PCM
│   ├── audiosocket/     #   Asterisk SLIN 8k
│   └── local/           #   mic/speaker
├── dsp/
│   ├── resample.py      # STATEFUL, per call, per direction
│   ├── rechunk.py       # transport framing → model framing
│   ├── jitter.py        # adaptive buffer (RTP paths)
│   ├── noise.py         # RNNoise — VAD path only
│   └── echo.py          # correlation gate / AEC when reference available
├── segmentation/
│   ├── vad.py           # Silero, one instance per call
│   ├── endpointer.py    # the state machine of §5.4
│   ├── turn/            # semantic tier
│   │   ├── audio.py     #   Smart Turn (no ASR dependency)
│   │   └── text.py      #   LM classifier (needs partials)
│   └── noise_floor.py   # rolling percentile
├── asr/
│   ├── streaming/       # RNN-T — partials
│   ├── batch/           # Whisper — finals / rescoring
│   └── stabilize.py     # LocalAgreement / prefix reranking
└── observability/
    ├── metrics.py       # per-stage histograms
    └── audio_health.py  # clipping, SNR, drops, jitter
```

**Why this shape:** the single most important boundary is between `transport/` (everything rate-, codec-, and loss-aware) and everything downstream (which sees only 512-sample 16 kHz int16 frames). Every failure mode in §2.3 lives above that line; every failure in §2.4 lives below it. Keeping them in separate packages means a telephony bug cannot become a VAD bug.

### 8.2 Process and thread model

```
┌─ ingress process (CPU, N per host) ────────────────────────────┐
│  asyncio loop                                                   │
│    ├─ per call: transport → dsp → segmentation                  │
│    ├─ batched VAD tick: all active calls, one ONNX call         │
│    └─ NEVER blocks: all inference dispatched out                │
│  Capacity: bounded by (VAD batch time) / (32 ms)                │
└──────────────────┬──────────────────────────────────────────────┘
                   │ utterance / partial-audio RPC
┌─ inference tier (GPU, autoscaled separately) ──────────────────┐
│  ├─ queue: SHORT utterances (< 5 s)   ← separate, never blocked │
│  ├─ queue: LONG utterances                                      │
│  ├─ queue: partials (droppable under load)                      │
│  └─ dynamic batching: max 8 or 10 ms, whichever first           │
└─────────────────────────────────────────────────────────────────┘
```

**Why separate the tiers:** ingress is CPU- and connection-bound; inference is GPU- and throughput-bound. They scale on different axes and fail in different ways. Colocating them means a GPU stall stops decoding audio for every call on the host — the worst possible coupling.

**Why separate queues by utterance length:** head-of-line blocking is the dominant p99 mechanism, and it is trivially avoidable.

**Why partials are droppable:** under load, a missing partial degrades responsiveness; a missing final breaks the conversation. Shed the former to protect the latter.

### 8.3 Observability — the specific metrics

Most voice pipelines log the wrong things. These are the ones that predict user-visible failure:

**Audio health (per call):**
- clipping ratio (samples ≥ 32700 / total)
- estimated SNR (speech RMS ÷ noise-floor RMS)
- frames dropped by backpressure — **count, not just log**
- inter-arrival p50/p99 (jitter proxy)
- clock-drift slope (packets × frame_ms vs wall-clock)

**Segmentation:**
- utterances committed vs discarded (`< min_speech_ms`)
- **false-endpoint proxy: fraction of turns whose transcript ends mid-clause** — a proxy but a good one
- barge-in rate, and barge-ins occurring within 500 ms of TTS onset (**echo indicator**)
- required-silence actually used (distribution — tells you if the semantic tier is firing)
- semantic-prediction rate and, when ground truth exists, precision

**ASR:**
- **UPWR / UPSR** **[C]** — the flicker metrics; nothing else measures perceived stability
- per-stage p50/p95/p99, separately
- GPU queue depth
- hallucination proxy: rate of known artifact strings ("Thank you for watching", etc.)

**Tracing.** One span per turn, child spans per stage, with the **VAD frame counter as the timeline reference** rather than wall-clock. This is the only way to correlate "the user felt interrupted" with which stage's timer fired.

### 8.4 Fault tolerance

| Failure | Response | Rationale |
|---|---|---|
| ASR provider 5xx | Retry with backoff, **only before any output emitted** | Retrying mid-stream duplicates text |
| ASR provider 429 | Rotate key / provider immediately, no retry | Rate limits do not clear in 400 ms |
| Transport drop | Reset all DSP state; discard in-flight utterance; keep conversation state | Stateful DSP cannot resume |
| GPU OOM | Shed partials first, then reject new calls | Protect in-flight calls |
| Turn model unavailable | Fall back to silence timer, log once | Degradation must be silent to the user |
| Queue depth > threshold | **Admission control — refuse new calls** | Better to reject one than degrade ten |

**[C] The `emitted` invariant** — never retry after output has been produced — appears in six places in VoiceOS and is the correct spine of the whole retry design. It generalizes exactly to ASR: once a partial has been shown, a retry that produces different text is worse than no retry.

---

## 9. Comparison tables

### 9.1 Endpointing

| Platform | Mechanism | Signal | Confidence |
|---|---|---|---|
| **Deepgram Flux** | Fused ASR+turn, single model | Joint acoustic + linguistic | **[C]** |
| **LiveKit** | Qwen2.5-0.5B, P(`<\|im_end\|>`), ≤6 turns/128 tok | Text (needs partials) | **[C]** |
| **Pipecat** | Smart Turn v3, Whisper-tiny encoder, 8 s | Audio-native | **[C]** |
| **VoiceOS** | Smart Turn v3 + adaptive silence + hysteresis | Audio-native | **[C]** source |
| **Vapi** | "suite of real-time models"; fusion claim unverified | Unknown | **[C]**/**[R]** |
| **OpenAI/Gemini Live** | Internal to the model | Unknown | **[L]** |
| **Retell / Bland** | Proprietary | Unknown | **[S]** |

### 9.2 Speculative execution

| Platform | Exposed? | Mechanism |
|---|---|---|
| **Deepgram Flux** | **Yes, as API** | `EagerEndOfTurn` → speculate; `TurnResumed` → cancel. 150–250 ms earlier, 50–70% more LLM calls **[C]** |
| **LiveKit** | Partially | Preemptive generation patterns **[L]** |
| **VoiceOS** | No | Has the primitive (`InterruptController` generation counter) but does not speculate |
| Others | Unknown | **[S]** |

### 9.3 VoiceOS versus the field — honest positioning

| Capability | Status |
|---|---|
| Hysteresis VAD | ✅ Matches best practice |
| Pre-roll | ✅ 300 ms, correctly moved-not-copied |
| Adaptive silence | ✅ Correct inverted logic |
| Semantic endpointing | ✅ Smart Turn v3, one-inference-per-pause |
| Barge-in + spoken-count history | ✅ **Better documented than most** |
| Stateful resampling | ✅ Correct |
| Multi-transport seam | ✅ Clean |
| Prewarm | ✅ Including real inference |
| **Streaming ASR** | ❌ Pseudo-streaming only (quadratic cost) |
| **Speculative execution** | ❌ Primitive exists, unused |
| **Backchannel classifier** | ❌ Absent |
| **Echo correlation gate** | ❌ Absent (threshold gate only) |
| **Jitter buffer** | ❌ Absent on all paths |
| **Reconnect DSP reset** | ❌ Absent |
| **Batched VAD** | ❌ Serialized per call — scaling ceiling |
| **UPWR/UPSR metrics** | ❌ Not measured |
| **Speaker gate (TV problem)** | ❌ Absent |
| **Browser AudioWorklet** | ❌ Uses deprecated ScriptProcessor |

---

## 10. What I would do next, in order

Ranked by (latency or quality win) ÷ (effort), grounded in the above:

1. **Fix `endpoint_predicted` staleness** — one-line guard removal. Real bug, silent premature commits.
2. **Migrate `live.html` to AudioWorklet** — ~30 lines, eliminates main-thread buffer drops.
3. **Echo cross-correlation gate on barge-in** — ~40 lines, kills the dominant false-barge-in mode on speakers.
4. **Backchannel suppression** — word-list + duration gate; large perceived-politeness win.
5. **Speculative turn commit** — you already own the cancellation primitive; this is the single biggest latency win available (150–250 ms **[C]** by analogy to Flux).
6. **Audio-health metrics** — clipping, SNR, drop count, inter-arrival p99. Cheap; tells you which of the above matters for *your* traffic.
7. **Batched VAD** — required before ~30 concurrent calls per process.
8. **True streaming ASR** — largest effort, removes the entire ASR term from the critical path.

---

## 11. Where this document is weakest

Stated plainly, because acting on a false claim is worse than acting on none:

- **Vendor internals for Retell, Bland, Gladia** are inference, not evidence. Confidence: low.
- **Latency numbers in §6.1** are engineering estimates, not measurements. The *shape* (endpoint wait dominates) is solid; the digits are not.
- **Nine claims went unverified** when the research run was interrupted (§0), notably whisper_streaming's LocalAgreement details and the streaming chunk-size/WER curve.
- **Three claims were refuted** and are excluded — but similar unverified vendor claims may have slipped through elsewhere. Anything marked **[L]** or **[S]** about a commercial platform deserves independent checking before you design around it.
- **No claim here about VoiceOS's own runtime performance is measured.** Every performance statement about your code is inferred from reading it. The concurrency ceiling in §1.3 in particular is a hypothesis, not a benchmark.

---

## 12. Sources

**Verified, load-bearing:**
- Vapi — `github.com/VapiAI/docs/blob/main/fern/how-vapi-works.mdx`
- Deepgram Flux — `deepgram.com/learn/introducing-flux-conversational-speech-recognition`
- LiveKit turn-detector — `huggingface.co/livekit/turn-detector`
- Partial-stability metrics (UPWR/UPSR) — `arxiv.org/pdf/2006.01416`
- Deflickering / prefix reranking — `bruguier.com/pub/deflickering.pdf`

**Consulted:**
- WebRTC NetEQ — `chromium.googlesource.com/external/webrtc/+/master/modules/audio_coding/neteq/g3doc/index.md`
- NetEQ explainer — `webrtchacks.com/how-webrtcs-neteq-jitter-buffer-provides-smooth-audio/`
- LiveKit turn detection — `livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection`
- LiveKit docs — `docs.livekit.io/agents/build/turns/`
- Smart Turn v3 — `daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/`
- Smart Turn source — `github.com/pipecat-ai/smart-turn`
- Pipecat RNNoise — `docs.pipecat.ai/api-reference/server/utilities/audio/rnnoise-filter`
- whisper_streaming — `github.com/ufal/whisper_streaming`
- Dynamic chunk convolution (Amazon) — `assets.amazon.science/18/80/2126d1f5416aa7143505694ae013/dynamic-chunk-convolution-for-unified-streaming-and-non-streaming-conformer-asr.pdf`
- RNNoise / suppression comparison — `forasoft.com/learn/audio-for-video/articles-audio/noise-suppression-rnnoise-krisp-rtx-voice`
- WebRTC audio pipeline overview — `forasoft.com/learn/audio-for-video/articles-audio/webrtc-audio-pipeline-end-to-end`
- Streaming Whisper / CTC hybrid — `arxiv.org/pdf/2501.11378`, `arxiv.org/html/2506.12154v1`, `arxiv.org/pdf/2307.14743`, `arxiv.org/pdf/2203.16758`
- faster-whisper streaming discussion — `github.com/SYSTRAN/faster-whisper/issues/843`
- NVIDIA speech ASR scaling — `huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents`
