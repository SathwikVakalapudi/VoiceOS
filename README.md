# VoiceOS

An open-source, self-hosted, modular voice AI engine. Think of it as the
foundation of your own Vapi — every stage is an independent async worker,
every model is swappable behind an interface, and everything communicates
through events.

## Phase 1: the core loop

```
Microphone
    │
Audio Queue
    │
VAD Worker ──────── Silero VAD        "speaking or silent?"
    │
Utterance Queue
    │
STT Worker ──────── faster-whisper    audio → transcript
    │
Transcript Queue
    │
LLM Worker ──────── Qwen 3 14B        transcript → response (streamed)
    │                                  (via Conversation Manager)
TTS Queue
    │
TTS Worker ──────── Svara-TTS         sentence → audio (streamed)
    │
Playback Queue
    │
Playback Worker
    │
Speaker
```

Latency is pipelined: the LLM response is split into sentences as it
streams, each sentence is synthesized while the next is still being
generated, and playback starts on the first audio chunk.

## Design rules

- **Audio** knows nothing about AI. **VAD** only answers speaking/silent.
  **STT**: audio in, text out. **LLM**: messages in, text out. **TTS**:
  text in, audio out.
- The **pipeline** never performs inference — it only wires queues,
  workers, the event bus, and the state machine.
- Nothing calls a model directly. Everything goes through
  `voiceos/interfaces/` (`BaseVAD`, `BaseSTT`, `BaseLLM`, `BaseTTS`), so
  swapping Whisper for Parakeet or Qwen for anything else never touches
  the pipeline.
- All progress is announced on the **EventBus** (`SpeechStarted`,
  `TranscriptReady`, `LLMFinished`, `PlaybackFinished`, ...) — the future
  home of barge-in, analytics, and telephony hooks.
- The **state machine** (`IDLE → LISTENING → THINKING → SPEAKING`) gates
  the mic so the assistant never transcribes its own voice. Barge-in
  later is just a relaxation of this gate.

## Getting started

Requires Python 3.10+ and two model servers running locally:

**1. LLM — any OpenAI-compatible server.** Easiest is [Ollama](https://ollama.com):

```
ollama pull qwen3:14b      # or qwen3:8b on smaller hardware
```

**2. TTS — the Svara-TTS inference server** ([Kenpath/svara-tts-inference](https://github.com/Kenpath/svara-tts-inference)),
which serves an OpenAI-compatible `/v1/audio/speech` on port 8080
(Docker Compose setup in that repo; needs a GPU for real-time synthesis).

**3. VoiceOS itself:**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Whisper and Silero VAD download automatically on first run. Use
`python main.py --list-devices` to pick audio devices.

## Configuration

Copy `.env.example` to `.env`. Every setting in
[voiceos/config/settings.py](voiceos/config/settings.py) is overridable
via `VOICEOS_`-prefixed environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VOICEOS_LLM__BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM endpoint (Ollama, vLLM, Groq, ...) |
| `VOICEOS_LLM__MODEL` | `qwen3:14b` | Model name |
| `VOICEOS_LLM__REASONING_EFFORT` | unset | `none` skips reasoning-model thinking (faster voice) |
| `VOICEOS_STT__PROVIDER` | `whisper` | `whisper` (local) or `sarvam` (hosted, Indian languages) |
| `VOICEOS_STT__MODEL` | `small` | faster-whisper size |
| `VOICEOS_STT__DEVICE` | `auto` | `cpu` / `cuda` |
| `VOICEOS_STT__SARVAM_API_KEY` | unset | key from dashboard.sarvam.ai |
| `VOICEOS_STT__FALLBACK` | `[]` | Backup STT providers tried in order if the primary fails |
| `VOICEOS_TTS__PROVIDER` | `svara` | `svara` (self-hosted), `cartesia` (fastest hosted), `edge` (free) |
| `VOICEOS_TTS__CARTESIA_API_KEY` | unset | key from play.cartesia.ai |
| `VOICEOS_TTS__BASE_URL` | `http://localhost:8080/v1` | Svara-TTS server |
| `VOICEOS_TTS__VOICE` | `en_female` | Svara voice: `{lang}_{gender}`, 19 languages |
| `VOICEOS_TTS__EDGE_VOICE` | `en-IN-NeerjaNeural` | edge-tts voice name |
| `VOICEOS_TTS__FALLBACK` | `[]` | Backup TTS providers tried in order if the primary fails before audio |
| `VOICEOS_VAD__MIN_SILENCE_MS` | `700` | Trailing silence that ends a turn |
| `VOICEOS_VAD__ADAPTIVE_SILENCE` | `false` | Wait longer after a brief utterance, respond faster once enough is said |
| `VOICEOS_VAD__PREDICTIVE_ENDPOINTING` | `false` | Transcribe while the user speaks and close early when the partial looks complete (CPU-heavy) |
| `VOICEOS_VAD__ADAPTIVE_NOISE` | `false` | Require frame energy above a rolling noise floor to resist echo/noise |
| `VOICEOS_PIPELINE__BACKCHANNEL` | `false` | Play a short "mm-hmm"/"right" filler while the assistant is still thinking |
| `VOICEOS_LLM__TOOLS_ENABLED` | `false` | Let the model call registered tools/functions before answering |
| `VOICEOS_LLM__FALLBACKS` | `[]` | Backup LLM endpoints tried in order if the primary is unreachable |
| `VOICEOS_MONITORING__ENABLED` | `false` | Serve a read-only JSON metrics dashboard over HTTP |

**No GPU?** The stack runs on a plain laptop with hosted/free services —
LLM via any OpenAI-compatible API (e.g. Groq) and `VOICEOS_TTS__PROVIDER=edge`;
VAD and Whisper stay local on CPU. Same code, different `.env`.

## Tests

```
pip install -e .[dev]
pytest
```

Tests cover the pure logic (event bus, history trimming, sentence
chunking, `<think>` filtering, speech segmentation with a fake VAD) and
run without any models or audio hardware.

## Roadmap

Phase 1 (this repo): mic → VAD → STT → LLM → TTS → speaker.
Later: barge-in, streaming STT, tool calling, memory, RAG, telephony,
dashboard, analytics, multi-agent, multi-GPU.
