# Fully local VoiceOS — your own Vapi, offline, on your machine

VoiceOS already *is* Vapi's architecture: a swappable **STT → LLM → TTS**
pipeline with the real-time orchestration Vapi charges for — predictive
**endpointing**, **barge-in**, **backchanneling**, and **sentence-level
streaming** (LLM streams tokens → sentences are cut and sent to TTS immediately
while generation continues). To run it **entirely on your machine** you just
swap the cloud providers for local ones:

| Stage | Cloud (current) | **Local (this guide)** |
|---|---|---|
| VAD | Silero (already local) | Silero — no change |
| STT | Sarvam (cloud) | **faster-whisper** (local, already built in) |
| LLM | Gemini (cloud) | **Ollama** (local, VoiceOS's default) |
| TTS | Cartesia (cloud) | **Piper** (local, new provider) |

Nothing leaves your machine; no API keys, no per-minute fees, works offline.

---

## 1. Install the three local engines

**Ollama (LLM)** — https://ollama.com
```bash
ollama pull qwen2.5:3b        # good multilingual 3B; fits ~2-3 GB
# (bigger = better but heavier: qwen2.5:7b needs ~5 GB)
ollama serve                  # exposes http://localhost:11434
```

**faster-whisper (STT)** — already in requirements. The model downloads on
first run (cached). `base`/`small` are the sizes to use on a 4 GB GPU.

**Piper (TTS)** — https://github.com/rhasspy/piper
```bash
pip install piper-tts
# download a voice (.onnx + .onnx.json) from
# https://huggingface.co/rhasspy/piper-voices
#   Hindi:   hi_IN-*        English: en_US-lessac-medium / en_GB-*
# put them somewhere and note the .onnx path + its sample_rate (in the .json)
```

## 2. Point `.env` at the local stack
Copy from `.env.local.example`:
```bash
# LLM — local Ollama (OpenAI-compatible)
VOICEOS_LLM__BASE_URL=http://localhost:11434/v1
VOICEOS_LLM__API_KEY=ollama
VOICEOS_LLM__MODEL=qwen2.5:3b
# remove any VOICEOS_LLM__REASONING_EFFORT line (Ollama models aren't thinking models)

# STT — local faster-whisper
VOICEOS_STT__PROVIDER=whisper
VOICEOS_STT__MODEL=small
VOICEOS_STT__DEVICE=cuda            # or cpu
VOICEOS_STT__COMPUTE_TYPE=int8_float16   # use int8 on cpu

# TTS — local Piper
VOICEOS_TTS__PROVIDER=piper
VOICEOS_TTS__PIPER_MODEL=C:/Users/sathw/piper-voices/hi_IN-priyamvada-medium.onnx
VOICEOS_TTS__PIPER_SAMPLE_RATE=22050    # match the voice's .onnx.json
```

## 3. Run it
```bash
ollama serve                                  # terminal 1
python serve_dashboard.py                      # terminal 2 -> http://localhost:8080/live
# or the phone stack: docker compose up ; python main.py for local mic
```
The dashboard/live tester and the whole telephony stack work unchanged — they
never knew the providers were cloud.

---

## Hardware reality (your laptop: GTX 1650 4 GB, i5-10300H, 8 GB RAM)
This runs, but **quality drops vs the cloud stack** — be honest with yourself:
- **LLM**: a 3B local model won't follow the intricate Hindi survey rules as
  reliably as Gemini. `qwen2.5:3b` is the best small multilingual option; a 7B
  is better if you can spare the VRAM/patience (partial CPU offload = slower).
- **STT**: whisper `small` is decent for Hindi; `base` is faster but weaker.
  Sarvam (cloud) is stronger on Indian languages.
- **TTS**: Piper Hindi voices are clear but flatter than Cartesia.
- **VRAM budget (4 GB)**: whisper `small` int8 (~1 GB) + a 3B model (~2.5 GB)
  is tight but fits; run whisper on CPU if you hit OOM.

**Best of both:** VoiceOS supports **fallbacks** and per-stage config — you can
run LLM+TTS local and keep Sarvam for STT, or vice-versa, by mixing providers
in `.env`. Fully local is free/private/offline; the cloud stack is higher
quality. Pick per call.
