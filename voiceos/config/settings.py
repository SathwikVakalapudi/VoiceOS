"""Application settings.

Everything is overridable via environment variables (or a `.env` file)
using the prefix ``VOICEOS_`` and ``__`` as the nesting delimiter, e.g.:

    VOICEOS_LLM__MODEL=qwen3:8b
    VOICEOS_STT__MODEL=medium
    VOICEOS_TTS__VOICE=hi_female
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseModel):
    """Microphone / speaker I/O. Knows nothing about AI."""

    input_device: int | None = None    # None -> system default
    output_device: int | None = None
    input_sample_rate: int = 16000     # required by Silero VAD + Whisper
    frame_size: int = 512              # samples per frame; Silero v5 needs 512 @ 16 kHz
    channels: int = 1
    queue_max_frames: int = 256        # mic backpressure: oldest frames are dropped


class VADSettings(BaseModel):
    threshold: float = 0.5             # prob above this STARTS speech
    # Hysteresis exit bar: once speech is latched, stay in speech while prob is
    # above this lower value, so a mid-word dip below `threshold` doesn't read as
    # silence (flicker, early cutoff, dropped short answers). None -> threshold-0.15,
    # Silero's own convention (and what dashboard/streaming_vad.py already uses).
    neg_threshold: float | None = None
    min_speech_ms: int = 150           # shorter bursts are discarded as noise
    min_silence_ms: int = 700          # this much trailing silence ends the utterance
    pre_roll_ms: int = 300             # audio kept from just before speech started
    max_utterance_s: float = 30.0      # hard cap; force-close runaway utterances
    use_onnx: bool = True

    # Adaptive endpointing: how long to wait before deciding a turn is over
    # depends on how much has been said. A brief utterance is likely mid-
    # thought, so wait longer; once the user has clearly said a lot, close
    # faster and respond sooner. Off by default (fixed min_silence_ms).
    adaptive_silence: bool = False
    min_silence_short_ms: int = 1100   # trailing silence required for a short utterance
    short_utterance_ms: int = 1200     # speech shorter than this counts as "short"

    # Adaptive noise gate: require frame energy to clear a rolling noise
    # floor before trusting the VAD, so steady background noise or echo is
    # less likely to false-trigger. Off by default (VAD probability only).
    adaptive_noise: bool = False
    noise_window_ms: int = 3000        # rolling window used to estimate the floor
    noise_margin: float = 2.0          # frame RMS must exceed floor * this to count

    # Predictive endpointing: transcribe the utterance-so-far while the user
    # speaks and, when the partial text looks like a complete turn, close
    # after a short silence instead of the full one — responding sooner.
    # CPU-heavy (re-runs STT on a rolling buffer) and off by default.
    predictive_endpointing: bool = False
    partial_interval_ms: int = 400     # how often to re-transcribe the buffer
    min_partial_speech_ms: int = 600   # don't transcribe until this much is said
    predicted_silence_ms: int = 250    # trailing silence needed after a "done" guess
    min_partial_chars: int = 12        # ignore partials shorter than this

    # Smart Turn v3: a local semantic end-of-turn model (raw waveform -> "turn
    # complete" probability, ~12 ms on CPU). On a short pause it decides whether
    # the user actually finished, so the turn closes fast when done and keeps
    # waiting when they paused mid-sentence — without the repeated STT calls
    # predictive endpointing needs. A positive prediction closes after the short
    # `predicted_silence_ms`. Off by default (needs the model file + transformers).
    smart_turn: bool = False
    smart_turn_model: str = "models/smart-turn-v3.2-cpu.onnx"
    smart_turn_pause_ms: int = 300     # pause that triggers a completion check
    smart_turn_threshold: float = 0.5  # prob >= this -> treat the turn as complete

    # Barge-in: interrupt the assistant by talking over it.
    # Works best with headphones — on loudspeakers the mic hears the
    # assistant's own voice and may self-interrupt.
    barge_in: bool = True
    barge_in_threshold: float = 0.7    # stricter than normal to resist echo/noise
    barge_in_speech_ms: int = 250      # sustained speech required to trigger
    # Echo gate: cross-correlate the mic against what was just played. Echo is
    # the same signal delayed, so it correlates strongly; another voice does
    # not. Lets barge-in stay on over loudspeakers instead of needing
    # headphones, without an adaptive filter.
    echo_gate: bool = False
    echo_gate_threshold: float = 0.35  # peak normalised correlation to call it echo
    echo_window_ms: int = 500          # must span the speaker->room->mic delay

    @model_validator(mode="after")
    def _default_neg_threshold(self) -> "VADSettings":
        # Fill the hysteresis exit bar relative to the (possibly overridden)
        # start threshold, unless the caller set it explicitly.
        if self.neg_threshold is None:
            self.neg_threshold = round(max(0.0, self.threshold - 0.15), 3)
        return self


class STTSettings(BaseModel):
    provider: str = "whisper"          # "whisper" (local) | "sarvam" (hosted API)
    # Ordered backup providers tried when the primary raises, e.g. ["whisper"]
    # so a hosted STT outage falls back to the local model. Same provider
    # names as `provider`; each reuses this settings block.
    fallback: list[str] = Field(default_factory=list)

    # whisper: local faster-whisper
    model: str = "small"               # faster-whisper size or local path
    device: str = "auto"               # cpu | cuda | auto
    compute_type: str = "default"      # int8 | float16 | default
    language: str | None = "en"        # None -> autodetect per utterance
    beam_size: int = 5
    # Hallucination guards. Whisper was trained on subtitle-heavy web audio, so
    # on near-silence it emits training artifacts ("Thank you for watching") or
    # loops the previous phrase. Re-feeding prior text is what sustains the
    # loop, hence the False default — it costs a little cross-segment coherence,
    # which single-utterance turns don't need anyway.
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6        # drop segments the model calls silence
    compression_ratio_threshold: float = 2.4  # reject degenerate repetition

    # sarvam: hosted Saaras/Saarika API (great for Indian accents/languages)
    sarvam_api_key: str = ""
    sarvam_model: str = "saarika:v2.5"  # transcription (saaras = translation!)
    sarvam_language: str | None = None  # BCP-47 like "en-IN"; None -> autodetect
    sarvam_timeout_s: float = 12.0     # a turn stalled longer than this is dead anyway
    # Streaming STT over WebSocket. Audio uploads while the user is still
    # talking, so when the turn ends only the tail is left to finalise —
    # removing most of the batch round-trip from perceived latency.
    sarvam_streaming: bool = False
    sarvam_streaming_model: str = "saaras:v3"   # or saarika:v2.5 (legacy)
    sarvam_streaming_sample_rate: int = 16000


class LLMEndpoint(BaseModel):
    """One OpenAI-compatible LLM endpoint used as a fallback brain.

    Only the connection differs from the primary; anything left unset is
    inherited from the primary LLMSettings at build time.
    """

    base_url: str
    model: str
    api_key: str = "not-needed"
    reasoning_effort: str | None = None


class LLMSettings(BaseModel):
    """Any OpenAI-compatible chat completions endpoint (Ollama, vLLM, Groq, ...)."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"
    # Extra keys to round-robin across (e.g. several free-tier accounts) so no
    # single key's rate limit is hit as fast. Empty -> just use api_key.
    api_keys: list[str] = Field(default_factory=list)
    model: str = "qwen3:14b"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_s: float = 120.0
    # For reasoning models on servers that support it (e.g. Groq qwen3):
    # "none" skips thinking entirely — much lower voice latency.
    reasoning_effort: str | None = None
    # Backup brains tried in order if the primary endpoint is unreachable,
    # e.g. a hosted Groq endpoint behind a local Ollama. Each rolls over only
    # before any tokens stream, so a mid-reply drop never repeats speech.
    fallbacks: list[LLMEndpoint] = Field(default_factory=list)

    # Tool calling: let the model invoke registered tools (APIs/functions)
    # before it speaks. Off by default; enabled per deployment.
    tools_enabled: bool = False
    max_tool_iterations: int = 4      # cap tool<->model round-trips per turn


class TTSSettings(BaseModel):
    """TTS provider selection plus per-provider settings."""

    provider: str = "svara"            # "svara" | "edge" | "cartesia" | "piper"
    # Ordered backup providers tried when the primary fails before emitting
    # audio, e.g. ["edge"] so a self-hosted TTS outage falls back to the free
    # cloud voice. All providers must share `sample_rate`.
    fallback: list[str] = Field(default_factory=list)
    sample_rate: int = 24000           # pipeline-wide TTS output rate

    # svara: self-hosted inference server (OpenAI-compatible /v1/audio/speech)
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model: str = "svara-tts-v1"
    voice: str = "en_female"
    timeout_s: float = 120.0

    # edge: free Microsoft neural voices (needs internet, no key)
    edge_voice: str = "en-IN-NeerjaNeural"

    # cartesia: Sonic — lowest-latency hosted TTS, streams raw PCM
    cartesia_api_key: str = ""
    cartesia_model: str = "sonic-3.5"
    cartesia_voice_id: str = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"  # Skylar
    cartesia_language: str = "en"
    cartesia_timeout_s: float = 30.0

    # piper: fully local/offline neural TTS (CPU). Voice from huggingface.co/
    # rhasspy/piper-voices; sample_rate must match the voice's .onnx.json.
    piper_binary: str = "piper"
    piper_model: str = ""              # path to the voice .onnx file
    piper_sample_rate: int = 22050
    piper_speaker: int | None = None   # multi-speaker voices only


class ConversationSettings(BaseModel):
    max_turns: int = 20                # history is trimmed beyond this many exchanges
    system_prompt: str | None = None   # None -> voiceos.llm.prompts.DEFAULT_SYSTEM_PROMPT
    first_message: str | None = None   # spoken by the assistant on start (outbound style)
    # Spoken when the LLM fails after all retries — keep it in the campaign language.
    error_message: str = "Sorry, I lost my connection for a moment. Say that again?"
    # JSON file with {"system_prompt", "first_message", "error_message"} — overrides the above.
    campaign_file: str | None = None
    # A respondent who goes quiet must be prompted, not waited on forever.
    # The assistant is nudged after this much silence; the campaign prompt
    # decides what to say, since it already defines the escalation.
    no_input_timeout_s: float = 5.0
    no_input_max_prompts: int = 3      # then hang up rather than nag


class PipelineSettings(BaseModel):
    sentence_min_chars: int = 24       # minimum chunk length streamed to TTS

    # Backchanneling: play a short filler ("mm-hmm", "right") if the assistant
    # is still THINKING after the delay below, so a slow turn isn't dead air.
    # Fires only while the mic is gated (THINKING), so it is never mistaken for
    # the user. Off by default; needs a working TTS to pre-render the fillers.
    backchannel: bool = False
    backchannel_delay_ms: int = 900    # thinking must exceed this before a filler
    backchannel_phrases: list[str] = Field(
        default_factory=lambda: ["mm-hmm", "right", "okay", "let me see"]
    )


class MonitoringSettings(BaseModel):
    """Read-only metrics dashboard (JSON over HTTP). Off by default."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8081
    # Per-call records for the dashboard Logs tab.
    calls_file: str = "results/calls.jsonl"
    # Optional JSON overriding the built-in provider price list, which is a
    # snapshot of published rates and goes stale without warning.
    pricing_file: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICEOS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
