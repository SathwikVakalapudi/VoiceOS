"""Known model identifiers for each pluggable stage.

Purely informational — nothing stops you from pointing settings at any
other model. This is the seed of a future model registry.
"""

WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "distil-large-v3",
    "large-v3-turbo",
)

# Any OpenAI-compatible server works; these are the tested Ollama tags.
QWEN_MODELS = (
    "qwen3:14b",
    "qwen3:8b",
    "qwen3:4b",
)

# Svara-TTS voice ids follow "{language_code}_{gender}" across 19 languages
# (18 Indic + Indian English). A few common ones:
SVARA_VOICES = (
    "en_female",
    "en_male",
    "hi_female",
    "hi_male",
    "bn_female",
    "bn_male",
    "ta_female",
    "ta_male",
    "te_female",
    "te_male",
)
