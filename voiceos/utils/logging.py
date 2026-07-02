"""Logging setup. Plain stdlib logging for now; structlog is a drop-in later."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party chatter we don't need at INFO
    for noisy in ("httpx", "httpcore", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
