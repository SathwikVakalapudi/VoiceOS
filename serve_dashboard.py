"""VoiceOS campaign dashboard.

A web UI + REST API to create, test (as text chat), dry-run, and review calling
campaigns. Runs independently of the telephony servers; it reads/writes the same
`campaigns/` files and reads the `results/` written by serve_telephony.py.

    pip install fastapi uvicorn
    python serve_dashboard.py                      # http://127.0.0.1:8080
    python serve_dashboard.py --host 0.0.0.0 --port 9000

The test sandbox needs a reachable LLM (configured via .env, same as calls).
"""

from __future__ import annotations

import argparse

from voiceos.config.settings import get_settings
from voiceos.dashboard.app import create_app
from voiceos.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceos-dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--campaigns-dir", default="campaigns")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    app = create_app(
        campaigns_dir=args.campaigns_dir,
        results_dir=args.results_dir,
        settings=settings,
    )

    import uvicorn

    print(f"VoiceOS dashboard on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
