"""Minimal monitoring dashboard.

Serves the collector's snapshot as JSON over a stdlib HTTP server in a
daemon thread — no extra dependencies. Enough for `curl localhost:8081`
or a browser to watch a live session; a richer UI can consume the same
JSON. Read-only.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from voiceos.monitoring.collector import MetricsCollector

logger = logging.getLogger(__name__)


def serve_metrics(
    collector: MetricsCollector, host: str = "127.0.0.1", port: int = 8081
) -> HTTPServer:
    """Start serving `collector.snapshot()` at http://host:port/ and return
    the server (call shutdown() to stop). Runs in a background daemon thread."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            body = json.dumps(collector.snapshot(), indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # silence per-request logging
            pass

    server = HTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("metrics dashboard at http://%s:%d", host, port)
    return server
