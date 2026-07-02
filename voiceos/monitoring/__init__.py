"""Monitoring — aggregate pipeline metrics and expose them for a dashboard.

Builds on the per-turn `LatencyMonitor` (which logs one turn) by
accumulating across turns into percentiles and counts, and optionally
serving them as JSON over HTTP.
"""

from voiceos.monitoring.collector import MetricsCollector

__all__ = ["MetricsCollector"]
