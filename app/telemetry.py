"""Central observability setup (Logfire).

Call setup_telemetry() once at application startup (before any service runs).
If LOGFIRE_TOKEN is set in .env, traces go to the Logfire Cloud dashboard;
otherwise everything stays local (console only) and nothing crashes.
"""
import os

import logfire

from app.config import settings  # importing app.config loads .env, so LOGFIRE_TOKEN is in os.environ


def setup_telemetry() -> str:
    """Configure Logfire. Returns 'cloud' or 'local'."""
    token = os.getenv("LOGFIRE_TOKEN")
    if token:
        logfire.configure(service_name="enterprise-rag")
        logfire.info("Telemetry: sending traces to Logfire Cloud.")
        return "cloud"
    logfire.configure(service_name="enterprise-rag", send_to_logfire=False)
    logfire.info("Telemetry: local only (no LOGFIRE_TOKEN in .env).")
    return "local"
