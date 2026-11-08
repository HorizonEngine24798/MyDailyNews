from __future__ import annotations

from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import AIConfig
from .base import AIClient


def create_ai_client(config: AIConfig, debug: DebugLogger | None = None) -> AIClient:
    if config.backend not in {"llama_cpp_server", "openai_compatible", "openai"}:
        raise ValueError(
            f"Unsupported ai backend: {config.backend}. "
            "Supported backends: llama_cpp_server, openai_compatible"
        )
    from .llama_cpp_server_client import LlamaCppServerClient

    return LlamaCppServerClient(config, debug)
