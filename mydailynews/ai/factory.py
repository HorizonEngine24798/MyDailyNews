from __future__ import annotations

from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import AIConfig
from .base import AIClient


LLAMA_CPP_BACKEND = "llama_cpp_server"
CODEX_AGENT_BACKEND = "codex_agent"


def create_ai_client(config: AIConfig, debug: DebugLogger | None = None) -> AIClient:
    if config.backend == CODEX_AGENT_BACKEND:
        from .codex_agent_client import CodexAgentClient

        return CodexAgentClient(config, debug)
    if config.backend != LLAMA_CPP_BACKEND:
        raise ValueError(f"Unsupported ai backend: {config.backend}")
    from .llama_cpp_server_client import LlamaCppServerClient

    return LlamaCppServerClient(config, debug)
