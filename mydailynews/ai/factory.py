from __future__ import annotations

from ..debug import DebugLogger
from ..models import AIConfig
from .base import AIClient


def normalize_backend(value: str) -> str:
    raw = (value or "transformers").strip().lower().replace("-", "_")
    aliases = {
        "hf": "transformers",
        "huggingface": "transformers",
        "llama_cpp": "llama_cpp_server",
        "llama_server": "llama_cpp_server",
        "llamacpp_server": "llama_cpp_server",
    }
    return aliases.get(raw, raw)


def create_ai_client(config: AIConfig, debug: DebugLogger | None = None) -> AIClient:
    backend = normalize_backend(config.backend)
    config.backend = backend
    if backend == "transformers":
        from .transformers_client import TransformersAIClient

        return TransformersAIClient(config, debug)
    if backend == "llama_cpp_server":
        from .llama_cpp_server_client import LlamaCppServerClient

        return LlamaCppServerClient(config, debug)
    raise ValueError(f"Unsupported ai backend: {config.backend}")
