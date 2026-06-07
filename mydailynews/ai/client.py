from __future__ import annotations

"""Convenience AI client exports.

Prefer importing from:
- mydailynews.ai.base
- mydailynews.ai.factory
- mydailynews.ai.llama_cpp_server_client
- mydailynews.ai.transformers_client (last-resort fallback only)
"""

from .base import AIBackendError, AIJsonError, AITransportError, JSONSchemaSpec, set_ai_artifact_root
from .factory import AutoFallbackAIClient, create_ai_client

def __getattr__(name: str):
    if name in {"TransformersAIClient", "LocalAIClient"}:
        from .transformers_client import LocalAIClient, TransformersAIClient

        return LocalAIClient if name == "LocalAIClient" else TransformersAIClient
    if name == "LlamaCppServerClient":
        from .llama_cpp_server_client import LlamaCppServerClient

        return LlamaCppServerClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AIBackendError",
    "AIJsonError",
    "AITransportError",
    "JSONSchemaSpec",
    "set_ai_artifact_root",
    "create_ai_client",
    "AutoFallbackAIClient",
    "TransformersAIClient",
    "LocalAIClient",
    "LlamaCppServerClient",
]
