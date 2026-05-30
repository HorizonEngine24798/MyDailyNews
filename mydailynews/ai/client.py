from __future__ import annotations

"""Convenience AI client exports.

Prefer importing from:
- mydailynews.ai.base
- mydailynews.ai.factory
- mydailynews.ai.transformers_client
- mydailynews.ai.llama_cpp_server_client
"""

from .base import AIBackendError, AIJsonError, AITransportError, JSONSchemaSpec, set_ai_artifact_root

try:
    from .transformers_client import LocalAIClient, TransformersAIClient
except Exception:  # pragma: no cover - optional dependency path
    LocalAIClient = None  # type: ignore[assignment]
    TransformersAIClient = None  # type: ignore[assignment]

try:
    from .llama_cpp_server_client import LlamaCppServerClient
except Exception:  # pragma: no cover - optional dependency path
    LlamaCppServerClient = None  # type: ignore[assignment]

__all__ = [
    "AIBackendError",
    "AIJsonError",
    "AITransportError",
    "JSONSchemaSpec",
    "set_ai_artifact_root",
    "TransformersAIClient",
    "LocalAIClient",
    "LlamaCppServerClient",
]
