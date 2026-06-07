"""AI backend adapters and prompt tooling."""

from .base import AIBackendError, AIClient, AIJsonError, AITransportError, JSONSchemaSpec, set_ai_artifact_root
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
    "AIClient",
    "AIJsonError",
    "AITransportError",
    "JSONSchemaSpec",
    "set_ai_artifact_root",
    "create_ai_client",
    "AutoFallbackAIClient",
    "LlamaCppServerClient",
    "TransformersAIClient",
    "LocalAIClient",
]
