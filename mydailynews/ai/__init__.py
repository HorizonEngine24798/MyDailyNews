"""AI backend adapters and prompt tooling."""

from .base import AIBackendError, AIClient, AIJsonError, AITransportError, JSONSchemaSpec, set_ai_artifact_root
from .factory import create_ai_client

# Optional convenience exports (best-effort when optional deps are installed).
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
    "AIClient",
    "AIJsonError",
    "AITransportError",
    "JSONSchemaSpec",
    "set_ai_artifact_root",
    "create_ai_client",
    "LlamaCppServerClient",
    "TransformersAIClient",
    "LocalAIClient",
]
