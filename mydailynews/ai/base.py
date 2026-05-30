from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


class AIBackendError(RuntimeError):
    """Base class for runtime AI backend errors."""


class AITransportError(AIBackendError):
    """Raised for HTTP/network issues talking to an external AI backend."""


class AIJsonError(AIBackendError):
    """Raised after retries when a backend cannot produce parseable JSON."""


@dataclass(frozen=True)
class JSONSchemaSpec:
    """Optional JSON schema request for structured generation."""

    name: str
    schema: Dict[str, Any]


class AIClient(Protocol):
    """Backend-agnostic interface used by the pipeline."""

    config: Any

    @property
    def max_input_tokens(self) -> int: ...

    @property
    def max_new_tokens(self) -> int: ...

    def estimate_tokens(self, text: str) -> int: ...

    def complete_json(
        self,
        system: str,
        user: str,
        label: str = "ai.complete_json",
        *,
        max_new_tokens: int | None = None,
        input_token_limit: int | None = None,
        json_schema: Optional[JSONSchemaSpec] = None,
    ) -> Dict[str, Any]: ...

    def unload(self) -> None: ...
