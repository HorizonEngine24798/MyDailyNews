from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Protocol


class AIBackendError(RuntimeError):
    """Base class for runtime AI backend errors."""


class AITransportError(AIBackendError):
    """Raised for HTTP/network issues talking to an external AI backend."""


class AIJsonError(AIBackendError):
    """Raised after retries when a backend cannot produce parseable JSON."""

    def __init__(
        self,
        message: str,
        *,
        artifact_path: str = "",
        raw_response_path: str = "",
        raw_response: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.artifact_path = artifact_path
        self.raw_response_path = raw_response_path
        self.raw_response = raw_response
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class JSONSchemaSpec:
    """Optional JSON schema request for structured generation."""

    name: str
    schema: Dict[str, Any]


_AI_ARTIFACT_ROOT = Path("output")


def set_ai_artifact_root(root: str | Path) -> None:
    """Configure where AI diagnostics artifacts are written."""
    global _AI_ARTIFACT_ROOT
    root_text = str(root or "").strip()
    _AI_ARTIFACT_ROOT = Path(root_text) if root_text else Path("output")


def write_ai_text_artifact(kind: str, label: str, text: str, suffix: str = ".txt") -> str:
    directory = _artifact_directory(kind)
    slug = _artifact_slug(label or kind)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    path = directory / f"{stamp}_{millis:03d}_{slug}{suffix}"
    path.write_text(text or "", encoding="utf-8")
    return str(path)


def write_ai_json_artifact(kind: str, label: str, payload: Dict[str, Any]) -> str:
    directory = _artifact_directory(kind)
    slug = _artifact_slug(label or kind)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    path = directory / f"{stamp}_{millis:03d}_{slug}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _artifact_directory(kind: str) -> Path:
    path = _AI_ARTIFACT_ROOT / "diagnostics" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "_", (value or "").lower()).strip("_")
    return slug[:80] or "artifact"


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
