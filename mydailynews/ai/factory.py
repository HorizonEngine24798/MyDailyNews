from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from ..debug import DebugLogger
from ..models import AIConfig
from .base import AIClient, AITransportError, JSONSchemaSpec


AUTO_BACKEND = "auto"
LLAMA_CPP_BACKEND = "llama_cpp_server"
TRANSFORMERS_BACKEND = "transformers"


def normalize_backend(value: str) -> str:
    raw = (value or AUTO_BACKEND).strip().lower().replace("-", "_")
    aliases = {
        "default": AUTO_BACKEND,
        "hf": TRANSFORMERS_BACKEND,
        "huggingface": TRANSFORMERS_BACKEND,
        "local": AUTO_BACKEND,
        "llama_cpp": LLAMA_CPP_BACKEND,
        "llama_server": LLAMA_CPP_BACKEND,
        "llamacpp_server": LLAMA_CPP_BACKEND,
    }
    return aliases.get(raw, raw)


def create_ai_client(config: AIConfig, debug: DebugLogger | None = None) -> AIClient:
    backend = normalize_backend(config.backend)
    config.backend = backend
    if backend == AUTO_BACKEND:
        return AutoFallbackAIClient(config, debug)
    return _create_specific_ai_client(config, backend, debug)


def _create_specific_ai_client(config: AIConfig, backend: str, debug: DebugLogger | None = None) -> AIClient:
    if backend == TRANSFORMERS_BACKEND:
        from .transformers_client import TransformersAIClient

        return TransformersAIClient(config, debug)
    if backend == LLAMA_CPP_BACKEND:
        from .llama_cpp_server_client import LlamaCppServerClient

        return LlamaCppServerClient(config, debug)
    raise ValueError(f"Unsupported ai backend: {config.backend}")


class AutoFallbackAIClient(AIClient):
    """Prefer llama.cpp, with lazy transformers fallback for backend startup/transport failure."""

    def __init__(self, config: AIConfig, debug: DebugLogger | None = None) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self._primary_config = replace(config, backend=LLAMA_CPP_BACKEND)
        self._fallback_config = replace(config, backend=TRANSFORMERS_BACKEND)
        self._primary_client: Optional[AIClient] = None
        self._fallback_client: Optional[AIClient] = None
        self._primary_failed = False

    @property
    def max_input_tokens(self) -> int:
        return max(512, int(self.config.max_input_tokens))

    @property
    def max_new_tokens(self) -> int:
        return max(64, int(self.config.max_new_tokens))

    def estimate_tokens(self, text: str) -> int:
        ratio = float(self.config.token_estimation_chars_per_token or 4.0)
        ratio = max(1.2, min(8.0, ratio))
        return max(1, int(round(len(text or "") / ratio)))

    def complete_json(
        self,
        system: str,
        user: str,
        label: str = "ai.complete_json",
        *,
        max_new_tokens: int | None = None,
        input_token_limit: int | None = None,
        json_schema: Optional[JSONSchemaSpec] = None,
    ) -> Dict[str, Any]:
        if not self._primary_failed:
            try:
                return self._get_primary_client().complete_json(
                    system,
                    user,
                    label,
                    max_new_tokens=max_new_tokens,
                    input_token_limit=input_token_limit,
                    json_schema=json_schema,
                )
            except (AITransportError, RuntimeError) as exc:
                self._primary_failed = True
                self.debug.log(
                    "ai.fallback",
                    "primary_failed",
                    label=label,
                    from_backend=LLAMA_CPP_BACKEND,
                    to_backend=TRANSFORMERS_BACKEND,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                self._close_client(self._primary_client)
                self._primary_client = None

        self.debug.log(
            "ai.fallback",
            "fallback_starting",
            label=label,
            backend=TRANSFORMERS_BACKEND,
            model=self._fallback_config.model_id,
        )
        try:
            return self._get_fallback_client().complete_json(
                system,
                user,
                label,
                max_new_tokens=max_new_tokens,
                input_token_limit=input_token_limit,
                json_schema=json_schema,
            )
        except Exception as exc:
            self.debug.log(
                "ai.fallback",
                "fallback_failed",
                label=label,
                backend=TRANSFORMERS_BACKEND,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    def unload(self) -> None:
        self._unload_client(self._primary_client)
        self._unload_client(self._fallback_client)

    def close(self) -> None:
        self._close_client(self._primary_client)
        self._close_client(self._fallback_client)
        self._primary_client = None
        self._fallback_client = None

    def _get_primary_client(self) -> AIClient:
        if self._primary_client is None:
            self._primary_client = _create_specific_ai_client(self._primary_config, LLAMA_CPP_BACKEND, self.debug)
        return self._primary_client

    def _get_fallback_client(self) -> AIClient:
        if self._fallback_client is None:
            self._fallback_client = _create_specific_ai_client(
                self._fallback_config,
                TRANSFORMERS_BACKEND,
                self.debug,
            )
        return self._fallback_client

    @staticmethod
    def _unload_client(client: Optional[AIClient]) -> None:
        if client is None:
            return
        try:
            client.unload()
        except Exception:
            pass

    @staticmethod
    def _close_client(client: Optional[AIClient]) -> None:
        if client is None:
            return
        close = getattr(client, "close", None)
        try:
            if callable(close):
                close()
            else:
                client.unload()
        except Exception:
            pass
