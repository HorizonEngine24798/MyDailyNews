from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from ..debug import DebugLogger
from ..models import AIConfig
from ..utils import safe_json_load
from .base import AIClient, AIJsonError, AITransportError, JSONSchemaSpec, write_ai_json_artifact, write_ai_text_artifact


class LlamaCppServerClient(AIClient):
    """LLM backend that targets llama-server's OpenAI-compatible chat endpoint."""

    def __init__(self, config: AIConfig, debug: DebugLogger | None = None) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.base_url = self._normalize_base_url(config.base_url)
        self.timeout_seconds = max(10, int(config.request_timeout_seconds))

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

    def unload(self) -> None:
        # External server lifecycle is managed outside this process.
        self.debug.log(
            "ai.unload",
            "noop",
            model=self.config.effective_model_label,
            backend=self.config.backend,
            endpoint=self.base_url,
        )

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
        _ = input_token_limit
        attempts = max(1, self.config.json_retries + 1)
        target_max_new = max(64, int(max_new_tokens or self.max_new_tokens))
        last_response_chars = 0
        last_error = ""
        last_response_text = ""
        last_failure: Dict[str, Any] = {}

        for attempt in range(1, attempts + 1):
            attempt_user = self._retry_user_prompt(user) if attempt > 1 else user
            payload = self._build_payload(system, attempt_user, target_max_new, json_schema=json_schema)
            input_tokens = self.estimate_tokens(f"System:\n{system}\n\nUser:\n{attempt_user}")
            self.debug.log(
                "ai.request",
                label,
                attempt=f"{attempt}/{attempts}",
                model=self.config.effective_model_label,
                backend=self.config.backend,
                endpoint=self.base_url,
                system_chars=len(system),
                user_chars=len(attempt_user),
                max_new_tokens=target_max_new,
                schema=bool(json_schema),
            )

            try:
                text = self._post_chat_completion(payload)
            except AITransportError as exc:
                last_error = str(exc)
                self.debug.log(
                    "ai.response",
                    label,
                    status="transport_error",
                    attempt=f"{attempt}/{attempts}",
                    error=last_error,
                )
                self.debug.record_ai(label=label, status="transport_error", input_tokens=input_tokens, estimated=True)
                continue

            last_response_chars = len(text)
            last_response_text = text
            output_tokens = self.estimate_tokens(text)
            parsed = safe_json_load(text)
            if parsed is not None:
                self.debug.log(
                    "ai.response",
                    label,
                    status="ok",
                    attempt=f"{attempt}/{attempts}",
                    response_chars=len(text),
                )
                self.debug.record_ai(
                    label=label,
                    status="ok",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_chars=len(text),
                    estimated=True,
                )
                return parsed

            last_failure = {
                "label": label,
                "backend": self.config.backend,
                "model": self.config.effective_model_label,
                "attempt": attempt,
                "attempts": attempts,
                "system_prompt": system,
                "user_prompt": attempt_user,
                "max_new_tokens_requested": target_max_new,
                "response_chars": len(text),
                "raw_response": text,
            }
            self.debug.log(
                "ai.response",
                label,
                status="invalid_json",
                attempt=f"{attempt}/{attempts}",
                response_chars=len(text),
            )
            self.debug.record_ai(
                label=label,
                status="invalid_json",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_chars=len(text),
                estimated=True,
            )

        if last_error:
            raise AITransportError(f"{label}: request failed after {attempts} attempt(s): {last_error}")
        artifact_path = ""
        raw_response_path = ""
        if last_failure:
            raw_response_path, artifact_path = self._write_invalid_json_artifacts(label, last_failure)
        raise AIJsonError(
            f"{label}: model did not return valid JSON after {attempts} attempt(s); "
            f"last response had {last_response_chars} characters"
            + (f"; raw response saved to {raw_response_path}" if raw_response_path else ""),
            artifact_path=artifact_path,
            raw_response_path=raw_response_path,
            raw_response=last_response_text,
            diagnostics=last_failure,
        )

    def _build_payload(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
        *,
        json_schema: Optional[JSONSchemaSpec],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.server_model or self.config.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_new_tokens,
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
        }

        if json_schema and (self.config.response_format in {"json_schema", "auto"}):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.name,
                    "schema": json_schema.schema,
                },
            }
            # Compatibility for llama.cpp builds that also check top-level json_schema.
            payload["json_schema"] = json_schema.schema
        else:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def _post_chat_completion(self, payload: Dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise AITransportError(f"POST {url}: {exc}") from exc

        if response.status_code >= 400:
            body_preview = (response.text or "").strip()
            if len(body_preview) > 280:
                body_preview = body_preview[:277] + "..."
            raise AITransportError(f"POST {url} -> {response.status_code}: {body_preview}")

        try:
            raw = response.json()
        except ValueError as exc:
            raise AITransportError(f"POST {url}: invalid JSON response body") from exc
        content = self._extract_content(raw)
        return content.strip()

    @staticmethod
    def _extract_content(raw: Dict[str, Any]) -> str:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AITransportError("chat completion response missing choices[0]")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)
            if chunks:
                return "\n".join(chunks)

        if isinstance(content, dict):
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)

        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _retry_user_prompt(user: str) -> str:
        return (
            f"{user}\n\n"
            "Retry instruction: your previous answer could not be parsed as one valid JSON object. "
            "Return exactly one JSON object only. Do not include markdown fences, explanations, or trailing text."
        )

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        base = (value or "http://127.0.0.1:8080/v1").strip().rstrip("/")
        if base.endswith("/v1"):
            return base
        return base + "/v1"

    def _write_invalid_json_artifacts(self, label: str, details: Dict[str, Any]) -> tuple[str, str]:
        try:
            raw_response = str(details.get("raw_response", ""))
            raw_response_path = write_ai_text_artifact("ai_invalid_json", label, raw_response)
            payload = dict(details)
            payload["raw_response_path"] = raw_response_path
            artifact_path = write_ai_json_artifact("ai_invalid_json", label, payload)
            self.debug.log(
                "ai.response",
                "invalid_json_saved",
                label=label,
                raw_response_path=raw_response_path,
                artifact_path=artifact_path,
            )
            return raw_response_path, artifact_path
        except Exception as exc:
            self.debug.log("ai.response", "invalid_json_save_failed", label=label, error=type(exc).__name__)
            return "", ""
