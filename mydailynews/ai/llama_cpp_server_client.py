from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import requests

from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import AIConfig
from mydailynews.common.utils import safe_json_load
from .base import AIClient, AIJsonError, AITransportError, JSONSchemaSpec, write_ai_json_artifact, write_ai_text_artifact
from .managed_llama_server import ManagedLlamaServerLease
from .prompts import GENERIC_JSON_RETRY_USER
from .token_budget import resolve_token_budget


class LlamaCppServerClient(AIClient):
    """LLM backend that targets llama-server's OpenAI-compatible chat endpoint."""

    def __init__(self, config: AIConfig, debug: DebugLogger | None = None) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.base_url = self._normalize_base_url(config.base_url)
        self.timeout_seconds = max(10, int(config.request_timeout_seconds))
        self.server_lease = ManagedLlamaServerLease(config=config, base_url=self.base_url, debug=self.debug)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def max_input_tokens(self) -> int:
        return max(512, int(self.config.max_input_tokens))

    @property
    def max_new_tokens(self) -> int:
        return max(64, int(self.config.max_new_tokens))

    def estimate_tokens(self, text: str) -> int:
        ratio = self._chars_per_token_ratio()
        return max(1, int(round(len(text or "") / ratio)))

    def unload(self) -> None:
        # Keep server warm within a run. Server shutdown is handled by close().
        if self.server_lease.enabled:
            self.debug.log(
                "ai.unload",
                "noop_managed_server",
                model=self.config.effective_model_label,
                backend=self.config.backend,
                endpoint=self.base_url,
            )
            return
        # External server lifecycle is managed outside this process.
        self.debug.log(
            "ai.unload",
            "noop",
            model=self.config.effective_model_label,
            backend=self.config.backend,
            endpoint=self.base_url,
        )

    def close(self) -> None:
        self.server_lease.release()

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
        self.server_lease.ensure_running()
        json_attempts = max(1, self.config.json_retries + 1)
        attempts = json_attempts + 1  # One extra attempt is reserved for parseable length-truncated JSON.
        budget_args = {
            "context_tokens": int(self.config.context_window_tokens),
            "max_input_tokens": int(self.config.max_input_tokens),
            "max_output_tokens": int(self.config.max_new_tokens),
        }
        initial_budget = resolve_token_budget(
            **budget_args,
            input_tokens=input_token_limit,
            output_tokens=max_new_tokens,
        )
        target_input_limit = initial_budget.input_tokens
        last_response_chars = 0
        last_error = ""
        last_response_text = ""
        last_failure: Dict[str, Any] = {}
        last_finish_reason = ""
        attempt_output_preference = initial_budget.output_tokens
        parsed_length_fallback: Dict[str, Any] | None = None
        length_retry_used = False

        for attempt in range(1, attempts + 1):
            attempt_user_raw = (
                self._retry_user_prompt(user, output_was_truncated=last_finish_reason == "length")
                if attempt > 1
                else user
            )
            attempt_budget = resolve_token_budget(
                **budget_args,
                input_tokens=target_input_limit,
                output_tokens=attempt_output_preference,
            )
            attempt_input_limit = attempt_budget.input_tokens
            (
                attempt_system,
                attempt_user,
                input_tokens,
                system_was_truncated,
                user_was_truncated,
            ) = self._fit_chat_to_input_limit(system, attempt_user_raw, attempt_input_limit)
            attempt_budget = resolve_token_budget(
                **budget_args,
                input_tokens=input_tokens,
                output_tokens=attempt_output_preference,
            )
            attempt_max_new = attempt_budget.output_tokens
            payload = self._build_payload(attempt_system, attempt_user, attempt_max_new, json_schema=json_schema)
            self.debug.log(
                "ai.request",
                label,
                attempt=f"{attempt}/{attempts}",
                model=self.config.effective_model_label,
                backend=self.config.backend,
                endpoint=self.base_url,
                system_chars=len(attempt_system),
                user_chars=len(attempt_user),
                system_chars_original=len(system),
                user_chars_original=len(attempt_user_raw),
                system_truncated=system_was_truncated,
                user_truncated=user_was_truncated,
                input_tokens=input_tokens,
                max_input_tokens=attempt_input_limit,
                max_new_tokens=attempt_max_new,
                context_tokens=attempt_budget.context_tokens,
                reserve_tokens=attempt_budget.reserve_tokens,
                schema=bool(json_schema),
            )

            request_started = time.perf_counter()
            try:
                text, finish_reason, usage, timings = self._post_chat_completion(payload)
            except AITransportError as exc:
                request_duration_ms = (time.perf_counter() - request_started) * 1000
                last_error = str(exc)
                self.debug.log(
                    "ai.response",
                    label,
                    status="transport_error",
                    attempt=f"{attempt}/{attempts}",
                    error=last_error,
                )
                self.debug.record_ai(
                    label=label,
                    status="transport_error",
                    input_tokens=input_tokens,
                    estimated=True,
                    retry=attempt > 1,
                    duration_ms=request_duration_ms,
                    input_budget_tokens=attempt_input_limit,
                    output_budget_tokens=attempt_max_new,
                )
                if parsed_length_fallback is not None:
                    return parsed_length_fallback
                if attempt >= json_attempts:
                    break
                continue

            request_duration_ms = (time.perf_counter() - request_started) * 1000
            last_response_chars = len(text)
            last_response_text = text
            last_finish_reason = finish_reason
            actual_input_tokens = self._token_count(usage.get("prompt_tokens"))
            actual_output_tokens = self._token_count(usage.get("completion_tokens"))
            recorded_input_tokens = actual_input_tokens if actual_input_tokens is not None else input_tokens
            output_tokens = (
                actual_output_tokens if actual_output_tokens is not None else self.estimate_tokens(text)
            )
            tokens_estimated = actual_input_tokens is None or actual_output_tokens is None
            parsed = safe_json_load(text)
            if parsed is not None:
                retry_output = attempt_max_new
                if finish_reason == "length" and not length_retry_used:
                    retry_budget = resolve_token_budget(
                        **budget_args,
                        input_tokens=recorded_input_tokens,
                        output_tokens=attempt_max_new * 2,
                    )
                    retry_output = retry_budget.output_tokens
                will_retry_length = (
                    finish_reason == "length"
                    and not length_retry_used
                    and retry_output > attempt_max_new
                    and attempt < attempts
                )
                self.debug.log(
                    "ai.response",
                    label,
                    status="length_retry" if will_retry_length else "ok",
                    attempt=f"{attempt}/{attempts}",
                    response_chars=len(text),
                    finish_reason=finish_reason,
                    input_tokens=recorded_input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=self._token_count(usage.get("total_tokens")),
                    timings=timings,
                )
                self.debug.record_ai(
                    label=label,
                    status="ok",
                    input_tokens=recorded_input_tokens,
                    output_tokens=output_tokens,
                    response_chars=len(text),
                    estimated=tokens_estimated,
                    retry=attempt > 1,
                    finish_reason=finish_reason,
                    duration_ms=request_duration_ms,
                    input_budget_tokens=attempt_input_limit,
                    output_budget_tokens=attempt_max_new,
                )
                if will_retry_length:
                    parsed_length_fallback = parsed
                    length_retry_used = True
                    attempt_output_preference = retry_output
                    continue
                if finish_reason == "length" and parsed_length_fallback is not None:
                    return parsed_length_fallback
                return parsed

            last_failure = {
                "label": label,
                "backend": self.config.backend,
                "model": self.config.effective_model_label,
                "attempt": attempt,
                "attempts": attempts,
                "system_prompt": attempt_system,
                "user_prompt": attempt_user,
                "system_prompt_original": system,
                "user_prompt_original": attempt_user_raw,
                "system_prompt_truncated": system_was_truncated,
                "user_prompt_truncated": user_was_truncated,
                "input_tokens_estimated": input_tokens,
                "max_new_tokens_requested": attempt_max_new,
                "input_token_limit_requested": target_input_limit,
                "input_token_limit_used": attempt_input_limit,
                "response_chars": len(text),
                "finish_reason": finish_reason,
                "usage": usage,
                "timings": timings,
                "raw_response": text,
            }
            self.debug.log(
                "ai.response",
                label,
                status="invalid_json",
                attempt=f"{attempt}/{attempts}",
                response_chars=len(text),
                finish_reason=finish_reason,
                input_tokens=recorded_input_tokens,
                output_tokens=output_tokens,
                total_tokens=self._token_count(usage.get("total_tokens")),
                timings=timings,
            )
            self.debug.record_ai(
                label=label,
                status="invalid_json",
                input_tokens=recorded_input_tokens,
                output_tokens=output_tokens,
                response_chars=len(text),
                estimated=tokens_estimated,
                retry=attempt > 1,
                finish_reason=finish_reason,
                duration_ms=request_duration_ms,
                input_budget_tokens=attempt_input_limit,
                output_budget_tokens=attempt_max_new,
            )
            if parsed_length_fallback is not None:
                return parsed_length_fallback
            if attempt >= json_attempts:
                break
            if finish_reason == "length":
                retry_budget = resolve_token_budget(
                    **budget_args,
                    input_tokens=recorded_input_tokens,
                    output_tokens=attempt_max_new * 2,
                )
                attempt_output_preference = retry_budget.output_tokens

        if parsed_length_fallback is not None:
            return parsed_length_fallback
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

    def _post_chat_completion(
        self,
        payload: Dict[str, Any],
    ) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
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
        choices = raw.get("choices")
        finish_reason = str(choices[0].get("finish_reason") or "") if isinstance(choices, list) and choices else ""
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}
        return content.strip(), finish_reason, dict(usage), dict(timings)

    @staticmethod
    def _token_count(value: Any) -> int | None:
        try:
            count = int(value)
        except (TypeError, ValueError):
            return None
        return count if count >= 0 else None

    @staticmethod
    def _extract_content(raw: Dict[str, Any]) -> str:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AITransportError("chat completion response missing choices[0]")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            if content.strip():
                return content
            reasoning_content = message.get("reasoning_content", "")
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                return reasoning_content

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
    def _retry_user_prompt(user: str, *, output_was_truncated: bool = False) -> str:
        prompt = GENERIC_JSON_RETRY_USER.format(user=user)
        if output_was_truncated:
            prompt = (
                "The previous response hit the output token limit. Preserve every required field and item, "
                "but keep text values concise enough to finish.\n"
                f"{prompt}"
            )
        return prompt

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

    def _fit_chat_to_input_limit(
        self,
        system: str,
        user: str,
        input_token_limit: int,
    ) -> Tuple[str, str, int, bool, bool]:
        target_limit = max(64, int(input_token_limit))
        fitted_system = system or ""
        fitted_user = user or ""
        ratio = self._chars_per_token_ratio()
        system_truncated = False
        user_truncated = False
        input_tokens = self._estimate_chat_input_tokens(fitted_system, fitted_user)
        if input_tokens <= target_limit:
            return fitted_system, fitted_user, input_tokens, system_truncated, user_truncated

        if fitted_user:
            empty_user_tokens = self._estimate_chat_input_tokens(fitted_system, "")
            if empty_user_tokens < target_limit:
                allowed_user_chars = int((target_limit - empty_user_tokens) * ratio)
                reduced_user = fitted_user[: max(0, allowed_user_chars)].rstrip()
                if reduced_user != fitted_user:
                    user_truncated = True
                    fitted_user = reduced_user
            else:
                if fitted_user:
                    user_truncated = True
                fitted_user = ""

        input_tokens = self._estimate_chat_input_tokens(fitted_system, fitted_user)
        if input_tokens > target_limit and fitted_system:
            empty_system_tokens = self._estimate_chat_input_tokens("", fitted_user)
            if empty_system_tokens < target_limit:
                allowed_system_chars = int((target_limit - empty_system_tokens) * ratio)
                reduced_system = fitted_system[: max(0, allowed_system_chars)].rstrip()
                if reduced_system != fitted_system:
                    system_truncated = True
                    fitted_system = reduced_system
            else:
                if fitted_system:
                    system_truncated = True
                fitted_system = ""

        input_tokens = self._estimate_chat_input_tokens(fitted_system, fitted_user)
        while input_tokens > target_limit and (fitted_user or fitted_system):
            overshoot_tokens = input_tokens - target_limit
            trim_chars = max(1, int(overshoot_tokens * ratio) + 4)
            if fitted_user:
                fitted_user = fitted_user[:-trim_chars].rstrip()
                user_truncated = True
            elif fitted_system:
                fitted_system = fitted_system[:-trim_chars].rstrip()
                system_truncated = True
            input_tokens = self._estimate_chat_input_tokens(fitted_system, fitted_user)

        return fitted_system, fitted_user, input_tokens, system_truncated, user_truncated

    def _estimate_chat_input_tokens(self, system: str, user: str) -> int:
        return self.estimate_tokens(f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n")

    def _chars_per_token_ratio(self) -> float:
        ratio = float(self.config.token_estimation_chars_per_token or 4.0)
        return max(1.2, min(8.0, ratio))
