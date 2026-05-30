from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from ..debug import DebugLogger
from ..models import AIConfig
from ..utils import safe_json_load
from .base import AIClient, AIJsonError, JSONSchemaSpec, write_ai_json_artifact, write_ai_text_artifact


class _GenerationProgressCriteria(StoppingCriteria):
    def __init__(
        self,
        *,
        debug: DebugLogger,
        label: str,
        input_tokens: int,
        target_new_tokens: int,
        log_every_tokens: int = 128,
        log_every_seconds: float = 20.0,
    ) -> None:
        self.debug = debug
        self.label = label
        self.input_tokens = max(0, int(input_tokens))
        self.target_new_tokens = max(1, int(target_new_tokens))
        self.log_every_tokens = max(16, int(log_every_tokens))
        self.log_every_seconds = max(2.0, float(log_every_seconds))
        self.started_at = time.perf_counter()
        self._next_token_log = self.log_every_tokens
        self._last_log_at = self.started_at

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
        total_tokens = int(input_ids.shape[-1]) if input_ids is not None else 0
        generated_tokens = max(0, total_tokens - self.input_tokens)
        if generated_tokens <= 0:
            return False

        now = time.perf_counter()
        should_log = generated_tokens >= self._next_token_log or (now - self._last_log_at) >= self.log_every_seconds
        if not should_log:
            return False

        elapsed = max(0.001, now - self.started_at)
        tokens_per_sec = generated_tokens / elapsed
        self.debug.log(
            "ai.generate",
            "progress",
            label=self.label,
            generated_tokens=generated_tokens,
            target_new_tokens=self.target_new_tokens,
            elapsed_sec=round(elapsed, 1),
            tokens_per_sec=round(tokens_per_sec, 2),
        )
        while self._next_token_log <= generated_tokens:
            self._next_token_log += self.log_every_tokens
        self._last_log_at = now
        return False


class TransformersAIClient(AIClient):
    """In-process local LLM backend using Hugging Face Transformers."""

    def __init__(self, config: AIConfig, debug: DebugLogger | None = None) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.tokenizer = None
        self.model = None
        self.device = self._resolve_device(config.device)

    @property
    def max_input_tokens(self) -> int:
        return max(512, int(self.config.max_input_tokens))

    @property
    def max_new_tokens(self) -> int:
        return max(64, int(self.config.max_new_tokens))

    def estimate_tokens(self, text: str) -> int:
        self._ensure_loaded()
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        input_ids = encoded.get("input_ids", [])
        return int(len(input_ids)) if isinstance(input_ids, list) else 0

    def unload(self) -> None:
        if self.model is None and self.tokenizer is None:
            return
        self.debug.log("ai.unload", "starting", model=self.config.model_id, backend=self.config.backend, device=self.device)
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._cleanup_cuda(reason="unload")
        self.debug.log("ai.unload", "complete", model=self.config.model_id, backend=self.config.backend, device=self.device)

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
        # Transformers path currently relies on prompt-instructed JSON + retries.
        # `json_schema` is accepted for interface consistency and cache fingerprinting.
        _ = json_schema
        self._ensure_loaded()
        attempts = max(1, self.config.json_retries + 1)
        last_response_chars = 0
        last_response_text = ""
        last_failure: Dict[str, Any] = {}

        target_max_new_tokens = max(64, int(max_new_tokens or self.max_new_tokens))
        target_input_limit = max(512, int(input_token_limit or self._input_limit_for_generation(target_max_new_tokens)))

        for attempt in range(1, attempts + 1):
            attempt_user = self._retry_user_prompt(user) if attempt > 1 else user
            prompt = self._format_prompt(system, attempt_user)
            self.debug.log(
                "ai.request",
                label,
                attempt=f"{attempt}/{attempts}",
                model=self.config.model_id,
                backend=self.config.backend,
                device=self.device,
                enable_thinking=self.config.enable_thinking if self._uses_qwen3_chat_template() else "n/a",
                system_chars=len(system),
                user_chars=len(attempt_user),
                max_new_tokens=target_max_new_tokens,
                max_input_tokens=target_input_limit,
            )

            text, input_tokens, generated_tokens, used_max_new_tokens, used_input_limit = self._generate_with_oom_backoff(
                prompt,
                max_new_tokens=target_max_new_tokens,
                input_token_limit=target_input_limit,
                label=label,
            )
            last_response_chars = len(text)
            last_response_text = text
            parsed = safe_json_load(text)
            if parsed is not None:
                self.debug.log(
                    "ai.response",
                    label,
                    status="ok",
                    attempt=f"{attempt}/{attempts}",
                    input_tokens=input_tokens,
                    response_chars=len(text),
                )
                self.debug.record_ai(
                    label=label,
                    status="ok",
                    input_tokens=input_tokens,
                    output_tokens=generated_tokens,
                    response_chars=len(text),
                )
                return parsed

            last_failure = {
                "label": label,
                "backend": self.config.backend,
                "model": self.config.model_id,
                "attempt": attempt,
                "attempts": attempts,
                "system_prompt": system,
                "user_prompt": attempt_user,
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "max_new_tokens_requested": target_max_new_tokens,
                "max_new_tokens_used": used_max_new_tokens,
                "input_token_limit_requested": target_input_limit,
                "input_token_limit_used": used_input_limit,
                "response_chars": len(text),
                "raw_response": text,
            }
            self.debug.log(
                "ai.response",
                label,
                status="invalid_json",
                attempt=f"{attempt}/{attempts}",
                input_tokens=input_tokens,
                response_chars=len(text),
            )
            self.debug.record_ai(
                label=label,
                status="invalid_json",
                input_tokens=input_tokens,
                output_tokens=generated_tokens,
                response_chars=len(text),
            )

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

    def _generate_with_oom_backoff(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        input_token_limit: int,
        label: str,
    ) -> Tuple[str, int, int, int, int]:
        current_max_new = max(64, int(max_new_tokens))
        current_input_limit = max(512, int(input_token_limit))

        for oom_attempt in range(1, 5):
            try:
                text, input_tokens, generated_tokens = self._generate(
                    prompt,
                    max_new_tokens=current_max_new,
                    input_token_limit=current_input_limit,
                    label=label,
                )
                return text, input_tokens, generated_tokens, current_max_new, current_input_limit
            except Exception as exc:
                if not self._is_oom_error(exc):
                    raise
                self._cleanup_cuda(reason=f"oom_recovery_{oom_attempt}")
                if oom_attempt == 4:
                    raise RuntimeError(f"{label}: CUDA OOM after adaptive retries") from exc

                current_max_new = max(128, int(current_max_new * 0.7))
                current_input_limit = max(1024, int(current_input_limit * 0.85))
                self.debug.log(
                    "ai.oom",
                    "retrying with reduced budget",
                    label=label,
                    oom_attempt=oom_attempt,
                    max_new_tokens=current_max_new,
                    max_input_tokens=current_input_limit,
                )
        raise RuntimeError(f"{label}: generation failed unexpectedly")

    def _generate(self, prompt: str, *, max_new_tokens: int, input_token_limit: int, label: str) -> Tuple[str, int, int]:
        self._cleanup_cuda(reason="pre_generate")
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_token_limit,
        )
        input_device = self._input_device()
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
        progress = _GenerationProgressCriteria(
            debug=self.debug,
            label=label,
            input_tokens=input_tokens,
            target_new_tokens=max_new_tokens,
        )
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([progress])
        self.debug.log(
            "ai.generate",
            "starting",
            label=label,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            device=str(input_device),
        )

        output = None
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model.generate(**encoded, **generation_kwargs)
            new_tokens = output[0][input_tokens:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated_tokens = int(new_tokens.shape[-1]) if hasattr(new_tokens, "shape") else len(new_tokens)
            elapsed = max(0.001, time.perf_counter() - started)
            self.debug.log(
                "ai.generate",
                "complete",
                label=label,
                generated_tokens=generated_tokens,
                max_new_tokens=max_new_tokens,
                elapsed_sec=round(elapsed, 2),
                tokens_per_sec=round(generated_tokens / elapsed, 2),
            )
            return text, input_tokens, generated_tokens
        finally:
            del encoded
            if output is not None:
                del output
            self._cleanup_cuda(reason="post_generate")

    def _write_invalid_json_artifacts(self, label: str, details: Dict[str, Any]) -> Tuple[str, str]:
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

    def _input_limit_for_generation(self, max_new_tokens: int) -> int:
        context_limit = self._context_limit_tokens()
        requested_input = self.max_input_tokens
        allowed_by_context = max(512, context_limit - max_new_tokens - 8)
        return max(512, min(requested_input, allowed_by_context))

    def _context_limit_tokens(self) -> int:
        self._ensure_loaded()
        configured_total = max(2048, self.max_input_tokens + self.max_new_tokens)
        explicit_limit = int(self.config.context_window_tokens or 0)
        if explicit_limit > 0:
            configured_total = min(configured_total, explicit_limit)

        model_limit = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", 0)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            model_limit = min(model_limit, tokenizer_limit) if model_limit > 0 else tokenizer_limit
        if model_limit <= 0:
            return configured_total
        return max(2048, min(configured_total, model_limit))

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        self.debug.log(
            "ai.load",
            "starting",
            model=self.config.model_id,
            backend=self.config.backend,
            device=self.device,
            dtype=self.config.torch_dtype,
            local_files_only=self.config.local_files_only,
            enable_thinking=self.config.enable_thinking if self._uses_qwen3_chat_template() else "n/a",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "low_cpu_mem_usage": True,
        }
        dtype = self._resolve_dtype(self.config.torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **model_kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.debug.log(
            "ai.load",
            "complete",
            model=self.config.model_id,
            backend=self.config.backend,
            device=str(self._input_device()),
            dtype=str(dtype),
            enable_thinking=self.config.enable_thinking if self._uses_qwen3_chat_template() else "n/a",
        )

    def _input_device(self):
        return next(self.model.parameters()).device

    def _format_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self._uses_qwen3_chat_template():
                kwargs["enable_thinking"] = bool(self.config.enable_thinking)
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                # Older or non-Qwen tokenizers may reject unknown kwargs.
                kwargs.pop("enable_thinking", None)
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    def _uses_qwen3_chat_template(self) -> bool:
        model_id = str(self.config.model_id or "").lower()
        preset = str(self.config.preset or "").lower()
        return "qwen3" in model_id or preset.startswith("qwen3")

    def _cleanup_cuda(self, reason: str) -> None:
        if self.device != "cuda" or not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        self.debug.log(
            "ai.cuda",
            "cleanup",
            reason=reason,
            allocated_mb=round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            reserved_mb=round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        )

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text and "cuda" in text

    @staticmethod
    def _retry_user_prompt(user: str) -> str:
        return (
            f"{user}\n\n"
            "Retry instruction: your previous answer could not be parsed as one valid JSON object. "
            "Return exactly one JSON object only. Do not include markdown fences, explanations, or trailing text."
        )

    def _resolve_dtype(self, value: str):
        value = (value or "auto").lower()
        if value == "auto":
            if self.device in {"cuda", "mps"}:
                return torch.float16
            return None
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if value not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {value}")
        return mapping[value]

    @staticmethod
    def _resolve_device(value: str) -> str:
        value = (value or "auto").lower()
        if value == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if value == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false")
        if value == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Config requested MPS, but torch MPS is unavailable")
        if value not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device: {value}")
        return value


# Backward compatibility alias used throughout the existing codebase and tools.
LocalAIClient = TransformersAIClient
