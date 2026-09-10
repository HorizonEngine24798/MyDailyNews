from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, TextIO
from urllib.parse import urlparse, urlunparse


class DebugAnalytics:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.started_at = time.perf_counter()
        self.durations: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.metrics: dict[str, Any] = {}
        self.ai_totals: dict[str, Any] = {
            "requests": 0,
            "ok": 0,
            "invalid_json": 0,
            "transport_error": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "response_chars": 0,
            "estimated_requests": 0,
            "retries": 0,
            "duration_ms": 0,
            "input_budget_tokens": 0,
            "output_budget_tokens": 0,
            "finish_reasons": {},
        }
        self.ai_buckets: dict[str, dict[str, Any]] = {}

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = self.durations.get(name, 0.0) + (time.perf_counter() - started)

    def increment(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def set_metric(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        self.metrics[name] = value

    def record_ai(
        self,
        *,
        label: str,
        status: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        response_chars: int | None = None,
        estimated: bool = False,
        retry: bool = False,
        finish_reason: str = "",
        duration_ms: float | None = None,
        input_budget_tokens: int | None = None,
        output_budget_tokens: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        bucket = self._bucket_for_label(label)
        totals = self.ai_totals
        stats = self.ai_buckets.setdefault(
            bucket,
            {
                "requests": 0,
                "ok": 0,
                "invalid_json": 0,
                "transport_error": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "response_chars": 0,
                "estimated_requests": 0,
                "retries": 0,
                "duration_ms": 0,
                "input_budget_tokens": 0,
                "output_budget_tokens": 0,
                "finish_reasons": {},
            },
        )
        totals["requests"] += 1
        stats["requests"] += 1
        if estimated:
            totals["estimated_requests"] += 1
            stats["estimated_requests"] += 1
        if retry:
            totals["retries"] += 1
            stats["retries"] += 1
        normalized_finish = str(finish_reason or "unknown").strip().lower()
        totals["finish_reasons"][normalized_finish] = totals["finish_reasons"].get(normalized_finish, 0) + 1
        stats["finish_reasons"][normalized_finish] = stats["finish_reasons"].get(normalized_finish, 0) + 1
        if duration_ms is not None:
            totals["duration_ms"] += max(0, round(float(duration_ms)))
            stats["duration_ms"] += max(0, round(float(duration_ms)))
        if input_budget_tokens is not None:
            totals["input_budget_tokens"] += int(input_budget_tokens)
            stats["input_budget_tokens"] += int(input_budget_tokens)
        if output_budget_tokens is not None:
            totals["output_budget_tokens"] += int(output_budget_tokens)
            stats["output_budget_tokens"] += int(output_budget_tokens)
        if status in {"ok", "invalid_json", "transport_error"}:
            totals[status] += 1
            stats[status] += 1
        if input_tokens is not None:
            totals["input_tokens"] += int(input_tokens)
            stats["input_tokens"] += int(input_tokens)
        if output_tokens is not None:
            totals["output_tokens"] += int(output_tokens)
            stats["output_tokens"] += int(output_tokens)
        if response_chars is not None:
            totals["response_chars"] += int(response_chars)
            stats["response_chars"] += int(response_chars)

    def payload(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "durations_sec": {key: round(value, 4) for key, value in sorted(self.durations.items())},
            "counts": dict(sorted(self.counts.items())),
            "metrics": dict(sorted(self.metrics.items())),
            "ai": {
                "totals": dict(self.ai_totals),
                "by_bucket": {key: dict(value) for key, value in sorted(self.ai_buckets.items())},
            },
        }

    def write_artifact(self, output_dir: str | Path) -> str:
        if not self.enabled:
            return ""
        root = Path(output_dir) / "diagnostics" / "analytics"
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = root / f"{stamp}_debug_analytics.json"
        path.write_text(json.dumps(self.payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _bucket_for_label(label: str) -> str:
        lowered = (label or "").lower()
        if lowered.startswith("headline scoring single replay"):
            return "headline_scoring_single_replay"
        if lowered.startswith("headline scoring"):
            return "headline_scoring"
        if lowered.startswith("final brief generation"):
            return "final_brief_generation"
        return lowered.replace(" ", "_") or "unknown"


class DebugLogger:
    def __init__(self, enabled: bool = False, *, stream: TextIO | None = None) -> None:
        self._enabled = bool(enabled)
        self.analytics = DebugAnalytics(True)
        self._logger = logging.Logger("mydailynews.debug", logging.DEBUG)
        self._logger.propagate = False
        handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
        handler.setFormatter(logging.Formatter("[debug] %(message)s"))
        self._logger.addHandler(handler)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def log(self, event: str, message: str = "", **fields: Any) -> None:
        if not self._enabled:
            return
        suffix = ""
        if fields:
            suffix = " | " + json.dumps(fields, default=str, ensure_ascii=False, sort_keys=True)
        self._logger.debug("%s | %s%s", event, message or "update", suffix)

    def increment(self, name: str, amount: int = 1) -> None:
        self.analytics.increment(name, amount)

    def set_metric(self, name: str, value: Any) -> None:
        self.analytics.set_metric(name, value)

    def record_ai(
        self,
        *,
        label: str,
        status: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        response_chars: int | None = None,
        estimated: bool = False,
        retry: bool = False,
        finish_reason: str = "",
        duration_ms: float | None = None,
        input_budget_tokens: int | None = None,
        output_budget_tokens: int | None = None,
    ) -> None:
        self.analytics.record_ai(
            label=label,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_chars=response_chars,
            estimated=estimated,
            retry=retry,
            finish_reason=finish_reason,
            duration_ms=duration_ms,
            input_budget_tokens=input_budget_tokens,
            output_budget_tokens=output_budget_tokens,
        )

    @contextmanager
    def span(self, name: str):
        with self.analytics.span(name):
            yield

    def analytics_payload(self) -> dict[str, Any]:
        return self.analytics.payload()

    def write_analytics_artifact(self, output_dir: str | Path) -> str:
        return self.analytics.write_artifact(output_dir)


def safe_url(url: str) -> str:
    parsed = urlparse(url or "")
    path = parsed.path
    if len(path) > 80:
        path = path[:77] + "..."
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
