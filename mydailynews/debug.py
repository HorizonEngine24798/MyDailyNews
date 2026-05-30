from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse, urlunparse


class DebugLogger:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._suppressed: dict[str, int] = {}
        self._last_emit: dict[str, float] = {}

    def log(self, event: str, message: str = "", **fields: Any) -> None:
        if not self.enabled:
            return
        if not self._should_emit(event, message, fields):
            return
        signature = f"{event}|{message}".strip()
        now = time.perf_counter()
        interval = self._min_interval_seconds(event, message)
        if interval > 0:
            last_emit = self._last_emit.get(signature, 0.0)
            if now - last_emit < interval:
                self._suppressed[signature] = self._suppressed.get(signature, 0) + 1
                return
            self._last_emit[signature] = now

        output_fields = dict(fields)
        suppressed = self._suppressed.pop(signature, 0)
        if suppressed > 0:
            output_fields["suppressed"] = suppressed

        rendered = [f"[debug] stage={event}"]
        rendered.append(f"action={message or 'update'}")
        if output_fields:
            rendered.append(" ".join(f"{key}={self._format(value)}" for key, value in output_fields.items()))
        print(" | ".join(rendered), flush=True)

    @staticmethod
    def _is_problem_signal(event: str, message: str, fields: dict[str, Any]) -> bool:
        text = " ".join([event, message, str(fields.get("status", "")), str(fields.get("error", ""))]).lower()
        return any(
            marker in text
            for marker in (
                "error",
                "failed",
                "exception",
                "invalid_json",
                "oom",
                "missing",
                "skipped",
                "http_",
                "http error",
            )
        )

    def _should_emit(self, event: str, message: str, fields: dict[str, Any]) -> bool:
        if self._is_problem_signal(event, message, fields):
            return True

        keep_messages: dict[str, set[str]] = {
            "pipeline": {"starting", "complete"},
            "brief.run": {"starting", "complete"},
            "snapshot": {"built"},
            "headline.fetch": {"complete", "reused_snapshot"},
            "headline.dedupe": {"complete"},
            "headline.heuristics": {"prefilter_complete", "title_dedupe"},
            "headline.limit": {"complete"},
            "headline.decisions": {"complete"},
            "headline.select": {"complete"},
            "headline.ai": {"starting batched scoring"},
            "headline.ai.batch": {"scoring", "cache_hit", "complete"},
            "brief.ai": {"synthesizing", "complete"},
            "brief.article.ai": {"chunking", "chunk_start", "starting_chunk", "chunk_complete"},
            "article": {"selected"},
            "article.fetch": {"batch_start", "batch_complete", "worker_exception"},
            "enrichment": {"starting", "not_needed", "complete", "skipped_heuristic_enough_context"},
            "prior_reports": {"complete", "skipped_disabled", "missing_output_dir"},
            "google_news.topic": {"complete"},
            "ai.load": {"starting", "complete"},
            "ai.unload": {"starting", "complete"},
            "ai.oom": {"retrying with reduced budget"},
            "ai.request": {"*"},
            "ai.response": {"*"},
            "ai.generate": {"starting", "progress", "complete"},
        }
        allowed = keep_messages.get(event)
        if allowed is None:
            return False
        if not message:
            return True
        if "*" in allowed:
            return True
        return message in allowed

    @staticmethod
    def _min_interval_seconds(event: str, message: str) -> float:
        throttles: dict[tuple[str, str], float] = {
            ("enrichment", "not_needed"): 2.0,
            ("enrichment", "starting"): 1.0,
            ("google_news.topic", "complete"): 0.5,
            ("headline.ai.batch", "complete"): 0.5,
            ("ai.generate", "progress"): 10.0,
        }
        return throttles.get((event, message), 0.0)

    @staticmethod
    def _format(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        text = str(value)
        if len(text) > 120:
            text = text[:117] + "..."
        if any(char.isspace() for char in text):
            return repr(text)
        return text


def safe_url(url: str) -> str:
    parsed = urlparse(url or "")
    path = parsed.path
    if len(path) > 80:
        path = path[:77] + "..."
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
