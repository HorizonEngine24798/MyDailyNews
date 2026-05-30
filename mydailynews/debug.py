from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse, urlunparse


class _DebugAnalytics:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.started_at = time.perf_counter()
        self.durations: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.metrics: dict[str, Any] = {}
        self.ai_totals: dict[str, int] = {
            "requests": 0,
            "ok": 0,
            "invalid_json": 0,
            "transport_error": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "response_chars": 0,
            "estimated_requests": 0,
        }
        self.ai_buckets: dict[str, dict[str, int]] = {}

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
            },
        )
        totals["requests"] += 1
        stats["requests"] += 1
        if estimated:
            totals["estimated_requests"] += 1
            stats["estimated_requests"] += 1
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

    def summary_lines(self) -> list[str]:
        if not self.enabled:
            return []

        durations = self.durations
        counts = self.counts
        metrics = self.metrics
        lines: list[str] = []

        def metric(name: str, default: Any = 0) -> Any:
            if name in metrics:
                return metrics[name]
            if name in counts:
                return counts[name]
            return default

        pipeline_total = durations.get("pipeline.total", 0.0)
        if pipeline_total > 0:
            lines.append(
                f"pipeline total={pipeline_total:.2f}s status={metric('pipeline.status', 'unknown')} "
                f"prior_reports={durations.get('prior_reports.fetch', 0.0):.2f}s "
                f"snapshot={durations.get('snapshot.total', 0.0):.2f}s "
                f"shared_scoring={durations.get('headline.shared.total', 0.0):.2f}s "
                f"outputs={metric('pipeline.outputs')} "
                f"warnings={metric('brief.general.warnings') + metric('brief.detailed.warnings')}"
            )

        snapshot_raw = metric("snapshot.raw_candidates")
        snapshot_unique = metric("snapshot.unique_candidates")
        if snapshot_raw or snapshot_unique:
            lines.append(
                f"snapshot raw={snapshot_raw} rss={metric('snapshot.rss_candidates')} "
                f"topic={metric('snapshot.topic_candidates')} unique={snapshot_unique} "
                f"dropped_duplicates={max(0, int(snapshot_raw) - int(snapshot_unique))}"
            )
            lines.append(
                f"snapshot timings rss_fetch={durations.get('snapshot.rss_fetch', 0.0):.2f}s "
                f"topic_fetch={durations.get('snapshot.topic_fetch', 0.0):.2f}s "
                f"merge={durations.get('snapshot.merge', 0.0):.2f}s"
            )

        shared_union = metric("headline.shared.union_candidates")
        if shared_union:
            shared_decisions = metric("headline.shared.decisions")
            lines.append(
                f"shared scoring union={shared_union} decisions={shared_decisions} "
                f"dropped_after_scoring={max(0, int(shared_union) - int(shared_decisions))}"
            )

        for brief_name in ("general", "detailed"):
            unique = metric(f"brief.{brief_name}.unique_candidates")
            limited = metric(f"brief.{brief_name}.limited_candidates")
            decisions = metric(f"brief.{brief_name}.decisions")
            selected = metric(f"brief.{brief_name}.selected")
            if not any([unique, limited, decisions, selected]):
                continue
            lines.append(
                f"{brief_name} funnel unique={unique} limited={limited} decisions={decisions} selected={selected} "
                f"dropped_prefilter={max(0, int(unique) - int(limited))} "
                f"dropped_scoring={max(0, int(limited) - int(decisions))} "
                f"dropped_selection={max(0, int(decisions) - int(selected))} "
                f"duration={durations.get(f'brief.{brief_name}.total', 0.0):.2f}s"
            )
            lines.append(
                f"{brief_name} timings prepare={durations.get(f'brief.{brief_name}.candidate_prepare', 0.0):.2f}s "
                f"limit={durations.get(f'brief.{brief_name}.headline_limit', 0.0):.2f}s "
                f"score={durations.get(f'brief.{brief_name}.headline_decisions', 0.0):.2f}s "
                f"select={durations.get(f'brief.{brief_name}.headline_select', 0.0):.2f}s "
                f"fetch={durations.get(f'brief.{brief_name}.article_fetch', 0.0):.2f}s "
                f"enrich={durations.get(f'brief.{brief_name}.enrichment', 0.0):.2f}s "
                f"final={durations.get(f'brief.{brief_name}.final_brief', 0.0):.2f}s "
                f"write={durations.get(f'brief.{brief_name}.write_output', 0.0):.2f}s"
            )
            article_attempted = metric(f"brief.{brief_name}.article_fetch.attempted")
            if article_attempted:
                lines.append(
                    f"{brief_name} article_fetch attempted={article_attempted} ok={metric(f'brief.{brief_name}.article_fetch.ok')} "
                    f"short_text={metric(f'brief.{brief_name}.article_fetch.short_text')} "
                    f"failed={metric(f'brief.{brief_name}.article_fetch.failed')} "
                    f"duration={durations.get(f'brief.{brief_name}.article_fetch', 0.0):.2f}s"
                )
            enrichment_articles = metric(f"brief.{brief_name}.enrichment.total_articles")
            if enrichment_articles:
                lines.append(
                    f"{brief_name} enrichment total={enrichment_articles} needed={metric(f'brief.{brief_name}.enrichment.needed')} "
                    f"skipped={metric(f'brief.{brief_name}.enrichment.skipped')} "
                    f"context_sources={metric(f'brief.{brief_name}.enrichment.context_sources')} "
                    f"duration={durations.get(f'brief.{brief_name}.enrichment', 0.0):.2f}s"
                )

        ai_totals = self.ai_totals
        if ai_totals["requests"] > 0:
            lines.append(
                f"ai requests={ai_totals['requests']} ok={ai_totals['ok']} invalid_json={ai_totals['invalid_json']} "
                f"transport_error={ai_totals['transport_error']} input_tokens={ai_totals['input_tokens']} "
                f"output_tokens={ai_totals['output_tokens']} estimated_requests={ai_totals['estimated_requests']}"
            )
            for bucket, stats in sorted(self.ai_buckets.items()):
                lines.append(
                    f"ai[{bucket}] requests={stats['requests']} ok={stats['ok']} invalid_json={stats['invalid_json']} "
                    f"input_tokens={stats['input_tokens']} output_tokens={stats['output_tokens']}"
                )
        return lines

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
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._suppressed: dict[str, int] = {}
        self._last_emit: dict[str, float] = {}
        self.analytics = _DebugAnalytics(enabled)

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
    ) -> None:
        self.analytics.record_ai(
            label=label,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_chars=response_chars,
            estimated=estimated,
        )

    @contextmanager
    def span(self, name: str):
        with self.analytics.span(name):
            yield

    def analytics_payload(self) -> dict[str, Any]:
        return self.analytics.payload()

    def analytics_summary_lines(self) -> list[str]:
        return self.analytics.summary_lines()

    def write_analytics_artifact(self, output_dir: str | Path) -> str:
        return self.analytics.write_artifact(output_dir)

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
