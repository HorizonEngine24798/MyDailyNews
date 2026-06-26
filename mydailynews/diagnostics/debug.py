from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
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
            limited_sources = metric(f"brief.{brief_name}.limited_sources")
            selected_sources = metric(f"brief.{brief_name}.selected_sources")
            if any([limited_sources, selected_sources]):
                lines.append(
                    f"{brief_name} source diversity limited_sources={limited_sources} "
                    f"selected_sources={selected_sources}"
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
                    f"story_threads_created={metric(f'brief.{brief_name}.enrichment.story_threads_created')} "
                    f"story_threads_enriched={metric(f'brief.{brief_name}.enrichment.story_threads_enriched')} "
                    f"story_threads_skipped={metric(f'brief.{brief_name}.enrichment.story_threads_skipped')} "
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


_DebugAnalytics = DebugAnalytics


@dataclass(frozen=True)
class DebugEventClassification:
    category: str
    level: str


class DebugEventEmitter:
    PROGRESS_EVENTS: dict[str, set[str]] = {
        "pipeline": {"starting", "complete", "stopped"},
        "brief.run": {"starting", "complete", "stopped"},
        "snapshot": {"built"},
        "headline.fetch": {"complete", "reused_snapshot"},
        "headline.dedupe": {"complete"},
        "headline.heuristics": {"prefilter_complete", "title_dedupe"},
        "headline.limit": {"complete", "reused_shared_prefilter"},
        "headline.decisions": {"complete", "reused_shared"},
        "headline.select": {"complete"},
        "headline.ai": {"starting batched scoring"},
        "brief.ai": {"synthesizing", "complete"},
        "article.fetch": {"batch_start", "batch_complete"},
        "enrichment": {"complete"},
        "analysis.evidence": {"starting_batched_distillation"},
        "analysis.delta": {"starting_batched_extraction", "deterministic_scaffold"},
        "prior_reports": {"complete", "skipped_disabled"},
        "ai.load": {"starting", "complete"},
        "ai.unload": {"starting", "complete", "noop", "noop_managed_server"},
        "ai.oom": {"retrying with reduced budget"},
        "ai.server": {
            "lease_acquired",
            "attached_existing",
            "spawned",
            "ready",
            "stopping",
            "stopped",
            "lease_released",
        },
        "ai.request": {"*"},
        "ai.response": {"*"},
        "ai.generate": {"starting", "progress", "complete"},
    }
    WARNING_EVENTS: dict[str, set[str]] = {
        "headline.ai.batch": {
            "incomplete",
            "recovered_invalid_json",
            "skipped_invalid_json",
            "recovered_missing_decisions",
        },
        "headline.select": {"final_budget_prune"},
        "analysis.evidence": {"dropped_tail_to_avoid_split"},
        "analysis.delta": {"dropped_tail_to_avoid_split"},
        "article.fetch": {
            "short_text",
            "google_news_resolve_failed",
        },
        "prior_reports": {"missing_output_dir"},
        "ai.server": {"process_exited", "close_failed"},
    }
    ERROR_EVENTS: dict[str, set[str]] = {
        "article.fetch": {
            "http_error",
            "extract_failed",
            "google_news_resolve_http_error",
        },
        "rss.source": {"failed", "worker_exception"},
        "google_news.topic": {"worker_exception"},
        "google_news.query": {"failed"},
        "enrichment": {"io_exception"},
        "analysis.evidence": {"failed"},
        "analysis.delta": {"failed"},
        "ai.response": {"invalid_json_save_failed"},
    }
    DETAIL_EVENTS: dict[str, set[str]] = {
        "article": {"selected"},
        "article.fetch": {
            "starting",
            "complete",
            "google_news_decoded",
            "google_news_resolved",
        },
        "headline.ai.batch": {"scoring", "cache_hit", "complete"},
        "analysis.evidence": {"skipped_disabled"},
        "analysis.evidence.batch": {"running", "cache_hit"},
        "analysis.evidence.prompt": {"budget_check"},
        "analysis.delta": {"skipped_disabled"},
        "analysis.delta.batch": {"running", "cache_hit"},
        "analysis.delta.prompt": {"budget_check"},
        "brief.ai": {"compacted_analysis_context"},
        "enrichment": {"skipped_disabled", "skipped_enough_context"},
        "google_news": {"skipped_disabled"},
        "google_news.topic": {"complete"},
        "google_news.query": {"fetching", "complete"},
        "rss.source": {"fetching", "complete"},
    }
    EVENT_CATEGORIES: dict[str, str] = {
        "pipeline": "pipeline",
        "brief.run": "pipeline",
        "snapshot": "pipeline",
        "prior_reports": "source_fetch",
        "rss.source": "source_fetch",
        "google_news": "source_fetch",
        "google_news.topic": "source_fetch",
        "google_news.query": "source_fetch",
        "headline.fetch": "source_fetch",
        "headline.dedupe": "headline_scoring",
        "headline.heuristics": "headline_scoring",
        "headline.limit": "headline_scoring",
        "headline.decisions": "headline_scoring",
        "headline.select": "selection",
        "headline.ai": "headline_scoring",
        "headline.ai.batch": "headline_scoring",
        "article": "selection",
        "article.fetch": "article_fetch",
        "enrichment": "enrichment",
        "analysis.evidence": "analysis",
        "analysis.evidence.batch": "analysis",
        "analysis.evidence.prompt": "analysis",
        "analysis.delta": "analysis",
        "analysis.delta.batch": "analysis",
        "analysis.delta.prompt": "analysis",
        "brief.ai": "ai_request",
        "ai.request": "ai_request",
        "ai.response": "ai_request",
        "ai.generate": "ai_request",
        "ai.load": "ai_server",
        "ai.unload": "ai_server",
        "ai.oom": "ai_server",
        "ai.server": "ai_server",
    }

    def __init__(
        self,
        enabled: bool = False,
        *,
        emit_detail: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.enabled = enabled
        self.emit_detail = emit_detail
        self.stream = stream if stream is not None else sys.stdout
        self._suppressed: dict[str, int] = {}
        self._last_emit: dict[str, float] = {}

    def emit(self, event: str, message: str = "", **fields: Any) -> None:
        if not self.enabled:
            return
        classification = self.classify(event, message, fields)
        if not self.should_emit(event, message, fields, classification=classification):
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

        rendered = [
            f"[debug] category={classification.category}",
            f"level={classification.level}",
            f"stage={event}",
        ]
        rendered.append(f"action={message or 'update'}")
        if output_fields:
            rendered.append(" ".join(f"{key}={self._format(value)}" for key, value in output_fields.items()))
        print(" | ".join(rendered), file=self.stream, flush=True)

    def classify(self, event: str, message: str = "", fields: dict[str, Any] | None = None) -> DebugEventClassification:
        fields = fields or {}
        category = self.category_for_event(event)
        explicit_level = self._explicit_level(event, message)
        if explicit_level in {"warning", "error"}:
            return DebugEventClassification(category=category, level=explicit_level)
        if self._is_problem_signal(event, message, fields):
            return DebugEventClassification(category=category, level=self._problem_level(event, message, fields))
        if explicit_level:
            return DebugEventClassification(category=category, level=explicit_level)
        return DebugEventClassification(category=category, level="detail")

    def should_emit(
        self,
        event: str,
        message: str,
        fields: dict[str, Any],
        *,
        classification: DebugEventClassification | None = None,
    ) -> bool:
        classification = classification or self.classify(event, message, fields)
        if classification.level in {"warning", "error", "progress"}:
            return True
        return bool(self.emit_detail)

    @classmethod
    def category_for_event(cls, event: str) -> str:
        if event in cls.EVENT_CATEGORIES:
            return cls.EVENT_CATEGORIES[event]
        if event.startswith("analysis."):
            return "analysis"
        if event.startswith("headline."):
            return "headline_scoring"
        if event.startswith("article."):
            return "article_fetch"
        if event.startswith("ai."):
            return "ai_request"
        if event.startswith("google_news") or event.startswith("rss."):
            return "source_fetch"
        return "artifact"

    @classmethod
    def _explicit_level(cls, event: str, message: str) -> str:
        for level, rules in (
            ("error", cls.ERROR_EVENTS),
            ("warning", cls.WARNING_EVENTS),
            ("progress", cls.PROGRESS_EVENTS),
            ("detail", cls.DETAIL_EVENTS),
        ):
            allowed = rules.get(event)
            if not allowed:
                continue
            if "*" in allowed or message in allowed or (not message and "" in allowed):
                return level
        return ""

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
                "http_",
                "http error",
            )
        )

    @classmethod
    def _problem_level(cls, event: str, message: str, fields: dict[str, Any]) -> str:
        text = " ".join([event, message, str(fields.get("status", "")), str(fields.get("error", ""))]).lower()
        if any(marker in text for marker in ("missing", "short_text")):
            return "warning"
        return "error"

    @staticmethod
    def _min_interval_seconds(event: str, message: str) -> float:
        throttles: dict[tuple[str, str], float] = {
            ("enrichment", "skipped_enough_context"): 2.0,
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


class DebugLogger:
    def __init__(self, enabled: bool = False, *, stream: TextIO | None = None, emit_detail: bool = False) -> None:
        self._enabled = bool(enabled)
        self.analytics = DebugAnalytics(self._enabled)
        self.events = DebugEventEmitter(self._enabled, stream=stream, emit_detail=emit_detail)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        if "analytics" in self.__dict__:
            self.analytics.enabled = self._enabled
        if "events" in self.__dict__:
            self.events.enabled = self._enabled

    def log(self, event: str, message: str = "", **fields: Any) -> None:
        self.events.emit(event, message, **fields)

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
        return DebugEventEmitter._is_problem_signal(event, message, fields)

    def _should_emit(self, event: str, message: str, fields: dict[str, Any]) -> bool:
        classification = self.events.classify(event, message, fields)
        return self.events.should_emit(event, message, fields, classification=classification)

    @staticmethod
    def _min_interval_seconds(event: str, message: str) -> float:
        return DebugEventEmitter._min_interval_seconds(event, message)

    @staticmethod
    def _format(value: Any) -> str:
        return DebugEventEmitter._format(value)


def safe_url(url: str) -> str:
    parsed = urlparse(url or "")
    path = parsed.path
    if len(path) > 80:
        path = path[:77] + "..."
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
