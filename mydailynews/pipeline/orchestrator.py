from __future__ import annotations

"""Horizon-inspired staged orchestrator for MyDailyNews.

This borrows the architecture of Horizon's fetch -> dedupe -> AI score -> enrich
-> summarize pipeline, but keeps only RSS/news discovery and local-first outputs.

Horizon: https://github.com/Thysrael/Horizon
License: MIT
Retained notice: see project LICENSE.
"""

import json
from datetime import date as date_type
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from mydailynews.ai.base import set_ai_artifact_root
from mydailynews.ai.factory import create_ai_client
from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.common.cache import HTTPCache, JSONCache
from mydailynews.pipeline.brief_execution import run_brief as run_brief_helper
from mydailynews.pipeline.enrichment_module import run_enrichment as run_enrichment_helper
from mydailynews.pipeline.narrative_brief import run_narrative_brief as run_narrative_brief_helper
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import (
    AppConfig,
    BriefOutput,
    EnrichmentOutput,
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    PipelineResult,
    PriorReport,
    RunSourceSnapshot,
    NarrativeBriefOutput,
    TopicConfig,
)
from mydailynews.pipeline.stages import PIPELINE_MODULES, PipelineRunOptions, validate_run_date_usage
from mydailynews.diagnostics.reporting import CliReporter
from mydailynews.retrieval.google_news import GoogleNewsQueryRetriever
from mydailynews.retrieval.article_cache import ArticleTextCache
from mydailynews.retrieval.reports import PriorReportRetriever
from mydailynews.scrapers.rss import RSSScraper
from mydailynews.pipeline.shared_headline_scoring import score_snapshot_headlines_once as score_snapshot_headlines_once_helper
from mydailynews.pipeline.snapshot_helpers import (
    build_snapshot as build_snapshot_helper,
)
from mydailynews.pipeline.stage_artifacts import build_stage_artifact, build_stage_payload, to_jsonable
from mydailynews.common.utils import datetime_to_iso, normalize_url, utc_now
from mydailynews.common.warnings import extend_prefixed_warnings, extend_warnings


class NewsOrchestrator:
    def __init__(self, config: AppConfig, debug: bool = False, reporter: CliReporter | None = None) -> None:
        self.config = config
        self.debug = DebugLogger(debug)
        self.reporter = reporter if reporter is not None else CliReporter(enabled=False)
        set_ai_artifact_root(config.output_dir)
        self.summary_ai_client = create_ai_client(config.ai_summary, self.debug)
        self.final_ai_client = create_ai_client(config.ai_final, self.debug)
        self.discovery_cache = HTTPCache(
            root_dir=config.cache.dir,
            namespace="discovery",
            enabled=config.cache.enabled,
            debug=self.debug,
        )
        self.discovery_cache.prune_older_than_days(config.cache.http_retention_days)
        self.enrichment_cache = HTTPCache(
            root_dir=config.cache.dir,
            namespace="enrichment",
            enabled=config.cache.enabled,
            debug=self.debug,
        )
        self.enrichment_cache.prune_older_than_days(config.cache.enrichment_retention_days)
        self.http_cache = self.enrichment_cache
        self.article_text_cache = ArticleTextCache(
            JSONCache(
                root_dir=config.cache.dir,
                namespace="article_text",
                enabled=config.cache.enabled,
            ),
            JSONCache(
                root_dir=config.cache.dir,
                namespace="article_aliases",
                enabled=config.cache.enabled,
            ),
            retention_days=config.cache.article_text_retention_days,
            debug=self.debug,
        )
        self.article_text_cache.prune()
        self.synth_cache = JSONCache(
            root_dir=config.cache.dir,
            namespace="synth",
            enabled=config.cache.enabled and config.cache.ai_enabled,
        )
        self.google_news_retriever = GoogleNewsQueryRetriever(
            config.google_news_source,
            config.user_agent,
            max_workers=config.runtime.max_http_workers,
            http_cache=self.discovery_cache,
            http_cache_mode=config.cache.discovery_mode,
            debug=self.debug,
        )
        self.prior_report_retriever = PriorReportRetriever(
            config.prior_reports_source,
            config.output_dir,
            self.debug,
        )
        self.warnings: List[str] = []
        self.run_options = PipelineRunOptions()
        self.stopped_after_stage: str = ""
        self.stage_artifact_paths: List[str] = []
        self._stage_run_label: str = ""
        self._stage_artifact_root = Path(config.output_dir) / "diagnostics" / "stages"

    def close(self) -> None:
        self._close_ai_client(self.summary_ai_client, role="summary")
        if self.final_ai_client is not self.summary_ai_client:
            self._close_ai_client(self.final_ai_client, role="final")

    def run(self, run_options: PipelineRunOptions | None = None) -> PipelineResult:
        self._prepare_run_options(run_options or PipelineRunOptions())
        self.warnings = []
        date = self._target_date()
        module = str(self.run_options.module or "series").strip().lower()
        if module == "series":
            return self.run_series(date=date)
        if module == "briefs":
            return self.run_briefs(date=date)
        if module == "enrichment":
            return self.run_enrichment(date=date)
        if module == "narrative_brief":
            return self.run_narrative_brief(date=date)
        raise ValueError(f"Unsupported pipeline module: {module}")

    def run_series(self, *, date: str) -> PipelineResult:
        outputs: List[BriefOutput] = []
        enrichment_outputs: List[EnrichmentOutput] = []
        narrative_outputs: List[NarrativeBriefOutput] = []
        for module in self._runtime_series():
            if module == "briefs":
                result = self.run_briefs(date=date)
                outputs.extend(result.outputs)
                enrichment_outputs.extend(result.enrichment_outputs)
                narrative_outputs.extend(result.narrative_outputs)
            elif module == "enrichment":
                if not self._module_enabled("enrichment"):
                    self.warnings.append("enrichment: module is disabled by config; skipped.")
                    continue
                result = self.run_enrichment(date=date, source_outputs=outputs, allow_disk_fallback=False)
                enrichment_outputs.extend(result.enrichment_outputs)
            elif module == "narrative_brief":
                if not self._module_enabled("narrative_brief"):
                    self.warnings.append("narrative_brief: module is disabled by config; skipped.")
                    continue
                enrichment_json_path = enrichment_outputs[-1].json_path if enrichment_outputs else ""
                result = self.run_narrative_brief(
                    date=date,
                    outputs=outputs,
                    enrichment_json_path=enrichment_json_path,
                    allow_disk_fallback=False,
                    use_enrichment=bool(enrichment_json_path),
                )
                narrative_outputs.extend(result.narrative_outputs)
            if self.stopped_after_stage:
                return self._stopped_result(
                    outputs=outputs,
                    enrichment_outputs=enrichment_outputs,
                    narrative_outputs=narrative_outputs,
                )
        return PipelineResult(
            outputs=outputs,
            enrichment_outputs=enrichment_outputs,
            narrative_outputs=narrative_outputs,
            warnings=self.warnings,
        )

    def run_briefs(self, *, date: str = "") -> PipelineResult:
        with self.debug.span("pipeline.total"):
            now = utc_now()
            today = self._date_object(date) if date else now.date()
            date = today.isoformat()
            general_topics = [topic for topic in self.config.general_topics if topic.enabled]
            detailed_topics = [topic for topic in self.config.topics_to_examine if topic.enabled]
            enabled_sources = len([source for source in self.config.rss_sources if source.enabled])
            self.debug.set_metric("pipeline.status", "running")
            self.debug.set_metric("pipeline.rss_sources", enabled_sources)
            self.debug.set_metric("pipeline.general_topics", len(general_topics))
            self.debug.set_metric("pipeline.detailed_topics", len(detailed_topics))
            self.debug.log(
                "pipeline",
                "starting",
                summary_model=self.config.ai_summary.effective_model_label,
                summary_backend=self.config.ai_summary.backend,
                final_model=self.config.ai_final.effective_model_label,
                final_backend=self.config.ai_final.backend,
                sources=enabled_sources,
                general_topics=len(general_topics),
                detailed_topics=len(detailed_topics),
                briefs=",".join(self.run_options.briefs),
                stop_after_stage=self.run_options.stop_after_stage or "none",
                save_intermediate=self.run_options.save_intermediate,
                enrichment=self.config.enrichment.enabled,
                use_shared_snapshot=self.config.runtime.use_shared_snapshot,
            )
            try:
                self.reporter.phase("Preparing prior reports...")
                with self.debug.span("prior_reports.fetch"):
                    prior_reports = self.fetch_prior_reports(today)
                self.debug.set_metric("prior_reports.count", len(prior_reports))
                self._record_stage_artifact(
                    stage="prior_reports",
                    payload=self._stage_payload(
                        stage="prior_reports",
                        brief_name="pipeline",
                        summary={
                            "prior_reports_count": len(prior_reports),
                            "prior_report_ids": [report.id for report in prior_reports],
                        },
                        next_stage_input={
                            "prior_reports": prior_reports,
                            "general_topics": general_topics,
                            "detailed_topics": detailed_topics,
                            "general_filtering": self.config.general_filtering,
                            "filtering": self.config.filtering,
                        },
                    ),
                )
                if self._stop_requested("prior_reports"):
                    return self._stopped_result()

                self.reporter.phase("Preparing source snapshot...")
                snapshot = self._build_snapshot(now, general_topics, detailed_topics)
                if snapshot is None:
                    snapshot_payload: Dict[str, Any] = {
                        "snapshot_enabled": False,
                        "reason": "runtime.use_shared_snapshot=false",
                    }
                else:
                    snapshot_payload = {
                        "snapshot_enabled": True,
                        "snapshot_since": datetime_to_iso(snapshot.fetched_since),
                        "rss_candidates": len(snapshot.rss_candidates),
                        "topic_candidates": len(snapshot.topic_candidates),
                        "merged_candidates": len(snapshot.merged_candidates),
                    }
                self._record_stage_artifact(
                    stage="snapshot",
                    payload=self._stage_payload(
                        stage="snapshot",
                        brief_name="pipeline",
                        summary=snapshot_payload,
                        next_stage_input={
                            "prior_reports": prior_reports,
                            "snapshot": snapshot,
                            "general_topics": general_topics,
                            "detailed_topics": detailed_topics,
                            "general_filtering": self.config.general_filtering,
                            "filtering": self.config.filtering,
                        },
                    ),
                )
                if self._stop_requested("snapshot"):
                    return self._stopped_result()

                shared_candidates_by_brief: Dict[str, List[NewsCandidate]] = {}
                shared_decisions: Dict[str, HeadlineDecision] | None = None
                if snapshot is not None:
                    self.reporter.phase("Scoring shared headline candidates...")
                    shared_candidates_by_brief, shared_decisions, shared_warnings = self._score_snapshot_headlines_once(
                        snapshot,
                        now,
                        general_topics,
                        detailed_topics,
                    )
                    extend_warnings(self.warnings, shared_warnings)
                    shared_payload = {
                        "used_shared_scoring": True,
                        "general_candidates_for_ai": len(shared_candidates_by_brief.get("general", [])),
                        "detailed_candidates_for_ai": len(shared_candidates_by_brief.get("detailed", [])),
                        "shared_decisions": len(shared_decisions or {}),
                    }
                else:
                    shared_payload = {
                        "used_shared_scoring": False,
                        "reason": "snapshot_unavailable",
                    }
                    shared_warnings = []
                general_brief_goal = (
                    "General daily news pass. Prefer breadth and usefulness over deep specialization. "
                    "Use the lower threshold to fill the brief with the strongest general stories, up to the configured article count. "
                    "Still avoid trivia, gossip, minor sports, and duplicate rewrites."
                )
                detailed_brief_goal = (
                    "Detailed topic investigation pass. Focus on the configured topics, identify major narratives, "
                    "compare with prior reports, and select sources that can deepen, challenge, or reshape those narratives."
                )
                self._record_stage_artifact(
                    stage="shared_headline_scoring",
                    payload=self._stage_payload(
                        stage="shared_headline_scoring",
                        brief_name="pipeline",
                        summary=shared_payload,
                        next_stage_input={
                            "prior_reports": prior_reports,
                            "snapshot": snapshot,
                            "shared_candidates_by_brief": shared_candidates_by_brief,
                            "shared_decisions": shared_decisions or {},
                            "shared_warnings": shared_warnings,
                            "general_topics": general_topics,
                            "detailed_topics": detailed_topics,
                            "general_filtering": self.config.general_filtering,
                            "filtering": self.config.filtering,
                            "brief_goals": {
                                "general": general_brief_goal,
                                "detailed": detailed_brief_goal,
                            },
                        },
                    ),
                )
                if self._stop_requested("shared_headline_scoring"):
                    return self._stopped_result()

                outputs: List[BriefOutput] = []
                if "general" in self.run_options.briefs:
                    general_output = self._run_brief(
                        name="general",
                        output_suffix="general",
                        topics=general_topics,
                        filtering=self.config.general_filtering,
                        prior_reports=prior_reports,
                        now=now,
                        date=date,
                        snapshot=snapshot,
                        brief_goal=general_brief_goal,
                        limited_candidates_override=shared_candidates_by_brief.get("general"),
                        shared_decisions=shared_decisions,
                    )
                    if general_output is not None:
                        outputs.append(general_output)
                    if self.stopped_after_stage:
                        return self._stopped_result(outputs=outputs)

                if "detailed" in self.run_options.briefs:
                    detailed_output = self._run_brief(
                        name="detailed",
                        output_suffix="detailed",
                        topics=detailed_topics,
                        filtering=self.config.filtering,
                        prior_reports=prior_reports,
                        now=now,
                        date=date,
                        snapshot=snapshot,
                        brief_goal=detailed_brief_goal,
                        limited_candidates_override=shared_candidates_by_brief.get("detailed"),
                        shared_decisions=shared_decisions,
                    )
                    if detailed_output is not None:
                        outputs.append(detailed_output)
                    if self.stopped_after_stage:
                        return self._stopped_result(outputs=outputs)

                self.debug.set_metric("pipeline.outputs", len(outputs))
                self.debug.set_metric("pipeline.narrative_outputs", 0)
                self.debug.set_metric("pipeline.status", "completed")
                self.debug.log(
                    "pipeline",
                    "complete",
                    outputs=len(outputs),
                    narrative_outputs=0,
                    warnings=len(self.warnings),
                )
                return PipelineResult(outputs=outputs, warnings=self.warnings)
            except Exception as exc:
                self.debug.set_metric("pipeline.status", "failed")
                self.debug.set_metric("pipeline.error", f"{type(exc).__name__}: {exc}")
                raise

    def _prepare_run_options(self, options: PipelineRunOptions) -> None:
        validate_run_date_usage(options.module, options.date)
        self.run_options = options
        self.stopped_after_stage = ""
        self.stage_artifact_paths = []
        self._stage_run_label = utc_now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = str(options.stage_artifact_dir or "").strip()
        if artifact_dir:
            self._stage_artifact_root = Path(artifact_dir)
        else:
            self._stage_artifact_root = Path(self.config.output_dir) / "diagnostics" / "stages"

    def _target_date(self) -> str:
        requested = str(getattr(self.run_options, "date", "") or "").strip()
        if not requested:
            return utc_now().date().isoformat()
        self._date_object(requested)
        return requested

    @staticmethod
    def _date_object(value: str) -> date_type:
        try:
            return date_type.fromisoformat(str(value or "").strip())
        except ValueError as exc:
            raise ValueError(f"Invalid --date value '{value}'. Expected YYYY-MM-DD.") from exc

    def _runtime_series(self) -> List[str]:
        configured = list(getattr(self.config.pipeline, "default_series", []) or ["briefs", "enrichment", "narrative_brief"])
        skip = set(getattr(self.run_options, "skip_modules", ()) or ())
        series: List[str] = []
        for module in configured:
            normalized = str(module or "").strip().lower().replace("-", "_")
            if normalized not in PIPELINE_MODULES:
                raise ValueError(f"Unsupported configured pipeline module: {module}")
            if normalized in skip:
                self.warnings.append(f"{normalized}: module skipped by run option.")
                continue
            if normalized not in series:
                series.append(normalized)
        return series

    def _module_enabled(self, module: str) -> bool:
        if module == "enrichment":
            mode = str(getattr(self.config.enrichment, "mode", "story_llm") or "story_llm").strip().lower()
            return bool(getattr(self.config.enrichment, "enabled", False)) and mode != "disabled"
        if module == "narrative_brief":
            return bool(getattr(self.config.narrative_briefing, "enabled", False))
        return True

    def _stopped_result(
        self,
        outputs: List[BriefOutput] | None = None,
        enrichment_outputs: List[EnrichmentOutput] | None = None,
        narrative_outputs: List[NarrativeBriefOutput] | None = None,
    ) -> PipelineResult:
        self.debug.set_metric("pipeline.outputs", len(outputs or []))
        self.debug.set_metric("pipeline.enrichment_outputs", len(enrichment_outputs or []))
        self.debug.set_metric("pipeline.narrative_outputs", len(narrative_outputs or []))
        self.debug.set_metric("pipeline.status", "stopped")
        self.debug.log(
            "pipeline",
            "stopped",
            stage=self.stopped_after_stage or self.run_options.stop_after_stage,
            outputs=len(outputs or []),
            enrichment_outputs=len(enrichment_outputs or []),
            narrative_outputs=len(narrative_outputs or []),
        )
        return PipelineResult(
            outputs=list(outputs or []),
            enrichment_outputs=list(enrichment_outputs or []),
            narrative_outputs=list(narrative_outputs or []),
            warnings=self.warnings,
        )

    def _stage_payload(
        self,
        *,
        stage: str,
        brief_name: str = "pipeline",
        summary: Dict[str, Any],
        next_stage_input: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return build_stage_payload(
            stage=stage,
            brief=brief_name,
            summary=summary,
            next_stage_input=next_stage_input,
        )

    def _stop_requested(self, stage: str) -> bool:
        requested = str(self.run_options.stop_after_stage or "").strip().lower()
        if not requested or requested != stage:
            return False
        self.mark_stopped_after_stage(stage)
        return True

    def mark_stopped_after_stage(self, stage: str) -> None:
        if self.stopped_after_stage:
            return
        self.stopped_after_stage = stage
        self.warnings.append(f"Run stopped after stage '{stage}' by request.")

    def _record_stage_artifact(self, *, stage: str, payload: Dict[str, Any], brief_name: str = "pipeline") -> str:
        should_dump = bool(
            self.run_options.dump_stage_artifacts
            or self.run_options.stop_after_stage
            or self.run_options.save_intermediate
        )
        if not should_dump:
            return ""
        try:
            target_dir = self._stage_artifact_root / self._stage_run_label / brief_name
            target_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = target_dir / f"{stage}.json"
            artifact_payload = build_stage_artifact(
                run_label=self._stage_run_label,
                brief=brief_name,
                stage=stage,
                generated_at=utc_now().isoformat(),
                summary=payload.get("summary", {}),
                next_stage_input=payload.get("next_stage_input", {}),
            )
            artifact_payload = to_jsonable(artifact_payload)
            artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            path_text = str(artifact_path)
            self.stage_artifact_paths.append(path_text)
            return path_text
        except Exception as exc:
            self.warnings.append(f"Stage artifact write failed ({brief_name}/{stage}): {type(exc).__name__}: {exc}")
            return ""

    def run_enrichment(
        self,
        *,
        date: str,
        source_outputs: List[BriefOutput] | None = None,
        allow_disk_fallback: bool = True,
    ) -> PipelineResult:
        output = run_enrichment_helper(
            self,
            date=date,
            source_outputs=list(source_outputs or []),
            allow_disk_fallback=allow_disk_fallback,
        )
        outputs = [output] if output is not None else []
        if self._stop_requested("enrichment"):
            return self._stopped_result(enrichment_outputs=outputs)
        return PipelineResult(enrichment_outputs=outputs, warnings=self.warnings)

    def run_narrative_brief(
        self,
        *,
        date: str,
        outputs: List[BriefOutput] | None = None,
        enrichment_json_path: str = "",
        allow_disk_fallback: bool = True,
        use_enrichment: bool | None = None,
    ) -> PipelineResult:
        if not self._module_enabled("narrative_brief"):
            self.warnings.append("narrative_brief: module is disabled by config; skipped.")
            return PipelineResult(outputs=list(outputs or []), warnings=self.warnings)
        if use_enrichment is None:
            use_enrichment = self._module_enabled("enrichment") and "enrichment" not in set(
                getattr(self.run_options, "skip_modules", ()) or ()
            )
        output = self._run_narrative_brief(
            outputs=list(outputs or []),
            date=date,
            enrichment_json_path=enrichment_json_path,
            allow_disk_fallback=allow_disk_fallback,
            use_enrichment=bool(use_enrichment),
        )
        narrative_outputs = [output] if output is not None else []
        if self._stop_requested("narrative_brief"):
            return self._stopped_result(
                outputs=list(outputs or []),
                narrative_outputs=narrative_outputs,
            )
        return PipelineResult(outputs=list(outputs or []), narrative_outputs=narrative_outputs, warnings=self.warnings)

    def _run_brief(
        self,
        name: str,
        output_suffix: str,
        topics: List[TopicConfig],
        filtering: FilteringConfig,
        prior_reports: List[PriorReport],
        now,
        date: str,
        snapshot: RunSourceSnapshot | None,
        brief_goal: str,
        limited_candidates_override: List[NewsCandidate] | None = None,
        shared_decisions: Dict[str, HeadlineDecision] | None = None,
    ) -> BriefOutput | None:
        return run_brief_helper(
            self,
            name=name,
            output_suffix=output_suffix,
            topics=topics,
            filtering=filtering,
            prior_reports=prior_reports,
            now=now,
            date=date,
            snapshot=snapshot,
            brief_goal=brief_goal,
            limited_candidates_override=limited_candidates_override,
            shared_decisions=shared_decisions,
        )

    def _run_narrative_brief(
        self,
        *,
        outputs: List[BriefOutput],
        date: str,
        enrichment_json_path: str = "",
        allow_disk_fallback: bool = True,
        use_enrichment: bool = True,
    ) -> NarrativeBriefOutput | None:
        return run_narrative_brief_helper(
            self,
            outputs=outputs,
            date=date,
            enrichment_json_path=enrichment_json_path,
            allow_disk_fallback=allow_disk_fallback,
            use_enrichment=use_enrichment,
        )

    def _close_ai_client(self, client, *, role: str) -> None:
        close_fn = getattr(client, "close", None)
        if not callable(close_fn):
            return
        try:
            close_fn()
        except Exception as exc:
            warning = f"AI client close failed ({role}): {type(exc).__name__}: {exc}"
            self.warnings.append(warning)
            self.debug.log("ai.server", "close_failed", role=role, error=type(exc).__name__)

    def _score_snapshot_headlines_once(
        self,
        snapshot: RunSourceSnapshot,
        now,
        general_topics: List[TopicConfig],
        detailed_topics: List[TopicConfig],
    ) -> tuple[Dict[str, List[NewsCandidate]], Dict[str, HeadlineDecision], List[str]]:
        return score_snapshot_headlines_once_helper(
            snapshot=snapshot,
            now=now,
            general_topics=general_topics,
            detailed_topics=detailed_topics,
            config=self.config,
            debug=self.debug,
            summary_ai_client=self.summary_ai_client,
            synth_cache=self.synth_cache,
            analyzer_cls=HeadlineAnalyzer,
        )

    def _build_snapshot(self, now, general_topics: List[TopicConfig], detailed_topics: List[TopicConfig]) -> RunSourceSnapshot | None:
        return build_snapshot_helper(
            use_shared_snapshot=self.config.runtime.use_shared_snapshot,
            now=now,
            general_topics=general_topics,
            detailed_topics=detailed_topics,
            general_filtering=self.config.general_filtering,
            detailed_filtering=self.config.filtering,
            debug=self.debug,
            fetch_headlines=self.fetch_headlines,
            fetch_topic_headlines=self.fetch_topic_headlines,
            merge_url_duplicates=self.merge_url_duplicates,
        )

    def fetch_headlines(self, since, max_headlines_per_source: int, warnings: List[str]) -> List[NewsCandidate]:
        scraper = RSSScraper(
            self.config.rss_sources,
            self.config.user_agent,
            max_headlines_per_source,
            max_workers=self.config.runtime.max_http_workers,
            http_cache=self.discovery_cache,
            http_cache_mode=self.config.cache.discovery_mode,
            debug=self.debug,
        )
        candidates = scraper.fetch(since)
        extend_prefixed_warnings(warnings, "RSS", scraper.errors)
        return candidates

    def fetch_topic_headlines(self, topics, since, warnings: List[str]) -> List[NewsCandidate]:
        candidates = self.google_news_retriever.fetch(topics, since)
        extend_prefixed_warnings(warnings, "Google News", self.google_news_retriever.errors)
        return candidates

    def fetch_prior_reports(self, today):
        reports = self.prior_report_retriever.fetch(today)
        extend_prefixed_warnings(self.warnings, "Prior reports", self.prior_report_retriever.errors)
        return reports

    @staticmethod
    def merge_url_duplicates(candidates: List[NewsCandidate]) -> List[NewsCandidate]:
        # Adapted from Horizon's non-LLM cross-source URL merge strategy (MIT):
        # https://github.com/Thysrael/Horizon
        # Group by canonical URL key, keep the richest item, and merge metadata.
        by_url: Dict[str, List[NewsCandidate]] = {}
        for candidate in candidates:
            key = NewsOrchestrator._dedupe_url_key(candidate.url)
            by_url.setdefault(key, []).append(candidate)

        merged: List[NewsCandidate] = []
        for group in by_url.values():
            if len(group) == 1:
                only_item = group[0]
                if "merged_latest_published_at" not in only_item.metadata:
                    only_item.metadata["merged_latest_published_at"] = datetime_to_iso(only_item.published_at)
                merged.append(group[0])
                continue

            primary = max(group, key=lambda item: len(item.snippet or ""))
            merged_sources = sorted({item.source for item in group if item.source})
            latest_published_at = max(
                (item.published_at for item in group if item.published_at is not None),
                default=None,
            )
            merged_tags: List[str] = []
            seen_tags: set[str] = set()

            for item in group:
                for tag in item.tags:
                    if tag and tag not in seen_tags:
                        seen_tags.add(tag)
                        merged_tags.append(tag)

                for meta_key, meta_value in item.metadata.items():
                    if meta_key not in primary.metadata or not primary.metadata[meta_key]:
                        primary.metadata[meta_key] = meta_value

                if item is not primary and len(item.snippet or "") > len(primary.snippet or ""):
                    primary.snippet = item.snippet

            primary.tags = merged_tags
            primary.metadata["merged_sources"] = merged_sources
            primary.metadata["merged_count"] = len(group)
            primary.metadata["merged_latest_published_at"] = datetime_to_iso(latest_published_at)
            merged.append(primary)
        return merged

    @staticmethod
    def _dedupe_url_key(url: str) -> str:
        normalized = normalize_url(url)
        parsed = urlparse(normalized)
        host = parsed.hostname or ""
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/")
        return f"{host}{path}"
