from __future__ import annotations

"""Horizon-inspired staged orchestrator for MyDailyNews.

This borrows the architecture of Horizon's fetch -> dedupe -> AI score -> enrich
-> summarize pipeline, but keeps only RSS/news discovery and local-first outputs.

Horizon: https://github.com/Thysrael/Horizon
License: MIT
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from .ai.base import set_ai_artifact_root
from .ai.factory import create_ai_client
from .ai.headline_analyzer import HeadlineAnalyzer
from .article_pipeline import (
    populate_article_texts as populate_article_texts_batch,
    record_article_fetch_metrics as record_article_fetch_metrics_helper,
    record_enrichment_metrics as record_enrichment_metrics_helper,
)
from .cache import HTTPCache, JSONCache
from .brief_execution import run_brief as run_brief_helper
from .debug import DebugLogger
from .headline_selection import (
    candidate_heuristic_score as candidate_heuristic_score_helper,
    candidate_topic_match as candidate_topic_match_helper,
    decisions_for_brief as decisions_for_brief_helper,
    dedupe_similar_titles as dedupe_similar_titles_helper,
    heuristic_ranked_candidates as heuristic_ranked_candidates_helper,
    limit_candidates_for_ai as limit_candidates_for_ai_helper,
    select_articles as select_articles_helper,
    sort_by_heuristic_then_time as sort_by_heuristic_then_time_helper,
    title_dedupe_key as title_dedupe_key_helper,
    tokenize_for_match as tokenize_for_match_helper,
    topic_is_enabled as topic_is_enabled_helper,
    union_candidates_by_id as union_candidates_by_id_helper,
)
from .intermediate_serialization import to_jsonable
from .models import (
    AppConfig,
    BriefOutput,
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    PipelineResult,
    PriorReport,
    RunSourceSnapshot,
    SelectedArticle,
    TopicConfig,
)
from .pipeline_stages import PipelineRunOptions
from .retrieval.article import ArticleRetriever
from .retrieval.google_news import GoogleNewsQueryRetriever
from .retrieval.reports import PriorReportRetriever
from .scrapers.rss import RSSScraper
from .shared_headline_scoring import score_snapshot_headlines_once as score_snapshot_headlines_once_helper
from .snapshot_helpers import (
    build_snapshot as build_snapshot_helper,
    candidate_in_window as candidate_in_window_helper,
    merge_topics_for_snapshot as merge_topics_for_snapshot_helper,
    snapshot_candidates_for_brief as snapshot_candidates_for_brief_helper,
)
from .utils import datetime_to_iso, normalize_url, utc_now


class NewsOrchestrator:
    def __init__(self, config: AppConfig, debug: bool = False) -> None:
        self.config = config
        self.debug = DebugLogger(debug)
        set_ai_artifact_root(config.output_dir)
        self.summary_ai_client = create_ai_client(config.ai_summary, self.debug)
        self.final_ai_client = create_ai_client(config.ai_final, self.debug)
        self.http_cache = HTTPCache(
            root_dir=config.cache.dir,
            namespace="shared",
            enabled=config.cache.enabled,
            debug=self.debug,
        )
        self.synth_cache = JSONCache(
            root_dir=config.cache.dir,
            namespace="synth",
            enabled=config.cache.enabled and config.cache.ai_enabled,
        )
        self.google_news_retriever = GoogleNewsQueryRetriever(
            config.google_news_source,
            config.user_agent,
            max_workers=config.runtime.max_http_workers,
            http_cache=self.http_cache,
            cache_fresh_seconds=config.cache.http_fresh_seconds,
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

    def run(self, run_options: PipelineRunOptions | None = None) -> PipelineResult:
        self._prepare_run_options(run_options or PipelineRunOptions())
        with self.debug.span("pipeline.total"):
            now = utc_now()
            today = now.date()
            date = today.isoformat()
            self.warnings = []
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
                summary_device=self.config.ai_summary.device,
                final_model=self.config.ai_final.effective_model_label,
                final_backend=self.config.ai_final.backend,
                final_device=self.config.ai_final.device,
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
                with self.debug.span("prior_reports.fetch"):
                    prior_reports = self.fetch_prior_reports(today)
                self.debug.set_metric("prior_reports.count", len(prior_reports))
                self._record_stage_artifact(
                    stage="prior_reports",
                    payload=self._stage_payload(
                        summary={
                            "prior_reports_count": len(prior_reports),
                            "prior_report_ids": [report.id for report in prior_reports],
                        },
                        intermediate={"prior_reports": prior_reports},
                    ),
                )
                if self._stop_requested("prior_reports"):
                    return self._stopped_result()

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
                        summary=snapshot_payload,
                        intermediate={"snapshot": snapshot},
                    ),
                )
                if self._stop_requested("snapshot"):
                    return self._stopped_result()

                shared_candidates_by_brief: Dict[str, List[NewsCandidate]] = {}
                shared_decisions: Dict[str, HeadlineDecision] | None = None
                if snapshot is not None:
                    shared_candidates_by_brief, shared_decisions, shared_warnings = self._score_snapshot_headlines_once(
                        snapshot,
                        now,
                        general_topics,
                        detailed_topics,
                    )
                    self.warnings.extend(shared_warnings)
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
                self._record_stage_artifact(
                    stage="shared_headline_scoring",
                    payload=self._stage_payload(
                        summary=shared_payload,
                        intermediate={
                            "shared_candidates_by_brief": shared_candidates_by_brief,
                            "shared_decisions": shared_decisions or {},
                            "shared_warnings": shared_warnings if snapshot is not None else [],
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
                        brief_goal=(
                            "General daily news pass. Prefer breadth and usefulness over deep specialization. "
                            "Use the lower threshold to fill the brief with the strongest general stories, up to the configured article count. "
                            "Still avoid trivia, gossip, minor sports, and duplicate rewrites."
                        ),
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
                        brief_goal=(
                            "Detailed topic investigation pass. Focus on the configured topics, identify major narratives, "
                            "compare with prior reports, and select sources that can deepen, challenge, or reshape those narratives."
                        ),
                        limited_candidates_override=shared_candidates_by_brief.get("detailed"),
                        shared_decisions=shared_decisions,
                    )
                    if detailed_output is not None:
                        outputs.append(detailed_output)
                    if self.stopped_after_stage:
                        return self._stopped_result(outputs=outputs)

                self.debug.set_metric("pipeline.outputs", len(outputs))
                self.debug.set_metric("pipeline.status", "completed")
                self.debug.log("pipeline", "complete", outputs=len(outputs), warnings=len(self.warnings))
                return PipelineResult(outputs=outputs, warnings=self.warnings)
            except Exception as exc:
                self.debug.set_metric("pipeline.status", "failed")
                self.debug.set_metric("pipeline.error", f"{type(exc).__name__}: {exc}")
                raise

    def _prepare_run_options(self, options: PipelineRunOptions) -> None:
        self.run_options = options
        self.stopped_after_stage = ""
        self.stage_artifact_paths = []
        self._stage_run_label = utc_now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = str(options.stage_artifact_dir or "").strip()
        if artifact_dir:
            self._stage_artifact_root = Path(artifact_dir)
        else:
            self._stage_artifact_root = Path(self.config.output_dir) / "diagnostics" / "stages"

    def _stopped_result(self, outputs: List[BriefOutput] | None = None) -> PipelineResult:
        self.debug.set_metric("pipeline.outputs", len(outputs or []))
        self.debug.set_metric("pipeline.status", "stopped")
        self.debug.log(
            "pipeline",
            "stopped",
            stage=self.stopped_after_stage or self.run_options.stop_after_stage,
            outputs=len(outputs or []),
        )
        return PipelineResult(outputs=list(outputs or []), warnings=self.warnings)

    def _stage_payload(
        self,
        *,
        summary: Dict[str, Any],
        intermediate: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = dict(summary)
        if self.run_options.save_intermediate and intermediate:
            payload["intermediate"] = intermediate
        return payload

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
            artifact_payload = {
                "run_label": self._stage_run_label,
                "brief": brief_name,
                "stage": stage,
                "generated_at": utc_now().isoformat(),
                "payload": to_jsonable(payload),
            }
            artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            path_text = str(artifact_path)
            self.stage_artifact_paths.append(path_text)
            return path_text
        except Exception as exc:
            self.warnings.append(f"Stage artifact write failed ({brief_name}/{stage}): {type(exc).__name__}: {exc}")
            return ""

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

    def _record_article_fetch_metrics(self, brief_name: str, selected: List[SelectedArticle]) -> None:
        record_article_fetch_metrics_helper(
            brief_name=brief_name,
            selected=selected,
            debug=self.debug,
        )

    def _record_enrichment_metrics(self, brief_name: str, selected: List[SelectedArticle]) -> None:
        record_enrichment_metrics_helper(
            brief_name=brief_name,
            selected=selected,
            debug=self.debug,
        )

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
            limit_candidates_for_ai=self.limit_candidates_for_ai,
            snapshot_candidates_for_brief=self._snapshot_candidates_for_brief,
            analyzer_cls=HeadlineAnalyzer,
        )

    @staticmethod
    def _union_candidates_by_id(*groups: List[NewsCandidate]) -> List[NewsCandidate]:
        return union_candidates_by_id_helper(*groups)

    @staticmethod
    def _decisions_for_brief(
        candidates: List[NewsCandidate],
        shared_decisions: Dict[str, HeadlineDecision],
        topics: List[TopicConfig],
    ) -> Dict[str, HeadlineDecision]:
        return decisions_for_brief_helper(candidates, shared_decisions, topics)

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

    @staticmethod
    def _merge_topics_for_snapshot(*topic_groups: List[TopicConfig]) -> List[TopicConfig]:
        return merge_topics_for_snapshot_helper(*topic_groups)

    def _snapshot_candidates_for_brief(
        self,
        snapshot: RunSourceSnapshot,
        since,
    ) -> tuple[List[NewsCandidate], List[NewsCandidate], List[NewsCandidate]]:
        return snapshot_candidates_for_brief_helper(snapshot, since)

    @staticmethod
    def _candidate_in_window(candidate: NewsCandidate, since) -> bool:
        return candidate_in_window_helper(candidate, since)

    def _populate_article_texts(
        self,
        brief_name: str,
        selected: List[SelectedArticle],
        article_retriever: ArticleRetriever,
        warnings: List[str],
    ) -> None:
        populate_article_texts_batch(
            brief_name=brief_name,
            selected=selected,
            article_retriever=article_retriever,
            warnings=warnings,
            max_article_workers=self.config.runtime.max_article_workers,
            debug=self.debug,
        )

    def fetch_headlines(self, since, max_headlines_per_source: int, warnings: List[str]) -> List[NewsCandidate]:
        scraper = RSSScraper(
            self.config.rss_sources,
            self.config.user_agent,
            max_headlines_per_source,
            max_workers=self.config.runtime.max_http_workers,
            http_cache=self.http_cache,
            cache_fresh_seconds=self.config.cache.http_fresh_seconds,
            debug=self.debug,
        )
        candidates = scraper.fetch(since)
        warnings.extend(f"RSS: {error}" for error in scraper.errors)
        return candidates

    def fetch_topic_headlines(self, topics, since, warnings: List[str]) -> List[NewsCandidate]:
        candidates = self.google_news_retriever.fetch(topics, since)
        warnings.extend(f"Google News: {error}" for error in self.google_news_retriever.errors)
        return candidates

    def fetch_prior_reports(self, today):
        reports = self.prior_report_retriever.fetch(today)
        self.warnings.extend(f"Prior reports: {error}" for error in self.prior_report_retriever.errors)
        return reports

    @staticmethod
    def merge_url_duplicates(candidates: List[NewsCandidate]) -> List[NewsCandidate]:
        # Borrowed from Horizon's non-LLM cross-source URL merge strategy (MIT):
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

    def limit_candidates_for_ai(self, candidates: List[NewsCandidate], topics, filtering: FilteringConfig, since) -> List[NewsCandidate]:
        return limit_candidates_for_ai_helper(
            candidates,
            topics,
            filtering,
            since,
            user_memory=self.config.user_memory,
            debug=self.debug,
        )

    @staticmethod
    def _sort_by_heuristic_then_time(
        candidates: List[NewsCandidate],
        score_by_id: Dict[str, float],
        fallback_date,
    ) -> List[NewsCandidate]:
        return sort_by_heuristic_then_time_helper(candidates, score_by_id, fallback_date)

    def _heuristic_ranked_candidates(self, candidates: List[NewsCandidate], topics, since) -> List[tuple[NewsCandidate, float]]:
        return heuristic_ranked_candidates_helper(candidates, topics, since, self.config.user_memory)

    def _candidate_heuristic_score(self, item: NewsCandidate, topics, since) -> float:
        return candidate_heuristic_score_helper(item, topics, since, user_memory=self.config.user_memory)

    @staticmethod
    def _topic_is_enabled(topics, topic_name: str) -> bool:
        return topic_is_enabled_helper(topics, topic_name)

    def _candidate_topic_match(self, item: NewsCandidate, topic) -> float:
        return candidate_topic_match_helper(item, topic)

    @staticmethod
    def _tokenize_for_match(text: str) -> List[str]:
        return tokenize_for_match_helper(text)

    def _dedupe_similar_titles(self, candidates: List[NewsCandidate]) -> List[NewsCandidate]:
        return dedupe_similar_titles_helper(candidates, self.debug)

    @staticmethod
    def _title_dedupe_key(title: str) -> str:
        return title_dedupe_key_helper(title)

    def select_articles(
        self,
        candidates: List[NewsCandidate],
        decisions: Dict[str, HeadlineDecision],
        topics: List[TopicConfig],
        filtering: FilteringConfig,
    ) -> List[SelectedArticle]:
        return select_articles_helper(
            candidates,
            decisions,
            topics,
            filtering,
            user_memory=self.config.user_memory,
        )
