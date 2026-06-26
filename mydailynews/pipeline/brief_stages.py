from __future__ import annotations

from typing import Any, Dict, List

from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.pipeline.article_pipeline import (
    populate_article_texts,
    record_article_fetch_metrics,
    record_enrichment_metrics,
    story_thread_enrichment_counts,
)
from mydailynews.enrichment.runner import StoryThreadEnricher
from mydailynews.briefing.final_budget import prune_selected_for_final_token_budget
from mydailynews.domain.headline_selection import (
    decisions_for_brief,
    limit_candidates_for_ai,
    select_articles,
    selection_reason_counters,
)
from mydailynews.app.models import HeadlineDecision, NewsCandidate, PriorReport, RunSourceSnapshot, SelectedArticle, TopicConfig
from mydailynews.retrieval.article import ArticleRetriever
from mydailynews.pipeline.stage_results import (
    ArticleFetchResult,
    CandidatePreparationResult,
    EnrichmentStageResult,
    HeadlineLimitResult,
    HeadlineScoringResult,
    SelectionResult,
    StoryGroupingStageResult,
)
from mydailynews.story_grouping.planner import StoryGroupingPlanner
from mydailynews.pipeline.snapshot_helpers import snapshot_candidates_for_brief
from mydailynews.common.warnings import extend_warnings


def _candidate_source_count(candidates: List[NewsCandidate]) -> int:
    sources = {
        str(candidate.source or "").strip().lower()
        for candidate in candidates
        if str(candidate.source or "").strip()
    }
    return len(sources)


def _selected_source_count(selected: List[SelectedArticle]) -> int:
    return _candidate_source_count([article.candidate for article in selected])


def _checkpoint_stage(
    orchestrator,
    *,
    brief_name: str,
    stage: str,
    summary: Dict[str, Any],
    next_stage_input: Dict[str, Any] | None = None,
) -> bool:
    payload = dict(summary)
    stage_payload_builder = getattr(orchestrator, "_stage_payload", None)
    if callable(stage_payload_builder):
        payload = stage_payload_builder(
            stage=stage,
            brief_name=brief_name,
            summary=summary,
            next_stage_input=next_stage_input,
        )
    elif next_stage_input:
        payload["next_stage_input"] = next_stage_input

    record_stage_artifact = getattr(orchestrator, "_record_stage_artifact", None)
    if callable(record_stage_artifact):
        record_stage_artifact(stage=stage, brief_name=brief_name, payload=payload)

    stop_requested = getattr(orchestrator, "_stop_requested", None)
    if callable(stop_requested) and stop_requested(stage):
        orchestrator.debug.set_metric(f"brief.{brief_name}.status", "stopped")
        orchestrator.debug.log("brief.run", "stopped", name=brief_name, stage=stage)
        return True
    return False


def _report_phase(orchestrator, message: str) -> None:
    reporter = getattr(orchestrator, "reporter", None)
    phase = getattr(reporter, "phase", None)
    if callable(phase):
        phase(message)


def _prepare_candidates_stage(
    orchestrator,
    *,
    brief_name: str,
    topics: List[TopicConfig],
    filtering,
    prior_reports: List[PriorReport],
    since,
    snapshot: RunSourceSnapshot | None,
) -> CandidatePreparationResult:
    warnings: List[str] = []
    unique_candidates: List[NewsCandidate]
    raw_count = 0
    rss_count = 0
    topic_count = 0
    rss_candidates_for_stage: List[NewsCandidate] = []
    topic_candidates_for_stage: List[NewsCandidate] = []

    _report_phase(orchestrator, f"Preparing {brief_name} candidates...")
    with orchestrator.debug.span(f"brief.{brief_name}.candidate_prepare"):
        if snapshot:
            extend_warnings(warnings, snapshot.metadata.get("warnings", []))
            rss_candidates, topic_candidates, unique_candidates = snapshot_candidates_for_brief(
                snapshot,
                since,
            )
            rss_candidates_for_stage = list(rss_candidates)
            topic_candidates_for_stage = list(topic_candidates)
            raw_count = len(rss_candidates) + len(topic_candidates)
            rss_count = len(rss_candidates)
            topic_count = len(topic_candidates)
            orchestrator.debug.log(
                "headline.fetch",
                "reused_snapshot",
                brief=brief_name,
                snapshot_since=snapshot.fetched_since,
                raw_candidates=raw_count,
                rss_candidates=rss_count,
                topic_candidates=topic_count,
                unique_candidates=len(unique_candidates),
                prior_reports=len(prior_reports),
            )
        else:
            with orchestrator.debug.span(f"brief.{brief_name}.headline_fetch"):
                rss_candidates_for_stage = orchestrator.fetch_headlines(
                    since,
                    filtering.max_headlines_per_source,
                    warnings,
                )
                topic_candidates_for_stage = orchestrator.fetch_topic_headlines(topics, since, warnings)
            candidates = list(rss_candidates_for_stage) + list(topic_candidates_for_stage)
            raw_count = len(candidates)
            rss_count = len(rss_candidates_for_stage)
            topic_count = len(topic_candidates_for_stage)
            orchestrator.debug.log(
                "headline.fetch",
                "complete",
                brief=brief_name,
                raw_candidates=raw_count,
                rss_candidates=rss_count,
                topic_candidates=topic_count,
                prior_reports=len(prior_reports),
            )
            unique_candidates = orchestrator.merge_url_duplicates(candidates)
            orchestrator.debug.log(
                "headline.dedupe",
                "complete",
                brief=brief_name,
                unique_candidates=len(unique_candidates),
            )

    orchestrator.debug.set_metric(f"brief.{brief_name}.raw_candidates", raw_count)
    orchestrator.debug.set_metric(f"brief.{brief_name}.rss_candidates", rss_count)
    orchestrator.debug.set_metric(f"brief.{brief_name}.topic_candidates", topic_count)
    orchestrator.debug.set_metric(f"brief.{brief_name}.unique_candidates", len(unique_candidates))
    return CandidatePreparationResult(
        raw_count=raw_count,
        rss_count=rss_count,
        topic_count=topic_count,
        unique_candidates=unique_candidates,
        warnings=warnings,
        rss_candidates=rss_candidates_for_stage,
        topic_candidates=topic_candidates_for_stage,
    )


def _limit_headlines_stage(
    orchestrator,
    *,
    brief_name: str,
    unique_candidates: List[NewsCandidate],
    topics: List[TopicConfig],
    filtering,
    since,
    limited_candidates_override: List[NewsCandidate] | None,
) -> HeadlineLimitResult:
    with orchestrator.debug.span(f"brief.{brief_name}.headline_limit"):
        if limited_candidates_override is None:
            limited_candidates = limit_candidates_for_ai(
                unique_candidates,
                topics,
                filtering,
                since,
                user_memory=orchestrator.config.user_memory,
                debug=orchestrator.debug,
            )
            orchestrator.debug.log(
                "headline.limit",
                "complete",
                brief=brief_name,
                candidates_for_ai=len(limited_candidates),
            )
        else:
            limited_candidates = list(limited_candidates_override)
            orchestrator.debug.log(
                "headline.limit",
                "reused_shared_prefilter",
                brief=brief_name,
                candidates_for_ai=len(limited_candidates),
            )

    source_count = _candidate_source_count(limited_candidates)
    orchestrator.debug.set_metric(f"brief.{brief_name}.limited_candidates", len(limited_candidates))
    orchestrator.debug.set_metric(f"brief.{brief_name}.limited_sources", source_count)
    return HeadlineLimitResult(
        limited_candidates=limited_candidates,
        limited_sources=source_count,
    )


def _score_headlines_stage(
    orchestrator,
    *,
    brief_name: str,
    limited_candidates: List[NewsCandidate],
    topics: List[TopicConfig],
    filtering,
    brief_goal: str,
    shared_decisions: Dict[str, HeadlineDecision] | None,
) -> HeadlineScoringResult:
    warnings: List[str] = []
    if shared_decisions is None:
        _report_phase(orchestrator, f"Scoring {brief_name} headlines...")
    with orchestrator.debug.span(f"brief.{brief_name}.headline_decisions"):
        if shared_decisions is None:
            # Batch size is configurable; smaller values trade speed for reliability on constrained hardware.
            headline_analyzer = HeadlineAnalyzer(
                orchestrator.summary_ai_client,
                max(1, int(filtering.max_headlines_per_ai_batch)),
                orchestrator.debug,
                cache=orchestrator.synth_cache,
                cache_ttl_seconds=orchestrator.config.cache.synth_fresh_seconds,
                input_token_limit=getattr(filtering, "headline_max_input_tokens", None),
                max_new_tokens=getattr(filtering, "headline_max_new_tokens", None),
                single_replay_max_new_tokens=getattr(
                    filtering,
                    "headline_single_replay_max_new_tokens",
                    None,
                ),
            )
            decisions = headline_analyzer.analyze(
                limited_candidates,
                orchestrator.config.user_memory,
                topics,
                brief_goal,
                brief_name=brief_name,
            )
            extend_warnings(warnings, headline_analyzer.warnings)
            orchestrator.debug.log("headline.decisions", "complete", brief=brief_name, decisions=len(decisions))
        else:
            decisions = decisions_for_brief(limited_candidates, shared_decisions, topics)
            orchestrator.debug.log("headline.decisions", "reused_shared", brief=brief_name, decisions=len(decisions))
    orchestrator.debug.set_metric(f"brief.{brief_name}.decisions", len(decisions))
    return HeadlineScoringResult(
        limited_candidates=limited_candidates,
        decisions=decisions,
        warnings=warnings,
    )


def _select_articles_stage(
    orchestrator,
    *,
    brief_name: str,
    limited_candidates: List[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
    topics: List[TopicConfig],
    filtering,
    prior_reports: List[PriorReport],
    brief_goal: str,
    date: str,
    include_enrichment_context: bool,
) -> SelectionResult:
    warnings: List[str] = []
    _report_phase(orchestrator, f"Selecting {brief_name} articles...")
    with orchestrator.debug.span(f"brief.{brief_name}.headline_select"):
        selected = select_articles(
            limited_candidates,
            decisions,
            topics,
            filtering,
            user_memory=orchestrator.config.user_memory,
        )
    selected = prune_selected_for_final_token_budget(
        orchestrator,
        brief_name=brief_name,
        selected=selected,
        filtering=filtering,
        topics=topics,
        prior_reports=prior_reports,
        brief_goal=brief_goal,
        date=date,
        include_enrichment_context=include_enrichment_context,
        run_warnings=warnings,
    )
    orchestrator.debug.set_metric(f"brief.{brief_name}.selected", len(selected))
    selection_counts = selection_reason_counters(decisions)
    selected_reason_counts = selection_counts.get("selected", {})
    skipped_reason_counts = selection_counts.get("skipped", {})
    for code, count in selected_reason_counts.items():
        orchestrator.debug.set_metric(f"brief.{brief_name}.selection.selected_reason.{code}", int(count))
    for code, count in skipped_reason_counts.items():
        orchestrator.debug.set_metric(f"brief.{brief_name}.selection.skipped_reason.{code}", int(count))
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.selection.selected_reason_total",
        sum(int(value) for value in selected_reason_counts.values()),
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.selection.skipped_reason_total",
        sum(int(value) for value in skipped_reason_counts.values()),
    )
    source_count = _selected_source_count(selected)
    orchestrator.debug.set_metric(f"brief.{brief_name}.selected_sources", source_count)
    orchestrator.debug.log(
        "headline.select",
        "complete",
        brief=brief_name,
        selected=len(selected),
        selected_sources=source_count,
        selected_reason_codes=selected_reason_counts,
        skipped_reason_codes=skipped_reason_counts,
    )
    return SelectionResult(
        selected=selected,
        selection_counts=selection_counts,
        warnings=warnings,
        selected_sources=source_count,
    )


def _fetch_articles_stage(
    orchestrator,
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    filtering,
) -> ArticleFetchResult:
    warnings: List[str] = []
    article_retriever = ArticleRetriever(
        orchestrator.config.user_agent,
        filtering.article_text_max_chars,
        http_cache=None,
        debug=orchestrator.debug,
    )
    for article in selected:
        orchestrator.debug.log(
            "article",
            "selected",
            brief=brief_name,
            score=article.decision.score,
            topic=article.decision.topic,
            selection_reason=article.selection_reason_code or article.decision.selection_reason_code,
            rank_score=article.selection_rank_score or article.decision.selection_rank_score,
            rank_mode=article.selection_rank_mode or article.decision.selection_rank_mode,
            source=article.candidate.source,
            title=article.candidate.title,
        )
    _report_phase(orchestrator, f"Fetching {brief_name} article text...")
    with orchestrator.debug.span(f"brief.{brief_name}.article_fetch"):
        populate_article_texts(
            brief_name=brief_name,
            selected=selected,
            article_retriever=article_retriever,
            warnings=warnings,
            max_article_workers=orchestrator.config.runtime.max_article_workers,
            debug=orchestrator.debug,
            article_text_cache=getattr(orchestrator, "article_text_cache", None),
        )
    record_article_fetch_metrics(
        brief_name=brief_name,
        selected=selected,
        debug=orchestrator.debug,
    )
    status_counts: Dict[str, int] = {}
    for item in selected:
        status = str(item.extraction_status or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return ArticleFetchResult(
        selected=selected,
        status_counts=status_counts,
        warnings=warnings,
    )


def _story_grouping_stage(
    orchestrator,
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    include_enrichment_context: bool,
    evidence_config,
    date: str = "",
) -> StoryGroupingStageResult:
    needs_enrichment_grouping = bool(include_enrichment_context)
    needs_evidence_grouping = bool(getattr(evidence_config, "enabled", False))
    warning_sink: List[str] = []

    def skipped(reason: str) -> StoryGroupingStageResult:
        result = StoryGroupingStageResult.skipped(selected=selected, reason=reason)
        _record_story_grouping_metrics(orchestrator, brief_name=brief_name, result=result)
        if reason:
            orchestrator.debug.set_metric(f"brief.{brief_name}.story_grouping.skipped_reason.{reason}", 1)
        orchestrator.debug.log(
            "story_grouping",
            "skipped",
            brief=brief_name,
            reason=reason,
            selected=len(selected),
            needs_enrichment=needs_enrichment_grouping,
            needs_evidence=needs_evidence_grouping,
        )
        return result

    if not selected:
        return skipped("no_selected_articles")
    if not needs_enrichment_grouping and not needs_evidence_grouping:
        return skipped("both_consumers_not_enabled")
    if not needs_enrichment_grouping:
        return skipped("enrichment_disabled")
    if not needs_evidence_grouping:
        return skipped("evidence_disabled")

    ai_client = getattr(orchestrator, "summary_ai_client", None)
    if ai_client is None:
        return skipped("no_ai_client")

    _report_phase(orchestrator, f"Planning {brief_name} story grouping...")
    planner = StoryGroupingPlanner(
        orchestrator.config,
        ai_client,
        cache=getattr(orchestrator, "synth_cache", None),
        debug=orchestrator.debug,
        warning_sink=warning_sink.append,
        brief_name=brief_name,
        date=date,
    )
    with orchestrator.debug.span(f"brief.{brief_name}.story_grouping"):
        planning = planner.plan(selected)
    if str(planning.artifact.get("status", "")).strip() == "skipped_budget":
        result = StoryGroupingStageResult.skipped(
            selected=selected,
            reason="budget_unfit",
            warnings=warning_sink,
            artifact=planning.artifact,
        )
    else:
        result = StoryGroupingStageResult.planned(
            selected=selected,
            story_groups=planning.story_groups,
            planner_artifact=planning.artifact,
            warnings=warning_sink,
            cache_hit=bool(getattr(planner, "cache_hits", 0)),
        )
    _record_story_grouping_metrics(orchestrator, brief_name=brief_name, result=result)
    orchestrator.debug.log(
        "story_grouping",
        "complete",
        brief=brief_name,
        status=result.artifact.get("status", ""),
        groups=len(result.story_groups),
        fallback_groups=len(result.artifact.get("fallback_groups", [])),
        cache_hit=bool(result.artifact.get("cache_hit", False)),
    )
    return result


def _record_story_grouping_metrics(
    orchestrator,
    *,
    brief_name: str,
    result: StoryGroupingStageResult,
) -> None:
    artifact = result.artifact
    orchestrator.debug.set_metric(f"brief.{brief_name}.story_grouping.enabled", bool(artifact.get("enabled", False)))
    orchestrator.debug.set_metric(f"brief.{brief_name}.story_grouping.groups", len(result.story_groups))
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.story_grouping.fallback_groups",
        len(artifact.get("fallback_groups", [])),
    )
    orchestrator.debug.set_metric(f"brief.{brief_name}.story_grouping.cache_hit", bool(artifact.get("cache_hit", False)))
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.story_grouping.skipped_reason",
        str(artifact.get("skipped_reason", "")),
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.story_grouping.requests",
        len(artifact.get("requests", [])),
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.story_grouping.split_requests",
        bool(artifact.get("split_requests", False)),
    )


def _enrich_articles_stage(
    orchestrator,
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    include_enrichment_context: bool,
    date: str = "",
    story_groups=None,
) -> EnrichmentStageResult:
    enricher = StoryThreadEnricher(
        orchestrator.config,
        http_cache=getattr(orchestrator, "enrichment_cache", None),
        debug=orchestrator.debug,
        ai_client=getattr(orchestrator, "summary_ai_client", None),
        cache=getattr(orchestrator, "synth_cache", None),
        brief_name=brief_name,
        date=date,
    )
    if include_enrichment_context:
        _report_phase(orchestrator, f"Enriching {brief_name} articles...")
    with orchestrator.debug.span(f"brief.{brief_name}.enrichment"):
        enricher.enrich_many(
            selected,
            story_groups=story_groups,
        )
    record_enrichment_metrics(
        brief_name=brief_name,
        selected=selected,
        debug=orchestrator.debug,
        story_thread_counts=(
            int(getattr(enricher, "story_threads_created", 0)),
            int(getattr(enricher, "story_threads_enriched", 0)),
            int(getattr(enricher, "story_threads_skipped", 0)),
        ),
    )
    story_threads_created, story_threads_enriched, story_threads_skipped = (
        int(getattr(enricher, "story_threads_created", 0)),
        int(getattr(enricher, "story_threads_enriched", 0)),
        int(getattr(enricher, "story_threads_skipped", 0)),
    )
    if not any((story_threads_created, story_threads_enriched, story_threads_skipped)):
        story_threads_created, story_threads_enriched, story_threads_skipped = story_thread_enrichment_counts(selected)
    return EnrichmentStageResult(
        selected=selected,
        enrichment_needed=sum(1 for article in selected if article.enrichment_needed),
        context_sources=sum(len(article.context_sources) for article in selected),
        story_threads_created=story_threads_created,
        story_threads_enriched=story_threads_enriched,
        story_threads_skipped=story_threads_skipped,
        artifact=dict(getattr(enricher, "artifact", {}) or {}),
        warnings=list(enricher.warnings),
    )
