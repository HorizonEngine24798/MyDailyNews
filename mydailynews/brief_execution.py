from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import re
from typing import Any, Dict, List

from .ai.headline_analyzer import HeadlineAnalyzer
from .analysis_pipeline import DeltaExtractor, EvidenceDistiller
from .brief import BriefGenerator, brief_metadata
from .enrichment import SimpleEnricher
from .headline_selection import selection_rationale_rows, selection_reason_counters
from .models import (
    AnalysisRolloutModeConfig,
    BriefOutput,
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
    HeadlineDecision,
    NewsCandidate,
    PriorReport,
    RunSourceSnapshot,
    SelectedArticle,
    TopicConfig,
)
from .output import write_json, write_markdown
from .retrieval.article import ArticleRetriever

ANALYSIS_ROLLOUT_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "safe_local": {
        "general": {
            "evidence_enabled": False,
            "delta_enabled": False,
        },
        "detailed": {
            "evidence_enabled": True,
            "delta_enabled": False,
            "evidence_max_input_tokens": 1500,
            "evidence_max_new_tokens": 360,
            "evidence_max_articles": 4,
            "evidence_max_article_chars": 420,
            "delta_max_input_tokens": 1200,
            "delta_max_new_tokens": 220,
            "delta_max_prior_reports": 2,
        },
    },
    "balanced_local": {
        "general": {
            "evidence_enabled": False,
            "delta_enabled": False,
        },
        "detailed": {
            "evidence_enabled": True,
            "delta_enabled": True,
            "evidence_max_input_tokens": 1900,
            "evidence_max_new_tokens": 520,
            "evidence_max_articles": 6,
            "evidence_max_article_chars": 560,
            "delta_max_input_tokens": 1450,
            "delta_max_new_tokens": 300,
            "delta_max_prior_reports": 3,
        },
    },
    "quality_focused": {
        "general": {
            "evidence_enabled": True,
            "delta_enabled": False,
            "evidence_max_input_tokens": 1700,
            "evidence_max_new_tokens": 420,
            "evidence_max_articles": 4,
            "evidence_max_article_chars": 460,
        },
        "detailed": {
            "evidence_enabled": True,
            "delta_enabled": True,
            "evidence_max_input_tokens": 2300,
            "evidence_max_new_tokens": 700,
            "evidence_max_articles": 8,
            "evidence_max_article_chars": 700,
            "delta_max_input_tokens": 1700,
            "delta_max_new_tokens": 380,
            "delta_max_prior_reports": 3,
        },
    },
}


def _tokenize_delta_text(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "latest",
        "today",
        "news",
        "major",
        "about",
        "after",
        "amid",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if token not in stop
    }


def _prior_headline_items(prior_reports: List[PriorReport], max_reports: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for report in prior_reports[: max(1, max_reports)]:
        major = report.major_headlines if isinstance(report.major_headlines, list) else []
        if not major:
            title = str(report.title or "").strip()
            if title:
                items.append(
                    {
                        "headline": title,
                        "report_id": report.id,
                        "report_date": report.date,
                        "tokens": _tokenize_delta_text(title),
                    }
                )
            continue
        for row in major[:8]:
            if not isinstance(row, dict):
                continue
            headline = str(row.get("headline") or row.get("title") or "").strip()
            if not headline:
                continue
            items.append(
                {
                    "headline": headline,
                    "report_id": report.id,
                    "report_date": report.date,
                    "tokens": _tokenize_delta_text(headline),
                }
            )
    return items


def _best_overlap(current_tokens: set[str], prior_items: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any] | None]:
    best_score = 0.0
    best_item: Dict[str, Any] | None = None
    if not current_tokens:
        return best_score, best_item
    for item in prior_items:
        prior_tokens = item.get("tokens", set())
        if not prior_tokens:
            continue
        overlap = len(current_tokens.intersection(prior_tokens))
        if overlap <= 0:
            continue
        score = overlap / max(1, min(len(current_tokens), len(prior_tokens)))
        if score > best_score:
            best_score = score
            best_item = item
    return best_score, best_item


def _delta_entry(article: SelectedArticle, summary: str) -> Dict[str, Any]:
    return {
        "item": str(article.candidate.title or "")[:100],
        "summary": summary[:180],
        "article_ids": [str(article.candidate.id)],
    }


def build_deterministic_delta_scaffold(
    selected: List[SelectedArticle],
    prior_reports: List[PriorReport],
    *,
    max_prior_reports: int = 3,
) -> Dict[str, Any]:
    if not selected:
        return {}

    prior_items = _prior_headline_items(prior_reports, max_prior_reports)
    coverage_note = (
        "No prior reports available; deterministic scaffold highlights only the current-run story shape."
        if not prior_reports
        else (
            f"Compared {len(selected)} current selected article(s) against "
            f"{len(prior_items)} prior headline anchor(s) from {min(len(prior_reports), max_prior_reports)} report(s)."
        )
    )

    escalated_terms = {
        "escalates",
        "escalation",
        "surge",
        "spike",
        "attack",
        "expands",
        "tightens",
        "sanctions",
        "deadline",
        "warning",
    }
    weakened_terms = {
        "decline",
        "drops",
        "drop",
        "eases",
        "eased",
        "ceasefire",
        "cools",
        "paused",
        "delay",
        "delayed",
    }

    new_items: List[Dict[str, Any]] = []
    escalated: List[Dict[str, Any]] = []
    weakened: List[Dict[str, Any]] = []
    reframed: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []

    overlap_count = 0
    for article in selected:
        title = str(article.candidate.title or "").strip()
        snippet = str(article.candidate.snippet or "").strip()
        current_tokens = _tokenize_delta_text(f"{title} {snippet}")
        overlap_score, prior_match = _best_overlap(current_tokens, prior_items)
        prior_label = ""
        prior_date = ""
        if prior_match:
            prior_label = str(prior_match.get("headline", "")).strip()
            prior_date = str(prior_match.get("report_date", "")).strip()
        if overlap_score >= 0.58 and prior_match:
            overlap_count += 1
            has_escalation = bool(current_tokens.intersection(escalated_terms))
            has_weakening = bool(current_tokens.intersection(weakened_terms))
            if has_escalation and not has_weakening:
                escalated.append(
                    _delta_entry(
                        article,
                        f"Likely continuation of prior coverage ({prior_date}: {prior_label}) with signs of escalation.",
                    )
                )
            elif has_weakening and not has_escalation:
                weakened.append(
                    _delta_entry(
                        article,
                        f"Likely continuation of prior coverage ({prior_date}: {prior_label}) with signs of easing.",
                    )
                )
            else:
                unchanged.append(
                    _delta_entry(
                        article,
                        f"Likely continuation of prior coverage ({prior_date}: {prior_label}).",
                    )
                )
            continue
        if overlap_score >= 0.34 and prior_match:
            overlap_count += 1
            reframed.append(
                _delta_entry(
                    article,
                    f"Partially overlaps prior coverage ({prior_date}: {prior_label}) but appears reframed.",
                )
            )
            continue
        new_items.append(
            _delta_entry(
                article,
                "No strong headline-level overlap found in prior report anchors.",
            )
        )

    evidence_gaps: List[Dict[str, Any]] = []
    if not prior_reports:
        evidence_gaps.append(
            {
                "gap": "No prior reports available for direct delta comparison.",
                "why_it_matters": "Change classification is approximate without a baseline brief.",
            }
        )
    elif not prior_items:
        evidence_gaps.append(
            {
                "gap": "Prior reports lacked reusable major-headline anchors.",
                "why_it_matters": "Deterministic overlap had to fall back to limited text anchors.",
            }
        )
    elif overlap_count == 0:
        evidence_gaps.append(
            {
                "gap": "No strong deterministic overlap between current and prior headline anchors.",
                "why_it_matters": "Stories may be genuinely new or token overlap may miss semantic continuation.",
            }
        )

    return {
        "baseline_coverage_note": coverage_note,
        "new": new_items[:6],
        "escalated": escalated[:5],
        "weakened": weakened[:5],
        "reframed": reframed[:5],
        "unchanged_but_important": unchanged[:6],
        "evidence_gaps": evidence_gaps[:4],
        "deterministic_scaffold": True,
    }


def _checkpoint_stage(
    orchestrator,
    *,
    brief_name: str,
    stage: str,
    summary: Dict[str, Any],
    intermediate: Dict[str, Any] | None = None,
) -> bool:
    payload = dict(summary)
    stage_payload_builder = getattr(orchestrator, "_stage_payload", None)
    if callable(stage_payload_builder):
        payload = stage_payload_builder(summary=summary, intermediate=intermediate)
    elif intermediate and bool(getattr(getattr(orchestrator, "run_options", None), "save_intermediate", False)):
        payload["intermediate"] = intermediate

    record_stage_artifact = getattr(orchestrator, "_record_stage_artifact", None)
    if callable(record_stage_artifact):
        record_stage_artifact(stage=stage, brief_name=brief_name, payload=payload)

    stop_requested = getattr(orchestrator, "_stop_requested", None)
    if callable(stop_requested) and stop_requested(stage):
        orchestrator.debug.set_metric(f"brief.{brief_name}.status", "stopped")
        orchestrator.debug.log("brief.run", "stopped", name=brief_name, stage=stage)
        return True
    return False


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _coerce_optional_int(value: Any, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    return max(minimum, int(value))


def _rollout_mode_from_object(raw: Any) -> AnalysisRolloutModeConfig:
    if raw is None:
        raw = {}
    if isinstance(raw, dict):
        getter = raw.get
    else:
        getter = lambda key, default=None: getattr(raw, key, default)
    return AnalysisRolloutModeConfig(
        evidence_enabled=_coerce_optional_bool(getter("evidence_enabled")),
        delta_enabled=_coerce_optional_bool(getter("delta_enabled")),
        evidence_max_input_tokens=_coerce_optional_int(getter("evidence_max_input_tokens"), minimum=256),
        evidence_max_new_tokens=_coerce_optional_int(getter("evidence_max_new_tokens"), minimum=64),
        evidence_max_articles=_coerce_optional_int(getter("evidence_max_articles"), minimum=1),
        evidence_max_article_chars=_coerce_optional_int(getter("evidence_max_article_chars"), minimum=120),
        delta_max_input_tokens=_coerce_optional_int(getter("delta_max_input_tokens"), minimum=256),
        delta_max_new_tokens=_coerce_optional_int(getter("delta_max_new_tokens"), minimum=64),
        delta_max_prior_reports=_coerce_optional_int(getter("delta_max_prior_reports"), minimum=1),
    )


def _merge_rollout_modes(base: AnalysisRolloutModeConfig, override: AnalysisRolloutModeConfig) -> AnalysisRolloutModeConfig:
    merged = AnalysisRolloutModeConfig(**vars(base))
    for key, value in vars(override).items():
        if value is None:
            continue
        setattr(merged, key, value)
    return merged


def _to_evidence_config(raw: Any) -> EvidenceDistillationConfig:
    defaults = EvidenceDistillationConfig()
    if raw is None:
        raw = defaults
    return EvidenceDistillationConfig(
        enabled=bool(getattr(raw, "enabled", defaults.enabled)),
        model_role=str(getattr(raw, "model_role", defaults.model_role)),
        include_reader_qa=bool(getattr(raw, "include_reader_qa", defaults.include_reader_qa)),
        max_input_tokens=max(256, int(getattr(raw, "max_input_tokens", defaults.max_input_tokens))),
        max_new_tokens=max(64, int(getattr(raw, "max_new_tokens", defaults.max_new_tokens))),
        max_articles=max(1, int(getattr(raw, "max_articles", defaults.max_articles))),
        max_article_chars=max(120, int(getattr(raw, "max_article_chars", defaults.max_article_chars))),
        max_context_sources_per_article=max(
            1,
            int(getattr(raw, "max_context_sources_per_article", defaults.max_context_sources_per_article)),
        ),
        max_story_clusters=max(1, int(getattr(raw, "max_story_clusters", defaults.max_story_clusters))),
        max_claims_per_cluster=max(1, int(getattr(raw, "max_claims_per_cluster", defaults.max_claims_per_cluster))),
        max_questions=max(0, int(getattr(raw, "max_questions", defaults.max_questions))),
        cache_ttl_seconds=max(0, int(getattr(raw, "cache_ttl_seconds", defaults.cache_ttl_seconds))),
    )


def _to_delta_config(raw: Any) -> DeltaExtractionConfig:
    defaults = DeltaExtractionConfig()
    if raw is None:
        raw = defaults
    return DeltaExtractionConfig(
        enabled=bool(getattr(raw, "enabled", defaults.enabled)),
        model_role=str(getattr(raw, "model_role", defaults.model_role)),
        input_source=str(getattr(raw, "input_source", defaults.input_source)),
        require_prior_reports=bool(getattr(raw, "require_prior_reports", defaults.require_prior_reports)),
        max_input_tokens=max(256, int(getattr(raw, "max_input_tokens", defaults.max_input_tokens))),
        max_new_tokens=max(64, int(getattr(raw, "max_new_tokens", defaults.max_new_tokens))),
        max_prior_reports=max(1, int(getattr(raw, "max_prior_reports", defaults.max_prior_reports))),
        cache_ttl_seconds=max(0, int(getattr(raw, "cache_ttl_seconds", defaults.cache_ttl_seconds))),
    )


def _apply_rollout_mode_overrides(
    evidence_config: EvidenceDistillationConfig,
    delta_config: DeltaExtractionConfig,
    mode: AnalysisRolloutModeConfig,
) -> None:
    if mode.evidence_enabled is not None:
        evidence_config.enabled = bool(mode.evidence_enabled)
    if mode.delta_enabled is not None:
        delta_config.enabled = bool(mode.delta_enabled)
    if mode.evidence_max_input_tokens is not None:
        evidence_config.max_input_tokens = min(
            int(evidence_config.max_input_tokens),
            int(mode.evidence_max_input_tokens),
        )
    if mode.evidence_max_new_tokens is not None:
        evidence_config.max_new_tokens = min(
            int(evidence_config.max_new_tokens),
            int(mode.evidence_max_new_tokens),
        )
    if mode.evidence_max_articles is not None:
        evidence_config.max_articles = min(
            int(evidence_config.max_articles),
            int(mode.evidence_max_articles),
        )
    if mode.evidence_max_article_chars is not None:
        evidence_config.max_article_chars = min(
            int(evidence_config.max_article_chars),
            int(mode.evidence_max_article_chars),
        )
    if mode.delta_max_input_tokens is not None:
        delta_config.max_input_tokens = min(
            int(delta_config.max_input_tokens),
            int(mode.delta_max_input_tokens),
        )
    if mode.delta_max_new_tokens is not None:
        delta_config.max_new_tokens = min(
            int(delta_config.max_new_tokens),
            int(mode.delta_max_new_tokens),
        )
    if mode.delta_max_prior_reports is not None:
        delta_config.max_prior_reports = min(
            int(delta_config.max_prior_reports),
            int(mode.delta_max_prior_reports),
        )


def resolve_analysis_stage_configs(analysis: Any, brief_name: str) -> tuple[EvidenceDistillationConfig, DeltaExtractionConfig, Dict[str, Any]]:
    evidence_config = _to_evidence_config(getattr(analysis, "evidence_distillation", None))
    delta_config = _to_delta_config(getattr(analysis, "delta_extraction", None))

    metadata: Dict[str, Any] = {
        "rollout_enabled": False,
        "rollout_profile": "",
        "rollout_mode": brief_name,
        "evidence_requested_enabled": bool(evidence_config.enabled),
        "delta_requested_enabled": bool(delta_config.enabled),
        "evidence_skip_reason": "disabled",
        "delta_skip_reason": "disabled",
    }
    rollout = getattr(analysis, "rollout", None)
    if rollout is None or not bool(getattr(rollout, "enabled", False)):
        metadata["evidence_effective_enabled"] = bool(evidence_config.enabled)
        metadata["delta_effective_enabled"] = bool(delta_config.enabled)
        metadata["evidence_skip_reason"] = "enabled" if evidence_config.enabled else "disabled"
        metadata["delta_skip_reason"] = "enabled" if delta_config.enabled else "disabled"
        return evidence_config, delta_config, metadata

    metadata["rollout_enabled"] = True
    profile = str(getattr(rollout, "profile", "safe_local")).strip().lower() or "safe_local"
    if profile not in ANALYSIS_ROLLOUT_PRESETS:
        profile = "safe_local"
    metadata["rollout_profile"] = profile

    mode_key = "detailed" if brief_name == "detailed" else "general"
    preset_mode_raw = ANALYSIS_ROLLOUT_PRESETS.get(profile, ANALYSIS_ROLLOUT_PRESETS["safe_local"]).get(mode_key, {})
    preset_mode = _rollout_mode_from_object(preset_mode_raw)
    explicit_mode = _rollout_mode_from_object(getattr(rollout, mode_key, None))
    effective_mode = _merge_rollout_modes(preset_mode, explicit_mode)

    _apply_rollout_mode_overrides(evidence_config, delta_config, effective_mode)
    metadata["evidence_effective_enabled"] = bool(evidence_config.enabled)
    metadata["delta_effective_enabled"] = bool(delta_config.enabled)
    metadata["evidence_skip_reason"] = "enabled" if evidence_config.enabled else "rollout_disabled"
    metadata["delta_skip_reason"] = "enabled" if delta_config.enabled else "rollout_disabled"
    return evidence_config, delta_config, metadata


def run_brief(
    orchestrator,
    *,
    name: str,
    output_suffix: str,
    topics: List[TopicConfig],
    filtering,
    prior_reports: List[PriorReport],
    now,
    date: str,
    snapshot: RunSourceSnapshot | None,
    brief_goal: str,
    limited_candidates_override: List[NewsCandidate] | None = None,
    shared_decisions: Dict[str, HeadlineDecision] | None = None,
) -> BriefOutput | None:
    with orchestrator.debug.span(f"brief.{name}.total"):
        since = now - timedelta(hours=filtering.time_window_hours)
        run_warnings: List[str] = []
        orchestrator.debug.set_metric(f"brief.{name}.status", "running")
        orchestrator.debug.log(
            "brief.run",
            "starting",
            name=name,
            topics=len(topics),
            max_candidates=filtering.max_candidates_for_ai,
            ai_batch_size=filtering.max_headlines_per_ai_batch,
            cutoff=filtering.headline_score_cutoff,
            max_selected=filtering.max_selected_articles,
            fill=filtering.fill_selected_articles,
        )

        try:
            unique_candidates: List[NewsCandidate]
            raw_candidate_count = 0
            rss_candidate_count = 0
            topic_candidate_count = 0
            rss_candidates_for_stage: List[NewsCandidate] = []
            topic_candidates_for_stage: List[NewsCandidate] = []
            with orchestrator.debug.span(f"brief.{name}.candidate_prepare"):
                if snapshot:
                    run_warnings.extend(str(item) for item in snapshot.metadata.get("warnings", []))
                    rss_candidates, topic_candidates, unique_candidates = orchestrator._snapshot_candidates_for_brief(snapshot, since)
                    rss_candidates_for_stage = list(rss_candidates)
                    topic_candidates_for_stage = list(topic_candidates)
                    raw_candidate_count = len(rss_candidates) + len(topic_candidates)
                    rss_candidate_count = len(rss_candidates)
                    topic_candidate_count = len(topic_candidates)
                    orchestrator.debug.log(
                        "headline.fetch",
                        "reused_snapshot",
                        brief=name,
                        snapshot_since=snapshot.fetched_since,
                        raw_candidates=raw_candidate_count,
                        rss_candidates=rss_candidate_count,
                        topic_candidates=topic_candidate_count,
                        unique_candidates=len(unique_candidates),
                        prior_reports=len(prior_reports),
                    )
                else:
                    with orchestrator.debug.span(f"brief.{name}.headline_fetch"):
                        rss_candidates_for_stage = orchestrator.fetch_headlines(
                            since,
                            filtering.max_headlines_per_source,
                            run_warnings,
                        )
                        topic_candidates_for_stage = orchestrator.fetch_topic_headlines(topics, since, run_warnings)
                    candidates = list(rss_candidates_for_stage) + list(topic_candidates_for_stage)
                    raw_candidate_count = len(candidates)
                    rss_candidate_count = len(rss_candidates_for_stage)
                    topic_candidate_count = len(topic_candidates_for_stage)
                    orchestrator.debug.log(
                        "headline.fetch",
                        "complete",
                        brief=name,
                        raw_candidates=raw_candidate_count,
                        rss_candidates=rss_candidate_count,
                        topic_candidates=topic_candidate_count,
                        prior_reports=len(prior_reports),
                    )
                    unique_candidates = orchestrator.merge_url_duplicates(candidates)
                    orchestrator.debug.log("headline.dedupe", "complete", brief=name, unique_candidates=len(unique_candidates))
            orchestrator.debug.set_metric(f"brief.{name}.raw_candidates", raw_candidate_count)
            orchestrator.debug.set_metric(f"brief.{name}.rss_candidates", rss_candidate_count)
            orchestrator.debug.set_metric(f"brief.{name}.topic_candidates", topic_candidate_count)
            orchestrator.debug.set_metric(f"brief.{name}.unique_candidates", len(unique_candidates))
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="candidate_prepare",
                summary={
                    "raw_candidates": raw_candidate_count,
                    "rss_candidates": rss_candidate_count,
                    "topic_candidates": topic_candidate_count,
                    "unique_candidates": len(unique_candidates),
                    "unique_candidate_ids": [candidate.id for candidate in unique_candidates],
                },
                intermediate={
                    "rss_candidates": rss_candidates_for_stage,
                    "topic_candidates": topic_candidates_for_stage,
                    "unique_candidates": unique_candidates,
                    "prior_reports": prior_reports,
                    "topics": topics,
                    "since": since,
                },
            ):
                return None

            if not unique_candidates:
                run_warnings.append(f"{name}: No live headline candidates were fetched.")
            with orchestrator.debug.span(f"brief.{name}.headline_limit"):
                if limited_candidates_override is None:
                    limited_candidates = orchestrator.limit_candidates_for_ai(unique_candidates, topics, filtering, since)
                    orchestrator.debug.log("headline.limit", "complete", brief=name, candidates_for_ai=len(limited_candidates))
                else:
                    limited_candidates = list(limited_candidates_override)
                    orchestrator.debug.log("headline.limit", "reused_shared_prefilter", brief=name, candidates_for_ai=len(limited_candidates))
            orchestrator.debug.set_metric(f"brief.{name}.limited_candidates", len(limited_candidates))
            limited_sources = {
                (str(item.source or "").strip().lower())
                for item in limited_candidates
                if str(item.source or "").strip()
            }
            limited_cluster_ids = {
                str(item.metadata.get("event_cluster_id", "")).strip()
                for item in limited_candidates
                if str(item.metadata.get("event_cluster_id", "")).strip()
            }
            limited_multi_source_clusters = {
                str(item.metadata.get("event_cluster_id", "")).strip()
                for item in limited_candidates
                if str(item.metadata.get("event_cluster_id", "")).strip()
                and bool(item.metadata.get("event_cluster_multi_source"))
            }
            orchestrator.debug.set_metric(f"brief.{name}.limited_sources", len(limited_sources))
            orchestrator.debug.set_metric(f"brief.{name}.limited_event_clusters", len(limited_cluster_ids))
            orchestrator.debug.set_metric(
                f"brief.{name}.limited_multi_source_clusters",
                len(limited_multi_source_clusters),
            )
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="headline_limit",
                summary={
                    "limited_candidates": len(limited_candidates),
                    "limited_candidate_ids": [candidate.id for candidate in limited_candidates],
                    "limited_sources": len(limited_sources),
                    "limited_event_clusters": len(limited_cluster_ids),
                    "limited_multi_source_clusters": len(limited_multi_source_clusters),
                },
                intermediate={
                    "limited_candidates": limited_candidates,
                    "unique_candidates": unique_candidates,
                    "topics": topics,
                },
            ):
                return None

            with orchestrator.debug.span(f"brief.{name}.headline_decisions"):
                if shared_decisions is None:
                    # Batch size is configurable; smaller values trade speed for reliability on constrained hardware.
                    headline_analyzer = HeadlineAnalyzer(
                        orchestrator.summary_ai_client,
                        max(1, int(filtering.max_headlines_per_ai_batch)),
                        orchestrator.debug,
                        cache=orchestrator.synth_cache,
                        cache_ttl_seconds=orchestrator.config.cache.synth_fresh_seconds,
                    )
                    decisions = headline_analyzer.analyze(
                        limited_candidates,
                        orchestrator.config.user_memory,
                        topics,
                        brief_goal,
                        brief_name=name,
                    )
                    run_warnings.extend(headline_analyzer.warnings)
                    orchestrator.debug.log("headline.decisions", "complete", brief=name, decisions=len(decisions))
                else:
                    decisions = orchestrator._decisions_for_brief(limited_candidates, shared_decisions, topics)
                    orchestrator.debug.log("headline.decisions", "reused_shared", brief=name, decisions=len(decisions))
            orchestrator.debug.set_metric(f"brief.{name}.decisions", len(decisions))
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="headline_decisions",
                summary={
                    "decisions": len(decisions),
                    "decision_ids": list(decisions.keys()),
                    "missing_decisions": max(0, len(limited_candidates) - len(decisions)),
                },
                intermediate={
                    "decisions": decisions,
                    "limited_candidates": limited_candidates,
                    "topics": topics,
                    "brief_goal": brief_goal,
                },
            ):
                return None

            with orchestrator.debug.span(f"brief.{name}.headline_select"):
                selected = orchestrator.select_articles(limited_candidates, decisions, topics, filtering)
            orchestrator.debug.set_metric(f"brief.{name}.selected", len(selected))
            selection_counts = selection_reason_counters(decisions)
            selected_reason_counts = selection_counts.get("selected", {})
            skipped_reason_counts = selection_counts.get("skipped", {})
            for code, count in selected_reason_counts.items():
                orchestrator.debug.set_metric(f"brief.{name}.selection.selected_reason.{code}", int(count))
            for code, count in skipped_reason_counts.items():
                orchestrator.debug.set_metric(f"brief.{name}.selection.skipped_reason.{code}", int(count))
            orchestrator.debug.set_metric(
                f"brief.{name}.selection.selected_reason_total",
                sum(int(value) for value in selected_reason_counts.values()),
            )
            orchestrator.debug.set_metric(
                f"brief.{name}.selection.skipped_reason_total",
                sum(int(value) for value in skipped_reason_counts.values()),
            )
            selected_sources = {
                (str(item.candidate.source or "").strip().lower())
                for item in selected
                if str(item.candidate.source or "").strip()
            }
            selected_cluster_ids = {
                str(item.candidate.metadata.get("event_cluster_id", "")).strip()
                for item in selected
                if str(item.candidate.metadata.get("event_cluster_id", "")).strip()
            }
            selected_multi_source_clusters = {
                str(item.candidate.metadata.get("event_cluster_id", "")).strip()
                for item in selected
                if str(item.candidate.metadata.get("event_cluster_id", "")).strip()
                and bool(item.candidate.metadata.get("event_cluster_multi_source"))
            }
            orchestrator.debug.set_metric(f"brief.{name}.selected_sources", len(selected_sources))
            orchestrator.debug.set_metric(f"brief.{name}.selected_event_clusters", len(selected_cluster_ids))
            orchestrator.debug.set_metric(
                f"brief.{name}.selected_multi_source_clusters",
                len(selected_multi_source_clusters),
            )
            orchestrator.debug.log(
                "headline.select",
                "complete",
                brief=name,
                selected=len(selected),
                selected_sources=len(selected_sources),
                selected_clusters=len(selected_cluster_ids),
                multi_source_clusters=len(selected_multi_source_clusters),
                selected_reason_codes=selected_reason_counts,
                skipped_reason_codes=skipped_reason_counts,
            )
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="headline_select",
                summary={
                    "selected": len(selected),
                    "selected_article_ids": [article.candidate.id for article in selected],
                    "selected_sources": len(selected_sources),
                    "selected_event_clusters": len(selected_cluster_ids),
                    "selected_multi_source_clusters": len(selected_multi_source_clusters),
                    "selected_reason_codes": selected_reason_counts,
                    "skipped_reason_codes": skipped_reason_counts,
                    "composite_ranking_enabled": bool(getattr(filtering, "use_multifactor_composite_ranking", False)),
                },
                intermediate={
                    "selected": selected,
                    "decisions": decisions,
                    "limited_candidates": limited_candidates,
                    "topics": topics,
                    "selection_rationale": selection_rationale_rows(limited_candidates, decisions),
                },
            ):
                return None
            if not selected:
                orchestrator.debug.set_metric(f"brief.{name}.status", "failed")
                raise RuntimeError(
                    f"{name}: selected 0 articles from {len(limited_candidates)} scored candidates; "
                    "aborting before final synthesis. Check output/diagnostics for scorer failure artifacts."
                )

            article_retriever = ArticleRetriever(
                orchestrator.config.user_agent,
                filtering.article_text_max_chars,
                http_cache=orchestrator.http_cache,
                cache_fresh_seconds=orchestrator.config.cache.http_fresh_seconds,
                debug=orchestrator.debug,
            )
            enricher = SimpleEnricher(
                orchestrator.config,
                http_cache=orchestrator.http_cache,
                debug=orchestrator.debug,
            )
            for article in selected:
                orchestrator.debug.log(
                    "article",
                    "selected",
                    brief=name,
                    score=article.decision.score,
                    topic=article.decision.topic,
                    selection_reason=article.selection_reason_code or article.decision.selection_reason_code,
                    rank_score=article.selection_rank_score or article.decision.selection_rank_score,
                    rank_mode=article.selection_rank_mode or article.decision.selection_rank_mode,
                    source=article.candidate.source,
                    title=article.candidate.title,
                )
            with orchestrator.debug.span(f"brief.{name}.article_fetch"):
                orchestrator._populate_article_texts(name, selected, article_retriever, run_warnings)
            orchestrator._record_article_fetch_metrics(name, selected)
            extraction_status_counts: Dict[str, int] = {}
            for item in selected:
                status = str(item.extraction_status or "unknown")
                extraction_status_counts[status] = extraction_status_counts.get(status, 0) + 1
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="article_fetch",
                summary={
                    "selected": len(selected),
                    "article_ids": [article.candidate.id for article in selected],
                    "extraction_status_counts": extraction_status_counts,
                },
                intermediate={
                    "selected": selected,
                },
            ):
                return None

            with orchestrator.debug.span(f"brief.{name}.enrichment"):
                enricher.enrich_many(selected, max_workers=orchestrator.config.runtime.max_enrichment_workers)
            orchestrator._record_enrichment_metrics(name, selected)
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="enrichment",
                summary={
                    "selected": len(selected),
                    "enrichment_needed": sum(1 for article in selected if article.enrichment_needed),
                    "context_sources": sum(len(article.context_sources) for article in selected),
                    "wikipedia_results": sum(len(article.wikipedia_context) for article in selected),
                    "past_news_results": sum(len(article.past_news_context) for article in selected),
                },
                intermediate={
                    "selected": selected,
                },
            ):
                return None

            evidence_packet: Dict[str, Any] = {}
            include_enrichment_context = bool(getattr(orchestrator.config.enrichment, "enabled", True))
            evidence_config, delta_config, analysis_rollout_meta = resolve_analysis_stage_configs(
                orchestrator.config.analysis,
                name,
            )
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.rollout.enabled",
                bool(analysis_rollout_meta.get("rollout_enabled", False)),
            )
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.rollout.profile",
                str(analysis_rollout_meta.get("rollout_profile", "")),
            )
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.evidence.enabled_requested",
                bool(analysis_rollout_meta.get("evidence_requested_enabled", False)),
            )
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.evidence.enabled",
                bool(evidence_config.enabled),
            )
            if evidence_config.enabled:
                evidence_client = (
                    orchestrator.summary_ai_client
                    if evidence_config.model_role == "summary"
                    else orchestrator.final_ai_client
                )
                if (
                    evidence_client is orchestrator.final_ai_client
                    and orchestrator.summary_ai_client is not orchestrator.final_ai_client
                ):
                    # Free scorer VRAM first when distillation is configured to run on the writer model.
                    orchestrator.summary_ai_client.unload()
                evidence_distiller = EvidenceDistiller(
                    evidence_client,
                    evidence_config,
                    include_enrichment_context=include_enrichment_context,
                    debug=orchestrator.debug,
                    cache=orchestrator.synth_cache,
                    cache_ttl_seconds=evidence_config.cache_ttl_seconds,
                )
                with orchestrator.debug.span(f"brief.{name}.evidence_distillation"):
                    try:
                        evidence_packet = evidence_distiller.distill(
                            selected,
                            orchestrator.config.user_memory,
                            topics,
                            prior_reports,
                            brief_goal,
                            date,
                            brief_name=name,
                        )
                    except Exception as exc:
                        warning = (
                            f"{name}: evidence distillation failed ({type(exc).__name__}): {exc}; "
                            "continuing without evidence packet."
                        )
                        run_warnings.append(warning)
                        orchestrator.debug.log(
                            "analysis.evidence",
                            "failed",
                            brief=name,
                            error=type(exc).__name__,
                        )
                        evidence_packet = {}
                run_warnings.extend(evidence_distiller.warnings)
                evidence_pressure_warnings = sum(
                    1
                    for warning in evidence_distiller.warnings
                    if "budget" in str(warning).lower() or "dropped lower-ranked article" in str(warning).lower()
                )
                orchestrator.debug.set_metric(
                    f"brief.{name}.analysis.evidence.prompt_pressure_warnings",
                    int(evidence_pressure_warnings),
                )
            else:
                reason = str(analysis_rollout_meta.get("evidence_skip_reason", "disabled"))
                orchestrator.debug.set_metric(f"brief.{name}.analysis.evidence.skipped_reason.{reason}", 1)
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.evidence.story_clusters",
                len(evidence_packet.get("story_clusters", [])) if evidence_packet else 0,
            )
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.evidence.reader_qa",
                len(evidence_packet.get("reader_qa", [])) if evidence_packet else 0,
            )
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="evidence_distillation",
                summary={
                    "enabled": bool(evidence_config.enabled),
                    "requested_enabled": bool(analysis_rollout_meta.get("evidence_requested_enabled", False)),
                    "rollout_profile": str(analysis_rollout_meta.get("rollout_profile", "")),
                    "story_clusters": len(evidence_packet.get("story_clusters", [])) if evidence_packet else 0,
                    "reader_qa": len(evidence_packet.get("reader_qa", [])) if evidence_packet else 0,
                    "global_watch_signals": len(evidence_packet.get("global_watch_signals", [])) if evidence_packet else 0,
                },
                intermediate={
                    "selected": selected,
                    "evidence_packet": evidence_packet,
                    "topics": topics,
                    "prior_reports": prior_reports,
                },
            ):
                return None

            delta_packet: Dict[str, Any] = {}
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.delta.enabled_requested",
                bool(analysis_rollout_meta.get("delta_requested_enabled", False)),
            )
            orchestrator.debug.set_metric(f"brief.{name}.analysis.delta.enabled", bool(delta_config.enabled))
            if delta_config.enabled:
                delta_client = (
                    orchestrator.summary_ai_client
                    if delta_config.model_role == "summary"
                    else orchestrator.final_ai_client
                )
                if delta_client is orchestrator.final_ai_client and orchestrator.summary_ai_client is not orchestrator.final_ai_client:
                    # Keep only one model resident when possible before running optional delta extraction.
                    orchestrator.summary_ai_client.unload()
                if (
                    delta_client is orchestrator.summary_ai_client
                    and orchestrator.summary_ai_client is not orchestrator.final_ai_client
                    and evidence_config.enabled
                    and evidence_config.model_role == "final"
                ):
                    # Distillation on final may have loaded the writer model; unload before scorer role.
                    orchestrator.final_ai_client.unload()
                delta_extractor = DeltaExtractor(
                    delta_client,
                    delta_config,
                    debug=orchestrator.debug,
                    cache=orchestrator.synth_cache,
                    cache_ttl_seconds=delta_config.cache_ttl_seconds,
                )
                with orchestrator.debug.span(f"brief.{name}.delta_extraction"):
                    try:
                        delta_packet = delta_extractor.extract(
                            selected,
                            orchestrator.config.user_memory,
                            topics,
                            prior_reports,
                            brief_goal,
                            date,
                            evidence_packet=evidence_packet,
                            brief_name=name,
                        )
                    except Exception as exc:
                        warning = (
                            f"{name}: delta extraction failed ({type(exc).__name__}): {exc}; "
                            "continuing without delta packet."
                        )
                        run_warnings.append(warning)
                        orchestrator.debug.log(
                            "analysis.delta",
                            "failed",
                            brief=name,
                            error=type(exc).__name__,
                        )
                        delta_packet = {}
                run_warnings.extend(delta_extractor.warnings)
                delta_pressure_warnings = sum(
                    1
                    for warning in delta_extractor.warnings
                    if "budget" in str(warning).lower() or "dropped lower-ranked article" in str(warning).lower()
                )
                orchestrator.debug.set_metric(
                    f"brief.{name}.analysis.delta.prompt_pressure_warnings",
                    int(delta_pressure_warnings),
                )
            else:
                reason = str(analysis_rollout_meta.get("delta_skip_reason", "disabled"))
                orchestrator.debug.set_metric(f"brief.{name}.analysis.delta.skipped_reason.{reason}", 1)

            deterministic_delta_packet = build_deterministic_delta_scaffold(
                selected,
                prior_reports,
                max_prior_reports=delta_config.max_prior_reports,
            )
            if deterministic_delta_packet and not delta_packet:
                delta_packet = deterministic_delta_packet
                scaffold_reason = "disabled"
                if delta_config.enabled:
                    scaffold_reason = "fallback_after_empty_or_failed_delta"
                orchestrator.debug.log(
                    "analysis.delta",
                    "deterministic_scaffold",
                    brief=name,
                    reason=scaffold_reason,
                    new=len(delta_packet.get("new", [])),
                    unchanged=len(delta_packet.get("unchanged_but_important", [])),
                )
                orchestrator.debug.set_metric(f"brief.{name}.analysis.delta.scaffold_used", True)
                orchestrator.debug.set_metric(
                    f"brief.{name}.analysis.delta.scaffold_reason.{scaffold_reason}",
                    1,
                )
            else:
                orchestrator.debug.set_metric(f"brief.{name}.analysis.delta.scaffold_used", False)
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.delta.new_items",
                len(delta_packet.get("new", [])) if delta_packet else 0,
            )
            orchestrator.debug.set_metric(
                f"brief.{name}.analysis.delta.evidence_gaps",
                len(delta_packet.get("evidence_gaps", [])) if delta_packet else 0,
            )
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="delta_extraction",
                summary={
                    "enabled": bool(delta_config.enabled),
                    "requested_enabled": bool(analysis_rollout_meta.get("delta_requested_enabled", False)),
                    "rollout_profile": str(analysis_rollout_meta.get("rollout_profile", "")),
                    "new_items": len(delta_packet.get("new", [])) if delta_packet else 0,
                    "escalated_items": len(delta_packet.get("escalated", [])) if delta_packet else 0,
                    "reframed_items": len(delta_packet.get("reframed", [])) if delta_packet else 0,
                    "evidence_gaps": len(delta_packet.get("evidence_gaps", [])) if delta_packet else 0,
                    "deterministic_scaffold": bool(delta_packet.get("deterministic_scaffold")) if delta_packet else False,
                },
                intermediate={
                    "selected": selected,
                    "delta_packet": delta_packet,
                    "evidence_packet": evidence_packet,
                    "prior_reports": prior_reports,
                    "topics": topics,
                },
            ):
                return None

            if orchestrator.summary_ai_client is not orchestrator.final_ai_client:
                orchestrator.summary_ai_client.unload()

            brief_generator = BriefGenerator(
                orchestrator.final_ai_client,
                orchestrator.config.enrichment.max_context_chars_per_article,
                input_token_limit=orchestrator.config.ai_final.max_input_tokens,
                max_new_tokens=orchestrator.config.ai_final.max_new_tokens,
                include_enrichment_context=include_enrichment_context,
                debug=orchestrator.debug,
            )
            with orchestrator.debug.span(f"brief.{name}.final_brief"):
                brief = brief_generator.generate(
                    selected,
                    orchestrator.config.user_memory,
                    topics,
                    prior_reports,
                    brief_goal,
                    date,
                    evidence_packet=evidence_packet,
                    delta_packet=delta_packet,
                    brief_name=name,
                )
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="final_brief",
                summary={
                    "title": str(brief.get("title", "")),
                    "topic_reports": len(brief.get("topic_reports", [])),
                    "sections": len(brief.get("sections", [])),
                    "knowns": len(brief.get("knowns", [])),
                    "unknowns": len(brief.get("unknowns", [])),
                    "watch_signals": len(brief.get("watch_signals", [])),
                },
                intermediate={
                    "brief": brief,
                    "selected": selected,
                    "topics": topics,
                    "prior_reports": prior_reports,
                    "evidence_packet": evidence_packet,
                    "delta_packet": delta_packet,
                    "brief_goal": brief_goal,
                    "date": date,
                    "brief_name": name,
                },
            ):
                return None
            orchestrator.final_ai_client.unload()
            run_warnings.extend(enricher.warnings)
            run_warnings.extend(brief_generator.warnings)
            brief["metadata"] = brief_metadata(
                date=date,
                model=f"{orchestrator.config.ai_summary.backend}:{orchestrator.config.ai_summary.effective_model_label} -> "
                f"{orchestrator.config.ai_final.backend}:{orchestrator.config.ai_final.effective_model_label}",
                candidate_count=len(unique_candidates),
                selected_count=len(selected),
                topics=[topic.name for topic in topics],
                prior_reports_count=len(prior_reports),
                brief_name=name,
                warnings=run_warnings,
            )
            brief["metadata"]["selection_reason_codes"] = selection_counts
            brief["metadata"]["composite_ranking_enabled"] = bool(
                getattr(filtering, "use_multifactor_composite_ranking", False)
            )
            brief["metadata"]["analysis_rollout"] = {
                "enabled": bool(analysis_rollout_meta.get("rollout_enabled", False)),
                "profile": str(analysis_rollout_meta.get("rollout_profile", "")),
                "mode": str(analysis_rollout_meta.get("rollout_mode", name)),
                "evidence_requested_enabled": bool(analysis_rollout_meta.get("evidence_requested_enabled", False)),
                "evidence_enabled": bool(evidence_config.enabled),
                "delta_requested_enabled": bool(analysis_rollout_meta.get("delta_requested_enabled", False)),
                "delta_enabled": bool(delta_config.enabled),
            }
            if evidence_packet:
                brief.setdefault("analysis", {})
                brief["analysis"]["evidence_packet"] = evidence_packet
                brief["analysis"]["evidence_model_role"] = evidence_config.model_role
            if delta_packet:
                brief.setdefault("analysis", {})
                brief["analysis"]["delta_packet"] = delta_packet
                brief["analysis"]["delta_model_role"] = (
                    "deterministic_scaffold"
                    if bool(delta_packet.get("deterministic_scaffold"))
                    else delta_config.model_role
                )

            output_dir = Path(orchestrator.config.output_dir)
            markdown_path = output_dir / f"{date}_{output_suffix}_brief.md"
            json_path = output_dir / f"{date}_{output_suffix}_brief.json"
            with orchestrator.debug.span(f"brief.{name}.write_output"):
                write_markdown(markdown_path, brief)
                write_json(json_path, brief)
            _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="write_output",
                summary={
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                    "candidate_count": len(unique_candidates),
                    "selected_count": len(selected),
                },
                intermediate={
                    "brief": brief,
                    "selected": selected,
                },
            )
            orchestrator.warnings.extend(run_warnings)
            orchestrator.debug.set_metric(f"brief.{name}.warnings", len(run_warnings))
            orchestrator.debug.set_metric(f"brief.{name}.status", "completed")
            orchestrator.debug.log("brief.run", "complete", name=name, markdown=markdown_path, json=json_path, warnings=len(run_warnings))

            return BriefOutput(
                name=name,
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                candidate_count=len(unique_candidates),
                selected_count=len(selected),
                warnings=run_warnings,
            )
        except Exception as exc:
            orchestrator.debug.set_metric(f"brief.{name}.status", "failed")
            orchestrator.debug.set_metric(f"brief.{name}.error", f"{type(exc).__name__}: {exc}")
            raise
