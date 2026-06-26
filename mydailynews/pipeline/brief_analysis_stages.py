from __future__ import annotations

from typing import Any, Dict, List

from mydailynews.analysis.delta import DeltaExtractor
from mydailynews.analysis.evidence import EvidenceDistiller
from mydailynews.pipeline.brief_stages import _report_phase
from mydailynews.analysis.deterministic_delta import build_deterministic_delta_scaffold
from mydailynews.app.models import DeltaExtractionConfig, EvidenceDistillationConfig, PriorReport, SelectedArticle, TopicConfig
from mydailynews.pipeline.stage_results import DeltaStageResult, EvidenceStageResult
from mydailynews.story_grouping.models import StoryGroup
from mydailynews.common.warnings import extend_warnings, prompt_pressure_warning_count


def _run_evidence_stage(
    orchestrator,
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    topics: List[TopicConfig],
    prior_reports: List[PriorReport],
    brief_goal: str,
    date: str,
    include_enrichment_context: bool,
    evidence_config: EvidenceDistillationConfig,
    analysis_rollout_meta: Dict[str, Any],
    story_groups: List[StoryGroup] | None = None,
) -> EvidenceStageResult:
    warnings: List[str] = []
    evidence_packet: Dict[str, Any] = {}
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.rollout.enabled",
        bool(analysis_rollout_meta.get("rollout_enabled", False)),
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.rollout.profile",
        str(analysis_rollout_meta.get("rollout_profile", "")),
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.evidence.enabled_requested",
        bool(analysis_rollout_meta.get("evidence_requested_enabled", False)),
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.evidence.enabled",
        bool(evidence_config.enabled),
    )
    if evidence_config.enabled:
        _report_phase(orchestrator, f"Running {brief_name} evidence analysis...")
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
        with orchestrator.debug.span(f"brief.{brief_name}.evidence_distillation"):
            try:
                evidence_packet = evidence_distiller.distill(
                    selected,
                    orchestrator.config.user_memory,
                    topics,
                    prior_reports,
                    brief_goal,
                    date,
                    brief_name=brief_name,
                    story_groups=story_groups,
                )
            except Exception as exc:
                warning = (
                    f"{brief_name}: evidence distillation failed ({type(exc).__name__}): {exc}; "
                    "continuing without evidence packet."
                )
                warnings.append(warning)
                orchestrator.debug.log(
                    "analysis.evidence",
                    "failed",
                    brief=brief_name,
                    error=type(exc).__name__,
                )
                evidence_packet = {}
        extend_warnings(warnings, evidence_distiller.warnings)
        evidence_pressure_warnings = prompt_pressure_warning_count(evidence_distiller.warnings)
        orchestrator.debug.set_metric(
            f"brief.{brief_name}.analysis.evidence.prompt_pressure_warnings",
            int(evidence_pressure_warnings),
        )
        orchestrator.debug.set_metric(
            f"brief.{brief_name}.analysis.evidence.shared_grouping_used",
            story_groups is not None,
        )
        orchestrator.debug.set_metric(
            f"brief.{brief_name}.analysis.evidence.group_boundary_warnings",
            int(getattr(evidence_distiller, "group_boundary_warning_count", 0)),
        )
    else:
        reason = str(analysis_rollout_meta.get("evidence_skip_reason", "disabled"))
        orchestrator.debug.set_metric(f"brief.{brief_name}.analysis.evidence.skipped_reason.{reason}", 1)
        orchestrator.debug.set_metric(f"brief.{brief_name}.analysis.evidence.shared_grouping_used", False)
        orchestrator.debug.set_metric(f"brief.{brief_name}.analysis.evidence.group_boundary_warnings", 0)
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.evidence.story_clusters",
        len(evidence_packet.get("story_clusters", [])) if evidence_packet else 0,
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.evidence.reader_qa",
        len(evidence_packet.get("reader_qa", [])) if evidence_packet else 0,
    )
    return EvidenceStageResult(evidence_packet=evidence_packet, warnings=warnings)


def _run_delta_stage(
    orchestrator,
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    topics: List[TopicConfig],
    prior_reports: List[PriorReport],
    brief_goal: str,
    date: str,
    evidence_packet: Dict[str, Any],
    evidence_config: EvidenceDistillationConfig,
    delta_config: DeltaExtractionConfig,
    analysis_rollout_meta: Dict[str, Any],
) -> DeltaStageResult:
    warnings: List[str] = []
    delta_packet: Dict[str, Any] = {}
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.delta.enabled_requested",
        bool(analysis_rollout_meta.get("delta_requested_enabled", False)),
    )
    orchestrator.debug.set_metric(f"brief.{brief_name}.analysis.delta.enabled", bool(delta_config.enabled))
    if delta_config.enabled:
        _report_phase(orchestrator, f"Running {brief_name} delta analysis...")
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
        with orchestrator.debug.span(f"brief.{brief_name}.delta_extraction"):
            try:
                delta_packet = delta_extractor.extract(
                    selected,
                    orchestrator.config.user_memory,
                    topics,
                    prior_reports,
                    brief_goal,
                    date,
                    evidence_packet=evidence_packet,
                    brief_name=brief_name,
                )
            except Exception as exc:
                warning = (
                    f"{brief_name}: delta extraction failed ({type(exc).__name__}): {exc}; "
                    "continuing without delta packet."
                )
                warnings.append(warning)
                orchestrator.debug.log(
                    "analysis.delta",
                    "failed",
                    brief=brief_name,
                    error=type(exc).__name__,
                )
                delta_packet = {}
        extend_warnings(warnings, delta_extractor.warnings)
        delta_pressure_warnings = prompt_pressure_warning_count(delta_extractor.warnings)
        orchestrator.debug.set_metric(
            f"brief.{brief_name}.analysis.delta.prompt_pressure_warnings",
            int(delta_pressure_warnings),
        )
    else:
        reason = str(analysis_rollout_meta.get("delta_skip_reason", "disabled"))
        orchestrator.debug.set_metric(f"brief.{brief_name}.analysis.delta.skipped_reason.{reason}", 1)

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
            brief=brief_name,
            reason=scaffold_reason,
            new=len(delta_packet.get("new", [])),
            unchanged=len(delta_packet.get("unchanged_but_important", [])),
        )
        orchestrator.debug.set_metric(f"brief.{brief_name}.analysis.delta.scaffold_used", True)
        orchestrator.debug.set_metric(
            f"brief.{brief_name}.analysis.delta.scaffold_reason.{scaffold_reason}",
            1,
        )
    else:
        orchestrator.debug.set_metric(f"brief.{brief_name}.analysis.delta.scaffold_used", False)
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.delta.new_items",
        len(delta_packet.get("new", [])) if delta_packet else 0,
    )
    orchestrator.debug.set_metric(
        f"brief.{brief_name}.analysis.delta.evidence_gaps",
        len(delta_packet.get("evidence_gaps", [])) if delta_packet else 0,
    )
    return DeltaStageResult(delta_packet=delta_packet, warnings=warnings)
