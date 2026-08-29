from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

from mydailynews.analysis.policy_filter import (
    filter_delta_packet_for_articles,
    filter_evidence_packet_for_articles,
    filter_prior_reports_for_articles,
)
from mydailynews.analysis.rollout import resolve_analysis_stage_configs
from mydailynews.briefing.generator import BriefGenerator, brief_metadata, no_material_changes_brief
from mydailynews.pipeline.brief_analysis_stages import _run_delta_stage, _run_evidence_stage
from mydailynews.pipeline.brief_stages import (
    _checkpoint_stage,
    _fetch_articles_stage,
    _limit_headlines_stage,
    _prepare_candidates_stage,
    _report_phase,
    _score_headlines_stage,
    _select_articles_stage,
    _story_grouping_stage,
)
from mydailynews.pipeline.handoff import write_brief_handoff
from mydailynews.domain.headline_selection import selection_rationale_rows
from mydailynews.app.models import BriefOutput, HeadlineDecision, NewsCandidate, PriorReport, RunSourceSnapshot, TopicConfig
from mydailynews.briefing.output import write_json, write_markdown
from mydailynews.common.warnings import extend_warnings
from mydailynews.memory.config import memory_enabled, memory_state_dir
from mydailynews.memory.coverage import CoverageMemoryStore
from mydailynews.memory.learned_preferences import LearnedPreferencesStore
from mydailynews.memory.recall import (
    apply_delta_signals_to_selected,
    build_recall_packet,
    partition_selected_for_brief,
    recall_packet_for_selected,
    save_recall_packet,
    selected_articles_represented_in_brief,
)
from mydailynews.memory.story_index import StoryIndexStore
from mydailynews.memory.story_ledger import StoryLedgerStore


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
        run_warnings_promoted = False
        memory_config = getattr(orchestrator.config, "memory", None)
        memory_is_enabled = memory_enabled(memory_config)
        recall_prompt_enabled = memory_is_enabled and bool(getattr(memory_config, "recall_prompt_enabled", True))
        save_recall_packets = memory_is_enabled and bool(getattr(memory_config, "save_recall_packets", True))
        memory_dir = memory_state_dir(orchestrator.config) if memory_is_enabled else None
        coverage_store = CoverageMemoryStore.from_state_dir(memory_dir) if memory_dir is not None else None
        story_index_store = StoryIndexStore.from_state_dir(memory_dir) if memory_dir is not None else None
        story_ledger_store = StoryLedgerStore.from_state_dir(memory_dir) if memory_dir is not None else None
        learned_preferences = (
            LearnedPreferencesStore.from_state_dir(memory_dir).read()
            if memory_dir is not None
            else None
        )
        recall_packet: dict = {}
        prompt_recall_packet: dict = {}
        recall_packet_path = ""

        def _promote_run_warnings() -> None:
            nonlocal run_warnings_promoted
            if run_warnings_promoted:
                return
            extend_warnings(orchestrator.warnings, run_warnings)
            run_warnings_promoted = True

        _report_phase(orchestrator, f"Generating {name} brief...")
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
            candidate_result = _prepare_candidates_stage(
                orchestrator,
                brief_name=name,
                topics=topics,
                filtering=filtering,
                prior_reports=prior_reports,
                since=since,
                snapshot=snapshot,
            )
            extend_warnings(run_warnings, candidate_result.warnings)
            unique_candidates = candidate_result.unique_candidates
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="candidate_prepare",
                summary={
                    "raw_candidates": candidate_result.raw_count,
                    "rss_candidates": candidate_result.rss_count,
                    "topic_candidates": candidate_result.topic_count,
                    "unique_candidates": len(unique_candidates),
                    "unique_candidate_ids": [candidate.id for candidate in unique_candidates],
                },
                next_stage_input={
                    "rss_candidates": candidate_result.rss_candidates,
                    "topic_candidates": candidate_result.topic_candidates,
                    "unique_candidates": unique_candidates,
                    "prior_reports": prior_reports,
                    "topics": topics,
                    "filtering": filtering,
                    "brief_goal": brief_goal,
                    "since": since,
                },
            ):
                return None

            if not unique_candidates:
                run_warnings.append(f"{name}: No live headline candidates were fetched.")

            headline_limit = _limit_headlines_stage(
                orchestrator,
                brief_name=name,
                unique_candidates=unique_candidates,
                topics=topics,
                filtering=filtering,
                since=since,
                limited_candidates_override=limited_candidates_override,
            )
            limited_candidates = headline_limit.limited_candidates
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="headline_limit",
                summary={
                    "limited_candidates": len(limited_candidates),
                    "limited_candidate_ids": [candidate.id for candidate in limited_candidates],
                    "limited_sources": headline_limit.limited_sources,
                },
                next_stage_input={
                    "limited_candidates": limited_candidates,
                    "unique_candidates": unique_candidates,
                    "topics": topics,
                    "filtering": filtering,
                    "brief_goal": brief_goal,
                    "shared_decisions": shared_decisions or {},
                    "since": since,
                },
            ):
                return None

            headline_scoring = _score_headlines_stage(
                orchestrator,
                brief_name=name,
                limited_candidates=limited_candidates,
                topics=topics,
                filtering=filtering,
                brief_goal=brief_goal,
                shared_decisions=shared_decisions,
            )
            extend_warnings(run_warnings, headline_scoring.warnings)
            decisions = headline_scoring.decisions
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="headline_decisions",
                summary={
                    "decisions": len(decisions),
                    "decision_ids": list(decisions.keys()),
                    "missing_decisions": max(0, len(limited_candidates) - len(decisions)),
                },
                next_stage_input={
                    "decisions": decisions,
                    "limited_candidates": limited_candidates,
                    "topics": topics,
                    "filtering": filtering,
                    "prior_reports": prior_reports,
                    "brief_goal": brief_goal,
                    "include_enrichment_context": False,
                },
            ):
                return None

            include_enrichment_context = False
            selection_result = _select_articles_stage(
                orchestrator,
                brief_name=name,
                limited_candidates=limited_candidates,
                decisions=decisions,
                topics=topics,
                filtering=filtering,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                include_enrichment_context=include_enrichment_context,
                coverage_store=coverage_store,
                story_index_store=story_index_store,
                story_ledger_store=story_ledger_store,
                learned_preferences=learned_preferences,
            )
            extend_warnings(run_warnings, selection_result.warnings)
            selected = selection_result.selected
            selection_counts = selection_result.selection_counts
            selected_reason_counts = selection_counts.get("selected", {})
            skipped_reason_counts = selection_counts.get("skipped", {})
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="headline_select",
                summary={
                    "selected": len(selected),
                    "selected_article_ids": [article.candidate.id for article in selected],
                    "selected_sources": selection_result.selected_sources,
                    "selected_reason_codes": selected_reason_counts,
                    "skipped_reason_codes": skipped_reason_counts,
                    "composite_ranking_enabled": bool(getattr(filtering, "use_multifactor_composite_ranking", False)),
                    "memory": selection_result.memory_summary,
                },
                next_stage_input={
                    "selected": selected,
                    "decisions": decisions,
                    "limited_candidates": limited_candidates,
                    "topics": topics,
                    "filtering": filtering,
                    "prior_reports": prior_reports,
                    "brief_goal": brief_goal,
                    "include_enrichment_context": include_enrichment_context,
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

            article_fetch_result = _fetch_articles_stage(
                orchestrator,
                brief_name=name,
                selected=selected,
                filtering=filtering,
            )
            extend_warnings(run_warnings, article_fetch_result.warnings)
            selected = article_fetch_result.selected
            evidence_config, delta_config, analysis_rollout_meta = resolve_analysis_stage_configs(
                orchestrator.config.analysis,
                name,
            )
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="article_fetch",
                summary={
                    "selected": len(selected),
                    "article_ids": [article.candidate.id for article in selected],
                    "extraction_status_counts": article_fetch_result.status_counts,
                },
                next_stage_input={
                    "selected": selected,
                    "filtering": filtering,
                    "include_enrichment_context": include_enrichment_context,
                    "evidence_config": evidence_config,
                    "delta_config": delta_config,
                    "analysis_rollout_meta": analysis_rollout_meta,
                },
            ):
                return None

            story_grouping_result = _story_grouping_stage(
                orchestrator,
                brief_name=name,
                selected=selected,
                include_enrichment_context=include_enrichment_context,
                evidence_config=evidence_config,
                date=date,
            )
            extend_warnings(run_warnings, story_grouping_result.warnings)
            story_groups = story_grouping_result.story_groups
            shared_story_grouping_ran = bool(story_grouping_result.artifact.get("enabled", False))
            shared_story_groups = story_groups if shared_story_grouping_ran else None
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="story_grouping",
                summary={
                    "enabled": bool(story_grouping_result.artifact.get("enabled", False)),
                    "status": str(story_grouping_result.artifact.get("status", "")),
                    "skipped_reason": str(story_grouping_result.artifact.get("skipped_reason", "")),
                    "shared_grouping_ran": shared_story_grouping_ran,
                    "selected": len(selected),
                    "story_groups": len(story_groups),
                    "fallback_groups": len(story_grouping_result.artifact.get("fallback_groups", [])),
                    "split_requests": bool(story_grouping_result.artifact.get("split_requests", False)),
                    "cache_hit": bool(story_grouping_result.artifact.get("cache_hit", False)),
                    "story_grouping": story_grouping_result.artifact,
                },
                next_stage_input={
                    "selected": selected,
                    "story_groups": story_groups,
                    "story_grouping": story_grouping_result.artifact,
                    "topics": topics,
                    "prior_reports": prior_reports,
                    "brief_goal": brief_goal,
                    "include_enrichment_context": include_enrichment_context,
                    "evidence_config": evidence_config,
                    "delta_config": delta_config,
                    "analysis_rollout_meta": analysis_rollout_meta,
                },
            ):
                return None

            evidence_result = _run_evidence_stage(
                orchestrator,
                brief_name=name,
                selected=selected,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                include_enrichment_context=include_enrichment_context,
                evidence_config=evidence_config,
                analysis_rollout_meta=analysis_rollout_meta,
                story_groups=shared_story_groups,
            )
            extend_warnings(run_warnings, evidence_result.warnings)
            evidence_packet = evidence_result.evidence_packet
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
                next_stage_input={
                    "selected": selected,
                    "evidence_packet": evidence_packet,
                    "topics": topics,
                    "prior_reports": prior_reports,
                    "brief_goal": brief_goal,
                    "include_enrichment_context": include_enrichment_context,
                    "evidence_config": evidence_config,
                    "delta_config": delta_config,
                    "analysis_rollout_meta": analysis_rollout_meta,
                    "story_groups": story_groups,
                    "shared_story_grouping_ran": shared_story_grouping_ran,
                    "story_grouping": story_grouping_result.artifact,
                },
            ):
                return None

            delta_result = _run_delta_stage(
                orchestrator,
                brief_name=name,
                selected=selected,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                evidence_packet=evidence_packet,
                evidence_config=evidence_config,
                delta_config=delta_config,
                analysis_rollout_meta=analysis_rollout_meta,
                story_groups=story_groups,
                story_index_store=story_index_store,
                story_ledger_store=story_ledger_store,
                coverage_store=coverage_store,
                coverage_window_days=int(getattr(memory_config, "coverage_window_days", 10)),
            )
            extend_warnings(run_warnings, delta_result.warnings)
            delta_packet = delta_result.delta_packet
            apply_delta_signals_to_selected(selected=selected, delta_packet=delta_packet)
            brief_selected, delta_omitted = partition_selected_for_brief(
                selected=selected,
                delta_packet=delta_packet,
            )
            allowed_article_ids = [article.candidate.id for article in brief_selected]
            brief_delta_packet = filter_delta_packet_for_articles(
                delta_packet,
                allowed_article_ids=allowed_article_ids,
                omitted_count=len(delta_omitted),
            )
            brief_evidence_packet = filter_evidence_packet_for_articles(
                evidence_packet,
                allowed_article_ids=allowed_article_ids,
                omitted_count=len(delta_omitted),
            )
            brief_prior_reports = filter_prior_reports_for_articles(
                prior_reports,
                selected=brief_selected,
                omitted_count=len(delta_omitted),
            )
            if memory_is_enabled:
                if recall_prompt_enabled or save_recall_packets:
                    recall_packet = build_recall_packet(
                        date=date,
                        brief_name=name,
                        candidates=limited_candidates,
                        decisions=decisions,
                    )
                prompt_recall_packet = (
                    recall_packet_for_selected(recall_packet, brief_selected)
                    if recall_prompt_enabled
                    else {}
                )
                if save_recall_packets and recall_packet and memory_dir is not None:
                    try:
                        recall_packet_path = str(
                            save_recall_packet(
                                state_dir=memory_dir,
                                date=date,
                                brief_name=name,
                                recall_packet=recall_packet,
                            )
                        )
                    except Exception as exc:
                        run_warnings.append(
                            f"{name}: memory recall packet write failed ({type(exc).__name__}: {exc})."
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
                    "omitted_as_unchanged": len(delta_omitted),
                    "recall_guidance": len(prompt_recall_packet.get("coverage_guidance", [])) if prompt_recall_packet else 0,
                    "recall_packet_saved": bool(recall_packet_path),
                },
                next_stage_input={
                    "selected": brief_selected,
                    "delta_packet": brief_delta_packet,
                    "evidence_packet": brief_evidence_packet,
                    "prior_reports": brief_prior_reports,
                    "topics": topics,
                    "brief_goal": brief_goal,
                    "include_enrichment_context": include_enrichment_context,
                    "evidence_config": evidence_config,
                    "delta_config": delta_config,
                    "analysis_rollout_meta": analysis_rollout_meta,
                    "recall_packet": prompt_recall_packet,
                },
            ):
                return None

            if orchestrator.summary_ai_client is not orchestrator.final_ai_client:
                orchestrator.summary_ai_client.unload()

            _report_phase(orchestrator, f"Writing {name} brief...")
            brief_generator = BriefGenerator(
                orchestrator.final_ai_client,
                orchestrator.config.enrichment.max_context_chars_per_article,
                input_token_limit=orchestrator.config.ai_final.max_input_tokens,
                max_new_tokens=orchestrator.config.ai_final.max_new_tokens,
                include_enrichment_context=include_enrichment_context,
                debug=orchestrator.debug,
            )
            with orchestrator.debug.span(f"brief.{name}.final_brief"):
                if brief_selected:
                    brief = brief_generator.generate(
                        brief_selected,
                        orchestrator.config.user_memory,
                        topics,
                        brief_prior_reports,
                        brief_goal,
                        date,
                        evidence_packet=brief_evidence_packet,
                        delta_packet=brief_delta_packet,
                        recall_packet=prompt_recall_packet,
                        brief_name=name,
                    )
                else:
                    brief = no_material_changes_brief(date)
                    orchestrator.debug.log(
                        "brief.ai",
                        "skipped_no_material_changes",
                        brief_name=name,
                        omitted=len(delta_omitted),
                    )
            rendered_selected = selected_articles_represented_in_brief(brief_selected, brief)
            output_dir = Path(orchestrator.config.output_dir)
            markdown_path = output_dir / f"{date}_{output_suffix}_brief.md"
            json_path = output_dir / f"{date}_{output_suffix}_brief.json"
            orchestrator.final_ai_client.unload()
            extend_warnings(run_warnings, brief_generator.warnings)
            brief["metadata"] = brief_metadata(
                date=date,
                model=f"{orchestrator.config.ai_summary.backend}:{orchestrator.config.ai_summary.effective_model_label} -> "
                f"{orchestrator.config.ai_final.backend}:{orchestrator.config.ai_final.effective_model_label}",
                candidate_count=len(unique_candidates),
                selected_count=len(rendered_selected),
                topics=[topic.name for topic in topics],
                prior_reports_count=len(brief_prior_reports),
                brief_name=name,
                warnings=run_warnings,
            )
            brief["metadata"]["selection_reason_codes"] = selection_counts
            brief["metadata"]["pre_delta_selected_count"] = len(selected)
            brief["metadata"]["delta_omitted_count"] = len(delta_omitted)
            brief["metadata"]["prompt_selected_count"] = len(brief_selected)
            brief["metadata"]["final_generation_skipped_no_material_changes"] = not bool(brief_selected)
            brief["metadata"]["composite_ranking_enabled"] = bool(
                getattr(filtering, "use_multifactor_composite_ranking", False)
            )
            brief["metadata"]["memory"] = {
                "enabled": memory_is_enabled,
                "recall_prompt_enabled": recall_prompt_enabled,
                "save_recall_packets": save_recall_packets,
                "coverage_guidance_used": bool(prompt_recall_packet.get("coverage_guidance", [])),
                "recall_packet_path": recall_packet_path,
                "recall_packet": prompt_recall_packet if memory_is_enabled else {},
                **(selection_result.memory_summary if memory_is_enabled else {}),
            }
            brief["metadata"]["analysis_rollout"] = {
                "enabled": bool(analysis_rollout_meta.get("rollout_enabled", False)),
                "profile": str(analysis_rollout_meta.get("rollout_profile", "")),
                "mode": str(analysis_rollout_meta.get("rollout_mode", name)),
                "evidence_requested_enabled": bool(analysis_rollout_meta.get("evidence_requested_enabled", False)),
                "evidence_enabled": bool(evidence_config.enabled),
                "delta_requested_enabled": bool(analysis_rollout_meta.get("delta_requested_enabled", False)),
                "delta_enabled": bool(delta_config.enabled),
            }
            if brief_evidence_packet:
                brief.setdefault("analysis", {})
                brief["analysis"]["evidence_packet"] = brief_evidence_packet
                brief["analysis"]["evidence_model_role"] = evidence_config.model_role
            if brief_delta_packet:
                brief.setdefault("analysis", {})
                brief["analysis"]["delta_packet"] = brief_delta_packet
                brief["analysis"]["delta_model_role"] = (
                    "deterministic_scaffold"
                    if bool(brief_delta_packet.get("deterministic_scaffold"))
                    else delta_config.model_role
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
                    "warnings": len(run_warnings),
                },
                next_stage_input={
                    "brief": brief,
                    "selected": rendered_selected,
                    "topics": topics,
                    "prior_reports": brief_prior_reports,
                    "evidence_packet": brief_evidence_packet,
                    "delta_packet": brief_delta_packet,
                    "brief_goal": brief_goal,
                    "brief_name": name,
                    "recall_packet": prompt_recall_packet,
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                },
            ):
                return None

            with orchestrator.debug.span(f"brief.{name}.write_output"):
                write_markdown(markdown_path, brief)
                write_json(json_path, brief)
            memory_write_summary = {}
            if memory_is_enabled and coverage_store is not None and story_index_store is not None:
                try:
                    story_records = story_index_store.update_selected(
                        selected=selected,
                        date=date,
                        story_groups=story_groups,
                        delta_packet=delta_packet,
                        stale_after_days=int(getattr(memory_config, "story_stale_after_days", 7)),
                        retention_days=int(getattr(memory_config, "story_retention_days", 30)),
                    )
                    ledger_records = (
                        story_ledger_store.update_selected(
                            selected=selected,
                            date=date,
                            visible_article_ids=[article.candidate.id for article in rendered_selected],
                            delta_packet=delta_packet,
                        )
                        if story_ledger_store is not None
                        else []
                    )
                    coverage_records = coverage_store.write_selected(
                        date=date,
                        brief_name=name,
                        selected=rendered_selected,
                    )
                    coverage_rows_pruned = coverage_store.prune(
                        as_of_date=date,
                        retention_days=int(getattr(memory_config, "coverage_retention_days", 30)),
                    )
                    memory_write_summary = {
                        "coverage_rows_written": len(coverage_records),
                        "coverage_rows_pruned": coverage_rows_pruned,
                        "story_index_records": len(story_records),
                        "story_index_stale_records": sum(1 for record in story_records if record.status == "stale"),
                        "story_ledger_records": len(ledger_records),
                        "story_ledger_source_facts": sum(len(record.facts) for record in ledger_records),
                    }
                    orchestrator.debug.set_metric(
                        f"brief.{name}.memory.coverage_rows_written",
                        len(coverage_records),
                    )
                    orchestrator.debug.set_metric(
                        f"brief.{name}.memory.coverage_rows_pruned",
                        coverage_rows_pruned,
                    )
                    orchestrator.debug.set_metric(
                        f"brief.{name}.memory.story_index_records",
                        len(story_records),
                    )
                    orchestrator.debug.set_metric(
                        f"brief.{name}.memory.story_index_stale_records",
                        memory_write_summary["story_index_stale_records"],
                    )
                    orchestrator.debug.set_metric(
                        f"brief.{name}.memory.story_ledger_records",
                        len(ledger_records),
                    )
                    orchestrator.debug.set_metric(
                        f"brief.{name}.memory.story_ledger_source_facts",
                        memory_write_summary["story_ledger_source_facts"],
                    )
                except Exception as exc:
                    memory_write_summary = {"write_error": f"{type(exc).__name__}: {exc}"}
                    run_warnings.append(f"{name}: memory writeback failed ({type(exc).__name__}: {exc}).")
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="write_output",
                summary={
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                    "candidate_count": len(unique_candidates),
                    "selected_count": len(rendered_selected),
                    "memory": memory_write_summary,
                },
                next_stage_input={
                    "brief": brief,
                    "selected": rendered_selected,
                    "memory": memory_write_summary,
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                },
            ):
                return None
            rendered_ids = {article.candidate.id for article in rendered_selected}
            handoff_story_groups = [
                replace(
                    group,
                    article_ids=[article_id for article_id in group.article_ids if article_id in rendered_ids],
                )
                for group in story_groups
                if any(article_id in rendered_ids for article_id in group.article_ids)
            ]
            handoff_written_path = write_brief_handoff(
                output_dir=output_dir,
                date=date,
                brief_name=name,
                json_path=json_path,
                markdown_path=markdown_path,
                topics=topics,
                prior_reports=brief_prior_reports,
                brief_goal=brief_goal,
                filtering=filtering,
                selected_articles=rendered_selected,
                story_groups=handoff_story_groups,
            )
            _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="write_handoff",
                summary={
                    "handoff_path": str(handoff_written_path),
                    "selected_count": len(rendered_selected),
                    "schema_version": "brief_handoff.v1",
                },
                next_stage_input={
                    "handoff_path": str(handoff_written_path),
                    "source_json_path": str(json_path),
                    "selected": rendered_selected,
                },
            )
            _promote_run_warnings()
            orchestrator.debug.set_metric(f"brief.{name}.warnings", len(run_warnings))
            orchestrator.debug.set_metric(f"brief.{name}.status", "completed")
            orchestrator.debug.log("brief.run", "complete", name=name, markdown=markdown_path, json=json_path, warnings=len(run_warnings))

            return BriefOutput(
                name=name,
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                candidate_count=len(unique_candidates),
                selected_count=len(rendered_selected),
                warnings=run_warnings,
                handoff_path=str(handoff_written_path),
            )
        except Exception as exc:
            orchestrator.debug.set_metric(f"brief.{name}.status", "failed")
            orchestrator.debug.set_metric(f"brief.{name}.error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _promote_run_warnings()
