from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, List

from mydailynews.analysis.rollout import resolve_analysis_stage_configs
from mydailynews.briefing.generator import BriefGenerator, brief_metadata
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
            )
            extend_warnings(run_warnings, delta_result.warnings)
            delta_packet = delta_result.delta_packet
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
                next_stage_input={
                    "selected": selected,
                    "delta_packet": delta_packet,
                    "evidence_packet": evidence_packet,
                    "prior_reports": prior_reports,
                    "topics": topics,
                    "brief_goal": brief_goal,
                    "include_enrichment_context": include_enrichment_context,
                    "evidence_config": evidence_config,
                    "delta_config": delta_config,
                    "analysis_rollout_meta": analysis_rollout_meta,
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
                    "selected": selected,
                    "topics": topics,
                    "prior_reports": prior_reports,
                    "evidence_packet": evidence_packet,
                    "delta_packet": delta_packet,
                    "brief_goal": brief_goal,
                    "brief_name": name,
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                },
            ):
                return None

            with orchestrator.debug.span(f"brief.{name}.write_output"):
                write_markdown(markdown_path, brief)
                write_json(json_path, brief)
            if _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="write_output",
                summary={
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                    "candidate_count": len(unique_candidates),
                    "selected_count": len(selected),
                },
                next_stage_input={
                    "brief": brief,
                    "selected": selected,
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                },
            ):
                return None
            handoff_written_path = write_brief_handoff(
                output_dir=output_dir,
                date=date,
                brief_name=name,
                json_path=json_path,
                markdown_path=markdown_path,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                filtering=filtering,
                selected_articles=selected,
            )
            _checkpoint_stage(
                orchestrator,
                brief_name=name,
                stage="write_handoff",
                summary={
                    "handoff_path": str(handoff_written_path),
                    "selected_count": len(selected),
                    "schema_version": "brief_handoff.v1",
                },
                next_stage_input={
                    "handoff_path": str(handoff_written_path),
                    "source_json_path": str(json_path),
                    "selected": selected,
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
                selected_count=len(selected),
                warnings=run_warnings,
                handoff_path=str(handoff_written_path),
            )
        except Exception as exc:
            orchestrator.debug.set_metric(f"brief.{name}.status", "failed")
            orchestrator.debug.set_metric(f"brief.{name}.error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _promote_run_warnings()
