from __future__ import annotations

from dataclasses import replace
from typing import List

from mydailynews.briefing.generator import FINAL_PROMPT_BUDGET_SAFETY_RATIO, BriefGenerator
from mydailynews.domain.candidate_annotations import set_selection_skip_annotation
from mydailynews.app.models import PriorReport, SelectedArticle, TopicConfig


def prune_selected_for_final_token_budget(
    orchestrator,
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    filtering,
    topics: List[TopicConfig],
    prior_reports: List[PriorReport],
    brief_goal: str,
    date: str,
    include_enrichment_context: bool,
    run_warnings: List[str],
) -> List[SelectedArticle]:
    if len(selected) <= 1:
        return selected

    final_input_limit = max(512, int(getattr(orchestrator.config.ai_final, "max_input_tokens", 0) or 0))
    prompt_budget_tokens = max(512, int(final_input_limit * FINAL_PROMPT_BUDGET_SAFETY_RATIO))
    base_context_chars = max(1, int(getattr(orchestrator.config.enrichment, "max_context_chars_per_article", 1600)))
    article_fetch_chars = max(1, int(getattr(filtering, "article_text_max_chars", base_context_chars)))
    context_chars = min(base_context_chars, article_fetch_chars)
    if include_enrichment_context:
        # The early estimate happens before enrichment exists, so reserve room for likely context notes/sources.
        context_chars += 512

    synthetic_text = ("estimated article context " * max(1, (context_chars // 26) + 1))[:context_chars]
    sorted_selected = sorted(
        selected,
        key=lambda item: (
            float(item.decision.score),
            float(item.selection_rank_score or item.decision.selection_rank_score or 0.0),
        ),
        reverse=True,
    )
    synthetic_by_id = {
        item.candidate.id: replace(item, article_text=synthetic_text, extraction_status="estimated")
        for item in sorted_selected
    }
    candidate_articles = [synthetic_by_id[item.candidate.id] for item in sorted_selected]
    estimator = BriefGenerator(
        orchestrator.final_ai_client,
        context_chars,
        input_token_limit=final_input_limit,
        max_new_tokens=orchestrator.config.ai_final.max_new_tokens,
        include_enrichment_context=include_enrichment_context,
    )
    estimated_tokens = 0
    dropped_ids: List[str] = []
    active_reports = prior_reports[:3]

    while len(candidate_articles) > 1:
        prompt = estimator._render_prompt(
            candidate_articles,
            context_chars,
            orchestrator.config.user_memory,
            topics,
            active_reports,
            brief_goal,
            date,
            evidence_packet={},
            delta_packet={},
        )
        estimated_tokens = estimator._estimate_final_input_tokens(prompt)
        if estimated_tokens <= prompt_budget_tokens:
            break
        dropped = candidate_articles.pop()
        dropped_ids.append(dropped.candidate.id)

    if not dropped_ids:
        orchestrator.debug.set_metric(f"brief.{brief_name}.selection.final_budget_estimated_tokens", int(estimated_tokens))
        orchestrator.debug.set_metric(f"brief.{brief_name}.selection.final_budget_tokens", int(prompt_budget_tokens))
        orchestrator.debug.set_metric(f"brief.{brief_name}.selection.final_budget_dropped", 0)
        return selected

    if candidate_articles:
        prompt = estimator._render_prompt(
            candidate_articles,
            context_chars,
            orchestrator.config.user_memory,
            topics,
            active_reports,
            brief_goal,
            date,
            evidence_packet={},
            delta_packet={},
        )
        estimated_tokens = estimator._estimate_final_input_tokens(prompt)

    original_by_id = {item.candidate.id: item for item in sorted_selected}
    pruned = [original_by_id[item.candidate.id] for item in candidate_articles if item.candidate.id in original_by_id]
    dropped = [original_by_id[item_id] for item_id in dropped_ids if item_id in original_by_id]
    effective_floor = min((float(item.decision.score) for item in pruned), default=0.0)
    for item in dropped:
        item.decision.selection_reason_code = "skipped_final_budget"
        set_selection_skip_annotation(item.candidate, "skipped_final_budget")

    warning = (
        f"{brief_name}: dynamic final-context budget raised effective headline score floor to "
        f"{effective_floor:.2f}; kept {len(pruned)}/{len(selected)} selected articles "
        f"({estimated_tokens}/{prompt_budget_tokens} estimated input tokens), dropped: "
        + ", ".join(dropped_ids)
    )
    run_warnings.append(warning)
    orchestrator.debug.log(
        "headline.select",
        "final_budget_prune",
        brief=brief_name,
        before=len(selected),
        after=len(pruned),
        dropped=len(dropped),
        estimated_tokens=estimated_tokens,
        budget_tokens=prompt_budget_tokens,
        effective_score_floor=effective_floor,
    )
    orchestrator.debug.set_metric(f"brief.{brief_name}.selection.final_budget_estimated_tokens", int(estimated_tokens))
    orchestrator.debug.set_metric(f"brief.{brief_name}.selection.final_budget_tokens", int(prompt_budget_tokens))
    orchestrator.debug.set_metric(f"brief.{brief_name}.selection.final_budget_dropped", len(dropped))
    orchestrator.debug.set_metric(f"brief.{brief_name}.selection.final_budget_effective_score_floor", effective_floor)
    return pruned
