from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, NewsCandidate, SelectedArticle
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.memory.ranking import memory_selection_summary


DELTA_MATERIALITY_KEYS = {
    "new": ("new", 0.95),
    "escalated": ("escalated", 0.9),
    "reframed": ("reframed", 0.85),
    "weakened": ("weakened", 0.75),
}

MIN_CONFIDENCE_FOR_OMISSION = 0.7


def partition_selected_for_brief(
    *,
    selected: List[SelectedArticle],
    delta_packet: Dict[str, Any] | None,
) -> tuple[List[SelectedArticle], List[SelectedArticle]]:
    """Split safe omissions from articles that should reach the writer.

    Unknown, malformed, duplicated, or conflicting decisions fail open. A weak
    model's formatting mistake should not silently hide a material update.
    """

    decisions = _unambiguous_story_decisions_by_article(delta_packet)
    included: List[SelectedArticle] = []
    omitted: List[SelectedArticle] = []
    for article in selected:
        decision = decisions.get(str(article.candidate.id))
        if _effective_disposition(decision) == "omit":
            omitted.append(article)
        else:
            included.append(article)
    return included, omitted


def selected_articles_represented_in_brief(
    selected: List[SelectedArticle],
    brief: Dict[str, Any] | None,
) -> List[SelectedArticle]:
    """Return articles actually exposed after final-prompt compaction."""

    rows = brief.get("selected_articles", []) if isinstance(brief, dict) else []
    if not isinstance(rows, list):
        return []
    represented_ids = {
        str(row.get("id", "") or "").strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("id", "") or "").strip()
    }
    return [article for article in selected if str(article.candidate.id) in represented_ids]


def recall_packet_for_selected(
    packet: Dict[str, Any] | None,
    selected: List[SelectedArticle],
) -> Dict[str, Any]:
    if not isinstance(packet, dict) or not packet:
        return {}
    story_keys = {
        annotation.story_key
        for article in selected
        for annotation in [candidate_memory_annotation(article.candidate)]
        if annotation is not None and annotation.story_key
    }
    rows = packet.get("coverage_guidance", [])
    if not isinstance(rows, list):
        rows = []
    filtered_rows = [
        dict(row)
        for row in rows
        if isinstance(row, dict) and str(row.get("story_key", "") or "") in story_keys
    ]
    return {**packet, "coverage_guidance": filtered_rows}


def apply_delta_signals_to_selected(
    *,
    selected: List[SelectedArticle],
    delta_packet: Dict[str, Any] | None,
) -> None:
    if not isinstance(delta_packet, dict) or not delta_packet:
        return
    by_article_id: Dict[str, tuple[str, float]] = {}
    decision_by_article = _unambiguous_story_decisions_by_article(delta_packet)
    for key, signal in DELTA_MATERIALITY_KEYS.items():
        change_type, materiality = signal
        for item in delta_packet.get(key, []):
            if not isinstance(item, dict):
                continue
            article_ids = item.get("article_ids", [])
            if not isinstance(article_ids, list):
                continue
            for article_id in article_ids:
                text_id = str(article_id or "").strip()
                if text_id and text_id not in by_article_id:
                    by_article_id[text_id] = (change_type, materiality)
    for article in selected:
        decision = decision_by_article.get(article.candidate.id)
        if decision is not None:
            annotation = candidate_memory_annotation(article.candidate)
            if annotation is not None:
                change_type = str(decision.get("change_type", "") or "").strip()
                materiality = _bounded_float(decision.get("materiality"), annotation.materiality)
                disposition = _effective_disposition(decision)
                relationship = str(decision.get("relationship", "") or "").strip()
                prior_story_key = str(decision.get("prior_story_key", "") or "").strip()
                set_memory_annotation(
                    article.candidate,
                    MemoryAnnotation(
                        story_key=(
                            prior_story_key
                            if relationship == "same_story" and prior_story_key
                            else annotation.story_key
                        ),
                        story_family_key=annotation.story_family_key,
                        story_title=annotation.story_title,
                        match_confidence=annotation.match_confidence,
                        recent_coverage_count=annotation.recent_coverage_count,
                        recent_lead_count=annotation.recent_lead_count,
                        covered_yesterday=annotation.covered_yesterday,
                        change_type=change_type or annotation.change_type,
                        materiality=max(float(annotation.materiality), materiality),
                        score_adjustment=annotation.score_adjustment,
                        today_policy=(
                            "capsule_unless_material_update"
                            if disposition == "continuing_bullet"
                            else "omit"
                            if disposition == "omit"
                            else annotation.today_policy
                        ),
                        reason=str(decision.get("reason", "") or annotation.reason).strip(),
                    ),
                )
            continue
        signal = by_article_id.get(article.candidate.id)
        if signal is None:
            continue
        annotation = candidate_memory_annotation(article.candidate)
        if annotation is None:
            continue
        change_type, materiality = signal
        set_memory_annotation(
            article.candidate,
            MemoryAnnotation(
                story_key=annotation.story_key,
                story_family_key=annotation.story_family_key,
                story_title=annotation.story_title,
                match_confidence=annotation.match_confidence,
                recent_coverage_count=annotation.recent_coverage_count,
                recent_lead_count=annotation.recent_lead_count,
                covered_yesterday=annotation.covered_yesterday,
                change_type=change_type,
                materiality=max(float(annotation.materiality), materiality),
                score_adjustment=annotation.score_adjustment,
                today_policy=(
                    "material_update_ok"
                    if annotation.recent_coverage_count > 0 and materiality >= 0.8
                    else annotation.today_policy
                ),
                reason=(
                    "Delta analysis marks this as a material update."
                    if annotation.recent_coverage_count > 0 and materiality >= 0.8
                    else annotation.reason
                ),
            ),
        )


def _bounded_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _unambiguous_story_decisions_by_article(
    delta_packet: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(delta_packet, dict):
        return {}
    rows = delta_packet.get("story_decisions", [])
    if not isinstance(rows, list):
        return {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for decision in rows:
        if not isinstance(decision, dict):
            continue
        article_ids = decision.get("article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for article_id in article_ids:
            text_id = str(article_id or "").strip()
            if text_id:
                grouped.setdefault(text_id, []).append(decision)

    output: Dict[str, Dict[str, Any]] = {}
    for article_id, decisions in grouped.items():
        signatures = {
            (
                str(item.get("relationship", "") or "").strip(),
                str(item.get("change_type", "") or "").strip(),
                str(item.get("disposition", "") or "").strip(),
            )
            for item in decisions
        }
        if len(signatures) == 1:
            output[article_id] = max(
                decisions,
                key=lambda item: _bounded_float(item.get("confidence"), 0.0),
            )
    return output


def _effective_disposition(decision: Dict[str, Any] | None) -> str:
    if not isinstance(decision, dict):
        return "full_report"
    disposition = str(decision.get("disposition", "") or "").strip()
    if disposition != "omit":
        return "continuing_bullet" if disposition == "continuing_bullet" else "full_report"
    relationship = str(decision.get("relationship", "") or "").strip()
    change_type = str(decision.get("change_type", "") or "").strip()
    confidence = _bounded_float(decision.get("confidence"), 0.0)
    if (
        relationship == "same_story"
        and change_type == "unchanged"
        and confidence >= MIN_CONFIDENCE_FOR_OMISSION
    ):
        return "omit"
    return "full_report"


def build_recall_packet(
    *,
    date: str,
    brief_name: str,
    candidates: List[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
) -> Dict[str, Any]:
    guidance_rows: List[Dict[str, Any]] = []
    seen_story_keys: set[str] = set()
    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            abs(float((candidate_memory_annotation(item) or MemoryAnnotation()).score_adjustment)),
            float((candidate_memory_annotation(item) or MemoryAnnotation()).recent_coverage_count),
            float(getattr(decisions.get(item.id), "selection_rank_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for candidate in ranked_candidates:
        annotation = candidate_memory_annotation(candidate)
        if annotation is None or not annotation.story_key:
            continue
        if annotation.story_key in seen_story_keys:
            continue
        if (
            annotation.recent_coverage_count <= 0
            and annotation.today_policy == "normal"
            and abs(float(annotation.score_adjustment)) < 1e-6
        ):
            continue
        seen_story_keys.add(annotation.story_key)
        decision = decisions.get(candidate.id)
        code = str(getattr(decision, "selection_reason_code", "") or "")
        guidance_rows.append(
            {
                "story_key": annotation.story_key,
                "story_family_key": annotation.story_family_key,
                "title": annotation.story_title or candidate.title,
                "recent_coverage": _recent_coverage_text(annotation),
                "today_policy": annotation.today_policy,
                "reason": annotation.reason,
                "score_adjustment": round(float(annotation.score_adjustment), 4),
                "materiality": round(float(annotation.materiality), 4),
                "selected": code.startswith("selected_"),
                "selection_reason_code": code,
            }
        )
        if len(guidance_rows) >= 12:
            break

    summary = memory_selection_summary(candidates, decisions)
    return {
        "schema_version": 1,
        "date": str(date or ""),
        "brief_name": str(brief_name or ""),
        "coverage_guidance": guidance_rows,
        "selection_summary": summary,
    }


def save_recall_packet(
    *,
    state_dir: Path | str,
    date: str,
    brief_name: str,
    recall_packet: Dict[str, Any],
) -> Path:
    path = Path(state_dir) / "recall_packets" / f"{date}_{brief_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recall_packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def combined_recall_packet_for_narrative(source_briefs: Iterable[Any]) -> Dict[str, Any]:
    guidance: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    names: List[str] = []
    for source in source_briefs:
        name = str(getattr(source, "name", "") or "")
        brief = getattr(source, "brief", {})
        if name:
            names.append(name)
        if not isinstance(brief, dict):
            continue
        memory_meta = brief.get("metadata", {}).get("memory", {}) if isinstance(brief.get("metadata"), dict) else {}
        packet = memory_meta.get("recall_packet", {}) if isinstance(memory_meta, dict) else {}
        if not isinstance(packet, dict):
            continue
        summary = packet.get("selection_summary", {})
        if isinstance(summary, dict):
            summaries.append({"brief_name": name, **summary})
        rows = packet.get("coverage_guidance", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            story_key = str(row.get("story_key", "") or "")
            key = f"{name}:{story_key}"
            if not story_key or key in seen:
                continue
            seen.add(key)
            guidance.append({"brief_name": name, **row})
            if len(guidance) >= 16:
                break
    if not guidance and not summaries:
        return {}
    return {
        "schema_version": 1,
        "source_briefs": names,
        "coverage_guidance": guidance,
        "selection_summaries": summaries,
    }


def _recent_coverage_text(annotation: MemoryAnnotation) -> str:
    if annotation.recent_lead_count > 0:
        return (
            f"Covered {annotation.recent_coverage_count} time(s) in the recent window; "
            f"led {annotation.recent_lead_count} time(s)."
        )
    if annotation.covered_yesterday:
        return "Covered yesterday."
    if annotation.recent_coverage_count > 0:
        return f"Covered {annotation.recent_coverage_count} time(s) in the recent window."
    return "No recent coverage memory."
