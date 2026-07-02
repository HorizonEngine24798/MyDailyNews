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


def apply_delta_signals_to_selected(
    *,
    selected: List[SelectedArticle],
    delta_packet: Dict[str, Any] | None,
) -> None:
    if not isinstance(delta_packet, dict) or not delta_packet:
        return
    by_article_id: Dict[str, tuple[str, float]] = {}
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
    if not by_article_id:
        return
    for article in selected:
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

