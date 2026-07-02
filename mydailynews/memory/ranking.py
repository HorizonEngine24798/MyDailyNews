from __future__ import annotations

from typing import Any, Dict, Iterable, List

from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, MemoryConfig, NewsCandidate
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.memory.coverage import CoverageMemoryStore
from mydailynews.memory.preference_learning import learned_preference_effect_from_candidate
from mydailynews.memory.story_index import MATCH_CONFIDENCE_THRESHOLD, StoryIndexStore
from mydailynews.memory.story_keys import StoryIdentity, story_identity_for_candidate, token_overlap_confidence


MATERIAL_ANGLE_TYPES = {
    "breakthrough",
    "escalation",
    "material_update",
    "major_update",
    "new_phase",
    "policy_change",
    "regulatory_action",
    "resolution",
    "reversal",
}


def annotate_candidates_with_memory(
    *,
    candidates: List[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
    memory_config: MemoryConfig | None,
    coverage_store: CoverageMemoryStore | None,
    story_index_store: StoryIndexStore | None,
    date: str,
) -> Dict[str, Any]:
    if not getattr(memory_config, "enabled", False):
        return {"enabled": False, "annotated": 0}

    run_identities: List[StoryIdentity] = []
    annotated = 0
    reduced_keys: set[str] = set()
    boosted_keys: set[str] = set()
    covered_keys: set[str] = set()

    for candidate in candidates:
        decision = decisions.get(candidate.id)
        base_identity = (
            story_index_store.match_candidate(candidate)
            if story_index_store is not None
            else story_identity_for_candidate(candidate)
        )
        identity = _match_same_run_story(base_identity, run_identities)
        run_identities.append(identity)

        summary = (
            coverage_store.recent_summary(
                story_key=identity.story_key,
                as_of_date=date,
                window_days=int(memory_config.coverage_window_days),
            )
            if coverage_store is not None
            else None
        )
        materiality = materiality_for_decision(decision)
        adjustment = 0.0
        today_policy = "normal"
        reason = ""
        change_type = str(getattr(decision, "angle_type", "") or "").strip()
        if not change_type:
            change_type = "material_update" if materiality >= 0.7 else "incremental_update"

        recent_count = int(getattr(summary, "recent_coverage_count", 0) or 0)
        recent_leads = int(getattr(summary, "recent_lead_count", 0) or 0)
        covered_yesterday = bool(getattr(summary, "covered_yesterday", False))
        if recent_count > 0:
            covered_keys.add(identity.story_key)
            penalty = min(float(memory_config.recent_story_penalty) * recent_count, float(memory_config.recent_story_penalty) * 2.0)
            penalty += min(float(memory_config.recent_lead_penalty) * recent_leads, float(memory_config.recent_lead_penalty) * 2.0)
            if covered_yesterday and recent_leads <= 0:
                penalty += float(memory_config.recent_story_penalty) * 0.5
            boost = float(memory_config.material_update_boost) * materiality if materiality >= 0.7 else 0.0
            adjustment = max(-3.0, min(1.5, boost - penalty))
            if boost > 0.0:
                boosted_keys.add(identity.story_key)
                today_policy = "material_update_ok"
                reason = "Recently covered, but headline signals a material update."
            elif recent_leads > 0 or covered_yesterday:
                today_policy = "capsule_unless_material_update"
                reason = "Recently prominent and current headline signal appears incremental."
            else:
                today_policy = "deprioritize_repeat"
                reason = "Recently covered in the memory window."
            if adjustment < 0.0:
                reduced_keys.add(identity.story_key)

        annotation = MemoryAnnotation(
            story_key=identity.story_key,
            story_family_key=identity.story_family_key,
            story_title=identity.story_title,
            match_confidence=identity.match_confidence,
            recent_coverage_count=recent_count,
            recent_lead_count=recent_leads,
            covered_yesterday=covered_yesterday,
            change_type=change_type,
            materiality=materiality,
            score_adjustment=round(adjustment, 4),
            today_policy=today_policy,
            reason=reason,
        )
        set_memory_annotation(candidate, annotation)
        annotated += 1

    return {
        "enabled": True,
        "annotated": annotated,
        "recent_story_keys": len(covered_keys),
        "stories_reduced_for_recent_coverage": len(reduced_keys),
        "stories_boosted_for_material_update": len(boosted_keys),
    }


def materiality_for_decision(decision: HeadlineDecision | None) -> float:
    if decision is None:
        return 0.0
    angle_type = str(decision.angle_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if angle_type in MATERIAL_ANGLE_TYPES:
        return 0.9
    novelty = _score_0_to_10(getattr(decision, "novelty", 5.0))
    impact = _score_0_to_10(getattr(decision, "impact", 5.0))
    urgency = _score_0_to_10(getattr(decision, "urgency", 5.0))
    if novelty >= 7.0 and impact >= 7.0:
        return round(min(1.0, ((novelty + impact + urgency) / 30.0) + 0.1), 4)
    if novelty >= 8.0 and urgency >= 8.0:
        return 0.75
    return round(max(novelty, impact, urgency) / 20.0, 4)


def memory_selection_summary(
    candidates: Iterable[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
) -> Dict[str, Any]:
    story_keys: set[str] = set()
    reduced: set[str] = set()
    boosted: set[str] = set()
    learned_adjusted = 0
    learned_positive = 0
    learned_negative = 0
    learned_topic_matches: set[str] = set()
    learned_source_matches: set[str] = set()
    skipped_story_cap = 0
    skipped_family_cap = 0
    for candidate in candidates:
        annotation = candidate_memory_annotation(candidate)
        if annotation is not None and annotation.story_key:
            story_keys.add(annotation.story_key)
            if annotation.recent_coverage_count > 0 and annotation.score_adjustment < 0:
                reduced.add(annotation.story_key)
            if annotation.recent_coverage_count > 0 and annotation.score_adjustment > 0:
                boosted.add(annotation.story_key)
        learned_effect = learned_preference_effect_from_candidate(candidate)
        if learned_effect is not None and learned_effect.changed:
            learned_adjusted += 1
            if learned_effect.score_adjustment > 0:
                learned_positive += 1
            elif learned_effect.score_adjustment < 0:
                learned_negative += 1
            learned_topic_matches.update(learned_effect.matched_topics or [])
            learned_source_matches.update(learned_effect.matched_sources or [])
        decision = decisions.get(candidate.id)
        code = str(getattr(decision, "selection_reason_code", "") or "")
        if code == "skipped_story_cap":
            skipped_story_cap += 1
        elif code == "skipped_story_family_cap":
            skipped_family_cap += 1
    return {
        "story_count": len(story_keys),
        "stories_reduced_for_recent_coverage": len(reduced),
        "stories_boosted_for_material_update": len(boosted),
        "stories_skipped_by_story_cap": skipped_story_cap,
        "stories_skipped_by_story_family_cap": skipped_family_cap,
        "learned_preference_adjusted_candidates": learned_adjusted,
        "learned_preference_positive_candidates": learned_positive,
        "learned_preference_negative_candidates": learned_negative,
        "learned_preference_topic_matches": len(learned_topic_matches),
        "learned_preference_source_matches": len(learned_source_matches),
    }


def _match_same_run_story(identity: StoryIdentity, existing: List[StoryIdentity]) -> StoryIdentity:
    best: tuple[float, StoryIdentity] | None = None
    for other in existing:
        confidence = token_overlap_confidence(identity.tokens, other.tokens)
        if confidence < MATCH_CONFIDENCE_THRESHOLD:
            continue
        if best is None or confidence > best[0]:
            best = (confidence, other)
    if best is None:
        return identity
    confidence, other = best
    return StoryIdentity(
        story_key=other.story_key,
        story_family_key=other.story_family_key,
        story_title=other.story_title,
        tokens=identity.tokens,
        match_confidence=confidence,
    )


def _score_0_to_10(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 5.0
    return max(0.0, min(10.0, number))
