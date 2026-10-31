from __future__ import annotations

from typing import Any, Dict, Iterable, List

from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, MemoryConfig, NewsCandidate
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.memory.coverage import CoverageMemoryStore
from mydailynews.memory.preference_learning import learned_preference_effect_from_candidate
from mydailynews.memory.story_store import MATCH_CONFIDENCE_THRESHOLD, StoryStore, provisional_story_key
from mydailynews.memory.story_keys import StoryIdentity, story_identity_for_candidate, token_overlap_confidence


MAX_PROVISIONAL_COVERAGE_PENALTY = 0.35


def annotate_candidates_with_memory(
    *,
    candidates: List[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
    memory_config: MemoryConfig | None,
    coverage_store: CoverageMemoryStore | None,
    story_store: StoryStore | None,
    date: str,
) -> Dict[str, Any]:
    if not getattr(memory_config, "enabled", False):
        return {"enabled": False, "annotated": 0}

    run_identities: List[StoryIdentity] = []
    annotated = 0
    reduced_keys: set[str] = set()
    covered_keys: set[str] = set()
    candidate_links = 0
    occupied_story_keys = {
        record.story_key
        for record in story_store.records()
    } if story_store is not None else set()

    for candidate in candidates:
        coverage_story_key = ""
        if story_store is not None:
            base_identity = story_identity_for_candidate(candidate)
            matches = story_store.candidate_stories(
                candidate,
                source_text=candidate.snippet,
                min_score=float(memory_config.story_candidate_threshold),
            )
            candidate.metadata["memory_prior_story_candidates"] = [match.metadata() for match in matches]
            candidate.metadata["memory_identity_state"] = "provisional"
            if matches:
                candidate_links += 1
                coverage_story_key = matches[0].record.story_key
            current_key = provisional_story_key(
                candidate,
                occupied_story_keys=[*occupied_story_keys, *(item.story_key for item in run_identities)],
            )
            identity = StoryIdentity(
                story_key=current_key,
                story_family_key=base_identity.story_family_key,
                story_title=base_identity.story_title,
                tokens=base_identity.tokens,
                match_confidence=(matches[0].score if matches else base_identity.match_confidence),
            )
        else:
            base_identity = story_identity_for_candidate(candidate)
            identity = _match_same_run_story(base_identity, run_identities)
            coverage_story_key = identity.story_key
        run_identities.append(identity)

        summary = (
            coverage_store.recent_summary(
                story_key=coverage_story_key or identity.story_key,
                as_of_date=date,
                window_days=int(memory_config.coverage_window_days),
            )
            if coverage_store is not None
            else None
        )
        adjustment = 0.0
        today_policy = "normal"
        reason = ""
        change_type = ""
        materiality = 0.0

        recent_count = int(getattr(summary, "recent_coverage_count", 0) or 0)
        recent_leads = int(getattr(summary, "recent_lead_count", 0) or 0)
        covered_yesterday = bool(getattr(summary, "covered_yesterday", False))
        if recent_count > 0:
            covered_keys.add(coverage_story_key or identity.story_key)
            penalty = min(float(memory_config.recent_story_penalty) * recent_count, float(memory_config.recent_story_penalty) * 2.0)
            penalty += min(float(memory_config.recent_lead_penalty) * recent_leads, float(memory_config.recent_lead_penalty) * 2.0)
            if covered_yesterday and recent_leads <= 0:
                penalty += float(memory_config.recent_story_penalty) * 0.5
            adjustment = max(-3.0, -penalty)
            today_policy = "deprioritize_repeat"
            reason = "Recently covered in the memory window; delta analysis has not established a new fact."
            if story_store is not None and adjustment < 0.0:
                adjustment = max(adjustment, -MAX_PROVISIONAL_COVERAGE_PENALTY)
                today_policy = "await_delta"
                reason = (
                    "Retrieved history was recently covered, but current facts have not yet "
                    "been classified against it."
                )
            if adjustment < 0.0:
                reduced_keys.add(coverage_story_key or identity.story_key)

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
        "candidates_with_prior_story_candidates": candidate_links,
    }


def memory_selection_summary(
    candidates: Iterable[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
) -> Dict[str, Any]:
    story_keys: set[str] = set()
    reduced: set[str] = set()
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
    identity_title_tokens = identity.story_key.split("-")
    for other in existing:
        # Same-run matching is only a conservative bridge for coverage memory.
        # Full identity tokens also contain topic and snippet text, which can
        # make unrelated headlines look identical when feed boilerplate is
        # shared.  The story key is derived from the headline, so compare that
        # narrower evidence here and leave broader grouping to the AI stage.
        confidence = token_overlap_confidence(
            identity_title_tokens,
            other.story_key.split("-"),
        )
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
