from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from mydailynews.app.models import HeadlineDecision, NewsCandidate
from mydailynews.common.utils import is_consent_interstitial_text
from mydailynews.domain.text_similarity import normalized_word_text
from mydailynews.memory.feedback import FeedbackEvent
from mydailynews.memory.learned_preferences import LearnedPreferences
from mydailynews.memory.multilingual_embeddings import (
    batch_query_passage_similarities,
    embedding_model_available,
    query_passage_similarity,
)
from mydailynews.memory.story_keys import story_identity_for_candidate


TOPIC_WEIGHT_SCALE = 0.2
SOURCE_WEIGHT_SCALE = 0.2
LIST_TOPIC_BONUS = 0.2
LIST_TOPIC_PENALTY = 0.35
LIST_SOURCE_BONUS = 0.15
LIST_SOURCE_PENALTY = 0.35
MAX_RANK_ADJUSTMENT = 1.0
MAX_COOLDOWN_RANK_ADJUSTMENT = 1.25

DEFAULT_PREFERENCE_HALF_LIFE_DAYS = 30.0
FEEDBACK_SEMANTIC_MAX_AGE_DAYS = 90.0
SEMANTIC_TOPIC_THRESHOLD = 0.845
SEMANTIC_TOPIC_MAX_SIM = 0.915
SEMANTIC_FEEDBACK_THRESHOLD = 0.835
SEMANTIC_FEEDBACK_MAX_SIM = 0.905
SEMANTIC_COOLDOWN_THRESHOLD = 0.875
SEMANTIC_TOPIC_CAP = 0.55
SEMANTIC_FEEDBACK_POS_CAP = 0.50
SEMANTIC_FEEDBACK_NEG_CAP = 0.60

STORY_COOLDOWN_HALF_LIFE_DAYS = 2.5
STORY_COOLDOWN_MAX_AGE_DAYS = 7.0
STORY_COOLDOWN_BASE_PENALTY = 0.85
FAMILY_COOLDOWN_SCALE = 0.6
MAX_COOLDOWN_PENALTY = 1.15

LEARNED_PREFERENCE_METADATA_KEYS = (
    "learned_preference_adjustment",
    "learned_preference_topic_weight",
    "learned_preference_source_weight",
    "learned_preference_list_adjustment",
    "learned_preference_semantic_adjustment",
    "learned_preference_cooldown_adjustment",
    "learned_preference_matched_topics",
    "learned_preference_matched_sources",
    "learned_preference_matched_semantic_topics",
    "learned_preference_matched_feedback_examples",
    "learned_preference_cooldown_story_key",
    "learned_preference_reason",
)


@dataclass(frozen=True)
class LearnedPreferenceDelta:
    topic_weights: Dict[str, float]
    source_weights: Dict[str, float]

    @property
    def changed(self) -> bool:
        return bool(self.topic_weights or self.source_weights)


@dataclass(frozen=True)
class StoryCooldownEntry:
    """Active repetitive-story suppression window derived from too_repetitive feedback."""

    story_key: str
    story_family_key: str
    title: str
    topic: str
    created_at: str
    age_days: float
    days_remaining: float
    penalty: float
    repeat_count: int = 1


@dataclass(frozen=True)
class LearnedPreferenceRankEffect:
    score_adjustment: float = 0.0
    matched_topics: List[str] | None = None
    matched_sources: List[str] | None = None
    topic_weight: float = 0.0
    source_weight: float = 0.0
    list_adjustment: float = 0.0
    semantic_adjustment: float = 0.0
    cooldown_adjustment: float = 0.0
    matched_semantic_topics: List[str] | None = None
    matched_feedback_examples: List[str] | None = None
    cooldown_story_key: str = ""
    reason: str = ""

    @property
    def changed(self) -> bool:
        return abs(float(self.score_adjustment or 0.0)) > 1e-6


def preference_delta_for_event(event: FeedbackEvent) -> LearnedPreferenceDelta:
    topic = _clean_label(event.topic, 120)
    source = _clean_label(event.source, 120)
    action = str(event.action or "").strip().lower()
    topic_weights: Dict[str, float] = {}
    source_weights: Dict[str, float] = {}
    if action == "more_like_this":
        if topic:
            topic_weights[topic] = 0.35
        if source:
            source_weights[source] = 0.2
    elif action == "not_interested_in_topic":
        if topic:
            topic_weights[topic] = -0.7
    elif action == "not_relevant":
        if topic:
            topic_weights[topic] = -0.3
        if source:
            source_weights[source] = -0.15
    return LearnedPreferenceDelta(topic_weights=topic_weights, source_weights=source_weights)


def temporal_decay_factor(
    created_at: str,
    *,
    as_of: datetime | str | None = None,
    half_life_days: float = DEFAULT_PREFERENCE_HALF_LIFE_DAYS,
) -> float:
    """Return exponential decay weight in (0, 1] relative to as_of (defaults to 1.0 when as_of omitted)."""
    if as_of is None or as_of == "" or half_life_days <= 0:
        return 1.0
    ref_dt = _parse_timestamp(as_of)
    event_dt = _parse_timestamp(created_at)
    if ref_dt is None or event_dt is None:
        return 1.0
    age_days = max(0.0, (ref_dt - event_dt).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / float(half_life_days)))


def apply_feedback_event(
    preferences: LearnedPreferences,
    event: FeedbackEvent,
    *,
    scale: float = 1.0,
) -> tuple[LearnedPreferences, LearnedPreferenceDelta]:
    delta = preference_delta_for_event(event)
    updated = _copy_preferences(preferences)
    factor = max(0.0, float(scale))
    for topic, change in delta.topic_weights.items():
        updated.topic_weights[topic] = _clamp_weight(
            float(updated.topic_weights.get(topic, 0.0)) + float(change) * factor
        )
    for source, change in delta.source_weights.items():
        updated.source_weights[source] = _clamp_weight(
            float(updated.source_weights.get(source, 0.0)) + float(change) * factor
        )
    return updated, delta


def rebuild_learned_preferences(
    events: Iterable[FeedbackEvent],
    base_preferences: LearnedPreferences | None = None,
    *,
    as_of: datetime | str | None = None,
    half_life_days: float = DEFAULT_PREFERENCE_HALF_LIFE_DAYS,
) -> LearnedPreferences:
    """Recompute learned weights from feedback history, applying half-life decay when as_of is given."""
    preferences = _copy_preferences(base_preferences) if base_preferences is not None else LearnedPreferences()
    for event in events:
        scale = temporal_decay_factor(event.created_at, as_of=as_of, half_life_days=half_life_days)
        if as_of is not None and as_of != "" and scale < 0.02:
            continue
        preferences, _ = apply_feedback_event(preferences, event, scale=scale)
    if as_of is not None and as_of != "":
        preferences.topic_weights = {
            k: v for k, v in preferences.topic_weights.items() if abs(float(v)) >= 0.01
        }
        preferences.source_weights = {
            k: v for k, v in preferences.source_weights.items() if abs(float(v)) >= 0.01
        }
    return preferences


def reset_learned_preferences_weights(
    preferences: LearnedPreferences,
    *,
    clear_manual_lists: bool = False,
) -> LearnedPreferences:
    """Clear learned weights while optionally preserving manual preferred/avoided topic & source tags."""
    return LearnedPreferences(
        schema_version=preferences.schema_version,
        updated_at=datetime.now(timezone.utc).isoformat(),
        preferred_topics=[] if clear_manual_lists else list(preferences.preferred_topics),
        avoided_topics=[] if clear_manual_lists else list(preferences.avoided_topics),
        preferred_sources=[] if clear_manual_lists else list(preferences.preferred_sources),
        avoided_sources=[] if clear_manual_lists else list(preferences.avoided_sources),
        topic_weights={},
        source_weights={},
        notes="" if clear_manual_lists else preferences.notes,
    )


def story_cooldowns_from_feedback(
    feedback_events: Sequence[FeedbackEvent] | None,
    *,
    as_of: datetime | str | None = None,
    half_life_days: float = STORY_COOLDOWN_HALF_LIFE_DAYS,
    max_age_days: float = STORY_COOLDOWN_MAX_AGE_DAYS,
) -> Dict[str, Any]:
    """Compute decaying story & family penalties for recent too_repetitive feedback events."""
    if not feedback_events:
        return {
            "story_penalties": {},
            "family_penalties": {},
            "entries": [],
            "semantic_targets": [],
        }

    ref_dt = _parse_timestamp(as_of) if (as_of is not None and as_of != "") else datetime.now(timezone.utc)
    if ref_dt is None:
        ref_dt = datetime.now(timezone.utc)

    by_story: Dict[str, Dict[str, Any]] = {}
    family_penalties: Dict[str, float] = {}
    semantic_targets: List[tuple[str, str, float]] = []

    for event in feedback_events:
        if str(getattr(event, "action", "") or "").strip().lower() != "too_repetitive":
            continue
        event_dt = _parse_timestamp(getattr(event, "created_at", "") or "")
        if event_dt is None:
            age_days = 0.0
        else:
            age_days = max(0.0, (ref_dt - event_dt).total_seconds() / 86400.0)
        if age_days > max_age_days:
            continue

        decay = float(0.5 ** (age_days / max(0.1, float(half_life_days))))
        event_penalty = round(STORY_COOLDOWN_BASE_PENALTY * decay, 4)
        if event_penalty < 0.04:
            continue

        story_key = _clean_label(getattr(event, "story_key", ""), 160)
        family_key = _clean_label(getattr(event, "story_family_key", ""), 160)
        title = _clean_label(getattr(event, "title", ""), 200)
        topic = _clean_label(getattr(event, "topic", ""), 120)
        days_remaining = round(max(0.0, max_age_days - age_days), 2)

        if title and len(title.split()) >= 3:
            semantic_targets.append((title, story_key or title, event_penalty))

        bucket_key = story_key or (f"family:{family_key}" if family_key else title.casefold())
        if bucket_key:
            existing = by_story.get(bucket_key)
            if existing is None:
                by_story[bucket_key] = {
                    "story_key": story_key,
                    "story_family_key": family_key,
                    "title": title,
                    "topic": topic,
                    "created_at": str(getattr(event, "created_at", "") or ""),
                    "age_days": round(age_days, 2),
                    "days_remaining": days_remaining,
                    "penalty": event_penalty,
                    "repeat_count": 1,
                }
            else:
                existing["penalty"] = round(min(MAX_COOLDOWN_PENALTY, existing["penalty"] + event_penalty * 0.6), 4)
                existing["repeat_count"] += 1
                existing["age_days"] = min(existing["age_days"], round(age_days, 2))
                existing["days_remaining"] = max(existing["days_remaining"], days_remaining)
                if str(getattr(event, "created_at", "") or "") > existing["created_at"]:
                    existing["created_at"] = str(getattr(event, "created_at", "") or "")
                    if title:
                        existing["title"] = title

        if family_key:
            family_penalties[family_key] = round(
                min(MAX_COOLDOWN_PENALTY * FAMILY_COOLDOWN_SCALE, family_penalties.get(family_key, 0.0) + event_penalty * FAMILY_COOLDOWN_SCALE),
                4,
            )

    story_penalties = {
        entry["story_key"]: entry["penalty"]
        for entry in by_story.values()
        if entry["story_key"]
    }
    entries = [
        StoryCooldownEntry(
            story_key=item["story_key"],
            story_family_key=item["story_family_key"],
            title=item["title"],
            topic=item["topic"],
            created_at=item["created_at"],
            age_days=item["age_days"],
            days_remaining=item["days_remaining"],
            penalty=item["penalty"],
            repeat_count=item["repeat_count"],
        )
        for item in sorted(by_story.values(), key=lambda row: (-row["penalty"], row["age_days"]))
    ]
    return {
        "story_penalties": story_penalties,
        "family_penalties": family_penalties,
        "entries": entries,
        "semantic_targets": semantic_targets,
    }


def candidate_semantic_passage(
    candidate: NewsCandidate,
    *,
    decision: HeadlineDecision | None = None,
) -> str:
    """Build clean headline+context passage text for multilingual E5 preference comparison."""
    title = " ".join(str(getattr(candidate, "title", "") or "").split()).strip()
    snippet = " ".join(str(getattr(candidate, "snippet", "") or "").split()).strip()
    if is_consent_interstitial_text(snippet):
        snippet = ""
    topic = " ".join(str(getattr(decision, "topic", "") or candidate.metadata.get("topic_name", "") or "").split()).strip()
    parts = [part for part in (title, snippet[:220]) if part]
    text = ". ".join(parts)
    if topic and topic.casefold() not in text.casefold():
        text = f"{text} ({topic})" if text else topic
    return text[:500]


def prefetch_preference_similarities(
    candidates: Sequence[NewsCandidate],
    learned_preferences: LearnedPreferences | None,
    feedback_events: Sequence[FeedbackEvent] | None = None,
    *,
    decisions: Mapping[str, HeadlineDecision] | None = None,
    as_of: datetime | str | None = None,
) -> Dict[tuple[str, str], float]:
    """Precompute E5 query-by-passage similarity lookup once before ranking candidates."""
    if not candidates or not embedding_model_available():
        return {}

    queries: List[str] = []
    if learned_preferences is not None:
        for topic, weight in learned_preferences.topic_weights.items():
            if abs(float(weight or 0.0)) > 0.05:
                queries.append(topic)
        queries.extend(learned_preferences.preferred_topics)
        queries.extend(learned_preferences.avoided_topics)

    if feedback_events:
        for event in feedback_events:
            title = _clean_label(getattr(event, "title", ""), 200)
            if not title or len(title.split()) < 3:
                continue
            action = str(getattr(event, "action", "") or "").strip().lower()
            if action == "too_repetitive":
                decay = temporal_decay_factor(
                    getattr(event, "created_at", "") or "",
                    as_of=as_of if (as_of is not None and as_of != "") else datetime.now(timezone.utc),
                    half_life_days=STORY_COOLDOWN_HALF_LIFE_DAYS,
                )
                if decay >= 0.05:
                    queries.append(title)
            elif action in {"more_like_this", "not_interested_in_topic", "not_relevant"}:
                decay = temporal_decay_factor(
                    getattr(event, "created_at", "") or "",
                    as_of=as_of,
                    half_life_days=DEFAULT_PREFERENCE_HALF_LIFE_DAYS,
                )
                if decay >= 0.12:
                    queries.append(title)

    if not queries:
        return {}

    passages = [
        candidate_semantic_passage(candidate, decision=(decisions or {}).get(candidate.id))
        for candidate in candidates
    ]
    return batch_query_passage_similarities(queries, passages)


def ranking_adjustment_for_candidate(
    candidate: NewsCandidate,
    learned_preferences: LearnedPreferences | None,
    *,
    decision: HeadlineDecision | None = None,
    feedback_events: Sequence[FeedbackEvent] | None = None,
    as_of: datetime | str | None = None,
    similarity_lookup: Mapping[tuple[str, str], float] | None = None,
    cooldown_state: Mapping[str, Any] | None = None,
) -> LearnedPreferenceRankEffect:
    """Compute lexical, semantic (multilingual E5), and cooldown ranking adjustment."""
    has_feedback = bool(feedback_events or cooldown_state)
    if learned_preferences is None and not has_feedback:
        return LearnedPreferenceRankEffect(matched_topics=[], matched_sources=[], matched_semantic_topics=[], matched_feedback_examples=[])

    prefs = learned_preferences if learned_preferences is not None else LearnedPreferences()
    matched_topics = _matched_topics(candidate, prefs, decision=decision)
    matched_sources = _matched_sources(candidate, prefs)
    topic_weight = sum(float(prefs.topic_weights.get(topic, 0.0)) for topic in matched_topics)
    source_weight = sum(float(prefs.source_weights.get(source, 0.0)) for source in matched_sources)

    list_adjustment = 0.0
    if _matches_any_topic(candidate, prefs.preferred_topics, decision=decision):
        list_adjustment += LIST_TOPIC_BONUS
    if _matches_any_topic(candidate, prefs.avoided_topics, decision=decision):
        list_adjustment -= LIST_TOPIC_PENALTY
    if _matches_any_source(candidate, prefs.preferred_sources):
        list_adjustment += LIST_SOURCE_BONUS
    if _matches_any_source(candidate, prefs.avoided_sources):
        list_adjustment -= LIST_SOURCE_PENALTY

    cand_passage = candidate_semantic_passage(candidate, decision=decision)
    sim_fn = _similarity_resolver(cand_passage, similarity_lookup)

    semantic_adjustment, semantic_topics, feedback_examples = _semantic_preference_adjustment(
        candidate=candidate,
        cand_passage=cand_passage,
        preferences=prefs,
        matched_lexical_topics=set(matched_topics),
        decision=decision,
        feedback_events=feedback_events,
        as_of=as_of,
        sim_fn=sim_fn,
    )

    state = cooldown_state or story_cooldowns_from_feedback(feedback_events, as_of=as_of)
    cooldown_adjustment, cooldown_story_key = _story_cooldown_adjustment(
        candidate=candidate,
        cand_passage=cand_passage,
        cooldown_state=state,
        sim_fn=sim_fn,
    )

    raw_adjustment = (
        topic_weight * TOPIC_WEIGHT_SCALE
        + source_weight * SOURCE_WEIGHT_SCALE
        + list_adjustment
        + semantic_adjustment
        + cooldown_adjustment
    )
    min_bound = -MAX_COOLDOWN_RANK_ADJUSTMENT if cooldown_adjustment < 0.0 else -MAX_RANK_ADJUSTMENT
    adjustment = round(max(min_bound, min(MAX_RANK_ADJUSTMENT, raw_adjustment)), 4)

    reason = _format_rank_reason(
        matched_topics=matched_topics,
        matched_sources=matched_sources,
        topic_weight=topic_weight,
        source_weight=source_weight,
        list_adjustment=list_adjustment,
        semantic_topics=semantic_topics,
        feedback_examples=feedback_examples,
        semantic_adjustment=semantic_adjustment,
        cooldown_adjustment=cooldown_adjustment,
        cooldown_story_key=cooldown_story_key,
    )

    return LearnedPreferenceRankEffect(
        score_adjustment=adjustment,
        matched_topics=matched_topics,
        matched_sources=matched_sources,
        topic_weight=round(topic_weight, 4),
        source_weight=round(source_weight, 4),
        list_adjustment=round(list_adjustment, 4),
        semantic_adjustment=round(semantic_adjustment, 4),
        cooldown_adjustment=round(cooldown_adjustment, 4),
        matched_semantic_topics=semantic_topics,
        matched_feedback_examples=feedback_examples,
        cooldown_story_key=cooldown_story_key,
        reason=reason,
    )


def _similarity_resolver(
    cand_passage: str,
    similarity_lookup: Mapping[tuple[str, str], float] | None,
) -> Callable[[str], float]:
    if not cand_passage:
        return lambda _query: 0.0
    if similarity_lookup is not None:
        return lambda query: float(similarity_lookup.get((" ".join(str(query or "").split()).strip(), cand_passage), 0.0))
    return lambda query: float(query_passage_similarity(query, cand_passage))


def _semantic_preference_adjustment(
    *,
    candidate: NewsCandidate,
    cand_passage: str,
    preferences: LearnedPreferences,
    matched_lexical_topics: set[str],
    decision: HeadlineDecision | None,
    feedback_events: Sequence[FeedbackEvent] | None,
    as_of: datetime | str | None,
    sim_fn: Callable[[str], float],
) -> tuple[float, List[str], List[str]]:
    if not cand_passage:
        return 0.0, [], []

    pos_topic_scores: List[tuple[float, str]] = []
    neg_topic_scores: List[tuple[float, str]] = []
    seen_labels: set[str] = {label.casefold() for label in matched_lexical_topics}

    # Include explicit weights first, then preferred/avoided list items not already weighted
    topic_targets: List[tuple[str, float]] = list(preferences.topic_weights.items())
    for label in preferences.preferred_topics:
        if label.casefold() not in {k.casefold() for k, _ in topic_targets} and not _matches_any_topic(
            candidate, [label], decision=decision
        ):
            topic_targets.append((label, 1.0))
    for label in preferences.avoided_topics:
        if label.casefold() not in {k.casefold() for k, _ in topic_targets} and not _matches_any_topic(
            candidate, [label], decision=decision
        ):
            topic_targets.append((label, -1.75))

    for label, weight in topic_targets:
        cleaned = _clean_label(label, 120)
        if not cleaned or cleaned.casefold() in seen_labels or abs(float(weight)) <= 0.05:
            continue
        seen_labels.add(cleaned.casefold())
        sim = sim_fn(cleaned)
        if sim < SEMANTIC_TOPIC_THRESHOLD:
            continue
        strength = min(1.0, max(0.35, (sim - SEMANTIC_TOPIC_THRESHOLD) / (SEMANTIC_TOPIC_MAX_SIM - SEMANTIC_TOPIC_THRESHOLD)))
        contribution = float(weight) * TOPIC_WEIGHT_SCALE * strength
        if contribution > 0:
            pos_topic_scores.append((contribution, cleaned))
        elif contribution < 0:
            neg_topic_scores.append((contribution, cleaned))

    topic_adj = 0.0
    matched_semantic_topics: List[str] = []
    if pos_topic_scores:
        pos_topic_scores.sort(key=lambda item: -item[0])
        topic_adj += pos_topic_scores[0][0] + (0.35 * pos_topic_scores[1][0] if len(pos_topic_scores) > 1 else 0.0)
        matched_semantic_topics.extend(item[1] for item in pos_topic_scores[:2])
    if neg_topic_scores:
        neg_topic_scores.sort(key=lambda item: item[0])
        topic_adj += neg_topic_scores[0][0] + (0.35 * neg_topic_scores[1][0] if len(neg_topic_scores) > 1 else 0.0)
        for _, name in neg_topic_scores[:2]:
            if name not in matched_semantic_topics:
                matched_semantic_topics.append(name)
    topic_adj = max(-SEMANTIC_TOPIC_CAP, min(SEMANTIC_TOPIC_CAP, topic_adj))

    # Semantic exemplar matching against specific liked/disliked FeedbackEvent headlines
    pos_examples: List[tuple[float, str]] = []
    neg_examples: List[tuple[float, str]] = []
    if feedback_events:
        for event in feedback_events:
            action = str(getattr(event, "action", "") or "").strip().lower()
            if action not in {"more_like_this", "not_interested_in_topic", "not_relevant"}:
                continue
            event_title = _clean_label(getattr(event, "title", ""), 200)
            if not event_title or len(event_title.split()) < 3:
                continue
            decay = temporal_decay_factor(
                getattr(event, "created_at", "") or "",
                as_of=as_of,
                half_life_days=DEFAULT_PREFERENCE_HALF_LIFE_DAYS,
            )
            if decay < 0.08:
                continue
            sim = sim_fn(event_title)
            if sim < SEMANTIC_FEEDBACK_THRESHOLD:
                continue
            strength = min(1.0, max(0.35, (sim - SEMANTIC_FEEDBACK_THRESHOLD) / (SEMANTIC_FEEDBACK_MAX_SIM - SEMANTIC_FEEDBACK_THRESHOLD)))
            if action == "more_like_this":
                pos_examples.append((0.45 * strength * decay, event_title[:80]))
            elif action == "not_interested_in_topic":
                neg_examples.append((-0.55 * strength * decay, event_title[:80]))
            elif action == "not_relevant":
                neg_examples.append((-0.35 * strength * decay, event_title[:80]))

    example_adj = 0.0
    matched_examples: List[str] = []
    if pos_examples:
        pos_examples.sort(key=lambda item: -item[0])
        example_adj += min(SEMANTIC_FEEDBACK_POS_CAP, pos_examples[0][0] + (0.25 * pos_examples[1][0] if len(pos_examples) > 1 else 0.0))
        matched_examples.append(pos_examples[0][1])
    if neg_examples:
        neg_examples.sort(key=lambda item: item[0])
        example_adj += max(-SEMANTIC_FEEDBACK_NEG_CAP, neg_examples[0][0] + (0.25 * neg_examples[1][0] if len(neg_examples) > 1 else 0.0))
        matched_examples.append(neg_examples[0][1])

    total_semantic = max(-MAX_RANK_ADJUSTMENT, min(MAX_RANK_ADJUSTMENT, topic_adj + example_adj))
    return round(total_semantic, 4), matched_semantic_topics, matched_examples


def _story_cooldown_adjustment(
    *,
    candidate: NewsCandidate,
    cand_passage: str,
    cooldown_state: Mapping[str, Any],
    sim_fn: Callable[[str], float],
) -> tuple[float, str]:
    story_penalties = cooldown_state.get("story_penalties") or {}
    family_penalties = cooldown_state.get("family_penalties") or {}
    semantic_targets = cooldown_state.get("semantic_targets") or []
    if not story_penalties and not family_penalties and not semantic_targets:
        return 0.0, ""

    cand_story_key, cand_family_key = _candidate_story_keys(candidate)
    if cand_story_key and cand_story_key in story_penalties:
        return -float(story_penalties[cand_story_key]), cand_story_key
    if cand_family_key and cand_family_key in family_penalties:
        return -float(family_penalties[cand_family_key]), cand_story_key or cand_family_key

    best_sem_penalty = 0.0
    best_sem_key = ""
    if cand_passage and semantic_targets:
        for title, target_key, penalty in semantic_targets:
            sim = sim_fn(title)
            if sim >= SEMANTIC_COOLDOWN_THRESHOLD:
                scaled = float(penalty) * 0.85
                if scaled > best_sem_penalty:
                    best_sem_penalty = scaled
                    best_sem_key = target_key
    if best_sem_penalty > 0.0:
        return -round(best_sem_penalty, 4), best_sem_key
    return 0.0, ""


def _candidate_story_keys(candidate: NewsCandidate) -> tuple[str, str]:
    story_key = str(candidate.metadata.get("memory_story_key", "") or candidate.metadata.get("story_key", "") or "").strip()
    family_key = str(candidate.metadata.get("memory_story_family_key", "") or candidate.metadata.get("story_family_key", "") or "").strip()
    if story_key or family_key:
        return story_key, family_key
    identity = story_identity_for_candidate(candidate)
    return identity.story_key, identity.story_family_key


def _format_rank_reason(
    *,
    matched_topics: List[str],
    matched_sources: List[str],
    topic_weight: float,
    source_weight: float,
    list_adjustment: float,
    semantic_topics: List[str],
    feedback_examples: List[str],
    semantic_adjustment: float,
    cooldown_adjustment: float,
    cooldown_story_key: str,
) -> str:
    parts: List[str] = []
    if matched_topics and abs(topic_weight) > 1e-6:
        parts.append(f"topic {', '.join(matched_topics[:2])} ({topic_weight * TOPIC_WEIGHT_SCALE:+.2f})")
    if matched_sources and abs(source_weight) > 1e-6:
        parts.append(f"source {', '.join(matched_sources[:2])} ({source_weight * SOURCE_WEIGHT_SCALE:+.2f})")
    if abs(list_adjustment) > 1e-6:
        parts.append(f"profile list ({list_adjustment:+.2f})")
    if semantic_topics and abs(semantic_adjustment) > 1e-6:
        parts.append(f"semantic {', '.join(semantic_topics[:2])} ({semantic_adjustment:+.2f})")
    elif feedback_examples and abs(semantic_adjustment) > 1e-6:
        parts.append(f"similar to '{feedback_examples[0][:40]}' ({semantic_adjustment:+.2f})")
    if cooldown_adjustment < -1e-6:
        parts.append(f"repetitive cooldown {cooldown_story_key[:32]} ({cooldown_adjustment:+.2f})")
    return "; ".join(parts)


def learned_preference_effect_payload(effect: LearnedPreferenceRankEffect) -> Dict[str, Any]:
    payload = asdict(effect)
    payload["matched_topics"] = list(effect.matched_topics or [])
    payload["matched_sources"] = list(effect.matched_sources or [])
    payload["matched_semantic_topics"] = list(effect.matched_semantic_topics or [])
    payload["matched_feedback_examples"] = list(effect.matched_feedback_examples or [])
    return payload


def write_learned_preference_effect(candidate: NewsCandidate, effect: LearnedPreferenceRankEffect) -> None:
    payload = learned_preference_effect_payload(effect)
    candidate.metadata["learned_preference_adjustment"] = payload["score_adjustment"]
    candidate.metadata["learned_preference_topic_weight"] = payload["topic_weight"]
    candidate.metadata["learned_preference_source_weight"] = payload["source_weight"]
    candidate.metadata["learned_preference_list_adjustment"] = payload["list_adjustment"]
    candidate.metadata["learned_preference_semantic_adjustment"] = payload["semantic_adjustment"]
    candidate.metadata["learned_preference_cooldown_adjustment"] = payload["cooldown_adjustment"]
    candidate.metadata["learned_preference_matched_topics"] = payload["matched_topics"]
    candidate.metadata["learned_preference_matched_sources"] = payload["matched_sources"]
    candidate.metadata["learned_preference_matched_semantic_topics"] = payload["matched_semantic_topics"]
    candidate.metadata["learned_preference_matched_feedback_examples"] = payload["matched_feedback_examples"]
    candidate.metadata["learned_preference_cooldown_story_key"] = payload["cooldown_story_key"]
    candidate.metadata["learned_preference_reason"] = payload["reason"]


def clear_learned_preference_effect(candidate: NewsCandidate) -> None:
    for key in LEARNED_PREFERENCE_METADATA_KEYS:
        candidate.metadata.pop(key, None)


def learned_preference_effect_from_candidate(candidate: NewsCandidate) -> LearnedPreferenceRankEffect | None:
    if "learned_preference_adjustment" not in candidate.metadata:
        return None
    return LearnedPreferenceRankEffect(
        score_adjustment=_float(candidate.metadata.get("learned_preference_adjustment"), 0.0),
        matched_topics=_string_list(candidate.metadata.get("learned_preference_matched_topics")),
        matched_sources=_string_list(candidate.metadata.get("learned_preference_matched_sources")),
        topic_weight=_float(candidate.metadata.get("learned_preference_topic_weight"), 0.0),
        source_weight=_float(candidate.metadata.get("learned_preference_source_weight"), 0.0),
        list_adjustment=_float(candidate.metadata.get("learned_preference_list_adjustment"), 0.0),
        semantic_adjustment=_float(candidate.metadata.get("learned_preference_semantic_adjustment"), 0.0),
        cooldown_adjustment=_float(candidate.metadata.get("learned_preference_cooldown_adjustment"), 0.0),
        matched_semantic_topics=_string_list(candidate.metadata.get("learned_preference_matched_semantic_topics")),
        matched_feedback_examples=_string_list(candidate.metadata.get("learned_preference_matched_feedback_examples")),
        cooldown_story_key=str(candidate.metadata.get("learned_preference_cooldown_story_key", "") or ""),
        reason=str(candidate.metadata.get("learned_preference_reason", "") or ""),
    )


def _matched_topics(
    candidate: NewsCandidate,
    preferences: LearnedPreferences,
    *,
    decision: HeadlineDecision | None,
) -> List[str]:
    fields = _candidate_topic_fields(candidate, decision=decision)
    matched: List[str] = []
    for label in preferences.topic_weights:
        if _topic_matches(label, fields):
            matched.append(label)
    return matched


def _matched_sources(candidate: NewsCandidate, preferences: LearnedPreferences) -> List[str]:
    source_keys = _candidate_source_keys(candidate)
    matched: List[str] = []
    for label in preferences.source_weights:
        if _normalize_source(label) in source_keys:
            matched.append(label)
    return matched


def _matches_any_topic(candidate: NewsCandidate, values: List[str], *, decision: HeadlineDecision | None) -> bool:
    fields = _candidate_topic_fields(candidate, decision=decision)
    return any(_topic_matches(value, fields) for value in values)


def _matches_any_source(candidate: NewsCandidate, values: List[str]) -> bool:
    source_keys = _candidate_source_keys(candidate)
    return any(_normalize_source(value) in source_keys for value in values)


def _candidate_topic_fields(candidate: NewsCandidate, *, decision: HeadlineDecision | None) -> List[str]:
    values = [
        str(getattr(decision, "topic", "") or ""),
        str(candidate.metadata.get("topic_name", "") or ""),
        str(candidate.category or ""),
        str(candidate.title or ""),
        str(candidate.snippet or ""),
    ]
    return [_normalize_text(value) for value in values if _normalize_text(value)]


def _candidate_source_keys(candidate: NewsCandidate) -> set[str]:
    keys: set[str] = set()
    direct = _normalize_source(candidate.source)
    if direct:
        keys.add(direct)
    merged_sources = candidate.metadata.get("merged_sources", [])
    if isinstance(merged_sources, list):
        for item in merged_sources:
            normalized = _normalize_source(str(item))
            if normalized:
                keys.add(normalized)
    parsed = urlparse(candidate.url or "")
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    if host:
        keys.add(host)
    return keys


def _topic_matches(label: str, normalized_fields: List[str]) -> bool:
    normalized = _normalize_text(label)
    if not normalized:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)")
    for field in normalized_fields:
        if normalized == field:
            return True
        if len(normalized) >= 4 and pattern.search(field):
            return True
    return False


def _copy_preferences(preferences: LearnedPreferences) -> LearnedPreferences:
    return LearnedPreferences(
        schema_version=preferences.schema_version,
        updated_at=preferences.updated_at,
        preferred_topics=list(preferences.preferred_topics),
        avoided_topics=list(preferences.avoided_topics),
        preferred_sources=list(preferences.preferred_sources),
        avoided_sources=list(preferences.avoided_sources),
        topic_weights=dict(preferences.topic_weights),
        source_weights=dict(preferences.source_weights),
        notes=preferences.notes,
    )


def _parse_timestamp(value: datetime | str | None) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text = f"{text}T23:59:59+00:00"
    elif text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_label(value: Any, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def _normalize_text(value: str) -> str:
    return normalized_word_text(value)


def _normalize_source(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _clamp_weight(value: float) -> float:
    return round(max(-3.0, min(3.0, float(value))), 4)


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
