from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

from mydailynews.app.models import HeadlineDecision, NewsCandidate
from mydailynews.domain.text_similarity import normalized_word_text
from mydailynews.memory.feedback import FeedbackEvent
from mydailynews.memory.learned_preferences import LearnedPreferences


TOPIC_WEIGHT_SCALE = 0.2
SOURCE_WEIGHT_SCALE = 0.2
LIST_TOPIC_BONUS = 0.2
LIST_TOPIC_PENALTY = 0.35
LIST_SOURCE_BONUS = 0.15
LIST_SOURCE_PENALTY = 0.35
MAX_RANK_ADJUSTMENT = 1.0
LEARNED_PREFERENCE_METADATA_KEYS = (
    "learned_preference_adjustment",
    "learned_preference_topic_weight",
    "learned_preference_source_weight",
    "learned_preference_list_adjustment",
    "learned_preference_matched_topics",
    "learned_preference_matched_sources",
)


@dataclass(frozen=True)
class LearnedPreferenceDelta:
    topic_weights: Dict[str, float]
    source_weights: Dict[str, float]

    @property
    def changed(self) -> bool:
        return bool(self.topic_weights or self.source_weights)


@dataclass(frozen=True)
class LearnedPreferenceRankEffect:
    score_adjustment: float = 0.0
    matched_topics: List[str] | None = None
    matched_sources: List[str] | None = None
    topic_weight: float = 0.0
    source_weight: float = 0.0
    list_adjustment: float = 0.0

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


def apply_feedback_event(preferences: LearnedPreferences, event: FeedbackEvent) -> tuple[LearnedPreferences, LearnedPreferenceDelta]:
    delta = preference_delta_for_event(event)
    updated = LearnedPreferences(
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
    for topic, change in delta.topic_weights.items():
        updated.topic_weights[topic] = _clamp_weight(float(updated.topic_weights.get(topic, 0.0)) + float(change))
    for source, change in delta.source_weights.items():
        updated.source_weights[source] = _clamp_weight(float(updated.source_weights.get(source, 0.0)) + float(change))
    return updated, delta


def rebuild_learned_preferences(
    events: Iterable[FeedbackEvent],
    base_preferences: LearnedPreferences | None = None,
) -> LearnedPreferences:
    preferences = base_preferences or LearnedPreferences()
    for event in events:
        preferences, _ = apply_feedback_event(preferences, event)
    return preferences


def ranking_adjustment_for_candidate(
    candidate: NewsCandidate,
    learned_preferences: LearnedPreferences | None,
    *,
    decision: HeadlineDecision | None = None,
) -> LearnedPreferenceRankEffect:
    if learned_preferences is None:
        return LearnedPreferenceRankEffect(matched_topics=[], matched_sources=[])

    matched_topics = _matched_topics(candidate, learned_preferences, decision=decision)
    matched_sources = _matched_sources(candidate, learned_preferences)
    topic_weight = sum(float(learned_preferences.topic_weights.get(topic, 0.0)) for topic in matched_topics)
    source_weight = sum(float(learned_preferences.source_weights.get(source, 0.0)) for source in matched_sources)

    list_adjustment = 0.0
    if _matches_any_topic(candidate, learned_preferences.preferred_topics, decision=decision):
        list_adjustment += LIST_TOPIC_BONUS
    if _matches_any_topic(candidate, learned_preferences.avoided_topics, decision=decision):
        list_adjustment -= LIST_TOPIC_PENALTY
    if _matches_any_source(candidate, learned_preferences.preferred_sources):
        list_adjustment += LIST_SOURCE_BONUS
    if _matches_any_source(candidate, learned_preferences.avoided_sources):
        list_adjustment -= LIST_SOURCE_PENALTY

    adjustment = topic_weight * TOPIC_WEIGHT_SCALE + source_weight * SOURCE_WEIGHT_SCALE + list_adjustment
    adjustment = round(max(-MAX_RANK_ADJUSTMENT, min(MAX_RANK_ADJUSTMENT, adjustment)), 4)
    return LearnedPreferenceRankEffect(
        score_adjustment=adjustment,
        matched_topics=matched_topics,
        matched_sources=matched_sources,
        topic_weight=round(topic_weight, 4),
        source_weight=round(source_weight, 4),
        list_adjustment=round(list_adjustment, 4),
    )


def learned_preference_effect_payload(effect: LearnedPreferenceRankEffect) -> Dict[str, Any]:
    payload = asdict(effect)
    payload["matched_topics"] = list(effect.matched_topics or [])
    payload["matched_sources"] = list(effect.matched_sources or [])
    return payload


def write_learned_preference_effect(candidate: NewsCandidate, effect: LearnedPreferenceRankEffect) -> None:
    payload = learned_preference_effect_payload(effect)
    candidate.metadata["learned_preference_adjustment"] = payload["score_adjustment"]
    candidate.metadata["learned_preference_topic_weight"] = payload["topic_weight"]
    candidate.metadata["learned_preference_source_weight"] = payload["source_weight"]
    candidate.metadata["learned_preference_list_adjustment"] = payload["list_adjustment"]
    candidate.metadata["learned_preference_matched_topics"] = payload["matched_topics"]
    candidate.metadata["learned_preference_matched_sources"] = payload["matched_sources"]


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
    for field in normalized_fields:
        if normalized == field:
            return True
        if len(normalized) >= 5 and normalized in field:
            return True
    return False


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
