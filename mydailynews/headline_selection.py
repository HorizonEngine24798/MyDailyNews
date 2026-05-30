from __future__ import annotations

from typing import Dict, List
import re

from .ai.headline_analyzer import HeadlineAnalyzer
from .models import (
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from .utils import utc_now


def union_candidates_by_id(*groups: List[NewsCandidate]) -> List[NewsCandidate]:
    merged: List[NewsCandidate] = []
    seen: set[str] = set()
    for group in groups:
        for candidate in group:
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            merged.append(candidate)
    return merged


def decisions_for_brief(
    candidates: List[NewsCandidate],
    shared_decisions: Dict[str, HeadlineDecision],
    topics: List[TopicConfig],
) -> Dict[str, HeadlineDecision]:
    decisions: Dict[str, HeadlineDecision] = {}
    for candidate in candidates:
        shared = shared_decisions.get(candidate.id)
        if shared is None:
            continue
        decisions[candidate.id] = HeadlineDecision(
            candidate_id=candidate.id,
            score=shared.score,
            topic=HeadlineAnalyzer.best_topic_for_candidate(candidate, topics),
        )
    return decisions


def limit_candidates_for_ai(
    candidates: List[NewsCandidate],
    topics: List[TopicConfig],
    filtering: FilteringConfig,
    since,
    *,
    user_memory: UserMemory,
    debug,
) -> List[NewsCandidate]:
    max_total = filtering.max_candidates_for_ai
    if max_total <= 0:
        return []

    candidates = dedupe_similar_titles(candidates, debug)
    scored = heuristic_ranked_candidates(candidates, topics, since, user_memory)
    if not scored:
        return []

    score_by_id = {item.id: score for item, score in scored}
    ranked = [item for item, _ in scored]
    pool_target = min(len(ranked), max_total * 2)
    nonnegative = [item for item in ranked if score_by_id.get(item.id, 0.0) >= 0.0]
    if len(nonnegative) < max_total:
        candidate_pool = ranked[:pool_target]
    else:
        candidate_pool = nonnegative[:pool_target]
    debug.log(
        "headline.heuristics",
        "prefilter_complete",
        input=len(candidates),
        pool=len(candidate_pool),
        max_total=max_total,
    )

    selected: List[NewsCandidate] = []
    selected_ids: set[str] = set()
    enabled_topics = [topic for topic in topics if topic.enabled]
    per_topic = max(1, max_total // max(1, len(enabled_topics))) if enabled_topics else 0

    for topic in enabled_topics:
        topic_items = [item for item in candidate_pool if candidate_topic_match(item, topic) > 0.0]
        topic_items = sort_by_heuristic_then_time(topic_items, score_by_id, since)
        for item in topic_items[:per_topic]:
            if item.id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.id)

    remaining = [item for item in candidate_pool if item.id not in selected_ids]
    for item in sort_by_heuristic_then_time(remaining, score_by_id, since):
        if len(selected) >= max_total:
            break
        selected.append(item)
        selected_ids.add(item.id)

    return selected[:max_total]


def sort_by_heuristic_then_time(
    candidates: List[NewsCandidate],
    score_by_id: Dict[str, float],
    fallback_date,
) -> List[NewsCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            score_by_id.get(item.id, -999.0),
            item.published_at or fallback_date,
        ),
        reverse=True,
    )


def heuristic_ranked_candidates(
    candidates: List[NewsCandidate],
    topics: List[TopicConfig],
    since,
    user_memory: UserMemory,
) -> List[tuple[NewsCandidate, float]]:
    scored = [
        (
            item,
            candidate_heuristic_score(
                item,
                topics,
                since,
                user_memory=user_memory,
            ),
        )
        for item in candidates
    ]
    scored.sort(
        key=lambda pair: (
            pair[1],
            pair[0].published_at or since,
        ),
        reverse=True,
    )
    return scored


def candidate_heuristic_score(
    item: NewsCandidate,
    topics: List[TopicConfig],
    since,
    *,
    user_memory: UserMemory,
) -> float:
    score = 0.0
    published_at = item.published_at or since
    age_hours = max(0.0, (utc_now() - published_at).total_seconds() / 3600.0)
    score += max(0.0, 3.0 - 0.07 * age_hours)

    topic_name = str(item.metadata.get("topic_name", "")).strip()
    if topic_name:
        score += 2.0

    if topic_name and topic_is_enabled(topics, topic_name):
        score += 1.0

    topic_match = 0.0
    for topic in topics:
        if getattr(topic, "enabled", False):
            topic_match = max(topic_match, candidate_topic_match(item, topic))
    score += min(3.0, topic_match * 1.2)

    merged_count = int(item.metadata.get("merged_count", 1) or 1)
    if merged_count > 1:
        score += min(1.5, 0.35 * (merged_count - 1))

    snippet_len = len(item.snippet or "")
    if snippet_len >= 260:
        score += 0.8
    elif snippet_len >= 120:
        score += 0.4
    elif snippet_len < 40:
        score -= 0.4

    title_len = len((item.title or "").strip())
    if 24 <= title_len <= 140:
        score += 0.5
    elif title_len < 12 or title_len > 180:
        score -= 0.8

    lowered_title = (item.title or "").lower()
    if any(needle in lowered_title for needle in ("live updates", "watch live", "photo gallery", "opinion:", "newsletter")):
        score -= 1.0

    preferred_sources = {source.lower() for source in user_memory.preferred_sources}
    avoided_sources = {source.lower() for source in user_memory.avoided_sources}
    source_name = (item.source or "").lower()
    if source_name in preferred_sources:
        score += 0.9
    if source_name in avoided_sources:
        score -= 2.5

    return round(score, 4)


def topic_is_enabled(topics: List[TopicConfig], topic_name: str) -> bool:
    for topic in topics:
        if getattr(topic, "enabled", False) and topic.name == topic_name:
            return True
    return False


def candidate_topic_match(item: NewsCandidate, topic: TopicConfig) -> float:
    item_topic = str(item.metadata.get("topic_name", "")).strip()
    if item_topic and item_topic == topic.name:
        return 1.0

    text = f"{item.title or ''} {item.snippet or ''}".lower()
    text_tokens = set(tokenize_for_match(text))
    if not text_tokens:
        return 0.0

    query_tokens = set(tokenize_for_match(topic.name))
    query_tokens.update(tokenize_for_match(topic.description))
    for query in topic.queries or []:
        query_tokens.update(tokenize_for_match(query))
    if not query_tokens:
        return 0.0
    overlap = len(text_tokens.intersection(query_tokens))
    if overlap <= 0:
        return 0.0
    return min(1.0, overlap / max(3, int(len(query_tokens) * 0.12)))


def tokenize_for_match(text: str) -> List[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "latest",
        "today",
        "news",
        "major",
        "about",
    }
    tokens = [token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in stop]
    return tokens


def dedupe_similar_titles(candidates: List[NewsCandidate], debug) -> List[NewsCandidate]:
    groups: Dict[str, List[NewsCandidate]] = {}
    for item in candidates:
        key = title_dedupe_key(item.title)
        if not key:
            groups.setdefault(item.id, []).append(item)
            continue
        groups.setdefault(key, []).append(item)

    deduped: List[NewsCandidate] = []
    removed = 0
    for group in groups.values():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        winner = max(
            group,
            key=lambda candidate: (
                len(candidate.snippet or ""),
                candidate.published_at or utc_now(),
            ),
        )
        related_ids = [candidate.id for candidate in group if candidate.id != winner.id]
        if related_ids:
            winner.metadata["headline_dupe_ids"] = related_ids
            winner.metadata["headline_dupe_count"] = len(group)
            removed += len(related_ids)
        deduped.append(winner)
    if removed > 0:
        debug.log("headline.heuristics", "title_dedupe", removed=removed, kept=len(deduped))
    return deduped


def title_dedupe_key(title: str) -> str:
    tokens = re.findall(r"[a-z0-9]{3,}", (title or "").lower())
    if not tokens:
        return ""
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "live",
        "latest",
        "news",
        "update",
        "updates",
        "says",
        "say",
    }
    core = [token for token in tokens if token not in stop]
    if len(core) < 4:
        core = tokens
    return " ".join(core[:10])


def select_articles(
    candidates: List[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
    topics: List[TopicConfig],
    filtering: FilteringConfig,
) -> List[SelectedArticle]:
    selected: List[SelectedArticle] = []
    seen_duplicate_targets: set[str] = set()
    topic_limits = {
        topic.name: topic.max_selected_articles
        for topic in topics
        if topic.enabled and topic.max_selected_articles is not None
    }
    topic_counts: Dict[str, int] = {}
    sorted_candidates = sorted(
        candidates,
        key=lambda item: decisions.get(item.id, HeadlineDecision(item.id, 0)).score,
        reverse=True,
    )

    def try_select(candidate: NewsCandidate, require_cutoff: bool) -> None:
        if len(selected) >= filtering.max_selected_articles:
            return
        decision = decisions.get(candidate.id)
        if not decision:
            return
        if require_cutoff and decision.score < filtering.headline_score_cutoff:
            return
        if candidate.id in seen_duplicate_targets:
            return
        topic = decision.topic or candidate.metadata.get("topic_name", "")
        topic_limit = topic_limits.get(topic)
        if topic_limit is not None and topic_counts.get(topic, 0) >= int(topic_limit):
            return
        seen_duplicate_targets.add(candidate.id)
        selected.append(SelectedArticle(candidate=candidate, decision=decision))
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    for candidate in sorted_candidates:
        try_select(candidate, require_cutoff=True)
        if len(selected) >= filtering.max_selected_articles:
            break

    if filtering.fill_selected_articles and len(selected) < filtering.max_selected_articles:
        for candidate in sorted_candidates:
            try_select(candidate, require_cutoff=False)
            if len(selected) >= filtering.max_selected_articles:
                break
    return selected
