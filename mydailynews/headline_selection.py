from __future__ import annotations

from typing import Dict, List
import re
from urllib.parse import urlparse

from .ai.headline_analyzer import HeadlineAnalyzer
from .models import (
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from .utils import datetime_to_iso, utc_now


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
    candidates = annotate_event_clusters(candidates, filtering, since, debug)
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
    cluster_size = int(item.metadata.get("event_cluster_size", 1) or 1)
    if cluster_size > 1:
        score += min(0.7, 0.1 * (cluster_size - 1))
    if bool(item.metadata.get("event_cluster_multi_source")):
        cluster_source_count = int(item.metadata.get("event_cluster_source_count", 2) or 2)
        score += min(1.1, 0.25 * cluster_source_count)

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


def _normalized_source_key(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "").strip().lower())
    return cleaned


def _source_keys_for_candidate(candidate: NewsCandidate) -> List[str]:
    keys: set[str] = set()
    primary = _normalized_source_key(candidate.source)
    if primary:
        keys.add(primary)
    merged_sources = candidate.metadata.get("merged_sources", [])
    if isinstance(merged_sources, list):
        for value in merged_sources:
            normalized = _normalized_source_key(str(value))
            if normalized:
                keys.add(normalized)
    if keys:
        return sorted(keys)
    parsed = urlparse(candidate.url or "")
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return [host] if host else []


def _primary_source_key(candidate: NewsCandidate) -> str:
    direct = _normalized_source_key(candidate.source)
    if direct:
        return direct
    keys = _source_keys_for_candidate(candidate)
    return keys[0] if keys else ""


def _event_url_key(url: str) -> str:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if segments:
        return f"{host}/{'/'.join(segments[:3])}"
    return host


def _event_candidate_features(candidate: NewsCandidate, fallback_date) -> Dict[str, object]:
    published = candidate.published_at or fallback_date
    title_key = title_dedupe_key(candidate.title)
    tokens = set(tokenize_for_match(title_key or candidate.title))
    topic_key = str(candidate.metadata.get("topic_name", "")).strip().lower()
    return {
        "candidate": candidate,
        "published_at": published,
        "topic_key": topic_key,
        "tokens": tokens,
        "url_key": _event_url_key(candidate.url),
        "source_keys": _source_keys_for_candidate(candidate),
    }


def _cluster_similarity(features: Dict[str, object], cluster: Dict[str, object], window_hours: int) -> float:
    published_at = features["published_at"]
    earliest = cluster["earliest_published_at"]
    latest = cluster["latest_published_at"]
    if published_at < earliest or published_at > latest:
        distance_hours = (
            (earliest - published_at).total_seconds() / 3600.0
            if published_at < earliest
            else (published_at - latest).total_seconds() / 3600.0
        )
        if distance_hours > float(window_hours):
            return -1.0

    cluster_topic_keys = cluster["topic_keys"]
    topic_key = str(features["topic_key"])
    has_topic_conflict = bool(topic_key and cluster_topic_keys and topic_key not in cluster_topic_keys)

    tokens = features["tokens"]
    cluster_tokens = cluster["prototype_tokens"] or cluster["all_tokens"]
    overlap = len(tokens.intersection(cluster_tokens))
    union = len(tokens.union(cluster_tokens))
    token_ratio = overlap / max(1, min(len(tokens), len(cluster_tokens)))
    jaccard = overlap / union if union > 0 else 0.0

    url_key = str(features["url_key"])
    url_match = bool(url_key) and url_key in cluster["url_keys"]
    strong_text_match = token_ratio >= 0.55
    medium_text_match = token_ratio >= 0.40 and (overlap >= 3 or jaccard >= 0.30)
    if not url_match and not strong_text_match and not medium_text_match:
        return -1.0

    topic_bonus = 0.15 if topic_key and topic_key in cluster_topic_keys else 0.0
    topic_penalty = 0.25 if has_topic_conflict else 0.0
    return token_ratio + jaccard + (0.9 if url_match else 0.0) + topic_bonus - topic_penalty


def annotate_event_clusters(
    candidates: List[NewsCandidate],
    filtering: FilteringConfig,
    since,
    debug,
) -> List[NewsCandidate]:
    if not candidates:
        return candidates

    window_hours = max(2, int(getattr(filtering, "event_cluster_time_window_hours", 18)))
    ordered = sorted(
        candidates,
        key=lambda item: ((item.published_at or since), item.id),
        reverse=True,
    )
    feature_by_id = {
        item.id: _event_candidate_features(item, since)
        for item in ordered
    }
    clusters: List[Dict[str, object]] = []

    for candidate in ordered:
        features = feature_by_id[candidate.id]
        best_index = -1
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            score = _cluster_similarity(features, cluster, window_hours)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index < 0:
            clusters.append(
                {
                    "members": [candidate],
                    "all_tokens": set(features["tokens"]),
                    "prototype_tokens": set(features["tokens"]),
                    "url_keys": {features["url_key"]} if features["url_key"] else set(),
                    "source_keys": set(features["source_keys"]),
                    "topic_keys": {features["topic_key"]} if features["topic_key"] else set(),
                    "earliest_published_at": features["published_at"],
                    "latest_published_at": features["published_at"],
                    "representative_id": candidate.id,
                }
            )
            continue

        cluster = clusters[best_index]
        cluster["members"].append(candidate)
        cluster["all_tokens"].update(features["tokens"])
        if features["url_key"]:
            cluster["url_keys"].add(features["url_key"])
        cluster["source_keys"].update(features["source_keys"])
        if features["topic_key"]:
            cluster["topic_keys"].add(features["topic_key"])
        if features["published_at"] < cluster["earliest_published_at"]:
            cluster["earliest_published_at"] = features["published_at"]
        if features["published_at"] > cluster["latest_published_at"]:
            cluster["latest_published_at"] = features["published_at"]

        representative = feature_by_id[cluster["representative_id"]]["candidate"]
        representative_rank = (
            len(representative.snippet or ""),
            representative.published_at or since,
        )
        candidate_rank = (
            len(candidate.snippet or ""),
            candidate.published_at or since,
        )
        if candidate_rank > representative_rank:
            cluster["representative_id"] = candidate.id
            cluster["prototype_tokens"] = set(features["tokens"])

    ordered_clusters = sorted(
        clusters,
        key=lambda cluster: (
            cluster["latest_published_at"],
            str(feature_by_id[cluster["representative_id"]]["candidate"].title or "").lower(),
        ),
        reverse=True,
    )

    multi_source_clusters = 0
    for index, cluster in enumerate(ordered_clusters, start=1):
        cluster_id = f"evt-{index:03d}"
        representative = feature_by_id[cluster["representative_id"]]["candidate"]
        label = (representative.title or "").strip()[:160] or cluster_id
        members = cluster["members"]
        source_count = len(cluster["source_keys"])
        is_multi_source = source_count >= 2
        if is_multi_source:
            multi_source_clusters += 1
        latest_iso = datetime_to_iso(cluster["latest_published_at"])
        cluster_size = len(members)
        for member in members:
            member.metadata["event_cluster_id"] = cluster_id
            member.metadata["event_cluster_label"] = label
            member.metadata["event_cluster_size"] = cluster_size
            member.metadata["event_cluster_source_count"] = source_count
            member.metadata["event_cluster_multi_source"] = is_multi_source
            member.metadata["event_cluster_latest_published_at"] = latest_iso

    debug.log(
        "headline.heuristics",
        "event_clusters",
        clusters=len(ordered_clusters),
        multi_source_clusters=multi_source_clusters,
        window_hours=window_hours,
    )
    return candidates


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
    source_counts: Dict[str, int] = {}
    cluster_counts: Dict[str, int] = {}
    source_cap = max(0, int(getattr(filtering, "max_selected_per_source", 0)))
    cluster_cap = max(0, int(getattr(filtering, "max_selected_per_event_cluster", 0)))
    prefer_multi_source = bool(getattr(filtering, "prefer_multi_source_clusters", False))
    multi_source_bonus = max(0.0, float(getattr(filtering, "multi_source_cluster_bonus", 0.0)))

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            decisions.get(item.id, HeadlineDecision(item.id, 0)).score
            + (
                multi_source_bonus
                if prefer_multi_source and bool(item.metadata.get("event_cluster_multi_source"))
                else 0.0
            ),
            decisions.get(item.id, HeadlineDecision(item.id, 0)).score,
            item.published_at or utc_now(),
        ),
        reverse=True,
    )

    def try_select(
        candidate: NewsCandidate,
        *,
        require_cutoff: bool,
        enforce_source_cap: bool,
        enforce_cluster_cap: bool,
    ) -> bool:
        if len(selected) >= filtering.max_selected_articles:
            return False
        decision = decisions.get(candidate.id)
        if not decision:
            return False
        if require_cutoff and decision.score < filtering.headline_score_cutoff:
            return False
        if candidate.id in seen_duplicate_targets:
            return False
        topic = decision.topic or candidate.metadata.get("topic_name", "")
        topic_limit = topic_limits.get(topic)
        if topic_limit is not None and topic_counts.get(topic, 0) >= int(topic_limit):
            return False
        source_key = _primary_source_key(candidate)
        if enforce_source_cap and source_cap > 0 and source_key and source_counts.get(source_key, 0) >= source_cap:
            return False
        cluster_id = str(candidate.metadata.get("event_cluster_id", "")).strip()
        if (
            enforce_cluster_cap
            and cluster_cap > 0
            and cluster_id
            and cluster_counts.get(cluster_id, 0) >= cluster_cap
        ):
            return False
        seen_duplicate_targets.add(candidate.id)
        selected.append(SelectedArticle(candidate=candidate, decision=decision))
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if source_key:
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
        if cluster_id:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        return True

    def select_from_ranked(
        *,
        require_cutoff: bool,
        enforce_source_cap: bool,
        enforce_cluster_cap: bool,
    ) -> None:
        for candidate in sorted_candidates:
            try_select(
                candidate,
                require_cutoff=require_cutoff,
                enforce_source_cap=enforce_source_cap,
                enforce_cluster_cap=enforce_cluster_cap,
            )
            if len(selected) >= filtering.max_selected_articles:
                break

    select_from_ranked(
        require_cutoff=True,
        enforce_source_cap=True,
        enforce_cluster_cap=True,
    )
    if filtering.fill_selected_articles and len(selected) < filtering.max_selected_articles:
        select_from_ranked(
            require_cutoff=False,
            enforce_source_cap=True,
            enforce_cluster_cap=True,
        )
    if filtering.fill_selected_articles and len(selected) < filtering.max_selected_articles and cluster_cap > 0:
        select_from_ranked(
            require_cutoff=False,
            enforce_source_cap=True,
            enforce_cluster_cap=False,
        )
    if filtering.fill_selected_articles and len(selected) < filtering.max_selected_articles and source_cap > 0:
        select_from_ranked(
            require_cutoff=False,
            enforce_source_cap=False,
            enforce_cluster_cap=False,
        )
    return selected
