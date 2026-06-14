from __future__ import annotations

from typing import Any, Dict, List
import re
from urllib.parse import urlparse

from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.domain.candidate_annotations import (
    candidate_event_cluster_annotation,
    candidate_event_cluster_id,
    candidate_profile_match_annotation,
    reset_selection_annotation,
    set_event_cluster_annotation,
    set_profile_match_annotation,
    set_selection_selected_annotation,
    set_selection_skip_annotation,
)
from mydailynews.app.models import (
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from mydailynews.common.utils import datetime_to_iso, utc_now

PROFILE_GEO_MATCH_BONUS = 0.7
PROFILE_WANTS_MATCH_BONUS = 0.35
PROFILE_BEAT_WEIGHT_SCALE = 0.25
PROFILE_AVOID_MATCH_PENALTY = 1.1
PROFILE_RANK_GEO_MATCH_BONUS = 0.25
PROFILE_RANK_WANTS_MATCH_BONUS = 0.15
PROFILE_RANK_BEAT_WEIGHT_SCALE = 0.1
PROFILE_RANK_AVOID_PENALTY = 0.75


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
            personal_relevance=shared.personal_relevance,
            impact=shared.impact,
            novelty=shared.novelty,
            urgency=shared.urgency,
            actionability=shared.actionability,
            confidence=shared.confidence,
            reason=shared.reason,
            skip_reason=shared.skip_reason,
            angle_type=shared.angle_type,
            selection_reason_code=shared.selection_reason_code,
            selection_rank_score=shared.selection_rank_score,
            selection_rank_mode=shared.selection_rank_mode,
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
    if max_total is None:
        max_total_label: int | str = "all"
    else:
        max_total = int(max_total)
        if max_total <= 0:
            return []
        max_total_label = max_total

    preferred_sources = {source.lower().strip() for source in user_memory.preferred_sources if str(source).strip()}
    avoided_sources = {source.lower().strip() for source in user_memory.avoided_sources if str(source).strip()}
    candidates = dedupe_similar_titles(candidates, debug)
    candidates = annotate_event_clusters(candidates, filtering, since, debug)
    for item in candidates:
        source_key = (item.source or "").strip().lower()
        profile_signals = _profile_signal_matches(item, user_memory)
        set_profile_match_annotation(
            item,
            source_preferred=bool(source_key and source_key in preferred_sources),
            source_avoided=bool(source_key and source_key in avoided_sources),
            geo_matches=profile_signals["geo_matches"],
            wants_matches=profile_signals["wants_matches"],
            avoid_matches=profile_signals["avoid_matches"],
            beat_matches=profile_signals["beat_matches"],
            beat_weight_sum=float(profile_signals["beat_weight_sum"]),
        )
    scored = heuristic_ranked_candidates(candidates, topics, since, user_memory)
    if not scored:
        return []

    score_by_id = {item.id: score for item, score in scored}
    ranked = [item for item, _ in scored]
    nonnegative = [item for item in ranked if score_by_id.get(item.id, 0.0) >= 0.0]
    if max_total is None:
        candidate_pool = nonnegative if nonnegative else ranked
    else:
        pool_target = min(len(ranked), max_total * 2)
        if len(nonnegative) < max_total:
            candidate_pool = ranked[:pool_target]
        else:
            candidate_pool = nonnegative[:pool_target]
    debug.log(
        "headline.heuristics",
        "prefilter_complete",
        input=len(candidates),
        pool=len(candidate_pool),
        max_total=max_total_label,
    )

    if max_total is None:
        return sort_by_heuristic_then_time(candidate_pool, score_by_id, since)

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
    event_cluster = candidate_event_cluster_annotation(item)
    cluster_size = event_cluster.size if event_cluster is not None else 1
    if cluster_size > 1:
        score += min(0.7, 0.1 * (cluster_size - 1))
    if event_cluster is not None and event_cluster.multi_source:
        cluster_source_count = event_cluster.source_count
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

    geo_match, wants_match_count, avoid_match_count, beat_weight_sum = _profile_scoring_signals(item, user_memory)
    if geo_match:
        score += PROFILE_GEO_MATCH_BONUS
    wants_bonus = min(
        1.2,
        PROFILE_WANTS_MATCH_BONUS * wants_match_count
        + PROFILE_BEAT_WEIGHT_SCALE * beat_weight_sum,
    )
    score += wants_bonus
    if avoid_match_count:
        score -= min(2.8, PROFILE_AVOID_MATCH_PENALTY * avoid_match_count)

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


def _normalized_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _normalized_profile_terms(values: List[str], *, max_items: int = 10, max_chars: int = 48) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for raw in values:
        cleaned = " ".join(str(raw or "").split()).strip().lower()
        if not cleaned:
            continue
        cleaned = cleaned[:max_chars]
        normalized = _normalized_match_text(cleaned)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
        if len(terms) >= max_items:
            break
    return terms


def _normalized_weighted_beats(beats: Dict[str, float]) -> List[tuple[str, float]]:
    rows: List[tuple[str, float]] = []
    for raw_name, raw_weight in (beats or {}).items():
        name = _normalized_match_text(" ".join(str(raw_name or "").split())[:48])
        if not name:
            continue
        try:
            weight = float(raw_weight)
        except Exception:
            weight = 0.0
        rows.append((name, max(0.0, min(3.0, weight))))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows[:8]


def _profile_term_matches(text: str, terms: List[str]) -> List[str]:
    if not text or not terms:
        return []
    matches: List[str] = []
    padded = f" {text} "
    for term in terms:
        if not term:
            continue
        phrase = f" {term} "
        if phrase in padded:
            matches.append(term)
            continue
        if term in text and len(term) >= 5:
            matches.append(term)
    return matches


def _profile_signal_text(candidate: NewsCandidate) -> str:
    event_cluster = candidate_event_cluster_annotation(candidate)
    fields: List[str] = [
        str(candidate.title or ""),
        str(candidate.snippet or ""),
        str(candidate.source or ""),
        str(candidate.url or ""),
        str(candidate.metadata.get("topic_name", "") or ""),
        str(event_cluster.label if event_cluster is not None else ""),
    ]
    tags = candidate.tags if isinstance(candidate.tags, list) else []
    fields.extend(str(tag) for tag in tags[:6])
    return _normalized_match_text(" ".join(fields))


def _profile_signal_matches(candidate: NewsCandidate, user_memory: UserMemory) -> Dict[str, Any]:
    text = _profile_signal_text(candidate)
    geo_terms = _normalized_profile_terms(user_memory.geography_focus, max_items=8)
    wants_terms = _normalized_profile_terms(user_memory.wants, max_items=10)
    avoid_terms = _normalized_profile_terms((user_memory.avoid or []) + (user_memory.avoided_topics or []), max_items=10)
    beats = _normalized_weighted_beats(user_memory.beats)

    geo_matches = _profile_term_matches(text, geo_terms)
    wants_matches = _profile_term_matches(text, wants_terms)
    avoid_matches = _profile_term_matches(text, avoid_terms)

    beat_matches: List[str] = []
    beat_weight_sum = 0.0
    for beat_name, weight in beats:
        if not beat_name:
            continue
        if _profile_term_matches(text, [beat_name]):
            beat_matches.append(beat_name)
            beat_weight_sum += float(weight)
    return {
        "geo_matches": geo_matches,
        "wants_matches": wants_matches,
        "avoid_matches": avoid_matches,
        "beat_matches": beat_matches,
        "beat_weight_sum": beat_weight_sum,
    }


def _profile_scoring_signals(candidate: NewsCandidate, user_memory: UserMemory) -> tuple[bool, int, int, float]:
    annotation = candidate_profile_match_annotation(candidate)
    if annotation is not None:
        return (
            bool(annotation.geo_match or annotation.geo_matches),
            max(0, int(annotation.wants_match_count or len(annotation.wants_matches))),
            max(0, int(annotation.avoid_match_count or len(annotation.avoid_matches))),
            float(annotation.beat_weight_sum or 0.0),
        )
    profile_signals = _profile_signal_matches(candidate, user_memory)
    return (
        bool(profile_signals["geo_matches"]),
        len(profile_signals["wants_matches"]),
        len(profile_signals["avoid_matches"]),
        float(profile_signals["beat_weight_sum"]),
    )


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
            set_event_cluster_annotation(
                member,
                cluster_id=cluster_id,
                label=label,
                size=cluster_size,
                source_count=source_count,
                multi_source=is_multi_source,
                latest_published_at=latest_iso,
            )

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


COMPOSITE_DIM_WEIGHTS: Dict[str, float] = {
    "personal_relevance": 0.30,
    "impact": 0.20,
    "novelty": 0.18,
    "actionability": 0.15,
    "urgency": 0.10,
    "confidence": 0.07,
}


def _decision_has_multifactor_signal(decision: HeadlineDecision) -> bool:
    if str(decision.reason or "").strip():
        return True
    if str(decision.angle_type or "").strip():
        return True
    if str(decision.skip_reason or "").strip():
        return True
    values = [
        float(decision.personal_relevance),
        float(decision.impact),
        float(decision.novelty),
        float(decision.actionability),
        float(decision.urgency),
        float(decision.confidence),
    ]
    return any(abs(value - 5.0) > 1e-6 for value in values)


def ranking_score_for_candidate(
    decision: HeadlineDecision,
    candidate: NewsCandidate,
    filtering: FilteringConfig,
    user_memory: UserMemory | None = None,
) -> tuple[float, str]:
    use_composite = bool(getattr(filtering, "use_multifactor_composite_ranking", False))
    if use_composite and _decision_has_multifactor_signal(decision):
        base_score = 0.0
        for key, weight in COMPOSITE_DIM_WEIGHTS.items():
            base_score += float(getattr(decision, key, 5.0)) * float(weight)
        rank_mode = "composite"
    else:
        base_score = float(decision.score)
        rank_mode = "score"

    adjusted = float(base_score)
    prefer_multi_source = bool(getattr(filtering, "prefer_multi_source_clusters", False))
    multi_source_bonus = max(0.0, float(getattr(filtering, "multi_source_cluster_bonus", 0.0)))
    event_cluster = candidate_event_cluster_annotation(candidate)
    if prefer_multi_source and event_cluster is not None and event_cluster.multi_source:
        adjusted += multi_source_bonus

    profile_annotation = candidate_profile_match_annotation(candidate)
    if profile_annotation is not None and profile_annotation.source_preferred:
        adjusted += max(0.0, float(getattr(filtering, "source_preference_bonus", 0.0)))
    if profile_annotation is not None and profile_annotation.source_avoided:
        adjusted -= max(0.0, float(getattr(filtering, "source_avoid_penalty", 0.0)))

    if user_memory is not None:
        geo_match, wants_match_count, avoid_match_count, beat_weight_sum = _profile_scoring_signals(candidate, user_memory)
        if geo_match:
            adjusted += PROFILE_RANK_GEO_MATCH_BONUS
        wants_bonus = min(
            0.65,
            PROFILE_RANK_WANTS_MATCH_BONUS * wants_match_count
            + PROFILE_RANK_BEAT_WEIGHT_SCALE * beat_weight_sum,
        )
        adjusted += wants_bonus
        if avoid_match_count:
            adjusted -= min(1.6, PROFILE_RANK_AVOID_PENALTY * avoid_match_count)

    return round(adjusted, 4), rank_mode


def selection_reason_counters(decisions: Dict[str, HeadlineDecision]) -> Dict[str, Dict[str, int]]:
    selected_counts: Dict[str, int] = {}
    skipped_counts: Dict[str, int] = {}
    for decision in decisions.values():
        code = str(decision.selection_reason_code or "").strip()
        if not code:
            continue
        if code.startswith("selected_"):
            selected_counts[code] = selected_counts.get(code, 0) + 1
            continue
        if code.startswith("skipped_"):
            skipped_counts[code] = skipped_counts.get(code, 0) + 1
            continue
    return {
        "selected": dict(sorted(selected_counts.items())),
        "skipped": dict(sorted(skipped_counts.items())),
    }


def selection_rationale_rows(
    candidates: List[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        decision = decisions.get(candidate.id)
        if decision is None:
            continue
        code = str(decision.selection_reason_code or "").strip()
        rows.append(
            {
                "candidate_id": candidate.id,
                "rank_score": round(float(decision.selection_rank_score), 4),
                "rank_mode": str(decision.selection_rank_mode or "score"),
                "reason_code": code,
                "selected": code.startswith("selected_"),
                "score": round(float(decision.score), 4),
                "topic": str(decision.topic or candidate.metadata.get("topic_name", "")),
                "composite_dimensions": {
                    "personal_relevance": round(float(decision.personal_relevance), 4),
                    "impact": round(float(decision.impact), 4),
                    "novelty": round(float(decision.novelty), 4),
                    "actionability": round(float(decision.actionability), 4),
                    "urgency": round(float(decision.urgency), 4),
                    "confidence": round(float(decision.confidence), 4),
                },
            }
        )
    rows.sort(
        key=lambda row: (
            float(row.get("rank_score", 0.0)),
            float(row.get("score", 0.0)),
        ),
        reverse=True,
    )
    return rows


def select_articles(
    candidates: List[NewsCandidate],
    decisions: Dict[str, HeadlineDecision],
    topics: List[TopicConfig],
    filtering: FilteringConfig,
    user_memory: UserMemory | None = None,
) -> List[SelectedArticle]:
    selected: List[SelectedArticle] = []
    max_selected = filtering.max_selected_articles
    if max_selected is not None:
        max_selected = int(max_selected)
        if max_selected <= 0:
            return []
    seen_duplicate_targets: set[str] = set()
    selected_ids: set[str] = set()
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
    novelty_floor = max(0.0, min(10.0, float(getattr(filtering, "min_novelty_for_selection", 0.0))))

    for candidate in candidates:
        decision = decisions.get(candidate.id)
        if decision is None:
            continue
        rank_score, rank_mode = ranking_score_for_candidate(
            decision,
            candidate,
            filtering,
            user_memory=user_memory,
        )
        decision.selection_rank_score = rank_score
        decision.selection_rank_mode = rank_mode
        decision.selection_reason_code = ""
        reset_selection_annotation(candidate, rank_score=rank_score, rank_mode=rank_mode)

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            decisions.get(item.id, HeadlineDecision(item.id, 0)).selection_rank_score,
            decisions.get(item.id, HeadlineDecision(item.id, 0)).score,
            item.published_at or utc_now(),
        ),
        reverse=True,
    )

    def mark_skip(candidate: NewsCandidate, decision: HeadlineDecision | None, code: str) -> None:
        if decision is None:
            return
        if decision.selection_reason_code.startswith("selected_"):
            return
        decision.selection_reason_code = code
        set_selection_skip_annotation(candidate, code)

    def mark_selected(candidate: NewsCandidate, decision: HeadlineDecision, code: str) -> None:
        decision.selection_reason_code = code
        set_selection_selected_annotation(candidate, code)

    def try_select(
        candidate: NewsCandidate,
        *,
        require_cutoff: bool,
        enforce_source_cap: bool,
        enforce_cluster_cap: bool,
    ) -> bool:
        decision = decisions.get(candidate.id)
        if not decision:
            return False
        if max_selected is not None and len(selected) >= max_selected:
            mark_skip(candidate, decision, "skipped_capacity")
            return False
        if require_cutoff and decision.score < filtering.headline_score_cutoff:
            mark_skip(candidate, decision, "skipped_below_cutoff")
            return False
        if (
            novelty_floor > 0.0
            and decision.novelty < novelty_floor
            and decision.impact < 6.5
            and decision.score < max(float(filtering.headline_score_cutoff), 7.0)
        ):
            mark_skip(candidate, decision, "skipped_low_novelty")
            return False
        if candidate.id in seen_duplicate_targets:
            if candidate.id not in selected_ids:
                mark_skip(candidate, decision, "skipped_duplicate")
            return False
        topic = decision.topic or candidate.metadata.get("topic_name", "")
        topic_limit = topic_limits.get(topic)
        if topic_limit is not None and topic_counts.get(topic, 0) >= int(topic_limit):
            mark_skip(candidate, decision, "skipped_topic_cap")
            return False
        source_key = _primary_source_key(candidate)
        if enforce_source_cap and source_cap > 0 and source_key and source_counts.get(source_key, 0) >= source_cap:
            mark_skip(candidate, decision, "skipped_source_cap")
            return False
        cluster_id = candidate_event_cluster_id(candidate)
        if (
            enforce_cluster_cap
            and cluster_cap > 0
            and cluster_id
            and cluster_counts.get(cluster_id, 0) >= cluster_cap
        ):
            mark_skip(candidate, decision, "skipped_cluster_cap")
            return False

        event_cluster = candidate_event_cluster_annotation(candidate)
        if prefer_multi_source and event_cluster is not None and event_cluster.multi_source:
            reason_code = "selected_cluster_diversity"
        else:
            reason_code = "selected_high_composite" if decision.selection_rank_mode == "composite" else "selected_high_score"
        mark_selected(candidate, decision, reason_code)

        seen_duplicate_targets.add(candidate.id)
        selected_ids.add(candidate.id)
        selected.append(
            SelectedArticle(
                candidate=candidate,
                decision=decision,
                selection_reason_code=reason_code,
                selection_rank_score=decision.selection_rank_score,
                selection_rank_mode=decision.selection_rank_mode,
            )
        )
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
            if max_selected is not None and len(selected) >= max_selected:
                break

    select_from_ranked(
        require_cutoff=True,
        enforce_source_cap=True,
        enforce_cluster_cap=True,
    )
    if filtering.fill_selected_articles and max_selected is not None and len(selected) < max_selected:
        select_from_ranked(
            require_cutoff=False,
            enforce_source_cap=True,
            enforce_cluster_cap=True,
        )
    if filtering.fill_selected_articles and max_selected is not None and len(selected) < max_selected and cluster_cap > 0:
        select_from_ranked(
            require_cutoff=False,
            enforce_source_cap=True,
            enforce_cluster_cap=False,
        )
    if filtering.fill_selected_articles and max_selected is not None and len(selected) < max_selected and source_cap > 0:
        select_from_ranked(
            require_cutoff=False,
            enforce_source_cap=False,
            enforce_cluster_cap=False,
        )

    for candidate in sorted_candidates:
        decision = decisions.get(candidate.id)
        if decision is None:
            continue
        if decision.selection_reason_code:
            continue
        if decision.score < filtering.headline_score_cutoff:
            mark_skip(candidate, decision, "skipped_below_cutoff")
            continue
        if (
            novelty_floor > 0.0
            and decision.novelty < novelty_floor
            and decision.impact < 6.5
            and decision.score < max(float(filtering.headline_score_cutoff), 7.0)
        ):
            mark_skip(candidate, decision, "skipped_low_novelty")
            continue
        mark_skip(candidate, decision, "skipped_capacity" if max_selected is not None else "skipped_not_selected")

    return selected
