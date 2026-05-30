from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from .models import (
    FilteringConfig,
    NewsCandidate,
    RunSourceSnapshot,
    TopicConfig,
)


def build_snapshot(
    *,
    use_shared_snapshot: bool,
    now,
    general_topics: List[TopicConfig],
    detailed_topics: List[TopicConfig],
    general_filtering: FilteringConfig,
    detailed_filtering: FilteringConfig,
    debug,
    fetch_headlines,
    fetch_topic_headlines,
    merge_url_duplicates,
) -> RunSourceSnapshot | None:
    if not use_shared_snapshot:
        debug.set_metric("snapshot.enabled", False)
        return None

    with debug.span("snapshot.total"):
        general_since = now - timedelta(hours=general_filtering.time_window_hours)
        detailed_since = now - timedelta(hours=detailed_filtering.time_window_hours)
        snapshot_since = min(general_since, detailed_since)
        max_headlines_per_source = max(
            general_filtering.max_headlines_per_source,
            detailed_filtering.max_headlines_per_source,
        )
        shared_topics = merge_topics_for_snapshot(general_topics, detailed_topics)

        snapshot_warnings: List[str] = []
        with debug.span("snapshot.rss_fetch"):
            rss_candidates = fetch_headlines(snapshot_since, max_headlines_per_source, snapshot_warnings)
        with debug.span("snapshot.topic_fetch"):
            topic_candidates = fetch_topic_headlines(shared_topics, snapshot_since, snapshot_warnings)
        with debug.span("snapshot.merge"):
            merged_candidates = merge_url_duplicates(rss_candidates + topic_candidates)
        raw_candidates = len(rss_candidates) + len(topic_candidates)
        debug.set_metric("snapshot.enabled", True)
        debug.set_metric("snapshot.rss_candidates", len(rss_candidates))
        debug.set_metric("snapshot.topic_candidates", len(topic_candidates))
        debug.set_metric("snapshot.raw_candidates", raw_candidates)
        debug.set_metric("snapshot.unique_candidates", len(merged_candidates))
        debug.log(
            "snapshot",
            "built",
            since=snapshot_since,
            rss_candidates=len(rss_candidates),
            topic_candidates=len(topic_candidates),
            merged_candidates=len(merged_candidates),
            warnings=len(snapshot_warnings),
        )
        return RunSourceSnapshot(
            fetched_since=snapshot_since,
            rss_candidates=rss_candidates,
            topic_candidates=topic_candidates,
            merged_candidates=merged_candidates,
            metadata={
                "warnings": snapshot_warnings,
                "max_headlines_per_source": max_headlines_per_source,
                "topic_count": len(shared_topics),
            },
        )


def merge_topics_for_snapshot(*topic_groups: List[TopicConfig]) -> List[TopicConfig]:
    merged: List[TopicConfig] = []
    seen: set[str] = set()
    for topics in topic_groups:
        for topic in topics:
            key = "|".join(
                [
                    topic.name.strip().lower(),
                    topic.description.strip().lower(),
                    ",".join(query.strip().lower() for query in topic.queries if query.strip()),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(topic)
    return merged


def snapshot_candidates_for_brief(
    snapshot: RunSourceSnapshot,
    since,
) -> tuple[List[NewsCandidate], List[NewsCandidate], List[NewsCandidate]]:
    rss_candidates = [candidate for candidate in snapshot.rss_candidates if candidate_in_window(candidate, since)]
    topic_candidates = [candidate for candidate in snapshot.topic_candidates if candidate_in_window(candidate, since)]
    merged_candidates = [candidate for candidate in snapshot.merged_candidates if candidate_in_window(candidate, since)]
    return rss_candidates, topic_candidates, merged_candidates


def candidate_in_window(candidate: NewsCandidate, since) -> bool:
    published_at = candidate.published_at
    latest_iso = str(candidate.metadata.get("merged_latest_published_at", "")).strip()
    if latest_iso:
        try:
            merged_latest = datetime.fromisoformat(latest_iso)
            if merged_latest.tzinfo is not None:
                if published_at is None or merged_latest > published_at:
                    published_at = merged_latest
        except ValueError:
            pass
    if published_at is None:
        return True
    return published_at >= since
