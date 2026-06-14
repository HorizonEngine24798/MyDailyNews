from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mydailynews.domain.candidate_annotations import (
    candidate_event_cluster_annotation,
    event_cluster_annotation_from_metadata,
)
from mydailynews.app.models import NewsCandidate, SelectedArticle


@dataclass(frozen=True)
class EventClusterScopeCounts:
    sources: int = 0
    clusters: int = 0
    multi_source_clusters: int = 0


def event_cluster_id(metadata: Mapping[str, Any]) -> str:
    annotation = event_cluster_annotation_from_metadata(metadata)
    return annotation.id if annotation is not None else ""


def candidate_event_cluster_id(candidate: NewsCandidate) -> str:
    annotation = candidate_event_cluster_annotation(candidate)
    return annotation.id if annotation is not None else ""


def candidate_event_cluster_payload(candidate: NewsCandidate) -> dict[str, Any]:
    return event_cluster_payload_from_annotation(candidate_event_cluster_annotation(candidate))


def event_cluster_payload_from_annotation(annotation) -> dict[str, Any]:
    if annotation is None or not annotation.id:
        return {}
    return {
        "id": annotation.id,
        "label": str(annotation.label or "")[:180],
        "size": max(1, int(annotation.size or 1)),
        "source_count": max(1, int(annotation.source_count or 1)),
        "multi_source": bool(annotation.multi_source),
        "latest_published_at": str(annotation.latest_published_at or "")[:64],
    }


def candidate_scope_counts(candidates: Sequence[NewsCandidate]) -> EventClusterScopeCounts:
    return _candidate_scope_counts(candidates)


def selected_scope_counts(selected: Sequence[SelectedArticle]) -> EventClusterScopeCounts:
    return _candidate_scope_counts(article.candidate for article in selected)


def _candidate_scope_counts(candidates) -> EventClusterScopeCounts:
    sources = set()
    cluster_ids = set()
    multi_source_clusters = set()
    for candidate in candidates:
        source = str(candidate.source or "").strip().lower()
        if source:
            sources.add(source)
        annotation = candidate_event_cluster_annotation(candidate)
        if annotation is None:
            continue
        cluster_ids.add(annotation.id)
        if annotation.multi_source:
            multi_source_clusters.add(annotation.id)
    return EventClusterScopeCounts(
        sources=len(sources),
        clusters=len(cluster_ids),
        multi_source_clusters=len(multi_source_clusters),
    )
