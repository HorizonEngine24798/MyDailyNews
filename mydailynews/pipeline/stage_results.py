from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from mydailynews.app.models import (
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
)
from mydailynews.story_grouping.models import StoryGroup
from mydailynews.story_grouping.payloads import selected_article_artifact, story_group_artifact


@dataclass
class CandidatePreparationResult:
    raw_count: int
    rss_count: int
    topic_count: int
    unique_candidates: List[NewsCandidate]
    warnings: List[str] = field(default_factory=list)
    rss_candidates: List[NewsCandidate] = field(default_factory=list)
    topic_candidates: List[NewsCandidate] = field(default_factory=list)


@dataclass
class HeadlineLimitResult:
    limited_candidates: List[NewsCandidate]
    warnings: List[str] = field(default_factory=list)
    limited_sources: int = 0


@dataclass
class HeadlineScoringResult:
    limited_candidates: List[NewsCandidate]
    decisions: Dict[str, HeadlineDecision]
    warnings: List[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    selected: List[SelectedArticle]
    selection_counts: Dict[str, Dict[str, int]]
    warnings: List[str] = field(default_factory=list)
    selected_sources: int = 0


@dataclass
class ArticleFetchResult:
    selected: List[SelectedArticle]
    status_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


@dataclass
class StoryGroupingStageResult:
    selected: List[SelectedArticle]
    story_groups: List[StoryGroup]
    artifact: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def planned(
        cls,
        *,
        selected: List[SelectedArticle],
        story_groups: List[StoryGroup],
        planner_artifact: Dict[str, Any] | None = None,
        warnings: List[str] | None = None,
        cache_hit: bool = False,
    ) -> "StoryGroupingStageResult":
        artifact = _story_grouping_artifact(
            selected=selected,
            story_groups=story_groups,
            status="ok" if story_groups else "empty",
            enabled=True,
            skipped_reason="",
            cache_hit=cache_hit,
            planner_artifact=planner_artifact,
        )
        return cls(
            selected=list(selected),
            story_groups=list(story_groups),
            artifact=artifact,
            warnings=list(warnings or []),
        )

    @classmethod
    def skipped(
        cls,
        *,
        selected: List[SelectedArticle],
        reason: str,
        warnings: List[str] | None = None,
        artifact: Dict[str, Any] | None = None,
    ) -> "StoryGroupingStageResult":
        merged_artifact = _story_grouping_artifact(
            selected=selected,
            story_groups=[],
            status="skipped",
            enabled=False,
            skipped_reason=reason,
            cache_hit=False,
            planner_artifact=artifact,
        )
        return cls(
            selected=list(selected),
            story_groups=[],
            artifact=merged_artifact,
            warnings=list(warnings or []),
        )

    @property
    def story_threads(self) -> List[StoryGroup]:
        return self.story_groups


@dataclass
class EnrichmentStageResult:
    selected: List[SelectedArticle]
    enrichment_needed: int
    context_sources: int
    story_threads_created: int = 0
    story_threads_enriched: int = 0
    story_threads_skipped: int = 0
    artifact: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvidenceStageResult:
    evidence_packet: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


@dataclass
class DeltaStageResult:
    delta_packet: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _story_grouping_artifact(
    *,
    selected: List[SelectedArticle],
    story_groups: List[StoryGroup],
    status: str,
    enabled: bool,
    skipped_reason: str,
    cache_hit: bool,
    planner_artifact: Dict[str, Any] | None,
) -> Dict[str, Any]:
    selected_articles = [selected_article_artifact(article) for article in selected]
    group_artifacts = [story_group_artifact(group) for group in story_groups]
    fallback_groups = [group for group in group_artifacts if bool(group.get("fallback", False))]
    requests = []
    if planner_artifact:
        raw_requests = planner_artifact.get("requests", [])
        if isinstance(raw_requests, list):
            requests = raw_requests
    artifact: Dict[str, Any] = {
        "enabled": enabled,
        "status": status,
        "skipped_reason": skipped_reason,
        "selected_articles": selected_articles,
        "selected_article_ids": [article["id"] for article in selected_articles],
        "story_groups": group_artifacts,
        "fallback_groups": fallback_groups,
        "cache_hit": cache_hit,
        "requests": requests,
        "split_requests": len(requests) > 1,
    }
    if planner_artifact:
        artifact["planner"] = planner_artifact
    return artifact
