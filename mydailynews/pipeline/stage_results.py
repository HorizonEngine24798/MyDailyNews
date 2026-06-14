from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from mydailynews.app.models import (
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
)


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
    limited_event_clusters: int = 0
    limited_multi_source_clusters: int = 0


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
    selected_event_clusters: int = 0
    selected_multi_source_clusters: int = 0


@dataclass
class ArticleFetchResult:
    selected: List[SelectedArticle]
    status_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


@dataclass
class EnrichmentStageResult:
    selected: List[SelectedArticle]
    enrichment_needed: int
    context_sources: int
    wikipedia_results: int
    past_news_results: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvidenceStageResult:
    evidence_packet: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


@dataclass
class DeltaStageResult:
    delta_packet: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
