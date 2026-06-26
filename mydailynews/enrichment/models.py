from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mydailynews.story_grouping.models import (
    ResearchQuestion,
    STORY_GROUPING_CACHE_VERSION,
    StoryGroup,
    StoryThread,
)

STORY_ENRICHMENT_CACHE_VERSION = 1


@dataclass
class ResearchResult:
    id: str
    query: str
    title: str
    url: str
    snippet: str
    source: str
    status: str = "search_result"
    text: str = ""
    effective_url: str = ""
    score: float = 0.0


@dataclass
class StoryEnrichment:
    story_id: str
    story_title: str
    internal_articles: list[dict[str, Any]]
    confirmed_facts: list[dict[str, Any]]
    conflicting_claims: list[dict[str, Any]]
    open_questions: list[dict[str, Any]]
