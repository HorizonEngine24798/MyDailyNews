from __future__ import annotations

from dataclasses import dataclass


STORY_GROUPING_CACHE_VERSION = 2


@dataclass
class ResearchQuestion:
    question: str
    queries: list[str]


@dataclass
class StoryGroup:
    story_id: str
    story_title: str
    article_ids: list[str]
    research_questions: list[ResearchQuestion]
    fallback: bool = False
    topic: str = ""
    disposition: str = "group"
