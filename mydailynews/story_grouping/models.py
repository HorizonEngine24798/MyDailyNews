from __future__ import annotations

from dataclasses import dataclass


STORY_GROUPING_CACHE_VERSION = 1


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


# Backward-compatible name while enrichment migrates from story threads to
# shared story groups.
StoryThread = StoryGroup
