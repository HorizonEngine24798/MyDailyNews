from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AIConfig:
    host: str = "http://localhost:11434"
    model: str = "qwen3:4b"
    timeout_seconds: int = 120
    temperature: float = 0.2


@dataclass
class FilteringConfig:
    time_window_hours: int = 36
    headline_score_cutoff: float = 7.0
    max_headlines_per_source: int = 12
    max_candidates_for_ai: int = 60
    max_selected_articles: int = 8
    article_text_max_chars: int = 7000


@dataclass
class EnrichmentConfig:
    enabled: bool = True
    past_news_days: int = 30
    max_past_news_results: int = 4
    max_wikipedia_results: int = 1
    max_context_chars_per_article: int = 2400


@dataclass
class UserMemory:
    preferred_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    avoided_sources: List[str] = field(default_factory=list)
    briefing_style: str = "Concise, explanatory, and skeptical of hype."
    custom_instructions: str = ""

    def to_prompt(self) -> str:
        return "\n".join(
            [
                f"Preferred topics: {', '.join(self.preferred_topics) or 'none specified'}",
                f"Avoided topics: {', '.join(self.avoided_topics) or 'none specified'}",
                f"Preferred sources: {', '.join(self.preferred_sources) or 'none specified'}",
                f"Avoided sources: {', '.join(self.avoided_sources) or 'none specified'}",
                f"Briefing style: {self.briefing_style}",
                f"Custom instructions: {self.custom_instructions or 'none'}",
            ]
        )


@dataclass
class RSSSourceConfig:
    name: str
    url: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AppConfig:
    output_dir: str = "output"
    user_agent: str = "MyDailyNews/0.2 (+local personal news brief)"
    ai: AIConfig = field(default_factory=AIConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    user_memory: UserMemory = field(default_factory=UserMemory)
    rss_sources: List[RSSSourceConfig] = field(default_factory=list)


@dataclass
class NewsCandidate:
    id: str
    source: str
    category: str
    title: str
    url: str
    snippet: str
    published_at: Optional[datetime]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadlineDecision:
    candidate_id: str
    score: float
    reason: str
    summary: str
    tags: List[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None


@dataclass
class WikipediaContext:
    title: str
    url: str
    summary: str


@dataclass
class PastNewsContext:
    title: str
    url: str
    source: str
    published_at: Optional[datetime]
    snippet: str


@dataclass
class SelectedArticle:
    candidate: NewsCandidate
    decision: HeadlineDecision
    article_text: str = ""
    extraction_status: str = "pending"
    enrichment_needed: bool = False
    enrichment_reason: str = ""
    wikipedia_query: str = ""
    past_news_query: str = ""
    wikipedia_context: List[WikipediaContext] = field(default_factory=list)
    past_news_context: List[PastNewsContext] = field(default_factory=list)


@dataclass
class PipelineResult:
    markdown_path: str
    json_path: str
    candidate_count: int
    selected_count: int
