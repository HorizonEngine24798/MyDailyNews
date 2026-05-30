from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AIConfig:
    backend: str = "transformers"
    preset: str = "qwen3-4b"
    model_id: str = "Qwen/Qwen3-4B"
    server_model: str = "local-gguf"
    base_url: str = "http://127.0.0.1:8080/v1"
    device: str = "auto"
    torch_dtype: str = "auto"
    context_window_tokens: int = 0
    max_input_tokens: int = 8192
    max_new_tokens: int = 1024
    json_retries: int = 1
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    response_format: str = "json_object"
    request_timeout_seconds: int = 300
    token_estimation_chars_per_token: float = 4.0
    trust_remote_code: bool = False
    local_files_only: bool = False
    enable_thinking: bool = False

    @property
    def effective_model_label(self) -> str:
        if self.backend == "llama_cpp_server":
            return self.server_model or self.model_id
        return self.model_id


@dataclass
class FilteringConfig:
    time_window_hours: int = 36
    headline_score_cutoff: float = 6.8
    max_headlines_per_source: int = 8
    max_candidates_for_ai: int = 12
    max_headlines_per_ai_batch: int = 4
    max_selected_articles: int = 6
    fill_selected_articles: bool = False
    article_text_max_chars: int = 4000


@dataclass
class EnrichmentConfig:
    enabled: bool = True
    past_news_days: int = 30
    max_past_news_results: int = 4
    max_wikipedia_results: int = 3
    max_entities: int = 4
    max_context_chars_per_article: int = 800


@dataclass
class CacheConfig:
    enabled: bool = True
    dir: str = ".cache/mydailynews"
    http_fresh_seconds: int = 900
    ai_enabled: bool = True
    synth_fresh_seconds: int = 604800


@dataclass
class RuntimeConfig:
    max_http_workers: int = 1
    max_article_workers: int = 1
    max_enrichment_workers: int = 1
    use_shared_snapshot: bool = True


@dataclass
class UserMemory:
    avoided_topics: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    avoided_sources: List[str] = field(default_factory=list)
    briefing_style: str = "Concise, explanatory, and skeptical of hype."
    custom_instructions: str = ""

    def to_prompt(self) -> str:
        return "\n".join(
            [
                f"Avoided topics: {', '.join(self.avoided_topics) or 'none specified'}",
                f"Preferred sources: {', '.join(self.preferred_sources) or 'none specified'}",
                f"Avoided sources: {', '.join(self.avoided_sources) or 'none specified'}",
                f"Briefing style: {self.briefing_style}",
                f"Custom instructions: {self.custom_instructions or 'none'}",
            ]
        )


@dataclass
class TopicConfig:
    name: str
    description: str = ""
    queries: List[str] = field(default_factory=list)
    enabled: bool = True
    max_results: Optional[int] = None
    max_selected_articles: Optional[int] = None

    def to_prompt(self) -> str:
        queries = ", ".join(self.queries) if self.queries else self.name
        return "\n".join(
            [
                f"Topic: {self.name}",
                f"Description: {self.description or 'none'}",
                f"Search queries: {queries}",
            ]
        )


@dataclass
class GoogleNewsSourceConfig:
    enabled: bool = True
    days: int = 7
    max_results_per_topic: int = 12
    language: str = "en-US"
    region: str = "US"
    ceid: str = "US:en"


@dataclass
class PriorReportsSourceConfig:
    enabled: bool = True
    days: int = 7
    max_reports: int = 5
    max_chars_per_report: int = 1800
    output_dir: str = ""


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
    user_agent: str = "MyDailyNews/0.4 (+local personal dual news brief)"
    ai_summary: AIConfig = field(
        default_factory=lambda: AIConfig(
            preset="qwen3-1.7b",
            model_id="Qwen/Qwen3-1.7B",
            max_input_tokens=3072,
            max_new_tokens=512,
        )
    )
    ai_final: AIConfig = field(
        default_factory=lambda: AIConfig(
            preset="qwen3-8b",
            model_id="Qwen/Qwen3-8B",
            max_input_tokens=8192,
            max_new_tokens=2048,
        )
    )
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    user_memory: UserMemory = field(default_factory=UserMemory)
    general_topics: List[TopicConfig] = field(default_factory=list)
    general_filtering: FilteringConfig = field(default_factory=FilteringConfig)
    topics_to_examine: List[TopicConfig] = field(default_factory=list)
    rss_sources: List[RSSSourceConfig] = field(default_factory=list)
    google_news_source: GoogleNewsSourceConfig = field(default_factory=GoogleNewsSourceConfig)
    prior_reports_source: PriorReportsSourceConfig = field(default_factory=PriorReportsSourceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


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
class RunSourceSnapshot:
    fetched_since: datetime
    rss_candidates: List[NewsCandidate] = field(default_factory=list)
    topic_candidates: List[NewsCandidate] = field(default_factory=list)
    merged_candidates: List[NewsCandidate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadlineDecision:
    candidate_id: str
    score: float
    topic: str = ""


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
class ContextSource:
    id: str
    parent_article_id: str
    kind: str
    title: str
    source: str
    url: str
    summary: str
    items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PriorReport:
    id: str
    date: str
    title: str
    path: str
    summary: str
    topics: List[str] = field(default_factory=list)
    major_headlines: List[Dict[str, Any]] = field(default_factory=list)


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
    extracted_entities: List[str] = field(default_factory=list)
    extracted_keywords: List[str] = field(default_factory=list)
    wikipedia_context: List[WikipediaContext] = field(default_factory=list)
    past_news_context: List[PastNewsContext] = field(default_factory=list)
    context_sources: List[ContextSource] = field(default_factory=list)


@dataclass
class BriefOutput:
    name: str
    markdown_path: str
    json_path: str
    candidate_count: int
    selected_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    outputs: List[BriefOutput] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
