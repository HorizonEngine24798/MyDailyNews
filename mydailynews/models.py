from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AIConfig:
    backend: str = "auto"
    preset: str = "qwen3-4b"
    model_id: str = "Qwen/Qwen3-4B"
    server_model: str = "local-gguf"
    base_url: str = "http://127.0.0.1:8080/v1"
    device: str = "auto"
    torch_dtype: str = "auto"
    context_window_tokens: int = 0
    max_input_tokens: int = 40960
    max_new_tokens: int = 8192
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
    manage_server: bool = False
    server_executable: str = "llama-server"
    server_model_path: str = ""
    server_arguments: List[str] = field(default_factory=list)
    server_log_dir: str = "output/diagnostics/llama_server"
    server_startup_timeout_seconds: int = 180
    server_shutdown_timeout_seconds: int = 15
    server_auto_stop: bool = True

    @property
    def effective_model_label(self) -> str:
        if self.backend in {"auto", "llama_cpp_server"}:
            return self.server_model or self.model_id
        return self.model_id


@dataclass
class FilteringConfig:
    time_window_hours: int = 36
    headline_score_cutoff: float = 6.8
    max_headlines_per_source: int = 8
    max_candidates_for_ai: int = 40
    max_headlines_per_ai_batch: int = 32
    headline_max_input_tokens: Optional[int] = None
    headline_max_new_tokens: Optional[int] = None
    headline_single_replay_max_new_tokens: Optional[int] = None
    max_selected_articles: int = 8
    fill_selected_articles: bool = False
    article_text_max_chars: int = 6000
    max_selected_per_source: int = 2
    max_selected_per_event_cluster: int = 2
    prefer_multi_source_clusters: bool = True
    multi_source_cluster_bonus: float = 0.35
    event_cluster_time_window_hours: int = 18
    use_multifactor_composite_ranking: bool = False
    min_novelty_for_selection: float = 0.0
    source_preference_bonus: float = 0.35
    source_avoid_penalty: float = 1.25


@dataclass
class EnrichmentConfig:
    enabled: bool = True
    past_news_days: int = 30
    max_past_news_results: int = 4
    max_wikipedia_results: int = 3
    max_entities: int = 4
    max_context_chars_per_article: int = 1600


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
class EvidenceDistillationConfig:
    enabled: bool = False
    model_role: str = "summary"
    include_reader_qa: bool = True
    max_input_tokens: int = 20000
    max_new_tokens: int = 2400
    max_articles: int = 12
    max_article_chars: int = 1200
    max_context_sources_per_article: int = 3
    max_story_clusters: int = 10
    max_claims_per_cluster: int = 6
    max_questions: int = 10
    cache_ttl_seconds: int = 604800


@dataclass
class DeltaExtractionConfig:
    enabled: bool = False
    model_role: str = "summary"
    input_source: str = "evidence_or_articles"
    require_prior_reports: bool = False
    max_input_tokens: int = 16000
    max_new_tokens: int = 1600
    max_prior_reports: int = 4
    cache_ttl_seconds: int = 604800


@dataclass
class AnalysisRolloutModeConfig:
    evidence_enabled: Optional[bool] = None
    delta_enabled: Optional[bool] = None
    evidence_max_input_tokens: Optional[int] = None
    evidence_max_new_tokens: Optional[int] = None
    evidence_max_articles: Optional[int] = None
    evidence_max_article_chars: Optional[int] = None
    delta_max_input_tokens: Optional[int] = None
    delta_max_new_tokens: Optional[int] = None
    delta_max_prior_reports: Optional[int] = None


@dataclass
class AnalysisRolloutConfig:
    enabled: bool = False
    profile: str = "safe_local"
    general: AnalysisRolloutModeConfig = field(default_factory=AnalysisRolloutModeConfig)
    detailed: AnalysisRolloutModeConfig = field(default_factory=AnalysisRolloutModeConfig)


@dataclass
class AnalysisConfig:
    evidence_distillation: EvidenceDistillationConfig = field(default_factory=EvidenceDistillationConfig)
    delta_extraction: DeltaExtractionConfig = field(default_factory=DeltaExtractionConfig)
    rollout: AnalysisRolloutConfig = field(default_factory=AnalysisRolloutConfig)


@dataclass
class UserMemory:
    avoided_topics: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    avoided_sources: List[str] = field(default_factory=list)
    role: str = ""
    geography_focus: List[str] = field(default_factory=list)
    time_horizon: str = "tactical"
    beats: Dict[str, float] = field(default_factory=dict)
    wants: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    portfolio_or_stake_notes: str = ""
    preferred_depth: str = "analytical"
    briefing_style: str = "Concise, explanatory, and skeptical of hype."
    custom_instructions: str = ""

    @staticmethod
    def _clean_text(value: Any, max_chars: int) -> str:
        text = " ".join(str(value or "").split())
        return text[:max_chars]

    @classmethod
    def _render_list(
        cls,
        values: List[str],
        *,
        max_items: int = 6,
        max_chars: int = 48,
    ) -> str:
        rendered: List[str] = []
        seen: set[str] = set()
        for raw in values:
            item = cls._clean_text(raw, max_chars)
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            rendered.append(item)
            if len(rendered) >= max_items:
                break
        return ", ".join(rendered) if rendered else "none specified"

    @classmethod
    def _render_beats(cls, beats: Dict[str, float]) -> str:
        ranked: List[tuple[str, float]] = []
        for raw_name, raw_weight in (beats or {}).items():
            name = cls._clean_text(raw_name, 44)
            if not name:
                continue
            try:
                weight = float(raw_weight)
            except Exception:
                weight = 0.0
            weight = max(0.0, min(3.0, weight))
            ranked.append((name, weight))
        if not ranked:
            return "none specified"
        ranked.sort(key=lambda row: (-row[1], row[0].lower()))
        compact = [f"{name}({weight:.2f})" for name, weight in ranked[:6]]
        return ", ".join(compact)

    def to_prompt(self) -> str:
        return "\n".join(
            [
                f"Role: {self._clean_text(self.role, 90) or 'none specified'}",
                f"Geography focus: {self._render_list(self.geography_focus)}",
                f"Time horizon: {self._clean_text(self.time_horizon, 24) or 'tactical'}",
                f"Preferred depth: {self._clean_text(self.preferred_depth, 24) or 'analytical'}",
                f"Priority beats: {self._render_beats(self.beats)}",
                f"Wants: {self._render_list(self.wants)}",
                f"Avoid classes: {self._render_list(self.avoid)}",
                f"Portfolio/stake notes: {self._clean_text(self.portfolio_or_stake_notes, 180) or 'none'}",
                f"Avoided topics: {self._render_list(self.avoided_topics)}",
                f"Preferred sources: {self._render_list(self.preferred_sources)}",
                f"Avoided sources: {self._render_list(self.avoided_sources)}",
                f"Briefing style: {self._clean_text(self.briefing_style, 180) or 'none specified'}",
                f"Custom instructions: {self._clean_text(self.custom_instructions, 220) or 'none'}",
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
            max_input_tokens=40960,
            max_new_tokens=8192,
        )
    )
    ai_final: AIConfig = field(
        default_factory=lambda: AIConfig(
            preset="qwen3-8b",
            model_id="Qwen/Qwen3-8B",
            max_input_tokens=40960,
            max_new_tokens=8192,
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
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


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
    personal_relevance: float = 5.0
    impact: float = 5.0
    novelty: float = 5.0
    urgency: float = 5.0
    actionability: float = 5.0
    confidence: float = 5.0
    reason: str = ""
    skip_reason: Optional[str] = None
    angle_type: str = ""
    selection_reason_code: str = ""
    selection_rank_score: float = 0.0
    selection_rank_mode: str = "score"


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
    selection_reason_code: str = ""
    selection_rank_score: float = 0.0
    selection_rank_mode: str = "score"
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
