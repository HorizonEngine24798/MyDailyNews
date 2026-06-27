from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any, Dict, List

from mydailynews.analysis.rollout import ANALYSIS_ROLLOUT_PROFILE_NAMES
from mydailynews.common.booleans import parse_bool, parse_optional_bool
from mydailynews.app.models import (
    AnalysisConfig,
    AnalysisRolloutConfig,
    AnalysisRolloutModeConfig,
    AIConfig,
    AppConfig,
    CacheConfig,
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
    EnrichmentConfig,
    FilteringConfig,
    GoogleNewsSourceConfig,
    NarrativeBriefingConfig,
    PipelineConfig,
    PriorReportsSourceConfig,
    RSSSourceConfig,
    RuntimeConfig,
    TopicConfig,
    UserMemory,
    default_general_filtering_config,
)

DEFAULT_LLAMA_CPP_MODEL_LABEL = "Qwen3-8B-Q4_K_M"
DEFAULT_CONTEXT_WINDOW_TOKENS = 16384
DEFAULT_MAX_INPUT_TOKENS = 12000
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_ENABLE_THINKING = False


def _defaults(config_obj: Any) -> Dict[str, Any]:
    return asdict(config_obj)


DEFAULT_OUTPUT_DIR = AppConfig().output_dir
DEFAULT_USER_AGENT = AppConfig().user_agent
DEFAULT_FILTERING = _defaults(FilteringConfig(max_headlines_per_source=16))
DEFAULT_GENERAL_FILTERING = _defaults(default_general_filtering_config())
DEFAULT_ENRICHMENT = _defaults(EnrichmentConfig())
DEFAULT_CACHE = _defaults(CacheConfig())
DEFAULT_RUNTIME = _defaults(RuntimeConfig())
DEFAULT_NARRATIVE_BRIEFING = _defaults(NarrativeBriefingConfig())
DEFAULT_PIPELINE = _defaults(PipelineConfig())
DEFAULT_ANALYSIS_EVIDENCE = _defaults(EvidenceDistillationConfig())
DEFAULT_ANALYSIS_DELTA = _defaults(DeltaExtractionConfig())
DEFAULT_ANALYSIS_ROLLOUT = _defaults(AnalysisRolloutConfig())

ROOT_CONFIG_KEYS = {
    "output_dir",
    "user_agent",
    "ai_summary",
    "ai_final",
    "user_memory",
    "general_topics",
    "general_filtering",
    "topics_to_examine",
    "filtering",
    "enrichment",
    "runtime",
    "narrative_briefing",
    "pipeline",
    "analysis",
    "cache",
    "sources",
}
AI_CONFIG_KEYS = set(_defaults(AIConfig()).keys())
AI_CONFIG_KEYS.update({"model", "gguf_model_path"})
FILTERING_CONFIG_KEYS = set(DEFAULT_FILTERING.keys())
ENRICHMENT_CONFIG_KEYS = set(DEFAULT_ENRICHMENT.keys())
CACHE_CONFIG_KEYS = set(DEFAULT_CACHE.keys())
RUNTIME_CONFIG_KEYS = set(DEFAULT_RUNTIME.keys())
NARRATIVE_BRIEFING_CONFIG_KEYS = set(DEFAULT_NARRATIVE_BRIEFING.keys())
PIPELINE_CONFIG_KEYS = set(DEFAULT_PIPELINE.keys())
PIPELINE_MODULE_NAMES = {"briefs", "enrichment", "narrative_brief"}
ANALYSIS_CONFIG_KEYS = {"evidence_distillation", "delta_extraction", "rollout"}
EVIDENCE_DISTILLATION_CONFIG_KEYS = set(DEFAULT_ANALYSIS_EVIDENCE.keys())
DELTA_EXTRACTION_CONFIG_KEYS = set(DEFAULT_ANALYSIS_DELTA.keys())
ANALYSIS_ROLLOUT_CONFIG_KEYS = set(DEFAULT_ANALYSIS_ROLLOUT.keys())
ANALYSIS_ROLLOUT_MODE_CONFIG_KEYS = {field.name for field in fields(AnalysisRolloutModeConfig)}
USER_MEMORY_CONFIG_KEYS = {field.name for field in fields(UserMemory)}
TOPIC_CONFIG_KEYS = {field.name for field in fields(TopicConfig)}
SOURCES_CONFIG_KEYS = {"rss", "google_news", "prior_reports"}
RSS_SOURCE_CONFIG_KEYS = {field.name for field in fields(RSSSourceConfig)}
GOOGLE_NEWS_SOURCE_CONFIG_KEYS = {field.name for field in fields(GoogleNewsSourceConfig)}
PRIOR_REPORTS_SOURCE_CONFIG_KEYS = {field.name for field in fields(PriorReportsSourceConfig)}


def _reject_unknown_keys(raw: Dict[str, Any], allowed: set[str], section_name: str) -> None:
    unknown = sorted(str(key) for key in raw.keys() if str(key) not in allowed)
    if not unknown:
        return
    if section_name == "config":
        raise ValueError(f"Config has unrecognized key(s): {', '.join(unknown)}")
    raise ValueError(f"Config section {section_name} has unrecognized key(s): {', '.join(unknown)}")


def _list(value: Any) -> List[str]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _normalize_profile_choice(value: Any, *, allowed: set[str], default: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in allowed:
        return raw
    return default


def _load_weighted_beats(value: Any) -> Dict[str, float]:
    beats: Dict[str, float] = {}
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = []
        for raw_item in value:
            if isinstance(raw_item, str):
                items.append((raw_item, 1.0))
                continue
            if isinstance(raw_item, dict):
                items.append((raw_item.get("name", ""), raw_item.get("weight", 1.0)))
    else:
        items = []

    for raw_name, raw_weight in items:
        name = " ".join(str(raw_name or "").split()).strip()
        if not name:
            continue
        try:
            weight = float(raw_weight)
        except Exception:
            weight = 0.0
        weight = max(0.0, min(3.0, weight))
        if name in beats:
            beats[name] = max(beats[name], weight)
            continue
        beats[name] = weight
    return beats


def _load_sources(raw: Dict[str, Any]) -> List[RSSSourceConfig]:
    sources_raw = raw["sources"]
    _reject_unknown_keys(sources_raw, SOURCES_CONFIG_KEYS, "sources")
    google_news_raw = sources_raw.get("google_news", {})
    prior_reports_raw = sources_raw.get("prior_reports", {})
    if not isinstance(google_news_raw, dict):
        raise ValueError("Config section sources.google_news must be an object")
    if not isinstance(prior_reports_raw, dict):
        raise ValueError("Config section sources.prior_reports must be an object")
    _reject_unknown_keys(google_news_raw, GOOGLE_NEWS_SOURCE_CONFIG_KEYS, "sources.google_news")
    _reject_unknown_keys(prior_reports_raw, PRIOR_REPORTS_SOURCE_CONFIG_KEYS, "sources.prior_reports")
    source_items = sources_raw.get("rss", [])
    if not isinstance(source_items, list):
        raise ValueError("Config section sources.rss must be a list")

    sources: List[RSSSourceConfig] = []
    for item in source_items:
        if not isinstance(item, dict):
            raise ValueError("Each sources.rss item must be an object")
        _reject_unknown_keys(item, RSS_SOURCE_CONFIG_KEYS, "sources.rss[]")
        sources.append(
            RSSSourceConfig(
                name=item["name"],
                url=item["url"],
                category=item.get("category", "general"),
                tags=_list(item.get("tags")),
                enabled=parse_bool(item.get("enabled", True), default=True, field_name="sources.rss[].enabled"),
            )
        )
    return sources


def _load_topics(raw: Dict[str, Any], key: str) -> List[TopicConfig]:
    topics_raw = raw.get(key, [])
    if not isinstance(topics_raw, list):
        raise ValueError(f"Config section {key} must be a list")

    topics: List[TopicConfig] = []
    for item in topics_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Each {key} item must be an object")
        _reject_unknown_keys(item, TOPIC_CONFIG_KEYS, f"{key}[]")
        topics.append(
            TopicConfig(
                name=item["name"],
                description=item.get("description", ""),
                queries=_list(item.get("queries")),
                enabled=parse_bool(item.get("enabled", True), default=True, field_name=f"{key}[].enabled"),
                max_results=_optional_int(item.get("max_results")),
                max_selected_articles=_optional_int(item.get("max_selected_articles")),
            )
        )
    return topics


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any, *, field_name: str = "boolean value") -> bool | None:
    return parse_optional_bool(value, field_name=field_name)


def _optional_pos_int(value: Any, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    return max(minimum, int(value))


def _optional_limit(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"all", "any", "unbounded", "unlimited", "none"}:
            return None
        if not normalized:
            raise ValueError(f"{field_name} must be an integer or 'all'")
        value = normalized
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or 'all'") from exc


def _load_ai_backend(value: Any, section_name: str) -> str:
    backend = str(value or "llama_cpp_server").strip().lower()
    if backend != "llama_cpp_server":
        raise ValueError(
            f"Unsupported {section_name}.backend '{backend}'. "
            "Supported backend: llama_cpp_server"
        )
    return backend


def _normalize_analysis_model_role(value: Any, field_name: str) -> str:
    role = str(value or "summary").strip().lower()
    if role not in {"summary", "final"}:
        raise ValueError(f"{field_name} must be 'summary' or 'final'")
    return role


def _normalize_delta_input_source(value: Any) -> str:
    mode = str(value or "evidence_or_articles").strip().lower()
    allowed = {"evidence_or_articles", "evidence_only", "articles_only"}
    if mode not in allowed:
        raise ValueError("analysis.delta_extraction.input_source must be one of: evidence_or_articles, evidence_only, articles_only")
    return mode


def _normalize_enrichment_mode(value: Any) -> str:
    mode = str(value or DEFAULT_ENRICHMENT["mode"]).strip().lower()
    allowed = {"story_llm", "disabled"}
    if mode not in allowed:
        raise ValueError("enrichment.mode must be one of: story_llm, disabled")
    return mode


def _normalize_pipeline_module(value: Any) -> str:
    module = str(value or "").strip().lower().replace("-", "_")
    if module not in PIPELINE_MODULE_NAMES:
        allowed = ", ".join(sorted(PIPELINE_MODULE_NAMES))
        raise ValueError(f"pipeline.default_series contains unsupported module '{value}'. Supported modules: {allowed}")
    return module


def _load_pipeline(raw: Dict[str, Any]) -> PipelineConfig:
    pipeline_raw = raw.get("pipeline", {})
    if pipeline_raw is None:
        pipeline_raw = {}
    if not isinstance(pipeline_raw, dict):
        raise ValueError("Config section pipeline must be an object")
    _reject_unknown_keys(pipeline_raw, PIPELINE_CONFIG_KEYS, "pipeline")
    raw_series = pipeline_raw.get("default_series", DEFAULT_PIPELINE["default_series"])
    if not isinstance(raw_series, list):
        raise ValueError("pipeline.default_series must be a list")
    series = [_normalize_pipeline_module(item) for item in raw_series]
    duplicates = sorted({item for item in series if series.count(item) > 1})
    if duplicates:
        raise ValueError(f"pipeline.default_series contains duplicate module(s): {', '.join(duplicates)}")
    if not series:
        raise ValueError("pipeline.default_series must contain at least one module")
    return PipelineConfig(default_series=series)


def _load_analysis_rollout_mode(value: Any, field_name: str) -> AnalysisRolloutModeConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {field_name} must be an object")
    _reject_unknown_keys(value, ANALYSIS_ROLLOUT_MODE_CONFIG_KEYS, field_name)
    return AnalysisRolloutModeConfig(
        evidence_enabled=_optional_bool(
            value.get("evidence_enabled"),
            field_name=f"{field_name}.evidence_enabled",
        ),
        delta_enabled=_optional_bool(
            value.get("delta_enabled"),
            field_name=f"{field_name}.delta_enabled",
        ),
        evidence_max_input_tokens=_optional_pos_int(value.get("evidence_max_input_tokens"), minimum=256),
        evidence_max_new_tokens=_optional_pos_int(value.get("evidence_max_new_tokens"), minimum=64),
        evidence_max_articles=_optional_pos_int(value.get("evidence_max_articles"), minimum=1),
        evidence_max_articles_per_batch=_optional_pos_int(value.get("evidence_max_articles_per_batch"), minimum=1),
        evidence_max_articles_dropped_to_avoid_split=_optional_pos_int(
            value.get("evidence_max_articles_dropped_to_avoid_split"),
            minimum=0,
        ),
        evidence_max_article_chars=_optional_pos_int(value.get("evidence_max_article_chars"), minimum=120),
        delta_max_input_tokens=_optional_pos_int(value.get("delta_max_input_tokens"), minimum=256),
        delta_max_new_tokens=_optional_pos_int(value.get("delta_max_new_tokens"), minimum=64),
        delta_max_articles=_optional_pos_int(value.get("delta_max_articles"), minimum=1),
        delta_max_articles_per_batch=_optional_pos_int(value.get("delta_max_articles_per_batch"), minimum=1),
        delta_max_articles_dropped_to_avoid_split=_optional_pos_int(
            value.get("delta_max_articles_dropped_to_avoid_split"),
            minimum=0,
        ),
        delta_max_article_chars=_optional_pos_int(value.get("delta_max_article_chars"), minimum=120),
        delta_max_prior_reports=_optional_pos_int(value.get("delta_max_prior_reports"), minimum=1),
    )


def _load_analysis_rollout(value: Any) -> AnalysisRolloutConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("Config section analysis.rollout must be an object")
    _reject_unknown_keys(value, ANALYSIS_ROLLOUT_CONFIG_KEYS, "analysis.rollout")
    profile = str(value.get("profile", DEFAULT_ANALYSIS_ROLLOUT["profile"])).strip().lower()
    profile = profile or str(DEFAULT_ANALYSIS_ROLLOUT["profile"])
    if profile not in ANALYSIS_ROLLOUT_PROFILE_NAMES:
        allowed_text = ", ".join(sorted(ANALYSIS_ROLLOUT_PROFILE_NAMES))
        raise ValueError(f"analysis.rollout.profile must be one of: {allowed_text}")
    return AnalysisRolloutConfig(
        enabled=parse_bool(
            value.get("enabled", DEFAULT_ANALYSIS_ROLLOUT["enabled"]),
            default=DEFAULT_ANALYSIS_ROLLOUT["enabled"],
            field_name="analysis.rollout.enabled",
        ),
        profile=profile,
        general=_load_analysis_rollout_mode(value.get("general", {}), "analysis.rollout.general"),
        detailed=_load_analysis_rollout_mode(value.get("detailed", {}), "analysis.rollout.detailed"),
    )


def _load_ai(ai_raw: Dict[str, Any], section_name: str = "ai") -> AIConfig:
    if ai_raw is None:
        ai_raw = {}
    if not isinstance(ai_raw, dict):
        raise ValueError(f"Config section {section_name} must be an object")
    _reject_unknown_keys(ai_raw, AI_CONFIG_KEYS, section_name)
    backend = _load_ai_backend(ai_raw.get("backend", "llama_cpp_server"), section_name)
    default_server_model = str(
        ai_raw.get(
            "server_model",
            ai_raw.get("model", ai_raw.get("model_id", DEFAULT_LLAMA_CPP_MODEL_LABEL)),
        )
    )
    default_model_id = str(ai_raw.get("model_id", default_server_model))

    config = AIConfig(
        backend=backend,
        model_id=default_model_id,
        server_model=default_server_model,
        base_url=str(ai_raw.get("base_url", "http://127.0.0.1:8080/v1")),
        context_window_tokens=int(ai_raw.get("context_window_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS)),
        max_input_tokens=int(ai_raw.get("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS)),
        max_new_tokens=int(ai_raw.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        json_retries=int(ai_raw.get("json_retries", 1)),
        temperature=float(ai_raw.get("temperature", DEFAULT_TEMPERATURE)),
        top_p=float(ai_raw.get("top_p", DEFAULT_TOP_P)),
        response_format=str(ai_raw.get("response_format", "json_object")),
        request_timeout_seconds=int(ai_raw.get("request_timeout_seconds", 300)),
        token_estimation_chars_per_token=float(ai_raw.get("token_estimation_chars_per_token", 4.0)),
        enable_thinking=parse_bool(
            ai_raw.get("enable_thinking", DEFAULT_ENABLE_THINKING),
            default=DEFAULT_ENABLE_THINKING,
            field_name=f"{section_name}.enable_thinking",
        ),
        manage_server=parse_bool(ai_raw.get("manage_server", False), default=False, field_name=f"{section_name}.manage_server"),
        server_executable=str(ai_raw.get("server_executable", "llama-server")),
        server_model_path=str(ai_raw.get("server_model_path", ai_raw.get("gguf_model_path", ""))),
        server_arguments=_string_list(ai_raw.get("server_arguments", [])),
        server_log_dir=str(ai_raw.get("server_log_dir", "output/diagnostics/llama_server")),
        server_startup_timeout_seconds=max(10, int(ai_raw.get("server_startup_timeout_seconds", 180))),
        server_shutdown_timeout_seconds=max(1, int(ai_raw.get("server_shutdown_timeout_seconds", 15))),
        server_auto_stop=parse_bool(ai_raw.get("server_auto_stop", True), default=True, field_name=f"{section_name}.server_auto_stop"),
    )
    _validate_ai_runtime(config, section_name)
    return config


def _validate_ai_runtime(config: AIConfig, section_name: str) -> None:
    if not config.manage_server:
        return
    if not str(config.server_executable or "").strip():
        raise ValueError(f"{section_name}.server_executable is required when manage_server=true")
    if not str(config.server_model_path or "").strip():
        raise ValueError(f"{section_name}.server_model_path is required when manage_server=true")


def _load_ai_sections(raw: Dict[str, Any]) -> tuple[AIConfig, AIConfig]:
    summary_raw = raw.get("ai_summary", {})
    final_raw = raw.get("ai_final", {})
    ai_summary = _load_ai(summary_raw, section_name="ai_summary")
    ai_final = _load_ai(final_raw, section_name="ai_final")
    return ai_summary, ai_final


def _load_filtering(raw: Dict[str, Any], defaults: Dict[str, Any], *, section_name: str) -> FilteringConfig:
    _reject_unknown_keys(raw, FILTERING_CONFIG_KEYS, section_name)
    return FilteringConfig(
        time_window_hours=int(raw.get("time_window_hours", defaults["time_window_hours"])),
        headline_score_cutoff=float(raw.get("headline_score_cutoff", defaults["headline_score_cutoff"])),
        max_headlines_per_source=int(raw.get("max_headlines_per_source", defaults["max_headlines_per_source"])),
        max_candidates_for_ai=_optional_limit(
            raw.get("max_candidates_for_ai", defaults["max_candidates_for_ai"]),
            field_name=f"{section_name}.max_candidates_for_ai",
        ),
        max_headlines_per_ai_batch=int(raw.get("max_headlines_per_ai_batch", defaults["max_headlines_per_ai_batch"])),
        headline_max_input_tokens=_optional_pos_int(
            raw.get("headline_max_input_tokens", defaults.get("headline_max_input_tokens")),
            minimum=256,
        ),
        headline_max_new_tokens=_optional_pos_int(
            raw.get("headline_max_new_tokens", defaults.get("headline_max_new_tokens")),
            minimum=64,
        ),
        headline_single_replay_max_new_tokens=_optional_pos_int(
            raw.get(
                "headline_single_replay_max_new_tokens",
                defaults.get("headline_single_replay_max_new_tokens"),
            ),
            minimum=64,
        ),
        max_selected_articles=_optional_limit(
            raw.get("max_selected_articles", defaults["max_selected_articles"]),
            field_name=f"{section_name}.max_selected_articles",
        ),
        fill_selected_articles=parse_bool(
            raw.get("fill_selected_articles", defaults["fill_selected_articles"]),
            default=defaults["fill_selected_articles"],
            field_name=f"{section_name}.fill_selected_articles",
        ),
        article_text_max_chars=int(raw.get("article_text_max_chars", defaults["article_text_max_chars"])),
        max_selected_per_source=max(0, int(raw.get("max_selected_per_source", defaults["max_selected_per_source"]))),
        use_multifactor_composite_ranking=parse_bool(
            raw.get("use_multifactor_composite_ranking", defaults["use_multifactor_composite_ranking"]),
            default=defaults["use_multifactor_composite_ranking"],
            field_name=f"{section_name}.use_multifactor_composite_ranking",
        ),
        min_novelty_for_selection=max(
            0.0,
            min(10.0, float(raw.get("min_novelty_for_selection", defaults["min_novelty_for_selection"]))),
        ),
        source_preference_bonus=max(
            0.0,
            float(raw.get("source_preference_bonus", defaults["source_preference_bonus"])),
        ),
        source_avoid_penalty=max(
            0.0,
            float(raw.get("source_avoid_penalty", defaults["source_avoid_penalty"])),
        ),
    )


def _load_analysis(raw: Dict[str, Any]) -> AnalysisConfig:
    analysis_raw = raw.get("analysis", {})
    if analysis_raw is None:
        analysis_raw = {}
    if not isinstance(analysis_raw, dict):
        raise ValueError("Config section analysis must be an object")
    _reject_unknown_keys(analysis_raw, ANALYSIS_CONFIG_KEYS, "analysis")

    evidence_raw = analysis_raw.get("evidence_distillation", {})
    if evidence_raw is None:
        evidence_raw = {}
    if not isinstance(evidence_raw, dict):
        raise ValueError("Config section analysis.evidence_distillation must be an object")
    _reject_unknown_keys(evidence_raw, EVIDENCE_DISTILLATION_CONFIG_KEYS, "analysis.evidence_distillation")

    delta_raw = analysis_raw.get("delta_extraction", {})
    if delta_raw is None:
        delta_raw = {}
    if not isinstance(delta_raw, dict):
        raise ValueError("Config section analysis.delta_extraction must be an object")
    _reject_unknown_keys(delta_raw, DELTA_EXTRACTION_CONFIG_KEYS, "analysis.delta_extraction")
    rollout_raw = analysis_raw.get("rollout", {})

    evidence_defaults = DEFAULT_ANALYSIS_EVIDENCE
    delta_defaults = DEFAULT_ANALYSIS_DELTA
    return AnalysisConfig(
        evidence_distillation=EvidenceDistillationConfig(
            enabled=parse_bool(
                evidence_raw.get("enabled", evidence_defaults["enabled"]),
                default=evidence_defaults["enabled"],
                field_name="analysis.evidence_distillation.enabled",
            ),
            model_role=_normalize_analysis_model_role(
                evidence_raw.get("model_role", evidence_defaults["model_role"]),
                "analysis.evidence_distillation.model_role",
            ),
            include_reader_qa=parse_bool(
                evidence_raw.get("include_reader_qa", evidence_defaults["include_reader_qa"]),
                default=evidence_defaults["include_reader_qa"],
                field_name="analysis.evidence_distillation.include_reader_qa",
            ),
            max_input_tokens=max(256, int(evidence_raw.get("max_input_tokens", evidence_defaults["max_input_tokens"]))),
            max_new_tokens=max(64, int(evidence_raw.get("max_new_tokens", evidence_defaults["max_new_tokens"]))),
            max_articles=max(1, int(evidence_raw.get("max_articles", evidence_defaults["max_articles"]))),
            max_articles_per_batch=max(
                1,
                int(evidence_raw.get("max_articles_per_batch", evidence_defaults["max_articles_per_batch"])),
            ),
            max_articles_dropped_to_avoid_split=max(
                0,
                int(
                    evidence_raw.get(
                        "max_articles_dropped_to_avoid_split",
                        evidence_defaults["max_articles_dropped_to_avoid_split"],
                    )
                ),
            ),
            max_article_chars=max(120, int(evidence_raw.get("max_article_chars", evidence_defaults["max_article_chars"]))),
            max_context_sources_per_article=max(
                1,
                int(evidence_raw.get("max_context_sources_per_article", evidence_defaults["max_context_sources_per_article"])),
            ),
            max_story_clusters=max(1, int(evidence_raw.get("max_story_clusters", evidence_defaults["max_story_clusters"]))),
            max_claims_per_cluster=max(
                1,
                int(evidence_raw.get("max_claims_per_cluster", evidence_defaults["max_claims_per_cluster"])),
            ),
            max_questions=max(0, int(evidence_raw.get("max_questions", evidence_defaults["max_questions"]))),
            cache_ttl_seconds=max(0, int(evidence_raw.get("cache_ttl_seconds", evidence_defaults["cache_ttl_seconds"]))),
        ),
        delta_extraction=DeltaExtractionConfig(
            enabled=parse_bool(
                delta_raw.get("enabled", delta_defaults["enabled"]),
                default=delta_defaults["enabled"],
                field_name="analysis.delta_extraction.enabled",
            ),
            model_role=_normalize_analysis_model_role(
                delta_raw.get("model_role", delta_defaults["model_role"]),
                "analysis.delta_extraction.model_role",
            ),
            input_source=_normalize_delta_input_source(delta_raw.get("input_source", delta_defaults["input_source"])),
            require_prior_reports=parse_bool(
                delta_raw.get("require_prior_reports", delta_defaults["require_prior_reports"]),
                default=delta_defaults["require_prior_reports"],
                field_name="analysis.delta_extraction.require_prior_reports",
            ),
            max_input_tokens=max(256, int(delta_raw.get("max_input_tokens", delta_defaults["max_input_tokens"]))),
            max_new_tokens=max(64, int(delta_raw.get("max_new_tokens", delta_defaults["max_new_tokens"]))),
            max_articles=max(1, int(delta_raw.get("max_articles", delta_defaults["max_articles"]))),
            max_articles_per_batch=max(1, int(delta_raw.get("max_articles_per_batch", delta_defaults["max_articles_per_batch"]))),
            max_articles_dropped_to_avoid_split=max(
                0,
                int(
                    delta_raw.get(
                        "max_articles_dropped_to_avoid_split",
                        delta_defaults["max_articles_dropped_to_avoid_split"],
                    )
                ),
            ),
            max_article_chars=max(120, int(delta_raw.get("max_article_chars", delta_defaults["max_article_chars"]))),
            max_prior_reports=max(1, int(delta_raw.get("max_prior_reports", delta_defaults["max_prior_reports"]))),
            cache_ttl_seconds=max(0, int(delta_raw.get("cache_ttl_seconds", delta_defaults["cache_ttl_seconds"]))),
        ),
        rollout=_load_analysis_rollout(rollout_raw),
    )


def _load_narrative_briefing(raw: Dict[str, Any]) -> NarrativeBriefingConfig:
    narrative_raw = raw.get("narrative_briefing", {})
    if narrative_raw is None:
        narrative_raw = {}
    if not isinstance(narrative_raw, dict):
        raise ValueError("Config section narrative_briefing must be an object")
    _reject_unknown_keys(narrative_raw, NARRATIVE_BRIEFING_CONFIG_KEYS, "narrative_briefing")
    return NarrativeBriefingConfig(
        enabled=parse_bool(
            narrative_raw.get("enabled", DEFAULT_NARRATIVE_BRIEFING["enabled"]),
            default=DEFAULT_NARRATIVE_BRIEFING["enabled"],
            field_name="narrative_briefing.enabled",
        ),
        max_input_tokens=_optional_pos_int(narrative_raw.get("max_input_tokens"), minimum=512),
        max_new_tokens=_optional_pos_int(narrative_raw.get("max_new_tokens"), minimum=128),
        target_words=max(300, int(narrative_raw.get("target_words", DEFAULT_NARRATIVE_BRIEFING["target_words"]))),
        editorial_style=str(
            narrative_raw.get(
                "editorial_style",
                DEFAULT_NARRATIVE_BRIEFING["editorial_style"],
            )
        ),
    )


def _worker_count(raw: Dict[str, Any], key: str, default_value: int) -> int:
    value = int(raw.get(key, default_value))
    if value < 1:
        return 1
    if value > 32:
        return 32
    return value


def _cache_mode(value: Any, *, field_name: str) -> str:
    mode = str(value or DEFAULT_CACHE["discovery_mode"]).strip().lower()
    allowed = {"cache_first", "network_first", "no_cache"}
    if mode not in allowed:
        raise ValueError(f"{field_name} must be one of: cache_first, network_first, no_cache")
    return mode


def _load_cache(raw: Dict[str, Any]) -> CacheConfig:
    cache_raw = raw.get("cache", {})
    if cache_raw is None:
        cache_raw = {}
    if not isinstance(cache_raw, dict):
        raise ValueError("Config section cache must be an object")
    _reject_unknown_keys(cache_raw, CACHE_CONFIG_KEYS, "cache")

    legacy_http_retention = max(
        0,
        int(cache_raw.get("http_retention_days", DEFAULT_CACHE["http_retention_days"])),
    )
    return CacheConfig(
        enabled=parse_bool(
            cache_raw.get("enabled", DEFAULT_CACHE["enabled"]),
            default=DEFAULT_CACHE["enabled"],
            field_name="cache.enabled",
        ),
        dir=cache_raw.get("dir", DEFAULT_CACHE["dir"]),
        http_retention_days=legacy_http_retention,
        discovery_mode=_cache_mode(
            cache_raw.get("discovery_mode", DEFAULT_CACHE["discovery_mode"]),
            field_name="cache.discovery_mode",
        ),
        article_text_retention_days=max(
            0,
            int(cache_raw.get("article_text_retention_days", DEFAULT_CACHE["article_text_retention_days"])),
        ),
        enrichment_retention_days=max(
            0,
            int(cache_raw.get("enrichment_retention_days", max(legacy_http_retention, 30))),
        ),
        ai_enabled=parse_bool(
            cache_raw.get("ai_enabled", DEFAULT_CACHE["ai_enabled"]),
            default=DEFAULT_CACHE["ai_enabled"],
            field_name="cache.ai_enabled",
        ),
        synth_fresh_seconds=int(cache_raw.get("synth_fresh_seconds", DEFAULT_CACHE["synth_fresh_seconds"])),
    )


def _require_sections(raw: Dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ValueError("Config root must be an object")
    _reject_unknown_keys(raw, ROOT_CONFIG_KEYS, "config")
    missing_ai_sections = [key for key in ("ai_summary", "ai_final") if key not in raw]
    if missing_ai_sections:
        raise ValueError(f"Config missing required AI section(s): {', '.join(missing_ai_sections)}")

    required_sections = [
        "user_memory",
        "general_topics",
        "general_filtering",
        "topics_to_examine",
        "filtering",
        "enrichment",
        "sources",
    ]
    missing = [section for section in required_sections if section not in raw]
    if missing:
        raise ValueError(f"Config missing required section(s): {', '.join(missing)}")


def load_config(path: Path) -> AppConfig:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    _require_sections(raw)

    ai_summary, ai_final = _load_ai_sections(raw)
    filtering_raw = raw["filtering"]
    general_filtering_raw = raw["general_filtering"]
    enrichment_raw = raw["enrichment"]
    memory_raw = raw["user_memory"]
    sources_raw = raw["sources"]
    runtime_raw = raw.get("runtime", {})
    analysis = _load_analysis(raw)
    narrative_briefing = _load_narrative_briefing(raw)
    pipeline = _load_pipeline(raw)
    if not isinstance(filtering_raw, dict):
        raise ValueError("Config section filtering must be an object")
    if not isinstance(general_filtering_raw, dict):
        raise ValueError("Config section general_filtering must be an object")
    if not isinstance(runtime_raw, dict):
        raise ValueError("Config section runtime must be an object")
    _reject_unknown_keys(runtime_raw, RUNTIME_CONFIG_KEYS, "runtime")
    if not isinstance(enrichment_raw, dict):
        raise ValueError("Config section enrichment must be an object")
    _reject_unknown_keys(enrichment_raw, ENRICHMENT_CONFIG_KEYS, "enrichment")
    if not isinstance(memory_raw, dict):
        raise ValueError("Config section user_memory must be an object")
    _reject_unknown_keys(memory_raw, USER_MEMORY_CONFIG_KEYS, "user_memory")
    if not isinstance(sources_raw, dict):
        raise ValueError("Config section sources must be an object")

    filtering = _load_filtering(filtering_raw, DEFAULT_FILTERING, section_name="filtering")
    general_filtering = _load_filtering(
        general_filtering_raw,
        DEFAULT_GENERAL_FILTERING,
        section_name="general_filtering",
    )

    return AppConfig(
        output_dir=raw.get("output_dir", DEFAULT_OUTPUT_DIR),
        user_agent=raw.get("user_agent", DEFAULT_USER_AGENT),
        ai_summary=ai_summary,
        ai_final=ai_final,
        filtering=filtering,
        general_filtering=general_filtering,
        enrichment=EnrichmentConfig(
            enabled=parse_bool(
                enrichment_raw.get("enabled", DEFAULT_ENRICHMENT["enabled"]),
                default=DEFAULT_ENRICHMENT["enabled"],
                field_name="enrichment.enabled",
            ),
            mode=_normalize_enrichment_mode(enrichment_raw.get("mode", DEFAULT_ENRICHMENT["mode"])),
            max_context_chars_per_article=int(
                enrichment_raw.get("max_context_chars_per_article", DEFAULT_ENRICHMENT["max_context_chars_per_article"])
            ),
            max_story_threads=max(
                1,
                int(enrichment_raw.get("max_story_threads", DEFAULT_ENRICHMENT["max_story_threads"])),
            ),
            planner_max_questions_per_story=max(
                0,
                int(
                    enrichment_raw.get(
                        "planner_max_questions_per_story",
                        DEFAULT_ENRICHMENT["planner_max_questions_per_story"],
                    )
                ),
            ),
            search_results_per_query=max(
                0,
                int(enrichment_raw.get("search_results_per_query", DEFAULT_ENRICHMENT["search_results_per_query"])),
            ),
            max_fetched_research_pages_per_story=max(
                0,
                int(
                    enrichment_raw.get(
                        "max_fetched_research_pages_per_story",
                        DEFAULT_ENRICHMENT["max_fetched_research_pages_per_story"],
                    )
                ),
            ),
            max_selected_article_excerpt_chars=max(
                0,
                int(
                    enrichment_raw.get(
                        "max_selected_article_excerpt_chars",
                        DEFAULT_ENRICHMENT["max_selected_article_excerpt_chars"],
                    )
                ),
            ),
            max_research_excerpt_chars=max(
                0,
                int(
                    enrichment_raw.get(
                        "max_research_excerpt_chars",
                        DEFAULT_ENRICHMENT["max_research_excerpt_chars"],
                    )
                ),
            ),
            cache_ttl_seconds=max(
                0,
                int(enrichment_raw.get("cache_ttl_seconds", DEFAULT_ENRICHMENT["cache_ttl_seconds"])),
            ),
        ),
        user_memory=UserMemory(
            avoided_topics=_list(memory_raw.get("avoided_topics")),
            preferred_sources=_list(memory_raw.get("preferred_sources")),
            avoided_sources=_list(memory_raw.get("avoided_sources")),
            role=str(memory_raw.get("role", "")),
            geography_focus=_list(memory_raw.get("geography_focus")),
            time_horizon=_normalize_profile_choice(
                memory_raw.get("time_horizon", "tactical"),
                allowed={"breaking", "tactical", "strategic"},
                default="tactical",
            ),
            beats=_load_weighted_beats(memory_raw.get("beats")),
            wants=_list(memory_raw.get("wants")),
            avoid=_list(memory_raw.get("avoid")),
            portfolio_or_stake_notes=str(memory_raw.get("portfolio_or_stake_notes", "")),
            preferred_depth=_normalize_profile_choice(
                memory_raw.get("preferred_depth", "analytical"),
                allowed={"brief", "analytical", "deep"},
                default="analytical",
            ),
            briefing_style=str(memory_raw.get("briefing_style", "Concise, explanatory, and skeptical of hype.")),
            custom_instructions=str(memory_raw.get("custom_instructions", "")),
        ),
        general_topics=_load_topics(raw, "general_topics"),
        topics_to_examine=_load_topics(raw, "topics_to_examine"),
        rss_sources=_load_sources(raw),
        google_news_source=GoogleNewsSourceConfig(
            enabled=parse_bool(
                sources_raw.get("google_news", {}).get("enabled", True),
                default=True,
                field_name="sources.google_news.enabled",
            ),
            days=int(sources_raw.get("google_news", {}).get("days", 7)),
            max_results_per_topic=int(sources_raw.get("google_news", {}).get("max_results_per_topic", 12)),
            language=sources_raw.get("google_news", {}).get("language", "en-US"),
            region=sources_raw.get("google_news", {}).get("region", "US"),
            ceid=sources_raw.get("google_news", {}).get("ceid", "US:en"),
        ),
        prior_reports_source=PriorReportsSourceConfig(
            enabled=parse_bool(
                sources_raw.get("prior_reports", {}).get("enabled", True),
                default=True,
                field_name="sources.prior_reports.enabled",
            ),
            days=int(sources_raw.get("prior_reports", {}).get("days", 7)),
            max_reports=int(sources_raw.get("prior_reports", {}).get("max_reports", 5)),
            max_chars_per_report=int(sources_raw.get("prior_reports", {}).get("max_chars_per_report", 1800)),
            output_dir=sources_raw.get("prior_reports", {}).get("output_dir", ""),
        ),
        cache=_load_cache(raw),
        runtime=RuntimeConfig(
            max_http_workers=_worker_count(runtime_raw, "max_http_workers", DEFAULT_RUNTIME["max_http_workers"]),
            max_article_workers=_worker_count(runtime_raw, "max_article_workers", DEFAULT_RUNTIME["max_article_workers"]),
            use_shared_snapshot=parse_bool(
                runtime_raw.get("use_shared_snapshot", DEFAULT_RUNTIME["use_shared_snapshot"]),
                default=DEFAULT_RUNTIME["use_shared_snapshot"],
                field_name="runtime.use_shared_snapshot",
            ),
        ),
        analysis=analysis,
        narrative_briefing=narrative_briefing,
        pipeline=pipeline,
    )
