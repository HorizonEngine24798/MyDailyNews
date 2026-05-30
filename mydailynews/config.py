from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List

from .models import (
    AIConfig,
    AppConfig,
    CacheConfig,
    EnrichmentConfig,
    FilteringConfig,
    GoogleNewsSourceConfig,
    PriorReportsSourceConfig,
    RSSSourceConfig,
    RuntimeConfig,
    TopicConfig,
    UserMemory,
)

DEFAULT_AI_PRESET = "qwen3-1.7b"

AI_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "parameter_count": "8.2B",
        "context_window_tokens": 32768,
        "max_generation_tokens": 32768,
        "max_input_tokens": 8192,
        "max_new_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "trust_remote_code": False,
        "local_files_only": False,
        "enable_thinking": False,
        "notes": "Qwen3-8B model card: native 32,768 context; 131,072 with YaRN.",
        "source": "https://huggingface.co/Qwen/Qwen3-8B",
    },
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "parameter_count": "4.0B",
        "context_window_tokens": 32768,
        "max_generation_tokens": 32768,
        "max_input_tokens": 8192,
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "trust_remote_code": False,
        "local_files_only": False,
        "enable_thinking": False,
        "notes": "Qwen3-4B model card: native 32,768 context; 131,072 with YaRN.",
        "source": "https://huggingface.co/Qwen/Qwen3-4B",
    },
    "qwen2.5-7b-instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "parameter_count": "7.61B",
        "context_window_tokens": 32768,
        "max_generation_tokens": 8192,
        "max_input_tokens": 28672,
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "trust_remote_code": False,
        "local_files_only": False,
        "enable_thinking": False,
        "notes": "Model card advertises 131,072 with YaRN; native config.json max_position_embeddings is 32,768.",
        "source": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    },
    "qwen2.5-3b-instruct": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "parameter_count": "3.09B",
        "context_window_tokens": 32768,
        "max_generation_tokens": 8192,
        "max_input_tokens": 28672,
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "trust_remote_code": False,
        "local_files_only": False,
        "enable_thinking": False,
        "notes": "Model card states full 32,768 context with up to 8,192 generation tokens.",
        "source": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
    },
    "qwen2.5-1.5b-instruct": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "parameter_count": "1.54B",
        "context_window_tokens": 32768,
        "max_generation_tokens": 8192,
        "max_input_tokens": 28672,
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "trust_remote_code": False,
        "local_files_only": False,
        "enable_thinking": False,
        "notes": "Model card states full 32,768 context with up to 8,192 generation tokens.",
        "source": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
    },
    "qwen3-1.7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "parameter_count": "1.7B",
        "context_window_tokens": 32768,
        "max_generation_tokens": 8192,
        "max_generation_tokens_note": "Not explicitly listed on model card; bounded by context window.",
        "max_input_tokens": 28672,
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "trust_remote_code": False,
        "local_files_only": False,
        "enable_thinking": False,
        "notes": "Model card lists 32,768 context length; config.json max_position_embeddings is 40,960.",
        "source": "https://huggingface.co/Qwen/Qwen3-1.7B",
    },
    "qwen2.5-0.5b-instruct": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "parameter_count": "0.49B",
        "context_window_tokens": 32768,
        "max_generation_tokens": 8192,
        "max_input_tokens": 28672,
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "trust_remote_code": False,
        "local_files_only": False,
        "enable_thinking": False,
        "notes": "Model card states full 32,768 context with up to 8,192 generation tokens.",
        "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
    },
}


def _list(value: Any) -> List[str]:
    return value if isinstance(value, list) else []


def _load_sources(raw: Dict[str, Any]) -> List[RSSSourceConfig]:
    source_items = raw["sources"].get("rss", [])

    sources: List[RSSSourceConfig] = []
    for item in source_items:
        sources.append(
            RSSSourceConfig(
                name=item["name"],
                url=item["url"],
                category=item.get("category", "general"),
                tags=_list(item.get("tags")),
                enabled=bool(item.get("enabled", True)),
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
        topics.append(
            TopicConfig(
                name=item["name"],
                description=item.get("description", ""),
                queries=_list(item.get("queries")),
                enabled=bool(item.get("enabled", True)),
                max_results=_optional_int(item.get("max_results")),
                max_selected_articles=_optional_int(item.get("max_selected_articles")),
            )
        )
    return topics


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_backend(value: Any) -> str:
    raw = str(value or "transformers").strip().lower().replace("-", "_")
    aliases = {
        "hf": "transformers",
        "huggingface": "transformers",
        "llama_cpp": "llama_cpp_server",
        "llama_server": "llama_cpp_server",
        "llamacpp_server": "llama_cpp_server",
    }
    return aliases.get(raw, raw)


def get_ai_model_presets() -> Dict[str, Dict[str, Any]]:
    return deepcopy(AI_MODEL_PRESETS)


def _load_ai(ai_raw: Dict[str, Any], section_name: str = "ai") -> AIConfig:
    backend = _normalize_backend(ai_raw.get("backend", "transformers"))
    if backend not in {"transformers", "llama_cpp_server"}:
        raise ValueError(f"Unsupported {section_name}.backend '{backend}'. Supported backends: transformers, llama_cpp_server")
    requested_preset = str(ai_raw.get("preset", DEFAULT_AI_PRESET) or DEFAULT_AI_PRESET).strip().lower()
    preset_name = requested_preset or DEFAULT_AI_PRESET
    preset = AI_MODEL_PRESETS.get(preset_name)
    if backend == "transformers" and preset is None:
        supported = ", ".join(sorted(AI_MODEL_PRESETS.keys()))
        raise ValueError(f"Unsupported {section_name}.preset '{requested_preset}'. Supported presets: {supported}")

    if preset is None:
        preset = AI_MODEL_PRESETS[DEFAULT_AI_PRESET]
        preset_name = DEFAULT_AI_PRESET

    default_model_id = str(ai_raw.get("model_id", preset["model_id"]))
    if backend == "llama_cpp_server":
        default_server_model = str(ai_raw.get("server_model", ai_raw.get("model", "local-gguf")))
    else:
        default_server_model = str(ai_raw.get("server_model", ai_raw.get("model", default_model_id)))

    return AIConfig(
        backend=backend,
        preset=preset_name,
        model_id=default_model_id,
        server_model=default_server_model,
        base_url=str(ai_raw.get("base_url", "http://127.0.0.1:8080/v1")),
        device=str(ai_raw.get("device", "auto")),
        torch_dtype=str(ai_raw.get("torch_dtype", "auto")),
        context_window_tokens=int(ai_raw.get("context_window_tokens", preset.get("context_window_tokens", 0))),
        max_input_tokens=int(ai_raw.get("max_input_tokens", preset["max_input_tokens"])),
        max_new_tokens=int(ai_raw.get("max_new_tokens", preset["max_new_tokens"])),
        json_retries=int(ai_raw.get("json_retries", 1)),
        temperature=float(ai_raw.get("temperature", preset["temperature"])),
        top_p=float(ai_raw.get("top_p", preset["top_p"])),
        do_sample=bool(ai_raw.get("do_sample", preset["do_sample"])),
        response_format=str(ai_raw.get("response_format", "json_object")),
        request_timeout_seconds=int(ai_raw.get("request_timeout_seconds", 300)),
        token_estimation_chars_per_token=float(ai_raw.get("token_estimation_chars_per_token", 4.0)),
        trust_remote_code=bool(ai_raw.get("trust_remote_code", preset["trust_remote_code"])),
        local_files_only=bool(ai_raw.get("local_files_only", preset["local_files_only"])),
        enable_thinking=bool(ai_raw.get("enable_thinking", preset.get("enable_thinking", False))),
    )


def _load_ai_sections(raw: Dict[str, Any]) -> tuple[AIConfig, AIConfig, AIConfig]:
    # Backward compatibility: old configs may only have "ai".
    legacy_raw = raw.get("ai", raw.get("ai_summary", {}))
    legacy_ai = _load_ai(legacy_raw, section_name="ai")
    summary_raw = raw.get("ai_summary", raw.get("ai", {}))
    final_raw = raw.get("ai_final", raw.get("ai", {}))
    ai_summary = _load_ai(summary_raw, section_name="ai_summary")
    ai_final = _load_ai(final_raw, section_name="ai_final")
    return legacy_ai, ai_summary, ai_final


def _load_filtering(raw: Dict[str, Any], defaults: Dict[str, Any]) -> FilteringConfig:
    return FilteringConfig(
        time_window_hours=int(raw.get("time_window_hours", defaults["time_window_hours"])),
        headline_score_cutoff=float(raw.get("headline_score_cutoff", defaults["headline_score_cutoff"])),
        max_headlines_per_source=int(raw.get("max_headlines_per_source", defaults["max_headlines_per_source"])),
        max_candidates_for_ai=int(raw.get("max_candidates_for_ai", defaults["max_candidates_for_ai"])),
        max_headlines_per_ai_batch=int(raw.get("max_headlines_per_ai_batch", defaults["max_headlines_per_ai_batch"])),
        max_selected_articles=int(raw.get("max_selected_articles", defaults["max_selected_articles"])),
        fill_selected_articles=bool(raw.get("fill_selected_articles", defaults["fill_selected_articles"])),
        article_text_max_chars=int(raw.get("article_text_max_chars", defaults["article_text_max_chars"])),
    )


def _worker_count(raw: Dict[str, Any], key: str, default_value: int) -> int:
    value = int(raw.get(key, default_value))
    if value < 1:
        return 1
    if value > 32:
        return 32
    return value


def _require_sections(raw: Dict[str, Any]) -> None:
    legacy_keys = {
        "database_path",
        "lookback_hours",
        "max_articles_per_feed",
        "target_articles",
        "rss_feeds",
        "preferred_topics",
    }
    present_legacy_keys = sorted(legacy_keys.intersection(raw.keys()))
    if present_legacy_keys:
        raise ValueError(f"Config uses legacy key(s): {', '.join(present_legacy_keys)}")

    if "topics to examine" in raw:
        raise ValueError("Use JSON key topics_to_examine, not 'topics to examine'")
    if "preferred_topics" in raw.get("user_memory", {}):
        raise ValueError("Config uses legacy user_memory.preferred_topics; move topics into general_topics or topics_to_examine")

    has_any_ai_section = any(key in raw for key in ("ai", "ai_summary", "ai_final"))
    if not has_any_ai_section:
        raise ValueError("Config missing AI section. Provide ai, or ai_summary + ai_final.")

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

    ai_legacy, ai_summary, ai_final = _load_ai_sections(raw)
    filtering_raw = raw["filtering"]
    general_filtering_raw = raw["general_filtering"]
    enrichment_raw = raw["enrichment"]
    memory_raw = raw["user_memory"]
    sources_raw = raw["sources"]
    runtime_raw = raw.get("runtime", {})
    if not isinstance(runtime_raw, dict):
        raise ValueError("Config section runtime must be an object")

    filtering = _load_filtering(
        filtering_raw,
        {
            "time_window_hours": 36,
            "headline_score_cutoff": 6.8,
            "max_headlines_per_source": 8,
            "max_candidates_for_ai": 12,
            "max_headlines_per_ai_batch": 4,
            "max_selected_articles": 6,
            "fill_selected_articles": False,
            "article_text_max_chars": 2500,
        },
    )
    general_filtering = _load_filtering(
        general_filtering_raw,
        {
            "time_window_hours": 36,
            "headline_score_cutoff": 5.5,
            "max_headlines_per_source": 12,
            "max_candidates_for_ai": 30,
            "max_headlines_per_ai_batch": 6,
            "max_selected_articles": 10,
            "fill_selected_articles": True,
            "article_text_max_chars": 2200,
        },
    )

    return AppConfig(
        output_dir=raw.get("output_dir", "output"),
        user_agent=raw.get("user_agent", "MyDailyNews/0.4 (+local personal dual news brief)"),
        ai=ai_legacy,
        ai_summary=ai_summary,
        ai_final=ai_final,
        filtering=filtering,
        general_filtering=general_filtering,
        enrichment=EnrichmentConfig(
            enabled=bool(enrichment_raw.get("enabled", True)),
            past_news_days=int(enrichment_raw.get("past_news_days", 30)),
            max_past_news_results=int(enrichment_raw.get("max_past_news_results", 4)),
            max_wikipedia_results=int(enrichment_raw.get("max_wikipedia_results", 3)),
            max_entities=int(enrichment_raw.get("max_entities", 4)),
            max_context_chars_per_article=int(enrichment_raw.get("max_context_chars_per_article", 800)),
        ),
        user_memory=UserMemory(
            avoided_topics=_list(memory_raw.get("avoided_topics")),
            preferred_sources=_list(memory_raw.get("preferred_sources")),
            avoided_sources=_list(memory_raw.get("avoided_sources")),
            briefing_style=memory_raw.get("briefing_style", "Concise, explanatory, and skeptical of hype."),
            custom_instructions=memory_raw.get("custom_instructions", ""),
        ),
        general_topics=_load_topics(raw, "general_topics"),
        topics_to_examine=_load_topics(raw, "topics_to_examine"),
        rss_sources=_load_sources(raw),
        google_news_source=GoogleNewsSourceConfig(
            enabled=bool(sources_raw.get("google_news", {}).get("enabled", True)),
            days=int(sources_raw.get("google_news", {}).get("days", 7)),
            max_results_per_topic=int(sources_raw.get("google_news", {}).get("max_results_per_topic", 12)),
            language=sources_raw.get("google_news", {}).get("language", "en-US"),
            region=sources_raw.get("google_news", {}).get("region", "US"),
            ceid=sources_raw.get("google_news", {}).get("ceid", "US:en"),
        ),
        prior_reports_source=PriorReportsSourceConfig(
            enabled=bool(sources_raw.get("prior_reports", {}).get("enabled", True)),
            days=int(sources_raw.get("prior_reports", {}).get("days", 7)),
            max_reports=int(sources_raw.get("prior_reports", {}).get("max_reports", 5)),
            max_chars_per_report=int(sources_raw.get("prior_reports", {}).get("max_chars_per_report", 1800)),
            output_dir=sources_raw.get("prior_reports", {}).get("output_dir", ""),
        ),
        cache=CacheConfig(
            enabled=bool(raw.get("cache", {}).get("enabled", True)),
            dir=raw.get("cache", {}).get("dir", ".cache/mydailynews"),
            http_fresh_seconds=int(raw.get("cache", {}).get("http_fresh_seconds", 900)),
            ai_enabled=bool(raw.get("cache", {}).get("ai_enabled", True)),
            synth_fresh_seconds=int(raw.get("cache", {}).get("synth_fresh_seconds", 604800)),
        ),
        runtime=RuntimeConfig(
            max_http_workers=_worker_count(runtime_raw, "max_http_workers", 1),
            max_article_workers=_worker_count(runtime_raw, "max_article_workers", 1),
            max_enrichment_workers=_worker_count(runtime_raw, "max_enrichment_workers", 1),
            use_shared_snapshot=bool(runtime_raw.get("use_shared_snapshot", True)),
        ),
    )
