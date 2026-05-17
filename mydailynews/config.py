from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .models import (
    AIConfig,
    AppConfig,
    EnrichmentConfig,
    FilteringConfig,
    RSSSourceConfig,
    UserMemory,
)


def _list(value: Any) -> List[str]:
    return value if isinstance(value, list) else []


def _load_sources(raw: Dict[str, Any]) -> List[RSSSourceConfig]:
    source_items = raw.get("sources", {}).get("rss")
    if source_items is None:
        source_items = raw.get("rss_feeds", [])

    sources: List[RSSSourceConfig] = []
    for item in source_items or []:
        sources.append(
            RSSSourceConfig(
                name=item["name"],
                url=item["url"],
                category=item.get("category") or (item.get("tags") or ["general"])[0],
                tags=_list(item.get("tags")),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return sources


def load_config(path: Path) -> AppConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ai_raw = raw.get("ai") or raw.get("ollama", {})
    filtering_raw = raw.get("filtering", {})
    enrichment_raw = raw.get("enrichment", {})
    memory_raw = raw.get("user_memory", {})

    filtering = FilteringConfig(
        time_window_hours=int(filtering_raw.get("time_window_hours", raw.get("lookback_hours", 36))),
        headline_score_cutoff=float(filtering_raw.get("headline_score_cutoff", 7.0)),
        max_headlines_per_source=int(filtering_raw.get("max_headlines_per_source", raw.get("max_articles_per_feed", 12))),
        max_candidates_for_ai=int(filtering_raw.get("max_candidates_for_ai", 60)),
        max_selected_articles=int(filtering_raw.get("max_selected_articles", raw.get("target_articles", 8))),
        article_text_max_chars=int(filtering_raw.get("article_text_max_chars", raw.get("article_text_max_chars", 7000))),
    )

    return AppConfig(
        output_dir=raw.get("output_dir", "output"),
        user_agent=raw.get("user_agent", "MyDailyNews/0.2 (+local personal news brief)"),
        ai=AIConfig(
            host=ai_raw.get("host", "http://localhost:11434"),
            model=ai_raw.get("model", "qwen3:4b"),
            timeout_seconds=int(ai_raw.get("timeout_seconds", 120)),
            temperature=float(ai_raw.get("temperature", 0.2)),
        ),
        filtering=filtering,
        enrichment=EnrichmentConfig(
            enabled=bool(enrichment_raw.get("enabled", True)),
            past_news_days=int(enrichment_raw.get("past_news_days", 30)),
            max_past_news_results=int(enrichment_raw.get("max_past_news_results", 4)),
            max_wikipedia_results=int(enrichment_raw.get("max_wikipedia_results", 1)),
            max_context_chars_per_article=int(enrichment_raw.get("max_context_chars_per_article", 2400)),
        ),
        user_memory=UserMemory(
            preferred_topics=_list(memory_raw.get("preferred_topics")),
            avoided_topics=_list(memory_raw.get("avoided_topics")),
            preferred_sources=_list(memory_raw.get("preferred_sources")),
            avoided_sources=_list(memory_raw.get("avoided_sources")),
            briefing_style=memory_raw.get("briefing_style", "Concise, explanatory, and skeptical of hype."),
            custom_instructions=memory_raw.get("custom_instructions", ""),
        ),
        rss_sources=_load_sources(raw),
    )
