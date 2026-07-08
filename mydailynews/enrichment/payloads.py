from __future__ import annotations

import re
from typing import Any

from mydailynews.app.models import ContextSource, SelectedArticle
from mydailynews.enrichment.models import ResearchResult, StoryEnrichment
from mydailynews.story_grouping.payloads import (
    clean_text,
    planner_article_payload,
    string_list,
)


def story_enrichment_payload(enrichment: StoryEnrichment) -> dict[str, Any]:
    return {
        "story_id": enrichment.story_id,
        "story_title": enrichment.story_title,
        "internal_articles": enrichment.internal_articles,
        "confirmed_facts": enrichment.confirmed_facts,
        "conflicting_claims": enrichment.conflicting_claims,
        "open_questions": enrichment.open_questions,
    }


def context_story_id(source: ContextSource) -> str:
    for item in source.items:
        if isinstance(item, dict) and item.get("story_id"):
            return str(item.get("story_id") or "").strip()
    return ""


def selected_source_payload(
    article: SelectedArticle,
    excerpt_chars: int,
    *,
    excerpt_strategy: str = "prefix",
    terms_text: str = "",
    lead_chars: int = 0,
    window_chars: int = 0,
    max_windows: int = 0,
) -> dict[str, Any]:
    payload = planner_article_payload(article, 0)
    if excerpt_chars > 0:
        payload["article_excerpt"] = excerpt_text(
            article.article_text or article.candidate.snippet,
            strategy=excerpt_strategy,
            terms_text=terms_text,
            max_chars=excerpt_chars,
            lead_chars=lead_chars,
            window_chars=window_chars,
            max_windows=max_windows,
        )
    payload["source_id"] = f"selected-{article.candidate.id}"
    payload["url"] = article.candidate.url
    return payload


def research_sources_payload(
    research_results: list[ResearchResult],
    *,
    fetched_count: int,
    excerpt_chars: int,
    search_results_per_query: int,
    excerpt_strategy: str = "prefix",
    terms_text: str = "",
    lead_chars: int = 0,
    window_chars: int = 0,
    max_windows: int = 0,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    fetched_used = 0
    for result in research_results[: max(8, int(search_results_per_query))]:
        item: dict[str, Any] = {
            "source_id": result.id,
            "query": result.query[:140],
            "title": result.title[:220],
            "source": result.source[:120],
            "url": result.effective_url or result.url,
            "snippet": result.snippet[:500],
            "status": result.status,
        }
        if result.text and fetched_used < fetched_count and excerpt_chars > 0:
            item["excerpt"] = excerpt_text(
                result.text,
                strategy=excerpt_strategy,
                terms_text=terms_text,
                max_chars=excerpt_chars,
                lead_chars=lead_chars,
                window_chars=window_chars,
                max_windows=max_windows,
            )
            fetched_used += 1
        payload.append(item)
    return payload


def excerpt_text(
    text: str,
    *,
    strategy: str,
    terms_text: str,
    max_chars: int,
    lead_chars: int,
    window_chars: int,
    max_windows: int,
) -> str:
    if str(strategy or "prefix").strip().lower() == "prefix":
        return clean_text(text, max_chars)
    return relevant_excerpt(
        text,
        terms_text=terms_text,
        max_chars=max_chars,
        lead_chars=lead_chars,
        window_chars=window_chars,
        max_windows=max_windows,
    )


def relevant_excerpt(
    text: str,
    *,
    terms_text: str,
    max_chars: int,
    lead_chars: int,
    window_chars: int,
    max_windows: int,
) -> str:
    max_chars = max(0, int(max_chars))
    cleaned = clean_text(text, max(len(str(text or "")), max_chars))
    if max_chars <= 0 or not cleaned:
        return ""
    lead = cleaned[: max(0, int(lead_chars))].strip()
    terms = _tokens(terms_text)
    paragraphs = [
        clean_text(part, max(0, int(window_chars)) or max_chars)
        for part in re.split(r"\n{2,}|(?<=[.!?])\s+", str(text or ""))
    ]
    scored: list[tuple[int, int, str]] = []
    for index, paragraph in enumerate(paragraph for paragraph in paragraphs if paragraph):
        overlap = len(terms.intersection(_tokens(paragraph))) if terms else 0
        if overlap:
            scored.append((overlap, -index, paragraph))
    picked = sorted(scored, reverse=True)[: max(0, int(max_windows))]
    windows = [paragraph for _, neg_index, paragraph in sorted(picked, key=lambda row: -row[1])]
    parts: list[str] = []
    for part in [lead, *windows]:
        if part and part not in parts:
            parts.append(part)
    excerpt = "\n\n".join(parts)
    return clean_text(excerpt or cleaned, max_chars)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))


def fact_list(value: Any, *, text_key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return output
    for raw in value:
        if not isinstance(raw, dict):
            continue
        text = clean_text(raw.get(text_key), 260)
        if not text:
            continue
        output.append(
            {
                text_key: text,
                "source_ids": string_list(raw.get("source_ids", []), max_items=12, max_chars=120),
            }
        )
        if len(output) >= 12:
            break
    return output


def confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    return "medium"
