from __future__ import annotations

from typing import Any

from mydailynews.app.models import ContextSource, SelectedArticle
from mydailynews.enrichment.models import ResearchResult, StoryEnrichment
from mydailynews.story_grouping.payloads import (
    clean_text,
    planner_article_payload,
    queries_for_story,
    selected_article_artifact,
    story_group_artifact,
    story_thread_artifact,
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


def selected_source_payload(article: SelectedArticle, excerpt_chars: int) -> dict[str, Any]:
    payload = planner_article_payload(article, excerpt_chars)
    payload["source_id"] = f"selected-{article.candidate.id}"
    payload["url"] = article.candidate.url
    return payload


def research_sources_payload(
    research_results: list[ResearchResult],
    *,
    fetched_count: int,
    excerpt_chars: int,
    search_results_per_query: int,
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
            item["excerpt"] = clean_text(result.text, excerpt_chars)
            fetched_used += 1
        payload.append(item)
    return payload


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
