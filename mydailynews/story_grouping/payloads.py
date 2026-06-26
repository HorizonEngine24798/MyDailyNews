from __future__ import annotations

import re
from typing import Any

from mydailynews.app.models import SelectedArticle
from mydailynews.common.utils import datetime_to_iso
from mydailynews.story_grouping.models import StoryGroup


def selected_article_artifact(article: SelectedArticle) -> dict[str, Any]:
    return {
        "id": article.candidate.id,
        "headline": article.candidate.title[:220],
        "source": article.candidate.source[:120],
        "url": article.candidate.url,
        "published_at": datetime_to_iso(article.candidate.published_at),
        "score": float(article.decision.score),
        "extraction_status": article.extraction_status,
    }


def story_group_artifact(story: StoryGroup) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "story_id": story.story_id,
        "story_title": story.story_title,
        "article_ids": story.article_ids,
        "fallback": story.fallback,
        "research_questions": [
            {"question": question.question, "queries": question.queries}
            for question in story.research_questions
        ],
    }
    if story.topic:
        payload["topic"] = story.topic
    return payload


def story_thread_artifact(story: StoryGroup) -> dict[str, Any]:
    return story_group_artifact(story)


def queries_for_story(story: StoryGroup) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for question in story.research_questions:
        for query in question.queries:
            text = clean_text(query, 140)
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            queries.append(text)
    if not queries:
        title = clean_text(story.story_title, 140)
        if title:
            queries.append(title)
    return queries


def planner_article_payload(article: SelectedArticle, excerpt_chars: int) -> dict[str, Any]:
    topic = article.decision.topic or article.candidate.metadata.get("topic_name", "")
    payload: dict[str, Any] = {
        "id": article.candidate.id,
        "headline": clean_text(article.candidate.title, 220),
        "source": clean_text(article.candidate.source, 80),
        "published_at": datetime_to_iso(article.candidate.published_at),
        "topic": clean_text(topic, 80),
        "score": float(article.decision.score),
        "snippet": clean_text(article.candidate.snippet, 280),
        "extraction_status": article.extraction_status,
    }
    if excerpt_chars > 0:
        payload["article_excerpt"] = clean_text(article.article_text or article.candidate.snippet, excerpt_chars)
    return payload


def clean_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[: max(0, int(max_chars))]


def string_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    raw_values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        text = clean_text(raw, max_chars)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= max_items:
            break
    return output
