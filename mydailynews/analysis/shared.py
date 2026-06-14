from __future__ import annotations

from typing import Any, Dict, List

from mydailynews.domain.event_clusters import candidate_event_cluster_id, candidate_event_cluster_payload
from mydailynews.app.models import PriorReport, SelectedArticle, TopicConfig
from mydailynews.common.utils import compact_json, datetime_to_iso


def _short_text(value: Any, max_chars: int) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def _dedupe_strings(values: List[Any], *, max_items: int, max_chars: int) -> List[str]:
    if max_items <= 0:
        return []
    output: List[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _short_text(raw, max_chars)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= max_items:
            break
    return output


def _dedupe_dicts_by_text(items: List[Any], *, text_key: str, max_items: int) -> List[Dict[str, Any]]:
    if max_items <= 0:
        return []
    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        text = _short_text(raw.get(text_key, ""), 220)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(raw))
        if len(output) >= max_items:
            break
    return output


def _article_rank_key(article: SelectedArticle) -> tuple[float, float, str]:
    return (
        float(article.decision.score),
        float(article.selection_rank_score or article.decision.selection_rank_score or 0.0),
        str(article.candidate.published_at or ""),
    )


def _article_group_key(article: SelectedArticle) -> str:
    cluster_id = candidate_event_cluster_id(article.candidate)
    if cluster_id:
        return f"cluster:{cluster_id}"
    return f"article:{article.candidate.id}"


def _ordered_article_groups(articles: List[SelectedArticle]) -> List[List[SelectedArticle]]:
    groups: Dict[str, List[SelectedArticle]] = {}
    for article in sorted(articles, key=_article_rank_key, reverse=True):
        groups.setdefault(_article_group_key(article), []).append(article)
    ordered = list(groups.values())
    ordered.sort(key=lambda group: max(_article_rank_key(article) for article in group), reverse=True)
    return [sorted(group, key=_article_rank_key, reverse=True) for group in ordered]


def _headline_context_payload(articles: List[SelectedArticle]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for article in sorted(articles, key=_article_rank_key, reverse=True):
        topic = article.decision.topic or article.candidate.metadata.get("topic_name", "")
        payload.append(
            {
                "id": article.candidate.id,
                "topic": str(topic)[:80],
                "headline": str(article.candidate.title or "")[:180],
                "source": str(article.candidate.source or "")[:80],
                "score": float(article.decision.score),
                "event_cluster": candidate_event_cluster_payload(article.candidate),
            }
        )
    return payload


def _append_headline_context(prompt: str, headline_context_articles: List[SelectedArticle], *, stage: str) -> str:
    if not headline_context_articles:
        return prompt
    payload = _headline_context_payload(headline_context_articles)
    if not payload:
        return prompt
    if stage == "delta":
        instruction = (
            "These are selected articles outside this batch with headline-only context. Treat them as weak awareness "
            "for avoiding duplicate or contradictory delta framing. Do not create new/escalated/weakened/reframed/"
            "unchanged entries solely for these headline-only articles; output should be grounded in the full evidence "
            "packet and full article excerpts above."
        )
    else:
        instruction = (
            "These are selected articles outside this batch with headline-only context. Treat them as weak awareness "
            "for clustering/framing, but do not create story clusters, key claims, reader Q&A, or watch signals solely "
            "for these headline-only articles; output should be grounded in the full article excerpts above."
        )
    return (
        prompt
        + "\n\nHeadline-only awareness for selected articles outside this batch:\n"
        + compact_json(payload)
        + "\n\n"
        + instruction
    )


def _article_ids(articles: List[SelectedArticle]) -> set[str]:
    return {article.candidate.id for article in articles}


def article_cache_payload(article: SelectedArticle) -> Dict[str, Any]:
    return {
        "id": article.candidate.id,
        "headline": article.candidate.title[:160],
        "source": article.candidate.source,
        "published_at": datetime_to_iso(article.candidate.published_at),
        "score": article.decision.score,
        "article_text": (article.article_text or article.candidate.snippet)[:220],
        "snippet": (article.candidate.snippet or "")[:120],
        "event_cluster": candidate_event_cluster_payload(article.candidate),
    }


def headline_context_cache_payload(article: SelectedArticle) -> Dict[str, Any]:
    return {
        "id": article.candidate.id,
        "headline": article.candidate.title[:160],
        "source": article.candidate.source,
        "score": article.decision.score,
        "topic": article.decision.topic or article.candidate.metadata.get("topic_name", ""),
        "event_cluster": candidate_event_cluster_payload(article.candidate),
    }


def topics_payload(topics: List[TopicConfig]) -> List[dict]:
    return [
        {
            "name": topic.name,
            "description": (topic.description or "")[:180],
            "queries": [query[:80] for query in (topic.queries or [topic.name])[:3]],
        }
        for topic in topics
        if topic.enabled
    ]


def prior_reports_payload(prior_reports: List[PriorReport]) -> List[dict]:
    return [
        {
            "id": report.id,
            "date": report.date,
            "title": report.title,
            "topics": report.topics[:4],
            "summary": report.summary[:320],
            "major_headlines": report.major_headlines[:4],
        }
        for report in prior_reports
    ]
