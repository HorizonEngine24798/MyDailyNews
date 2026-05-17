from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from .ai.client import LocalAIClient
from .ai.prompts import BRIEF_SYSTEM, BRIEF_USER
from .models import SelectedArticle, UserMemory
from .utils import datetime_to_iso


class BriefGenerator:
    def __init__(self, client: LocalAIClient, max_context_chars: int) -> None:
        self.client = client
        self.max_context_chars = max_context_chars

    def generate(self, articles: List[SelectedArticle], memory: UserMemory, date: str) -> Dict[str, Any]:
        payload = [self._article_payload(article) for article in articles]
        result = self.client.complete_json(
            BRIEF_SYSTEM,
            BRIEF_USER.format(
                memory=memory.to_prompt(),
                date=date,
                articles=json.dumps(payload, ensure_ascii=False, indent=2),
            ),
        )
        if not result:
            return self._fallback(payload, date)
        result.setdefault("title", f"Daily Brief - {date}")
        result.setdefault("lead", "")
        result.setdefault("sections", [])
        result.setdefault("articles", [])
        result.setdefault("major_headlines", [])
        return result

    def _article_payload(self, article: SelectedArticle) -> Dict[str, Any]:
        wiki = [
            {
                "title": item.title,
                "url": item.url,
                "summary": item.summary,
            }
            for item in article.wikipedia_context
        ]
        past_news = [
            {
                "title": item.title,
                "url": item.url,
                "source": item.source,
                "published_at": datetime_to_iso(item.published_at),
                "snippet": item.snippet,
            }
            for item in article.past_news_context
        ]
        return {
            "id": article.candidate.id,
            "headline": article.candidate.title,
            "source": article.candidate.source,
            "url": article.candidate.url,
            "published_at": datetime_to_iso(article.candidate.published_at),
            "score": article.decision.score,
            "why_selected": article.decision.reason,
            "headline_summary": article.decision.summary,
            "tags": article.decision.tags,
            "article_text": article.article_text[: self.max_context_chars],
            "extraction_status": article.extraction_status,
            "enrichment": {
                "needed": article.enrichment_needed,
                "reason": article.enrichment_reason,
                "wikipedia": wiki,
                "past_news": past_news,
            },
        }

    @staticmethod
    def _fallback(payload: List[Dict[str, Any]], date: str) -> Dict[str, Any]:
        articles = []
        for item in payload:
            articles.append(
                {
                    "id": item["id"],
                    "headline": item["headline"],
                    "source": item["source"],
                    "url": item["url"],
                    "score": item["score"],
                    "summary": item["headline_summary"] or item["article_text"][:400],
                    "why_it_matters": item["why_selected"],
                    "key_context": item["enrichment"]["reason"],
                    "tags": item["tags"],
                }
            )
        return {
            "title": f"Daily Brief - {date}",
            "lead": "The local model could not generate a synthesized brief, so this fallback lists selected stories.",
            "sections": [],
            "articles": articles,
            "major_headlines": [
                {
                    "headline": item["headline"],
                    "source": item["source"],
                    "url": item["url"],
                    "score": item["score"],
                }
                for item in payload
            ],
        }


def brief_metadata(date: str, model: str, candidate_count: int, selected_count: int) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "date": date,
        "model": model,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
    }
