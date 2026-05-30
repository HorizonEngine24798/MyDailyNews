from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from .ai.base import AIClient
from .ai.prompts import BRIEF_SYSTEM, BRIEF_USER
from .ai.schemas import FINAL_BRIEF_JSON_SCHEMA
from .debug import DebugLogger
from .models import PriorReport, SelectedArticle, TopicConfig, UserMemory
from .utils import datetime_to_iso


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class BriefGenerator:
    def __init__(
        self,
        client: AIClient,
        max_context_chars: int,
        input_token_limit: int | None = None,
        max_new_tokens: int | None = None,
        debug: DebugLogger | None = None,
    ) -> None:
        self.client = client
        self.max_context_chars = max(200, max_context_chars)
        self.input_token_limit = input_token_limit
        self.max_new_tokens = max_new_tokens
        self.debug = debug or DebugLogger(False)
        self.warnings: List[str] = []

    def generate(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        brief_name: str = "",
    ) -> Dict[str, Any]:
        self.warnings = []
        prompt, used_articles = self._build_prompt(
            articles,
            memory,
            topics,
            prior_reports,
            brief_goal,
            date,
        )
        self.debug.log("brief.ai", "synthesizing", articles=len(used_articles), prompt_chars=len(prompt))
        label = "final brief generation"
        if brief_name:
            label = f"{label} ({brief_name})"
        result = self.client.complete_json(
            BRIEF_SYSTEM,
            prompt,
            label=label,
            max_new_tokens=self.max_new_tokens,
            input_token_limit=self.input_token_limit,
            json_schema=FINAL_BRIEF_JSON_SCHEMA,
        )
        required = {"title", "lead", "topic_reports", "sections"}
        missing = required.difference(result.keys())
        if missing:
            raise ValueError(f"final brief generation: missing key(s): {', '.join(sorted(missing))}")

        result.setdefault("title", f"Daily Brief - {date}")
        result["major_headlines"] = self._major_headlines_payload(used_articles)
        result["selected_articles"] = self._selected_articles_payload(used_articles)
        self.debug.log("brief.ai", "complete", articles=len(used_articles))
        return result

    def _build_prompt(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
    ) -> tuple[str, List[SelectedArticle]]:
        target_input_tokens = max(1024, int(self.input_token_limit or self.client.max_input_tokens))
        prompt_budget_tokens = max(900, int(target_input_tokens * 0.9))
        ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)
        active_reports = prior_reports[:3]
        excerpt_options = [
            self.max_context_chars,
            min(self.max_context_chars, 650),
            min(self.max_context_chars, 450),
            280,
        ]
        dropped_article_ids: list[str] = []
        used_articles = ordered_articles[:]
        prompt = ""

        for excerpt_chars in excerpt_options:
            candidate_articles = used_articles[:]
            candidate_reports = active_reports[:]
            while candidate_articles:
                prompt = self._render_prompt(
                    candidate_articles,
                    excerpt_chars,
                    memory,
                    topics,
                    candidate_reports,
                    brief_goal,
                    date,
                )
                estimated_tokens = self.client.estimate_tokens(prompt)
                self.debug.log(
                    "brief.prompt",
                    "budget_check",
                    articles=len(candidate_articles),
                    prior_reports=len(candidate_reports),
                    excerpt_chars=excerpt_chars,
                    estimated_tokens=estimated_tokens,
                    budget_tokens=prompt_budget_tokens,
                )
                if estimated_tokens <= prompt_budget_tokens:
                    used_articles = candidate_articles
                    if dropped_article_ids:
                        self.warnings.append(
                            "final brief prompt dropped lower-ranked article(s) to stay within the local model budget: "
                            + ", ".join(dropped_article_ids)
                        )
                    return prompt, used_articles
                if len(candidate_reports) > 1:
                    candidate_reports = candidate_reports[:-1]
                    continue
                if len(candidate_articles) > 4:
                    dropped = candidate_articles.pop()
                    dropped_article_ids.append(dropped.candidate.id)
                    continue
                used_articles = candidate_articles
                break

        if dropped_article_ids:
            self.warnings.append(
                "final brief prompt dropped lower-ranked article(s) to stay within the local model budget: "
                + ", ".join(dropped_article_ids)
            )
        if used_articles and prompt:
            estimated_tokens = self.client.estimate_tokens(prompt)
            if estimated_tokens > prompt_budget_tokens:
                self.warnings.append(
                    f"final brief prompt still estimated above budget ({estimated_tokens}>{prompt_budget_tokens}); "
                    "the backend input limit may truncate the prompt."
                )
            return prompt, used_articles
        return self._render_prompt([], 0, memory, topics, active_reports[:1], brief_goal, date), []

    def _render_prompt(
        self,
        articles: List[SelectedArticle],
        excerpt_chars: int,
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
    ) -> str:
        payload = [self._article_payload(article, excerpt_chars) for article in articles]
        return BRIEF_USER.format(
            memory=memory.to_prompt(),
            date=date,
            brief_goal=brief_goal,
            topics=_compact_json(self._topics_payload(topics)),
            prior_reports=_compact_json(self._prior_reports_payload(prior_reports)),
            articles=_compact_json(payload),
        )

    def _article_payload(self, article: SelectedArticle, excerpt_chars: int) -> Dict[str, Any]:
        topic = article.decision.topic or article.candidate.metadata.get("topic_name", "")
        context_sources = [
            {
                "kind": item.kind,
                "source": item.source,
                "title": item.title[:120],
                "summary": item.summary[:180],
                "items": item.items[:3],
            }
            for item in article.context_sources[:2]
        ]
        return {
            "id": article.candidate.id,
            "topic": topic,
            "headline": article.candidate.title,
            "source": article.candidate.source,
            "url": article.candidate.url,
            "published_at": datetime_to_iso(article.candidate.published_at),
            "score": article.decision.score,
            "article_text": (article.article_text or article.candidate.snippet)[:excerpt_chars],
            "extraction_status": article.extraction_status,
            "context_note": article.enrichment_reason,
            "context_sources": context_sources,
        }

    @staticmethod
    def _topics_payload(topics: List[TopicConfig]) -> List[dict]:
        return [
            {
                "name": topic.name,
                "description": (topic.description or "")[:180],
                "queries": [query[:80] for query in (topic.queries or [topic.name])[:3]],
            }
            for topic in topics
            if topic.enabled
        ]

    @staticmethod
    def _prior_reports_payload(prior_reports: List[PriorReport]) -> List[dict]:
        return [
            {
                "id": report.id,
                "date": report.date,
                "title": report.title,
                "topics": report.topics[:4],
                "summary": report.summary[:420],
                "major_headlines": report.major_headlines[:5],
            }
            for report in prior_reports
        ]

    @staticmethod
    def _major_headlines_payload(articles: List[SelectedArticle]) -> List[dict]:
        return [
            {
                "headline": article.candidate.title,
                "source": article.candidate.source,
                "url": article.candidate.url,
                "score": article.decision.score,
                "topic": article.decision.topic or article.candidate.metadata.get("topic_name", ""),
            }
            for article in articles
        ]

    @staticmethod
    def _selected_articles_payload(articles: List[SelectedArticle]) -> List[dict]:
        return [
            {
                "id": article.candidate.id,
                "headline": article.candidate.title,
                "source": article.candidate.source,
                "url": article.candidate.url,
                "score": article.decision.score,
                "topic": article.decision.topic or article.candidate.metadata.get("topic_name", ""),
                "snippet": (article.candidate.snippet or "")[:180],
            }
            for article in articles
        ]


def brief_metadata(
    date: str,
    model: str,
    candidate_count: int,
    selected_count: int,
    topics: List[str] | None = None,
    prior_reports_count: int = 0,
    brief_name: str = "",
    warnings: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "2.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "date": date,
        "brief_name": brief_name,
        "model": model,
        "topics": topics or [],
        "prior_reports_count": prior_reports_count,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "warnings": warnings or [],
    }
