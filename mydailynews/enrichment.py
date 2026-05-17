from __future__ import annotations

from .ai.client import LocalAIClient
from .ai.prompts import ENRICHMENT_SYSTEM, ENRICHMENT_USER
from .models import AppConfig, SelectedArticle
from .retrieval.past_news import PastNewsRetriever
from .retrieval.wikipedia import WikipediaRetriever


class EnrichmentPlanner:
    def __init__(self, client: LocalAIClient) -> None:
        self.client = client

    def plan(self, article: SelectedArticle) -> None:
        result = self.client.complete_json(
            ENRICHMENT_SYSTEM,
            ENRICHMENT_USER.format(
                title=article.candidate.title,
                source=article.candidate.source,
                url=article.candidate.url,
                summary=article.decision.summary,
                text=(article.article_text or article.candidate.snippet)[:2200],
            ),
        )
        if not result:
            article.enrichment_needed = False
            article.enrichment_reason = "AI enrichment planner failed."
            return

        article.enrichment_needed = bool(result.get("needed", False))
        article.enrichment_reason = str(result.get("reason", ""))
        article.wikipedia_query = str(result.get("wikipedia_query", "") or "")
        article.past_news_query = str(result.get("past_news_query", "") or "")


class SimpleEnricher:
    def __init__(self, config: AppConfig, client: LocalAIClient) -> None:
        self.config = config
        self.planner = EnrichmentPlanner(client)
        self.wikipedia = WikipediaRetriever(config.user_agent)
        self.past_news = PastNewsRetriever(config.user_agent)

    def enrich(self, article: SelectedArticle) -> None:
        if not self.config.enrichment.enabled:
            return

        self.planner.plan(article)
        if not article.enrichment_needed:
            return

        wiki_query = article.wikipedia_query or article.candidate.title
        past_news_query = article.past_news_query or article.candidate.title
        article.wikipedia_context = self.wikipedia.search(
            wiki_query,
            self.config.enrichment.max_wikipedia_results,
        )
        article.past_news_context = self.past_news.search(
            past_news_query,
            self.config.enrichment.past_news_days,
            self.config.enrichment.max_past_news_results,
            exclude_url=article.candidate.url,
        )
