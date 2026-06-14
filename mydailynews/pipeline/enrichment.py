from __future__ import annotations

from threading import Lock

from mydailynews.common.cache import HTTPCache
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import AppConfig, ContextSource, SelectedArticle
from mydailynews.common.parallel import ordered_parallel_map
from mydailynews.retrieval.past_news import PastNewsRetriever
from mydailynews.retrieval.wikipedia import WikipediaRetriever
from mydailynews.common.utils import datetime_to_iso, stable_id


class SimpleEnricher:
    def __init__(
        self,
        config: AppConfig,
        http_cache: HTTPCache | None = None,
        wikipedia_cache: HTTPCache | None = None,
        debug: DebugLogger | None = None,
    ) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.warnings: list[str] = []
        self._warning_lock = Lock()
        self.wikipedia = WikipediaRetriever(
            config.user_agent,
            http_cache=wikipedia_cache or http_cache,
            debug=self.debug,
        )
        self.past_news = PastNewsRetriever(
            config.user_agent,
            http_cache=http_cache,
            debug=self.debug,
        )

    def enrich_many(self, articles: list[SelectedArticle], max_workers: int = 1) -> None:
        if not articles:
            return
        if not self.config.enrichment.enabled:
            for article in articles:
                self.debug.log("enrichment", "skipped_disabled", article_id=article.candidate.id)
            return

        io_targets: list[SelectedArticle] = []
        for article in articles:
            if not self._needs_external_context(article):
                article.enrichment_needed = False
                article.enrichment_reason = "Skipped external context: article text already has enough detail."
                self.debug.log("enrichment", "skipped_enough_context", article_id=article.candidate.id)
                continue

            article.enrichment_needed = True
            article.enrichment_reason = "Added external context because the local article text was short or thin."
            article.extracted_entities = self._candidate_entities(article)
            article.extracted_keywords = self._candidate_keywords(article)
            article.wikipedia_query = ", ".join(article.extracted_entities)
            article.past_news_query = " ".join(article.extracted_keywords)
            io_targets.append(article)

        if not io_targets:
            return

        worker_count = max(1, int(max_workers))
        worker_count = min(worker_count, len(io_targets))
        warnings = ordered_parallel_map(
            io_targets,
            worker_count,
            self._populate_external_context_safe,
        )
        for warning in warnings:
            if warning:
                self._push_warning(warning)

    @staticmethod
    def _needs_external_context(article: SelectedArticle) -> bool:
        text = (article.article_text or "").strip()
        if not text:
            return True
        if article.extraction_status != "ok":
            return True
        if len(text) < 1200:
            return True
        sentence_count = text.count(".") + text.count("!") + text.count("?")
        return sentence_count < 8

    def _candidate_entities(self, article: SelectedArticle) -> list[str]:
        entities: list[str] = []
        topic = (article.decision.topic or article.candidate.metadata.get("topic_name", "")).strip()
        if topic:
            entities.append(topic)
        source = (article.candidate.source or "").strip()
        if source and source not in entities:
            entities.append(source)
        title = (article.candidate.title or "").strip()
        if title and title not in entities:
            entities.append(title[:120])
        return entities[: self.config.enrichment.max_entities]

    def _candidate_keywords(self, article: SelectedArticle) -> list[str]:
        topic = (article.decision.topic or article.candidate.metadata.get("topic_name", "")).strip()
        title = (article.candidate.title or "").strip()
        queries = [item for item in [topic, title] if item]
        return queries[: self.config.enrichment.max_entities]

    def _retrieve_wikipedia(self, entities: list[str]):
        contexts = []
        seen_urls = set()
        for entity in entities:
            remaining = self.config.enrichment.max_wikipedia_results - len(contexts)
            if remaining <= 0:
                break
            for item in self.wikipedia.search(entity, limit=1):
                if item.url in seen_urls:
                    continue
                seen_urls.add(item.url)
                contexts.append(item)
                break
        return contexts

    def _populate_external_context_safe(self, article: SelectedArticle) -> str:
        try:
            entities = (article.extracted_entities or [article.candidate.title])[: self.config.enrichment.max_entities]
            article.wikipedia_context = self._retrieve_wikipedia(entities)

            news_query = article.past_news_query or " ".join(entities)
            article.past_news_context = self.past_news.search(
                news_query,
                self.config.enrichment.past_news_days,
                self.config.enrichment.max_past_news_results,
                exclude_url=article.candidate.url,
            )
            article.context_sources = self._build_context_sources(article)
            self.debug.log(
                "enrichment",
                "complete",
                article_id=article.candidate.id,
                wiki=len(article.wikipedia_context),
                past_news=len(article.past_news_context),
                context_sources=len(article.context_sources),
            )
            return ""
        except Exception as exc:
            self.debug.log("enrichment", "io_exception", article_id=article.candidate.id, error=type(exc).__name__)
            return f"enrichment io {article.candidate.id}: {type(exc).__name__}: {exc}"

    def _push_warning(self, warning: str) -> None:
        with self._warning_lock:
            self.warnings.append(warning)

    @staticmethod
    def _build_context_sources(article: SelectedArticle) -> list[ContextSource]:
        sources: list[ContextSource] = []
        for item in article.wikipedia_context:
            sources.append(
                ContextSource(
                    id=stable_id(article.candidate.id, "wikipedia", item.url or item.title),
                    parent_article_id=article.candidate.id,
                    kind="wikipedia_summary",
                    title=f"Context for {article.candidate.title}: {item.title}",
                    source="Wikipedia",
                    url=item.url,
                    summary=item.summary,
                    items=[{"title": item.title, "url": item.url, "summary": item.summary}],
                )
            )

        if article.past_news_context:
            headlines = [
                {
                    "title": item.title,
                    "source": item.source,
                    "url": item.url,
                    "published_at": datetime_to_iso(item.published_at),
                }
                for item in article.past_news_context
            ]
            summary = "Recent related headlines: " + "; ".join(item["title"] for item in headlines)
            sources.append(
                ContextSource(
                    id=stable_id(article.candidate.id, "past_news", article.past_news_query),
                    parent_article_id=article.candidate.id,
                    kind="related_news_headlines",
                    title=f"Recent related headlines for {article.candidate.title}",
                    source="Google News RSS",
                    url="",
                    summary=summary,
                    items=headlines,
                )
            )
        return sources
