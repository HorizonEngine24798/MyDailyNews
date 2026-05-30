from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .cache import HTTPCache
from .debug import DebugLogger
from .models import AppConfig, ContextSource, SelectedArticle
from .retrieval.past_news import PastNewsRetriever
from .retrieval.wikipedia import WikipediaRetriever
from .utils import datetime_to_iso, stable_id


class SimpleEnricher:
    def __init__(
        self,
        config: AppConfig,
        http_cache: HTTPCache | None = None,
        debug: DebugLogger | None = None,
    ) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.warnings: list[str] = []
        self._warning_lock = Lock()
        self.wikipedia = WikipediaRetriever(
            config.user_agent,
            http_cache=http_cache,
            cache_fresh_seconds=config.cache.http_fresh_seconds,
        )
        self.past_news = PastNewsRetriever(
            config.user_agent,
            http_cache=http_cache,
            cache_fresh_seconds=config.cache.http_fresh_seconds,
        )

    def enrich(self, article: SelectedArticle) -> None:
        self.enrich_many([article], max_workers=1)

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
        if worker_count <= 1:
            for article in io_targets:
                warning = self._populate_external_context_safe(article)
                if warning:
                    self._push_warning(warning)
            return

        ordered_warnings: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._populate_external_context_safe, article): index
                for index, article in enumerate(io_targets)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                warning = future.result()
                if warning:
                    ordered_warnings[index] = warning

        for index in range(len(io_targets)):
            warning = ordered_warnings.get(index)
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
