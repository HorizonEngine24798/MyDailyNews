from __future__ import annotations

import re
from typing import Callable
from urllib.parse import urlparse

from mydailynews.app.models import SelectedArticle
from mydailynews.common.utils import normalize_url
from mydailynews.diagnostics.debug import safe_url
from mydailynews.enrichment.models import ResearchResult
from mydailynews.retrieval.article import ArticleRetriever
from mydailynews.retrieval.ddg import DuckDuckGoSearchRetriever


class StoryResearchCollector:
    """Collect and cheaply rank external research for one story thread."""

    def __init__(
        self,
        search_retriever: DuckDuckGoSearchRetriever,
        article_retriever: ArticleRetriever,
        *,
        warning_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.search_retriever = search_retriever
        self.article_retriever = article_retriever
        self.warning_sink = warning_sink or (lambda warning: None)

    def collect(
        self,
        *,
        queries: list[str],
        story_title: str,
        story_articles: list[SelectedArticle],
        search_results_per_query: int,
        max_fetched_research_pages_per_story: int,
    ) -> list[ResearchResult]:
        if not queries:
            return []

        metadata_results = self._search_metadata(
            queries,
            per_query_limit=max(0, int(search_results_per_query)),
        )
        ranked = self.rank(metadata_results, story_title=story_title, story_articles=story_articles)
        return self._fetch_ranked_results(
            ranked,
            story_articles=story_articles,
            max_fetched_research_pages_per_story=max_fetched_research_pages_per_story,
        )

    def _search_metadata(self, queries: list[str], *, per_query_limit: int) -> list[ResearchResult]:
        metadata_results: list[ResearchResult] = []
        seen_urls: set[str] = set()
        for query in queries:
            query_text = str(query or "").strip()
            if not query_text:
                continue
            prior_errors = len(self.search_retriever.errors)
            search_results = self.search_retriever.search(query_text, per_query_limit)
            for warning in self.search_retriever.errors[prior_errors:]:
                self.warning_sink(warning)
            for search_result in search_results:
                key = normalize_url(search_result.url)
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                metadata_results.append(
                    ResearchResult(
                        id=f"research-{len(metadata_results) + 1}",
                        query=search_result.query,
                        title=search_result.title,
                        url=search_result.url,
                        snippet=search_result.snippet,
                        source=search_result.source,
                    )
                )
        return metadata_results

    def rank(
        self,
        results: list[ResearchResult],
        *,
        story_title: str,
        story_articles: list[SelectedArticle],
    ) -> list[ResearchResult]:
        story_text = " ".join(
            [story_title]
            + [article.candidate.title for article in story_articles]
            + [article.candidate.snippet for article in story_articles]
        )
        story_tokens = _tokens(story_text)
        selected_urls = {normalize_url(article.candidate.url) for article in story_articles}
        source_counts: dict[str, int] = {}
        ranked: list[ResearchResult] = []
        for result in results:
            result_tokens = _tokens(f"{result.title} {result.snippet}")
            overlap = len(story_tokens.intersection(result_tokens))
            host = _source_from_url(result.url).lower()
            duplicate_source_penalty = 0.4 * source_counts.get(host, 0)
            selected_url_penalty = 4.0 if normalize_url(result.url) in selected_urls else 0.0
            result.score = float(overlap) - duplicate_source_penalty - selected_url_penalty
            source_counts[host] = source_counts.get(host, 0) + 1
            ranked.append(result)
        ranked.sort(key=lambda item: item.score, reverse=True)
        for index, result in enumerate(ranked, start=1):
            result.id = f"research-{index}"
        return ranked

    def _fetch_ranked_results(
        self,
        ranked: list[ResearchResult],
        *,
        story_articles: list[SelectedArticle],
        max_fetched_research_pages_per_story: int,
    ) -> list[ResearchResult]:
        selected_urls = {normalize_url(article.candidate.url) for article in story_articles}
        fetch_limit = max(0, int(max_fetched_research_pages_per_story))
        output: list[ResearchResult] = []
        fetched_count = 0
        for result in ranked:
            if normalize_url(result.url) in selected_urls:
                result.status = "selected_article_duplicate"
                output.append(result)
                continue
            if fetched_count >= fetch_limit:
                output.append(result)
                continue
            try:
                text, status, effective_url = self.article_retriever.fetch_text_with_url(result.url)
                result.status = status
                result.text = text or ""
                result.effective_url = effective_url or result.url
                fetched_count += 1
                output.append(result)
            except Exception as exc:
                result.status = f"fetch_exception_{type(exc).__name__}"
                output.append(result)
                self.warning_sink(f"research fetch {safe_url(result.url)}: {type(exc).__name__}: {exc}")
        return output


def _source_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))
