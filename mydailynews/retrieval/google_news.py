from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import List
from urllib.parse import quote_plus

import feedparser

from ..cache import CachedHttpClient, HTTPCache
from ..debug import DebugLogger, safe_url
from ..models import GoogleNewsSourceConfig, NewsCandidate, TopicConfig
from ..scrapers.rss import RSSScraper
from ..utils import normalize_url, normalize_whitespace, stable_id, strip_html


class GoogleNewsQueryRetriever:
    """Fetch topic-focused headline candidates from Google News RSS search."""

    def __init__(
        self,
        config: GoogleNewsSourceConfig,
        user_agent: str,
        max_workers: int = 1,
        http_cache: HTTPCache | None = None,
        cache_fresh_seconds: int = 900,
        debug: DebugLogger | None = None,
    ) -> None:
        self.config = config
        self.user_agent = user_agent
        self.max_workers = max(1, int(max_workers))
        self.debug = debug or DebugLogger(False)
        self.errors: List[str] = []
        self._error_lock = Lock()
        self.http = CachedHttpClient(
            user_agent=user_agent,
            cache=http_cache,
            fresh_seconds=cache_fresh_seconds,
            debug=self.debug,
        )

    def fetch(self, topics: List[TopicConfig], since: datetime) -> List[NewsCandidate]:
        self.errors = []
        if not self.config.enabled:
            self.debug.log("google_news", "skipped_disabled")
            return []

        enabled_topics = [topic for topic in topics if topic.enabled]
        if not enabled_topics:
            return []

        worker_count = min(self.max_workers, len(enabled_topics))
        candidates: List[NewsCandidate] = []
        if worker_count <= 1:
            for topic in enabled_topics:
                candidates.extend(self._fetch_topic(topic, since))
            return candidates

        by_index: dict[int, List[NewsCandidate]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._fetch_topic, topic, since): index
                for index, topic in enumerate(enabled_topics)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                topic = enabled_topics[index]
                try:
                    by_index[index] = future.result()
                except Exception as exc:
                    self._push_error(f"{topic.name}: worker_exception={type(exc).__name__}")
                    self.debug.log("google_news.topic", "worker_exception", topic=topic.name, error=type(exc).__name__)
                    by_index[index] = []

        for index in range(len(enabled_topics)):
            candidates.extend(by_index.get(index, []))
        return candidates

    def _fetch_topic(self, topic: TopicConfig, since: datetime) -> List[NewsCandidate]:
        queries = [query.strip() for query in (topic.queries or [topic.name]) if query.strip()]
        if not queries:
            return []
        max_results = int(topic.max_results or self.config.max_results_per_topic)
        if max_results <= 0:
            return []

        candidates: List[NewsCandidate] = []
        seen_urls: set[str] = set()
        per_query_limit = max(1, (max_results + len(queries) - 1) // len(queries))
        for query in queries:
            if len(candidates) >= max_results:
                break
            limit = min(per_query_limit, max_results - len(candidates))
            candidates.extend(
                self._fetch_query(
                    topic=topic,
                    query=query,
                    since=since,
                    limit=limit,
                    seen_urls=seen_urls,
                )
            )
            candidates = candidates[:max_results]

        self.debug.log("google_news.topic", "complete", topic=topic.name, candidates=len(candidates))
        return candidates

    def _fetch_query(
        self,
        topic: TopicConfig,
        query: str,
        since: datetime,
        limit: int,
        seen_urls: set[str],
    ) -> List[NewsCandidate]:
        search_query = self._with_window(query)
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(search_query)}"
            f"&hl={quote_plus(self.config.language)}"
            f"&gl={quote_plus(self.config.region)}"
            f"&ceid={quote_plus(self.config.ceid)}"
        )
        self.debug.log("google_news.query", "fetching", topic=topic.name, query=query, url=safe_url(rss_url))
        response = self.http.get_text(rss_url, timeout=20, allow_redirects=True)
        if not response.ok:
            self._push_error(f"{topic.name}: HTTP error status={response.status_code}")
            self.debug.log("google_news.query", "failed", topic=topic.name, status=response.status_code)
            return []

        parsed = feedparser.parse(response.text)
        candidates: List[NewsCandidate] = []
        for entry in parsed.entries:
            if len(candidates) >= limit:
                break

            published_at = RSSScraper._parse_date(entry)
            if published_at and published_at < since:
                continue

            url = normalize_url(entry.get("link", ""))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = normalize_whitespace(strip_html(entry.get("title", "Untitled")))
            source = self._entry_source(entry) or f"Google News: {topic.name}"
            snippet = strip_html(entry.get("summary", "") or entry.get("description", ""))
            entry_key = entry.get("id") or entry.get("guid") or url or title
            candidates.append(
                NewsCandidate(
                    id=stable_id("google_news", topic.name, query, entry_key),
                    source=source,
                    category="topic_search",
                    title=title,
                    url=url,
                    snippet=snippet,
                    published_at=published_at,
                    tags=["google_news", topic.name],
                    metadata={
                        "discovery_source": "google_news_query",
                        "topic_name": topic.name,
                        "topic_description": topic.description,
                        "query": query,
                        "feed_url": rss_url,
                    },
                )
            )

        self.debug.log(
            "google_news.query",
            "complete",
            topic=topic.name,
            query=query,
            entries_seen=len(parsed.entries),
            candidates=len(candidates),
            cache=response.cache_state,
        )
        return candidates

    def _with_window(self, query: str) -> str:
        text = query.strip()
        if "when:" in text:
            return text
        return f"{text} when:{self.config.days}d"

    @staticmethod
    def _entry_source(entry) -> str:
        source = entry.get("source")
        if isinstance(source, dict):
            return str(source.get("title", "")).strip()
        title = getattr(source, "title", "")
        return str(title).strip()

    def _push_error(self, text: str) -> None:
        with self._error_lock:
            self.errors.append(text)
