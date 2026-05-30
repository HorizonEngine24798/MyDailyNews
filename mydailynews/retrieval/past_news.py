from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List
from urllib.parse import quote_plus

import feedparser

from ..cache import CachedHttpClient, HTTPCache
from ..models import PastNewsContext
from ..scrapers.rss import RSSScraper
from ..utils import normalize_url, strip_html


class PastNewsRetriever:
    def __init__(
        self,
        user_agent: str,
        http_cache: HTTPCache | None = None,
        cache_fresh_seconds: int = 900,
    ) -> None:
        self.user_agent = user_agent
        self.http = CachedHttpClient(
            user_agent=user_agent,
            cache=http_cache,
            fresh_seconds=cache_fresh_seconds,
        )

    def search(self, query: str, days: int, limit: int, exclude_url: str = "") -> List[PastNewsContext]:
        if not query.strip() or limit <= 0:
            return []
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query + ' when:' + str(days) + 'd')}&hl=en-US&gl=US&ceid=US:en"
        )
        response = self.http.get_text(rss_url, timeout=20, allow_redirects=True)
        if not response.ok:
            return []

        parsed = feedparser.parse(response.text)
        contexts: List[PastNewsContext] = []
        excluded = normalize_url(exclude_url)
        for entry in parsed.entries:
            url = normalize_url(entry.get("link", ""))
            if not url or url == excluded:
                continue
            published_at = RSSScraper._parse_date(entry)
            if published_at and published_at < since:
                continue
            contexts.append(
                PastNewsContext(
                    title=strip_html(entry.get("title", "Untitled")),
                    url=url,
                    source="Google News",
                    published_at=published_at,
                    snippet=strip_html(entry.get("summary", "") or entry.get("description", "")),
                )
            )
            if len(contexts) >= limit:
                break
        return contexts
