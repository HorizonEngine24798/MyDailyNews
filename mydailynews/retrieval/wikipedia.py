from __future__ import annotations

import json
from typing import List
from urllib.parse import quote

from ..cache import CachedHttpClient, HTTPCache
from ..models import WikipediaContext
from ..utils import normalize_whitespace


class WikipediaRetriever:
    API_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

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

    def search(self, query: str, limit: int) -> List[WikipediaContext]:
        if not query.strip() or limit <= 0:
            return []
        response = self.http.get_text(
            self.API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": limit,
            },
            timeout=15,
            allow_redirects=True,
        )
        if not response.ok:
            return []
        try:
            payload = json.loads(response.text)
            titles = [item["title"] for item in payload.get("query", {}).get("search", [])]
        except Exception:
            return []

        contexts: List[WikipediaContext] = []
        for title in titles[:limit]:
            summary = self._summary(title)
            if summary:
                contexts.append(summary)
        return contexts

    def _summary(self, title: str) -> WikipediaContext | None:
        response = self.http.get_text(
            self.SUMMARY_URL.format(title=quote(title.replace(" ", "_"))),
            timeout=15,
            allow_redirects=True,
        )
        if not response.ok:
            return None
        try:
            raw = json.loads(response.text)
            extract = raw.get("extract", "")
            first_paragraph = next((part.strip() for part in extract.split("\n") if part.strip()), extract)
            return WikipediaContext(
                title=raw.get("title", title),
                url=raw.get("content_urls", {}).get("desktop", {}).get("page", ""),
                summary=normalize_whitespace(first_paragraph),
            )
        except Exception:
            return None
