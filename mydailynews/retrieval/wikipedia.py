from __future__ import annotations

from typing import List
from urllib.parse import quote

import requests

from ..models import WikipediaContext
from ..utils import normalize_whitespace


class WikipediaRetriever:
    API_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent

    def search(self, query: str, limit: int) -> List[WikipediaContext]:
        if not query.strip() or limit <= 0:
            return []
        try:
            response = requests.get(
                self.API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": limit,
                },
                headers={"User-Agent": self.user_agent},
                timeout=15,
            )
            response.raise_for_status()
            titles = [item["title"] for item in response.json().get("query", {}).get("search", [])]
        except Exception:
            return []

        contexts: List[WikipediaContext] = []
        for title in titles[:limit]:
            summary = self._summary(title)
            if summary:
                contexts.append(summary)
        return contexts

    def _summary(self, title: str) -> WikipediaContext | None:
        try:
            response = requests.get(
                self.SUMMARY_URL.format(title=quote(title.replace(" ", "_"))),
                headers={"User-Agent": self.user_agent},
                timeout=15,
            )
            response.raise_for_status()
            raw = response.json()
            return WikipediaContext(
                title=raw.get("title", title),
                url=raw.get("content_urls", {}).get("desktop", {}).get("page", ""),
                summary=normalize_whitespace(raw.get("extract", "")),
            )
        except Exception:
            return None
