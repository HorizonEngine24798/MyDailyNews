from __future__ import annotations

from typing import Tuple

import trafilatura

from ..cache import CachedHttpClient, HTTPCache
from ..debug import DebugLogger, safe_url
from ..utils import normalize_whitespace


class ArticleRetriever:
    def __init__(
        self,
        user_agent: str,
        max_chars: int,
        http_cache: HTTPCache | None = None,
        cache_fresh_seconds: int = 900,
        debug: DebugLogger | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.max_chars = max_chars
        self.debug = debug or DebugLogger(False)
        self.http = CachedHttpClient(
            user_agent=user_agent,
            cache=http_cache,
            fresh_seconds=cache_fresh_seconds,
            debug=self.debug,
        )

    def fetch_text(self, url: str) -> Tuple[str, str]:
        self.debug.log("article.fetch", "starting", url=safe_url(url), max_chars=self.max_chars)
        response = self.http.get_text(url, timeout=20, allow_redirects=True)
        if not response.ok:
            self.debug.log("article.fetch", "http_error", status=response.status_code, url=safe_url(url))
            return "", f"http_{response.status_code}" if response.status_code else "request_failed"
        try:
            text = trafilatura.extract(
                response.text,
                url=url,
                include_comments=False,
                include_images=False,
                include_links=False,
                include_tables=False,
            )
            cleaned = normalize_whitespace(text or "")
            if len(cleaned) < 250:
                self.debug.log("article.fetch", "short_text", chars=len(cleaned), url=safe_url(url), cache=response.cache_state)
                return cleaned[: self.max_chars], "short_text"
            self.debug.log(
                "article.fetch",
                "complete",
                chars=len(cleaned),
                used_chars=min(len(cleaned), self.max_chars),
                cache=response.cache_state,
            )
            return cleaned[: self.max_chars], "ok"
        except Exception:
            self.debug.log("article.fetch", "extract_failed", url=safe_url(url))
            return "", "extract_failed"
