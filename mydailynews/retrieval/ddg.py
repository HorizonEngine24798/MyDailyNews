from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from mydailynews.common.cache import CachedHttpClient, HTTPCache
from mydailynews.common.utils import normalize_url, strip_html
from mydailynews.diagnostics.debug import DebugLogger


DDG_HTML_URL = "https://duckduckgo.com/html/"


@dataclass
class DDGSearchResult:
    query: str
    title: str
    url: str
    snippet: str
    source: str


class DuckDuckGoSearchRetriever:
    """Cached DDG HTML metadata search.

    This intentionally returns search-result metadata only. Full article fetch
    and story-specific ranking stay in the enrichment pipeline.
    """

    def __init__(
        self,
        user_agent: str,
        http_cache: HTTPCache | None = None,
        debug: DebugLogger | None = None,
    ) -> None:
        self.debug = debug or DebugLogger(False)
        self.http = CachedHttpClient(user_agent, http_cache, debug=self.debug)
        self.errors: list[str] = []

    def search(self, query: str, limit: int) -> list[DDGSearchResult]:
        if limit <= 0:
            return []
        response = self.http.get_text(
            DDG_HTML_URL,
            params={"q": query},
            timeout=20,
            allow_redirects=True,
        )
        self.debug.increment(f"cache.enrichment.{response.cache_state}")
        if not response.ok:
            error = f"DDG search failed for query '{query[:80]}': status={response.status_code}"
            self.errors.append(error)
            self.debug.log("enrichment.search", "failed", query=query[:80], status=response.status_code)
            return []
        results = self.parse_html(query, response.text, limit)
        self.debug.log("enrichment.search", "complete", query=query[:80], results=len(results), cache=response.cache_state)
        return results

    @classmethod
    def parse_html(cls, query: str, html_text: str, limit: int) -> list[DDGSearchResult]:
        results: list[DDGSearchResult] = []
        seen_urls: set[str] = set()
        pattern = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(html_text or ""):
            if len(results) >= limit:
                break
            url = cls.decode_href(match.group(1))
            key = normalize_url(url)
            if not url or not key or key in seen_urls:
                continue
            seen_urls.add(key)
            title = strip_html(match.group(2))
            tail = html_text[match.end() : match.end() + 2400]
            snippet_match = re.search(
                r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
                r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
                tail,
                flags=re.IGNORECASE | re.DOTALL,
            )
            snippet = ""
            if snippet_match:
                snippet = strip_html(snippet_match.group(1) or snippet_match.group(2) or "")
            results.append(
                DDGSearchResult(
                    query=query,
                    title=title[:220],
                    url=url,
                    snippet=snippet[:500],
                    source=cls.source_from_url(url),
                )
            )
        return results

    @staticmethod
    def decode_href(href: str) -> str:
        text = unescape(href or "").strip()
        if not text:
            return ""
        text = urljoin("https://duckduckgo.com", text)
        parsed = urlparse(text)
        params = parse_qs(parsed.query)
        if "uddg" in params and params["uddg"]:
            return unquote(params["uddg"][0]).strip()
        if parsed.scheme in {"http", "https"} and parsed.netloc and "duckduckgo.com" not in parsed.netloc.lower():
            return text
        return text if parsed.scheme in {"http", "https"} and parsed.netloc else ""

    @staticmethod
    def source_from_url(url: str) -> str:
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            host = ""
        if host.startswith("www."):
            host = host[4:]
        return host or "unknown"
