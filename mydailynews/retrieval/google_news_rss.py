from __future__ import annotations

from datetime import datetime, timezone, timedelta
import re
import time
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import quote_plus, urlparse

import feedparser

from mydailynews.common.cache import CachedHttpClient, HTTPCache
from mydailynews.common.utils import (
    GOOGLE_CONSENT_HEADERS,
    canonical_article_url,
    normalize_url,
    normalize_whitespace,
    stable_id,
    strip_html,
)
from mydailynews.diagnostics.debug import DebugLogger, safe_url
from mydailynews.scrapers.rss import RSSScraper


GOOGLE_NEWS_SEARCH_BASE = "https://news.google.com/rss/search"

# Standard country to primary language mapping for Google News editions
COUNTRY_DEFAULT_LANGUAGE = {
    "US": "en-US",
    "GB": "en-GB",
    "CA": "en-CA",
    "AU": "en-AU",
    "NZ": "en-NZ",
    "IN": "en-IN",
    "FR": "fr",
    "DE": "de",
    "IT": "it",
    "ES": "es",
    "JP": "ja",
    "KR": "ko",
    "BR": "pt-BR",
    "MX": "es-419",
    "AR": "es-419",
    "RU": "ru",
    "UA": "uk",
    "CN": "zh-CN",
    "TW": "zh-TW",
    "HK": "zh-HK",
    "IL": "he",
    "TR": "tr",
    "SA": "ar",
    "AE": "ar",
    "EG": "ar",
    "ZA": "en-ZA",
    "NG": "en-NG",
    "KE": "en-KE",
}


class GoogleNewsRssCoverageRetriever:
    """Free, multi-country international news coverage retriever powered by Google News RSS."""

    def __init__(
        self,
        user_agent: str,
        http_cache: HTTPCache | None = None,
        http_cache_mode: str = CachedHttpClient.CACHE_FIRST,
        debug: DebugLogger | None = None,
        progress_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.debug = debug or DebugLogger(False)
        self.progress_sink = progress_sink or (lambda message: None)
        self.http = CachedHttpClient(
            user_agent=user_agent,
            cache=http_cache,
            cache_mode=http_cache_mode,
            debug=self.debug,
        )
        self._last_request_at = 0.0

    def search(
        self,
        query: str,
        *,
        timespan_days: int = 7,
        max_records: int = 25,
        source_countries: List[str] | None = None,
        source_languages: List[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        cleaned_query = normalize_whitespace(query or "")
        if not cleaned_query:
            return [], ["google_news_rss: empty search query; skipped."]

        countries = [
            str(c).strip().upper()
            for c in (source_countries or [])
            if str(c).strip() and len(str(c).strip()) == 2
        ]
        if not countries:
            countries = ["US"]

        warnings: List[str] = []
        all_articles: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()

        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(timespan_days or 1)))
        per_country_limit = max(3, max_records // len(countries))

        for country in countries:
            language = self._resolve_language(country, source_languages)
            ceid = f"{country}:{language}"
            search_query = f"{cleaned_query} when:{max(1, int(timespan_days or 1))}d"
            rss_url = (
                f"{GOOGLE_NEWS_SEARCH_BASE}?"
                f"q={quote_plus(search_query)}"
                f"&hl={quote_plus(language)}"
                f"&gl={quote_plus(country)}"
                f"&ceid={quote_plus(ceid)}"
            )

            self._throttle()
            self.debug.increment("provider.google_news_rss.requests")
            response = self.http.get_text(
                rss_url,
                timeout=20,
                allow_redirects=True,
                headers=GOOGLE_CONSENT_HEADERS,
            )

            if not response.ok:
                warnings.append(f"google_news_rss: {country} request failed (status {response.status_code}).")
                continue

            parsed = feedparser.parse(response.text)
            country_count = 0

            for entry in parsed.entries:
                if country_count >= per_country_limit or len(all_articles) >= max_records:
                    break

                published_at = RSSScraper._parse_date(entry)
                if published_at and published_at < cutoff:
                    continue

                raw_url = normalize_url(entry.get("link", ""))
                canonical_url = canonical_article_url(raw_url) or raw_url
                if not canonical_url or canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)

                raw_title = normalize_whitespace(strip_html(entry.get("title", "Untitled")))
                source_name, title = self._split_source_from_title(raw_title, entry)
                snippet = strip_html(entry.get("summary", "") or entry.get("description", ""))

                article_id = stable_id("gnews_rss", country, canonical_url, title)
                published_str = published_at.isoformat() if published_at else ""

                all_articles.append(
                    {
                        "article_id": article_id,
                        "provider": "google_news_rss",
                        "url": raw_url,
                        "canonical_url": canonical_url,
                        "domain": self._extract_domain(canonical_url),
                        "source_id": f"gnews_{country.lower()}",
                        "source_name": source_name,
                        "source_country": country,
                        "source_language": language.split("-")[0].lower(),
                        "title": title,
                        "snippet": snippet,
                        "published_at": published_str,
                        "retrieval_query": query,
                        "retrieval_query_type": "free_coverage",
                    }
                )
                country_count += 1

        return all_articles[:max_records], warnings

    def _throttle(self) -> None:
        min_interval = 0.5
        remaining = min_interval - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _resolve_language(country: str, requested_languages: List[str] | None) -> str:
        if requested_languages:
            first_lang = str(requested_languages[0]).strip().lower()
            if first_lang:
                return first_lang
        return COUNTRY_DEFAULT_LANGUAGE.get(country, "en-US")

    @staticmethod
    def _split_source_from_title(raw_title: str, entry: Any) -> Tuple[str, str]:
        source_name = ""
        entry_source = entry.get("source")
        if isinstance(entry_source, dict):
            source_name = str(entry_source.get("title") or "").strip()
        elif hasattr(entry_source, "title"):
            source_name = str(entry_source.title or "").strip()

        title = raw_title
        if " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            candidate_source = parts[1].strip()
            if not source_name and len(candidate_source) < 40:
                source_name = candidate_source
            if len(candidate_source) < 40:
                title = parts[0].strip()

        return source_name or "Google News", title

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower().removeprefix("www.")
        except Exception:
            return ""
