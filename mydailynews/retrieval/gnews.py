from __future__ import annotations

import json
import math
import re
import time
from datetime import timedelta
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlparse

from mydailynews.common.cache import CachedHttpClient, HTTPCache, retry_after_seconds
from mydailynews.common.utils import canonical_article_url, normalize_whitespace, stable_id, utc_now
from mydailynews.diagnostics.debug import DebugLogger


GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"
GNEWS_RETRY_SLEEP_SECONDS = 2
GNEWS_THROTTLE_SECONDS = 1.1
GNEWS_RETRY_STATUS_CODES = {429, 500, 503}
QUERY_TEXT_RE = re.compile(r"[^\w\s\"().-]", flags=re.UNICODE)
LANGUAGE_ALIASES = {
    "arabic": "ar",
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "japanese": "ja",
    "chinese": "zh",
}
LANGUAGE_NAMES = {value: key.title() for key, value in LANGUAGE_ALIASES.items()}
COUNTRY_ALIASES = {
    "unitedstates": "us",
    "usa": "us",
    "uk": "gb",
    "unitedkingdom": "gb",
    "france": "fr",
    "germany": "de",
    "japan": "jp",
    "india": "in",
    "uae": "ae",
}


class GNewsRetriever:
    def __init__(
        self,
        user_agent: str,
        api_key: str,
        http_cache: HTTPCache | None = None,
        http_cache_mode: str = CachedHttpClient.CACHE_FIRST,
        debug: DebugLogger | None = None,
        progress_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.debug = debug or DebugLogger(False)
        self.progress_sink = progress_sink or (lambda message: None)
        self.http = CachedHttpClient(user_agent=user_agent, cache=http_cache, cache_mode=http_cache_mode, debug=self.debug)
        self.rate_limited = False

    def search(
        self,
        query: str,
        *,
        timespan_days: int,
        max_records: int,
        source_countries: List[str] | None = None,
        source_languages: List[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if self.rate_limited:
            self.debug.increment("provider.gnews.skipped_rate_limited")
            return [], []
        if not self.api_key:
            return [], ["gnews: missing API key; set perspectives_report.gnews_api_key or GNEWS_API_KEY."]
        gnews_query = clean_gnews_query(query)
        if not gnews_query:
            return [], ["gnews: empty coverage query; skipped."]
        filters = _request_filters(source_countries or [], source_languages or [])
        request_limit = max(1, min(100, math.ceil(max(1, int(max_records or 1)) / len(filters))))
        warnings: List[str] = []
        articles: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for offset, request_filter in enumerate(filters):
            if offset:
                time.sleep(GNEWS_THROTTLE_SECONDS)
            page, page_warnings = self._search_page(gnews_query, timespan_days, request_limit, request_filter)
            warnings.extend(page_warnings)
            if any(warning.startswith(("gnews: quota exhausted", "gnews: rate limited", "gnews: unauthorized")) for warning in page_warnings):
                return articles, warnings
            for article in page:
                key = article["canonical_url"] or article["url"] or article["provider_id"]
                if key in seen:
                    continue
                seen.add(key)
                articles.append(article)
                if len(articles) >= max_records:
                    return articles, warnings
        return articles, warnings

    def _search_page(self, query: str, timespan_days: int, max_records: int, request_filter: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        params = {
            "q": query,
            "max": max(1, min(100, int(max_records or 1))),
            "sortby": "publishedAt",
            "in": "title,description,content",
            "from": (utc_now() - timedelta(days=max(1, int(timespan_days or 1)))).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        params.update(request_filter)
        warnings: List[str] = []
        last_warning = ""
        for attempt in range(2):
            self.debug.increment("provider.gnews.requests")
            response = self.http.get_text(
                GNEWS_SEARCH_URL,
                timeout=20,
                allow_redirects=True,
                params=params,
                headers={"X-Api-Key": self.api_key},
            )
            if not response.ok:
                last_warning = _status_warning(query, response.status_code, response.text)
                if attempt == 0 and int(response.status_code) in GNEWS_RETRY_STATUS_CODES:
                    retry_delay = retry_after_seconds(getattr(response, "headers", {}), GNEWS_RETRY_SLEEP_SECONDS)
                    retry_warning = f"{last_warning} Retrying in {retry_delay:g}s."
                    warnings.append(retry_warning)
                    self.progress_sink(retry_warning)
                    time.sleep(retry_delay)
                    continue
                if int(response.status_code) == 429:
                    self.rate_limited = True
                    self.debug.increment("provider.gnews.rate_limited")
                return [], [*warnings, last_warning]
            try:
                payload = json.loads(response.text)
                break
            except json.JSONDecodeError as exc:
                return [], [f"gnews: invalid JSON for '{query}' ({exc.msg})."]
        else:
            return [], [last_warning or f"gnews: request failed for '{query}'."]
        rows = payload.get("articles") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return [], [f"gnews: response did not include an article list for '{query}'."]
        return [_normalize_article(row) for row in rows if isinstance(row, dict)], warnings


def clean_gnews_query(value: Any) -> str:
    return normalize_whitespace(QUERY_TEXT_RE.sub(" ", str(value or "")))[:200].strip()


def _request_filters(countries: List[str], languages: List[str]) -> List[Dict[str, str]]:
    country_codes = [_code(value, COUNTRY_ALIASES) for value in countries]
    language_codes = [_code(value, LANGUAGE_ALIASES) for value in languages]
    country_codes = [code for offset, code in enumerate(country_codes) if code and code not in country_codes[:offset]]
    language_codes = [code for offset, code in enumerate(language_codes) if code and code not in language_codes[:offset]]
    if country_codes:
        lang = language_codes[0] if len(language_codes) == 1 else ""
        return [{"country": country, **({"lang": lang} if lang else {})} for country in country_codes]
    if language_codes:
        return [{"lang": language} for language in language_codes]
    return [{}]


def _code(value: str, aliases: Dict[str, str]) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    return cleaned if len(cleaned) == 2 else aliases.get(cleaned, "")


def _normalize_article(row: Dict[str, Any]) -> Dict[str, Any]:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    url = str(row.get("url", "") or "").strip()
    canonical_url = canonical_article_url(url)
    parsed = urlparse(str(source.get("url", "") or "") or canonical_url or url)
    domain = (parsed.hostname or "").strip().lower()
    source_name = normalize_whitespace(str(source.get("name", "") or domain or "Unknown source"))
    source_country = str(source.get("country", "") or "").strip().upper()
    lang = str(row.get("lang", "") or "").strip().lower()
    provider_id = str(row.get("id", "") or stable_id("gnews", canonical_url or url, row.get("title", "")))
    return {
        "article_id": stable_id("gnews", provider_id, canonical_url or url),
        "provider": "gnews",
        "provider_id": provider_id,
        "url": url,
        "canonical_url": canonical_url or url,
        "domain": domain,
        "source_name": source_name,
        "source_country": source_country,
        "source_language": LANGUAGE_NAMES.get(lang, lang.upper()),
        "section_hint": parsed.path or "",
        "source_key": "|".join(["gnews", source_name or domain, source_country, lang, parsed.path or ""]),
        "title": normalize_whitespace(str(row.get("title", "") or "")),
        "snippet": normalize_whitespace(str(row.get("description", "") or row.get("content", "") or "")),
        "published_at": str(row.get("publishedAt", "") or "").strip(),
        "image_url": str(row.get("image", "") or "").strip(),
        "context_status": "metadata_only",
        "exact_duplicate_of": "",
    }


def _status_warning(query: str, status_code: int, body: str) -> str:
    if int(status_code) == 403:
        reason = "quota exhausted"
    elif int(status_code) == 429:
        reason = "rate limited"
    elif int(status_code) == 401:
        reason = "unauthorized"
    else:
        reason = "request failed"
    return f"gnews: {reason} for '{query}' (status {status_code})."
