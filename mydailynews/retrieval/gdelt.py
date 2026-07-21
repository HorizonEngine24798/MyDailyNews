from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlparse

from mydailynews.common.cache import CachedHttpClient, HTTPCache, retry_after_seconds
from mydailynews.common.utils import canonical_article_url, normalize_whitespace, stable_id
from mydailynews.diagnostics.debug import DebugLogger


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY_TEXT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
OPERATOR_VALUE_RE = re.compile(r"[^a-z0-9]", flags=re.IGNORECASE)
RETRY_STATUS_CODES = {0, 429, 500, 502, 503, 504}
GDELT_RETRY_SLEEP_SECONDS = 6
GDELT_MIN_REQUEST_INTERVAL_SECONDS = 1.0


class GdeltDocRetriever:
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
        self.http = CachedHttpClient(user_agent=user_agent, cache=http_cache, cache_mode=http_cache_mode, debug=self.debug)
        self.rate_limited = False
        self._last_request_at = 0.0

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
            self.debug.increment("provider.gdelt.skipped_rate_limited")
            return [], []
        gdelt_query = build_gdelt_query(query, source_countries=source_countries or [], source_languages=source_languages or [])
        if not gdelt_query:
            return [], ["gdelt_doc: empty coverage query; skipped."]
        params = {
            "query": gdelt_query,
            "mode": "artlist",
            "format": "json",
            "timespan": f"{max(1, int(timespan_days or 1))}d",
            "maxrecords": max(1, min(250, int(max_records or 1))),
            "sort": "datedesc",
        }
        warnings: List[str] = []
        last_warning = ""
        for attempt in range(2):
            self._throttle()
            self.debug.increment("provider.gdelt.requests")
            response = self.http.get_text(GDELT_DOC_URL, timeout=20, allow_redirects=True, params=params)
            if not response.ok:
                last_warning = f"gdelt_doc: request failed for '{gdelt_query}' (status {response.status_code})."
                if attempt == 0 and int(response.status_code) in RETRY_STATUS_CODES:
                    retry_delay = retry_after_seconds(getattr(response, "headers", {}), GDELT_RETRY_SLEEP_SECONDS)
                    retry_warning = f"{last_warning} Retrying in {retry_delay:g}s."
                    warnings.append(retry_warning)
                    self.progress_sink(retry_warning)
                    time.sleep(retry_delay)
                    continue
                if int(response.status_code) == 429:
                    self.rate_limited = True
                    self.debug.increment("provider.gdelt.rate_limited")
                return [], [*warnings, last_warning]
            try:
                payload = json.loads(response.text)
                break
            except json.JSONDecodeError as exc:
                last_warning = f"gdelt_doc: invalid JSON for '{gdelt_query}' ({exc.msg})."
                if attempt == 0:
                    retry_warning = f"gdelt_doc: non-JSON response for '{gdelt_query}'; retrying in {GDELT_RETRY_SLEEP_SECONDS}s."
                    warnings.append(retry_warning)
                    self.progress_sink(retry_warning)
                    time.sleep(GDELT_RETRY_SLEEP_SECONDS)
                    continue
                return [], [*warnings, last_warning]
        else:
            return [], [last_warning or f"gdelt_doc: request failed for '{gdelt_query}'."]
        rows = payload.get("articles") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return [], [f"gdelt_doc: response did not include an article list for '{gdelt_query}'."]
        return [_normalize_article(row) for row in rows if isinstance(row, dict)], warnings

    def _throttle(self) -> None:
        remaining = GDELT_MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


def build_gdelt_query(query: str, *, source_countries: List[str] | None = None, source_languages: List[str] | None = None) -> str:
    parts = [clean_gdelt_query(query)]
    for operator, values in (("sourcecountry", source_countries or []), ("sourcelang", source_languages or [])):
        clause = _operator_clause(operator, values)
        if clause:
            parts.append(clause)
    return " ".join(part for part in parts if part)


def clean_gdelt_query(value: Any) -> str:
    text = normalize_whitespace(QUERY_TEXT_RE.sub(" ", str(value or "")))
    tokens = [token for token in text.split() if len(token) >= 3 or (token.isupper() and len(token) >= 2)]
    return " ".join(tokens[:12])


def _normalize_article(row: Dict[str, Any]) -> Dict[str, Any]:
    url = str(row.get("url", "") or "").strip()
    canonical_url = canonical_article_url(url)
    parsed = urlparse(canonical_url or url)
    domain = str(row.get("domain", "") or parsed.hostname or "").strip().lower()
    source_name = str(row.get("source", "") or row.get("source_name", "") or domain or "Unknown source").strip()
    title = normalize_whitespace(str(row.get("title", "") or ""))
    source_country = str(row.get("sourcecountry", "") or row.get("source_country", "") or "").strip().upper()
    source_language = str(row.get("language", "") or row.get("sourcelang", "") or row.get("source_language", "") or "").strip()
    provider_id = stable_id("gdelt_doc", canonical_url or url, title)
    return {
        "article_id": provider_id,
        "provider": "gdelt_doc",
        "provider_id": provider_id,
        "url": url,
        "canonical_url": canonical_url or url,
        "domain": domain,
        "source_name": source_name,
        "source_country": source_country,
        "source_language": source_language,
        "section_hint": parsed.path or "",
        "source_key": "|".join(["gdelt_doc", source_name or domain, source_country, source_language, parsed.path or ""]),
        "title": title,
        "snippet": normalize_whitespace(str(row.get("snippet", "") or row.get("summary", "") or "")),
        "published_at": str(row.get("seendate", "") or row.get("date", "") or "").strip(),
        "image_url": str(row.get("socialimage", "") or row.get("image", "") or "").strip(),
        "context_status": "metadata_only",
        "exact_duplicate_of": "",
    }


def _operator_clause(operator: str, values: List[str]) -> str:
    cleaned: List[str] = []
    for value in values:
        text = OPERATOR_VALUE_RE.sub("", str(value or "").lower())
        if text and text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        return ""
    terms = [f"{operator}:{value}" for value in cleaned]
    return terms[0] if len(terms) == 1 else f"({' OR '.join(terms)})"
