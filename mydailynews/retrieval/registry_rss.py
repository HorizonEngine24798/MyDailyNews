from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import re
from typing import Any, Dict, List, Tuple

import feedparser

from mydailynews.common.cache import CachedHttpClient, HTTPCache
from mydailynews.common.utils import canonical_article_url, normalize_whitespace, stable_id, strip_html
from mydailynews.diagnostics.debug import DebugLogger


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
SEARCH_STOPWORDS = {
    "about", "after", "again", "against", "amid", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "new", "of", "on", "over", "says", "the", "to", "with",
    "need", "additional", "latest", "global", "news", "today", "year", "years", "us", "uk", "eu", "ai",
}


class RegistryRssRetriever:
    def __init__(
        self,
        source_registry: List[Dict[str, Any]],
        user_agent: str,
        http_cache: HTTPCache | None = None,
        http_cache_mode: str = CachedHttpClient.CACHE_FIRST,
        debug: DebugLogger | None = None,
        progress_sink=None,
    ) -> None:
        self.sources = {str(source.get("source_id") or ""): source for source in source_registry if bool(source.get("enabled", True))}
        self.http = CachedHttpClient(user_agent=user_agent, cache=http_cache, cache_mode=http_cache_mode, debug=debug or DebugLogger(False))
        self._articles_by_source: Dict[str, List[Dict[str, Any]]] = {}
        self._warnings_by_source: Dict[str, List[str]] = {}

    def search(
        self,
        query: str,
        *,
        timespan_days: int,
        max_records: int,
        source_countries: List[str] | None = None,
        source_languages: List[str] | None = None,
        source_ids: List[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        sources = self._sources(source_ids or [], source_countries or [], source_languages or [])
        tokens = _tokens(query)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(timespan_days or 1)))
        rows: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for source in sources:
            articles, source_warnings = self._source_articles(source)
            warnings.extend(source_warnings)
            for article in articles:
                if not _fresh_enough(article.get("published_at"), cutoff):
                    continue
                title_hits = len(tokens.intersection(_tokens(article.get("title"))))
                snippet_hits = len(tokens.intersection(_tokens(article.get("snippet"))))
                score = title_hits * 3 + snippet_hits
                if tokens and not (title_hits >= 2 or (title_hits >= 1 and snippet_hits >= 1)):
                    continue
                rows.append({**_normalize_article(source, article), "retrieval_match_score": score, "_score": score})
        rows.sort(key=lambda row: (int(row.pop("_score", 0)), str(row.get("published_at") or "")), reverse=True)
        return rows[: max(1, int(max_records or 1))], warnings

    def _sources(self, source_ids: List[str], countries: List[str], languages: List[str]) -> List[Dict[str, Any]]:
        ids = [str(source_id or "") for source_id in source_ids if str(source_id or "")]
        if ids:
            return [self.sources[source_id] for source_id in ids if source_id in self.sources]
        country_set = {str(country or "").upper() for country in countries if str(country or "").strip()}
        language_set = {str(language or "").lower() for language in languages if str(language or "").strip()}
        return [
            source
            for source in self.sources.values()
            if (not country_set or str(source.get("country") or "").upper() in country_set)
            and (not language_set or str(source.get("language") or "").lower() in language_set)
        ]

    def _source_articles(self, source: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        source_id = str(source.get("source_id") or "")
        if source_id in self._articles_by_source:
            return self._articles_by_source[source_id], self._warnings_by_source.get(source_id, [])
        articles: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for feed_url in source.get("feed_urls") or []:
            response = self.http.get_text(str(feed_url), timeout=20, allow_redirects=True)
            if not response.ok:
                warnings.append(f"registry_rss: {source_id} feed failed ({response.status_code}).")
                continue
            articles.extend(parse_feed_articles(response.text, feed_url=str(feed_url), source=source))
        self._articles_by_source[source_id] = articles
        self._warnings_by_source[source_id] = warnings
        return articles, warnings


def _normalize_article(source: Dict[str, Any], article: Dict[str, Any]) -> Dict[str, Any]:
    url = str(article.get("url") or "")
    canonical = canonical_article_url(url)
    source_id = str(source.get("source_id") or "")
    title = normalize_whitespace(str(article.get("title") or ""))
    snippet = normalize_whitespace(str(article.get("snippet") or ""))
    feed_content = normalize_whitespace(str(article.get("feed_content") or ""))
    feed_summary = normalize_whitespace(str(article.get("feed_summary") or ""))
    article_id = stable_id("registry_rss", source_id, canonical or url, title)
    return {
        "article_id": article_id,
        "provider": "registry_rss",
        "provider_id": article_id,
        "url": url,
        "canonical_url": canonical or url,
        "domain": "",
        "source_id": source_id,
        "source_name": str(source.get("name") or source_id or "Unknown source"),
        "source_country": str(source.get("country") or "").upper(),
        "source_language": str(source.get("language") or "").lower(),
        "source_key": "|".join(["registry_rss", source_id, str(source.get("country") or "").upper(), str(source.get("language") or "").lower()]),
        "title": title,
        "snippet": snippet,
        "feed_content": feed_content,
        "feed_summary": feed_summary,
        "published_at": str(article.get("published_at") or ""),
        "image_url": "",
        "context_status": "snippet_only" if snippet else "metadata_only",
        "context_text": snippet,
        "exact_duplicate_of": "",
    }


def _tokens(value: Any) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(str(value or ""))
        if len(token) >= 3 and token.lower() not in SEARCH_STOPWORDS and not token.isdigit()
    }


def parse_feed_articles(feed_text: str, *, feed_url: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    stripped = (feed_text or "").lstrip()
    if stripped.startswith("{"):
        return _parse_json_feed(stripped, source)
    parsed = feedparser.parse(feed_text)
    articles: List[Dict[str, Any]] = []
    for entry in parsed.entries:
        url = str(entry.get("link") or feed_url or "").strip()
        title = normalize_whitespace(strip_html(str(entry.get("title") or "")))
        if not url or not title:
            continue
        feed_summary, feed_content, snippet = _entry_texts(entry)
        articles.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet,
                "feed_content": feed_content,
                "feed_summary": feed_summary,
                "published_at": _entry_date(entry),
            }
        )
    return articles


def _parse_json_feed(feed_text: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(feed_text)
    except json.JSONDecodeError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else []
    output: List[Dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("external_url") or "").strip()
        title = normalize_whitespace(strip_html(str(item.get("title") or "")))
        if not url or not title:
            continue
        feed_summary, feed_content, snippet = _json_feed_texts(item)
        output.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet,
                "feed_content": feed_content,
                "feed_summary": feed_summary,
                "published_at": _parse_date_string(str(item.get("date_published") or item.get("date_modified") or "")),
            }
        )
    return output


def _entry_texts(entry: Any) -> tuple[str, str, str]:
    summaries = [_clean_feed_text(entry.get(field)) for field in ("summary", "description")]
    content = entry.get("content") or []
    full_texts = [
        _clean_feed_text(item.get("value"))
        for item in content
        if isinstance(item, dict)
    ]
    return _choose_feed_text(summaries, full_texts)


def _json_feed_texts(item: Dict[str, Any]) -> tuple[str, str, str]:
    return _choose_feed_text(
        [_clean_feed_text(item.get("summary"))],
        [_clean_feed_text(item.get("content_text")), _clean_feed_text(item.get("content_html"))],
    )


def _choose_feed_text(summary_candidates: List[str], content_candidates: List[str]) -> tuple[str, str, str]:
    summary = max((text for text in summary_candidates if text), key=len, default="")
    content = max((text for text in content_candidates if text), key=len, default="")
    materially_longer = len(content) >= max(len(summary) + 80, int(len(summary) * 1.35))
    return summary, content, content if materially_longer or not summary else summary


def _clean_feed_text(value: Any) -> str:
    return normalize_whitespace(strip_html(str(value or "")))


def _entry_date(entry: Any) -> str:
    for field in ("published", "updated", "created"):
        parsed_field = f"{field}_parsed"
        try:
            if entry.get(parsed_field):
                return datetime.fromtimestamp(calendar.timegm(entry[parsed_field]), tz=timezone.utc).isoformat()
            if entry.get(field):
                return _parse_date_string(str(entry.get(field)))
        except Exception:
            continue
    return ""


def _parse_date_string(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except Exception:
            return text
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()


def _fresh_enough(value: Any, cutoff: datetime) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)) >= cutoff
