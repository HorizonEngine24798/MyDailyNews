from __future__ import annotations

"""RSS scraper adapted from Horizon's RSSSource -> ContentItem normalization.

Horizon: https://github.com/Thysrael/Horizon
License: MIT
Retained notice: see project LICENSE.
This implementation is intentionally smaller and supports bounded threadpool fetches.
"""

import calendar
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import feedparser

from mydailynews.app.models import NewsCandidate, RSSSourceConfig
from mydailynews.common.cache import CachedHttpClient, HTTPCache
from mydailynews.common.parallel import ordered_parallel_map
from mydailynews.common.utils import (
    GOOGLE_CONSENT_HEADERS,
    normalize_url,
    normalize_whitespace,
    stable_id,
    strip_html,
)
from mydailynews.diagnostics.debug import DebugLogger, safe_url


SYNTHETIC_TEST_TLDS = (".example", ".test", ".invalid", ".localhost")
RSS_LINK_TAG_RE = re.compile(
    r"<link\b[^>]*rel=[\"'][^\"']*alternate[^\"']*[\"'][^>]*>",
    re.IGNORECASE,
)
RSS_TYPE_RE = re.compile(
    r"type=[\"']application/(?:rss|atom)\+xml[\"']",
    re.IGNORECASE,
)
HREF_RE = re.compile(r"href=[\"']([^\"']+)[\"']", re.IGNORECASE)


def _is_real_public_url(url: str) -> bool:
    try:
        host = (urlparse(str(url or "").strip()).hostname or "").lower()
    except Exception:
        return False
    if not host or "." not in host or host == "localhost":
        return False
    return not host.endswith(SYNTHETIC_TEST_TLDS)


def _publisher_domain(feed_url: str, homepage_url: str = "") -> Tuple[str, str]:
    for candidate in (homepage_url, feed_url):
        try:
            parsed = urlparse(str(candidate or "").strip())
        except Exception:
            continue
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            host = parsed.hostname.lower()
            clean_host = host[4:] if host.startswith("www.") else host
            root_url = f"{parsed.scheme}://{parsed.netloc}/"
            return clean_host, root_url
    return "", ""


def discover_rss_links_from_html(html_text: str, base_url: str) -> List[str]:
    discovered: List[str] = []
    for tag_match in RSS_LINK_TAG_RE.finditer(html_text or ""):
        tag = tag_match.group(0)
        if not RSS_TYPE_RE.search(tag):
            continue
        href_match = HREF_RE.search(tag)
        if not href_match:
            continue
        resolved = urljoin(base_url, href_match.group(1).strip())
        if resolved.startswith(("http://", "https://")) and resolved not in discovered:
            discovered.append(resolved)
    return discovered


def repair_or_fallback_rss_feed(
    http: CachedHttpClient,
    feed_url: str,
    *,
    homepage_url: str = "",
    debug: DebugLogger | None = None,
) -> Tuple[str, str, str]:
    """Attempt self-healing when an RSS feed fails or returns 0 entries.

    Returns (feed_text, effective_feed_url, repair_mode) where repair_mode is:
    - 'autodiscovered_html_link' if an alternate RSS/Atom link on the publisher's homepage worked
    - 'google_news_site_fallback' if Google News site:<domain> RSS fallback succeeded
    - '' if repair was not applicable or unsuccessful
    """
    if not _is_real_public_url(feed_url):
        return "", feed_url, ""

    domain, root_url = _publisher_domain(feed_url, homepage_url)
    if not domain:
        return "", feed_url, ""

    # Step 1: Try HTML <link rel="alternate" type="application/rss+xml"> discovery on homepage
    for page_url in dict.fromkeys(filter(None, [homepage_url, root_url])):
        page_resp = http.get_text(page_url, timeout=12, allow_redirects=True)
        if not page_resp.ok or not page_resp.text:
            continue
        for candidate_feed_url in discover_rss_links_from_html(page_resp.text, page_url)[:3]:
            if candidate_feed_url == feed_url:
                continue
            cand_resp = http.get_text(candidate_feed_url, timeout=15, allow_redirects=True)
            if not cand_resp.ok or not cand_resp.text:
                continue
            parsed = feedparser.parse(cand_resp.text)
            if parsed.entries:
                if debug:
                    debug.log(
                        "rss.source",
                        "repaired_via_autodiscovery",
                        original_url=safe_url(feed_url),
                        repaired_url=safe_url(candidate_feed_url),
                        entries=len(parsed.entries),
                    )
                return cand_resp.text, candidate_feed_url, "autodiscovered_html_link"

    # Step 2: Universal Google News site:<domain> RSS fallback
    query = quote_plus(f"site:{domain} when:3d")
    gnews_fallback_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    gnews_resp = http.get_text(
        gnews_fallback_url,
        timeout=15,
        allow_redirects=True,
        headers=GOOGLE_CONSENT_HEADERS,
    )
    if gnews_resp.ok and gnews_resp.text:
        parsed = feedparser.parse(gnews_resp.text)
        if parsed.entries:
            if debug:
                debug.log(
                    "rss.source",
                    "repaired_via_google_news_site_fallback",
                    original_url=safe_url(feed_url),
                    domain=domain,
                    entries=len(parsed.entries),
                )
            return gnews_resp.text, gnews_fallback_url, "google_news_site_fallback"

    return "", feed_url, ""


def strip_google_news_source_suffix(title: str) -> str:
    parts = str(title or "").rsplit(" - ", 1)
    if len(parts) == 2 and parts[0].strip() and len(parts[1].strip()) <= 45:
        return parts[0].strip()
    return str(title or "")


def parse_rss_date_string(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except Exception:
            return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def parse_rss_entry_datetime(entry: Any) -> Optional[datetime]:
    for field in ("published", "updated", "created"):
        parsed_field = f"{field}_parsed"
        try:
            if parsed_field in entry and entry[parsed_field]:
                return datetime.fromtimestamp(calendar.timegm(entry[parsed_field]), tz=timezone.utc)
            if field in entry and entry[field]:
                parsed = parse_rss_date_string(str(entry[field]))
                if parsed is not None:
                    return parsed
        except Exception:
            continue
    return None


class RSSScraper:
    def __init__(
        self,
        sources: List[RSSSourceConfig],
        user_agent: str,
        max_per_source: int,
        max_workers: int = 1,
        http_cache: HTTPCache | None = None,
        http_cache_mode: str = "cache_first",
        debug: DebugLogger | None = None,
    ) -> None:
        self.sources = sources
        self.user_agent = user_agent
        self.max_per_source = max_per_source
        self.max_workers = max(1, int(max_workers))
        self.errors: List[str] = []
        self.debug = debug or DebugLogger(False)
        self.http = CachedHttpClient(
            user_agent=user_agent,
            cache=http_cache,
            debug=self.debug,
            cache_mode=http_cache_mode,
        )

    def fetch(self, since: datetime) -> List[NewsCandidate]:
        self.errors = []
        enabled_sources = [source for source in self.sources if source.enabled]
        if not enabled_sources:
            return []

        worker_count = min(self.max_workers, len(enabled_sources))

        def handle_exception(_index: int, source: RSSSourceConfig, exc: Exception) -> List[NewsCandidate]:
            self.errors.append(f"{source.name}: worker_exception={type(exc).__name__}")
            self.debug.log("rss.source", "worker_exception", source=source.name, error=type(exc).__name__)
            return []

        results = ordered_parallel_map(
            enabled_sources,
            worker_count,
            lambda source: self._fetch_source(source, since),
            on_exception=handle_exception,
        )
        items: List[NewsCandidate] = []
        for result in results:
            items.extend(result)
        return items

    def _fetch_source(self, source: RSSSourceConfig, since: datetime) -> List[NewsCandidate]:
        feed_url = self._expand_env_vars(source.url)
        self.debug.log("rss.source", "fetching", source=source.name, url=safe_url(feed_url))
        response = self.http.get_text(feed_url, timeout=20, allow_redirects=True)
        self.debug.increment(f"cache.discovery.{response.cache_state}")

        feed_text = response.text if response.ok else ""
        feed = feedparser.parse(feed_text) if feed_text else None
        effective_feed_url = feed_url
        repair_mode = ""

        if not response.ok or not feed or not feed.entries:
            repaired_text, repaired_url, repair_mode = repair_or_fallback_rss_feed(
                self.http,
                feed_url,
                debug=self.debug,
            )
            if repaired_text:
                feed = feedparser.parse(repaired_text)
                effective_feed_url = repaired_url

        if not feed or not feed.entries:
            if not response.ok:
                self.errors.append(f"{source.name}: HTTP error status={response.status_code}")
                self.debug.log("rss.source", "failed", source=source.name, status=response.status_code)
            return []

        candidates: List[NewsCandidate] = []
        for entry in feed.entries[: self.max_per_source]:
            published_at = self._parse_date(entry)
            if published_at and published_at < since:
                continue

            url = normalize_url(entry.get("link", effective_feed_url))
            raw_title = normalize_whitespace(strip_html(entry.get("title", "Untitled")))
            title = self._strip_google_news_source_suffix(raw_title) if repair_mode == "google_news_site_fallback" else raw_title
            snippet = self._extract_content(entry)
            if repair_mode == "google_news_site_fallback" and snippet.lower().startswith(title.lower()):
                snippet = normalize_whitespace(snippet[len(title) :].lstrip(" -–—:"))
            entry_id = str(entry.get("id", "") or "").strip()
            entry_guid = str(entry.get("guid", "") or "").strip()
            entry_key = entry_id or entry_guid or url or title

            metadata = {
                "feed_url": effective_feed_url,
                "entry_id": entry_id,
                "entry_guid": entry_guid,
            }
            if repair_mode:
                metadata["rss_repair_mode"] = repair_mode
                metadata["original_feed_url"] = feed_url

            candidates.append(
                NewsCandidate(
                    id=stable_id(source.name, entry_key),
                    source=source.name,
                    category=source.category,
                    title=title,
                    url=url,
                    snippet=snippet,
                    published_at=published_at,
                    tags=[*source.tags, *self._entry_tags(entry)],
                    metadata=metadata,
                )
            )
        self.debug.log(
            "rss.source",
            "complete",
            source=source.name,
            entries_seen=len(feed.entries),
            candidates=len(candidates),
            cache=response.cache_state,
            repair_mode=repair_mode or "none",
        )
        return candidates

    @staticmethod
    def _strip_google_news_source_suffix(title: str) -> str:
        return strip_google_news_source_suffix(title)

    @staticmethod
    def _expand_env_vars(url: str) -> str:
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)).strip(), url)

    @staticmethod
    def _parse_date(entry: Any) -> Optional[datetime]:
        return parse_rss_entry_datetime(entry)

    @staticmethod
    def _entry_tags(entry: Any) -> List[str]:
        tags = []
        for tag in entry.get("tags", []) or []:
            term = tag.get("term") if isinstance(tag, dict) else getattr(tag, "term", "")
            if term:
                tags.append(str(term))
        return tags

    @staticmethod
    def _extract_content(entry: Any) -> str:
        if entry.get("summary"):
            return strip_html(entry.summary)
        if entry.get("description"):
            return strip_html(entry.description)
        if entry.get("content"):
            content = entry.content[0].get("value", "")
            return strip_html(content)
        return ""
