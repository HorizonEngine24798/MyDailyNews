from __future__ import annotations

"""RSS scraper inspired by Horizon's RSSSource -> ContentItem normalization.

Horizon: https://github.com/Thysrael/Horizon
License: MIT
This implementation is intentionally smaller and synchronous for local/mobile portability.
"""

import calendar
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, List, Optional

import feedparser
import requests

from ..models import NewsCandidate, RSSSourceConfig
from ..utils import normalize_url, normalize_whitespace, stable_id, strip_html


class RSSScraper:
    def __init__(self, sources: List[RSSSourceConfig], user_agent: str, max_per_source: int) -> None:
        self.sources = sources
        self.user_agent = user_agent
        self.max_per_source = max_per_source

    def fetch(self, since: datetime) -> List[NewsCandidate]:
        items: List[NewsCandidate] = []
        for source in self.sources:
            if source.enabled:
                items.extend(self._fetch_source(source, since))
        return items

    def _fetch_source(self, source: RSSSourceConfig, since: datetime) -> List[NewsCandidate]:
        feed_url = self._expand_env_vars(source.url)
        try:
            response = requests.get(
                feed_url,
                headers={"User-Agent": self.user_agent},
                timeout=20,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []

        feed = feedparser.parse(response.text)
        candidates: List[NewsCandidate] = []
        for entry in feed.entries[: self.max_per_source]:
            published_at = self._parse_date(entry)
            if published_at and published_at < since:
                continue

            url = normalize_url(entry.get("link", feed_url))
            title = normalize_whitespace(strip_html(entry.get("title", "Untitled")))
            snippet = self._extract_content(entry)
            entry_key = entry.get("id") or entry.get("guid") or url or title

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
                    metadata={"feed_url": feed_url},
                )
            )
        return candidates

    @staticmethod
    def _expand_env_vars(url: str) -> str:
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)).strip(), url)

    @staticmethod
    def _parse_date(entry: Any) -> Optional[datetime]:
        for field in ("published", "updated", "created"):
            parsed_field = f"{field}_parsed"
            try:
                if parsed_field in entry and entry[parsed_field]:
                    return datetime.fromtimestamp(calendar.timegm(entry[parsed_field]), tz=timezone.utc)
                if field in entry and entry[field]:
                    parsed = parsedate_to_datetime(entry[field])
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except Exception:
                continue
        return None

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
