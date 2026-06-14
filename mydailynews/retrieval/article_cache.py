from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from mydailynews.app.models import NewsCandidate
from mydailynews.common.cache import JSONCache
from mydailynews.common.utils import datetime_to_iso
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.domain.article_identity import (
    article_aliases_for_candidate,
    article_url_alias,
    merge_article_aliases,
    stable_article_id,
)


ARTICLE_TEXT_CACHE_SCHEMA = "article_text_cache.v1"
ARTICLE_ALIAS_CACHE_SCHEMA = "article_alias_cache.v1"
CACHEABLE_EXTRACTION_STATUSES = {"ok", "short_text"}


class ArticleTextCache:
    def __init__(
        self,
        text_cache: JSONCache,
        alias_cache: JSONCache,
        retention_days: int,
        debug: DebugLogger | None = None,
    ) -> None:
        self.text_cache = text_cache
        self.alias_cache = alias_cache
        self.retention_days = max(0, int(retention_days))
        self.debug = debug or DebugLogger(False)
        self.enabled = bool(text_cache.enabled and alias_cache.enabled and self.retention_days > 0)

    def get_by_aliases(self, aliases: Iterable[str]) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        for alias in merge_article_aliases(aliases):
            alias_record = self.alias_cache.get(self._alias_key(alias), max_age_seconds=self._ttl_seconds)
            article_id = str((alias_record or {}).get("article_id", "")).strip()
            if not article_id:
                continue
            record = self.text_cache.get(article_id, max_age_seconds=self._ttl_seconds)
            if self._is_article_record(record):
                self.debug.increment("cache.article_text.hit")
                self.debug.log("cache.article_text", "hit", article_id=article_id)
                return record
        self.debug.increment("cache.article_text.miss")
        return None

    def store(
        self,
        *,
        candidate: NewsCandidate,
        aliases: Iterable[str],
        article_text: str,
        extraction_status: str,
        resolved_url: str,
    ) -> str:
        if not self.enabled:
            return ""
        status = str(extraction_status or "").strip()
        text = str(article_text or "").strip()
        if status not in CACHEABLE_EXTRACTION_STATUSES or not text:
            return ""

        resolved_alias = article_url_alias(resolved_url)
        candidate_alias = article_url_alias(candidate.url)
        all_aliases = merge_article_aliases(aliases, article_aliases_for_candidate(candidate), [resolved_alias])
        primary_alias = resolved_alias or candidate_alias or (all_aliases[0] if all_aliases else "")
        if not primary_alias:
            return ""

        article_id = stable_article_id(primary_alias)
        record = {
            "schema_version": ARTICLE_TEXT_CACHE_SCHEMA,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "article_id": article_id,
            "aliases": all_aliases,
            "source": candidate.source,
            "title": candidate.title,
            "url": candidate.url,
            "resolved_url": resolved_url or candidate.url,
            "published_at": datetime_to_iso(candidate.published_at),
            "article_text": text,
            "extraction_status": status,
        }
        self.text_cache.put(article_id, record)
        for alias in all_aliases:
            self.alias_cache.put(
                self._alias_key(alias),
                {
                    "schema_version": ARTICLE_ALIAS_CACHE_SCHEMA,
                    "alias": alias,
                    "article_id": article_id,
                },
            )
        self.debug.increment("cache.article_text.stored")
        self.debug.log(
            "cache.article_text",
            "stored",
            article_id=article_id,
            aliases=len(all_aliases),
            status=status,
        )
        return article_id

    def prune(self) -> int:
        if not self.enabled:
            return 0
        removed = self.text_cache.prune_older_than_days(self.retention_days)
        removed += self.alias_cache.prune_older_than_days(self.retention_days)
        if removed:
            self.debug.increment("cache.article_text.pruned", removed)
            self.debug.log("cache.article_text", "pruned", removed=removed, retention_days=self.retention_days)
        return removed

    @property
    def _ttl_seconds(self) -> int:
        return self.retention_days * 24 * 60 * 60

    def _alias_key(self, alias: str) -> str:
        return self.alias_cache.make_key(alias)

    @staticmethod
    def _is_article_record(record: Dict[str, Any] | None) -> bool:
        if not isinstance(record, dict):
            return False
        if record.get("schema_version") != ARTICLE_TEXT_CACHE_SCHEMA:
            return False
        return bool(str(record.get("article_text", "")).strip())
