from __future__ import annotations

import hashlib
from typing import Iterable, List

from mydailynews.app.models import NewsCandidate
from mydailynews.common.utils import normalize_url, normalize_whitespace


def article_url_alias(url: str) -> str:
    normalized = normalize_url(url)
    return f"url:{normalized}" if normalized else ""


def article_aliases_for_candidate(candidate: NewsCandidate) -> List[str]:
    aliases: List[str] = []
    metadata = candidate.metadata or {}

    def add(alias: str) -> None:
        alias = normalize_whitespace(alias)
        if alias and alias not in aliases:
            aliases.append(alias)

    add(article_url_alias(candidate.url))
    for key in ("original_url", "resolved_url", "canonical_url"):
        add(article_url_alias(str(metadata.get(key, ""))))

    if candidate.id:
        add(f"candidate:{candidate.id}")

    source_key = _identity_part(candidate.source)
    feed_url = article_url_alias(str(metadata.get("feed_url", "")))
    for key in ("entry_id", "entry_guid", "guid", "google_news_entry_id", "google_news_guid"):
        value = _identity_part(metadata.get(key))
        if not value:
            continue
        if source_key:
            add(f"source_entry:{source_key}:{value}")
        if feed_url:
            add(f"feed_entry:{feed_url}:{value}")
        if key.startswith("google_news"):
            add(f"google_news:{value}")

    return aliases


def merge_article_aliases(*alias_groups: Iterable[str]) -> List[str]:
    merged: List[str] = []
    for group in alias_groups:
        for alias in group:
            normalized = normalize_whitespace(alias)
            if normalized and normalized not in merged:
                merged.append(normalized)
    return merged


def stable_article_id(primary_alias: str) -> str:
    return hashlib.sha1(primary_alias.encode("utf-8")).hexdigest()


def _identity_part(value: object) -> str:
    return normalize_whitespace(str(value or "")).lower()
