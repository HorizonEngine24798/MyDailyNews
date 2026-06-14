from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import requests

from mydailynews.app.models import NewsCandidate
from mydailynews.common.cache import CachedHttpClient, HTTPCache, JSONCache
from mydailynews.domain.article_identity import article_aliases_for_candidate, article_url_alias
from mydailynews.retrieval.article_cache import ArticleTextCache


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "cache_strategy"


class FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = {"Content-Type": "text/plain"}


class CacheStrategyTests(unittest.TestCase):
    def test_network_first_ignores_existing_cache_when_network_succeeds(self) -> None:
        temp_dir = self._temp_dir()
        cache = HTTPCache(temp_dir, "discovery")
        url = "https://example.test/rss"
        cache.put(url, 200, "cached feed")
        client = CachedHttpClient("test-agent", cache, cache_mode=CachedHttpClient.NETWORK_FIRST)

        with patch("mydailynews.common.cache.requests.get", return_value=FakeResponse(200, "live feed")) as get:
            response = client.get_text(url)

        self.assertTrue(response.ok)
        self.assertEqual(response.text, "live feed")
        self.assertEqual(response.cache_state, "network")
        self.assertEqual(cache.get(url).body, "live feed")
        get.assert_called_once()

    def test_network_first_falls_back_to_cache_only_on_network_failure(self) -> None:
        temp_dir = self._temp_dir()
        cache = HTTPCache(temp_dir, "discovery")
        url = "https://example.test/rss"
        cache.put(url, 200, "cached feed")
        client = CachedHttpClient("test-agent", cache, cache_mode=CachedHttpClient.NETWORK_FIRST)

        with patch("mydailynews.common.cache.requests.get", side_effect=requests.RequestException):
            response = client.get_text(url)

        self.assertTrue(response.ok)
        self.assertEqual(response.text, "cached feed")
        self.assertEqual(response.cache_state, "cached_fallback")

    def test_article_text_cache_returns_fresh_records(self) -> None:
        temp_dir = self._temp_dir()
        cache = self._article_cache(temp_dir)
        candidate = self._candidate("https://publisher.example/story")
        aliases = article_aliases_for_candidate(candidate)

        cache.store(
            candidate=candidate,
            aliases=aliases,
            article_text="Extracted article body with enough useful context.",
            extraction_status="ok",
            resolved_url=candidate.url,
        )
        record = cache.get_by_aliases(aliases)

        self.assertIsNotNone(record)
        self.assertEqual(record["article_text"], "Extracted article body with enough useful context.")
        self.assertEqual(record["extraction_status"], "ok")

    def test_article_text_cache_ignores_and_prunes_stale_records(self) -> None:
        temp_dir = self._temp_dir()
        cache = self._article_cache(temp_dir)
        candidate = self._candidate("https://publisher.example/stale")
        aliases = article_aliases_for_candidate(candidate)
        cache.store(
            candidate=candidate,
            aliases=aliases,
            article_text="Old extracted article body.",
            extraction_status="ok",
            resolved_url=candidate.url,
        )
        self._age_json_cache(Path(temp_dir), days=4)

        self.assertIsNone(cache.get_by_aliases(aliases))
        self.assertGreaterEqual(cache.prune(), 2)

    def test_alias_lookup_maps_google_wrapper_to_resolved_publisher_url(self) -> None:
        temp_dir = self._temp_dir()
        cache = self._article_cache(temp_dir)
        google_url = "https://news.google.com/articles/CBMi-test"
        publisher_url = "https://publisher.example/world/story"
        candidate = self._candidate(
            google_url,
            source="Google News",
            metadata={
                "feed_url": "https://news.google.com/rss/search?q=world",
                "google_news_entry_id": "CBMi-test",
                "entry_id": "CBMi-test",
            },
        )
        wrapper_aliases = article_aliases_for_candidate(candidate)

        cache.store(
            candidate=candidate,
            aliases=wrapper_aliases,
            article_text="Resolved publisher article text.",
            extraction_status="ok",
            resolved_url=publisher_url,
        )

        wrapper_hit = cache.get_by_aliases(wrapper_aliases)
        publisher_hit = cache.get_by_aliases([article_url_alias(publisher_url)])

        self.assertIsNotNone(wrapper_hit)
        self.assertIsNotNone(publisher_hit)
        self.assertEqual(wrapper_hit["article_id"], publisher_hit["article_id"])
        self.assertEqual(publisher_hit["resolved_url"], publisher_url)

    def _temp_dir(self) -> str:
        path = TEMP_ROOT / self.id().rsplit(".", 1)[-1] / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return str(path)

    @staticmethod
    def _article_cache(root_dir: str) -> ArticleTextCache:
        return ArticleTextCache(
            JSONCache(root_dir, "article_text"),
            JSONCache(root_dir, "article_aliases"),
            retention_days=3,
        )

    @staticmethod
    def _candidate(url: str, source: str = "Example News", metadata: dict | None = None) -> NewsCandidate:
        return NewsCandidate(
            id="candidate-1",
            source=source,
            category="world",
            title="Example headline",
            url=url,
            snippet="Example snippet",
            published_at=datetime(2026, 6, 14, tzinfo=timezone.utc),
            metadata=metadata or {"feed_url": "https://example.test/rss", "entry_id": "entry-1"},
        )

    @staticmethod
    def _age_json_cache(root_dir: Path, days: int) -> None:
        cached_at = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        for path in root_dir.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cached_at"] = cached_at
            if isinstance(payload.get("value"), dict):
                payload["value"]["cached_at"] = cached_at
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
