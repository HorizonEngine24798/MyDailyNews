from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mydailynews.retrieval.gnews import GNEWS_SEARCH_URL, GNewsRetriever


class FakeHttp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get_text(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return SimpleNamespace(ok=True, status_code=200, text=json.dumps(self.payload))


class RateLimitedHttp:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_text(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return SimpleNamespace(ok=False, status_code=429, text="", headers={"Retry-After": "3"})


class GNewsRetrieverTests(unittest.TestCase):
    def test_search_uses_api_key_filters_and_normalizes_rows(self) -> None:
        retriever = GNewsRetriever(user_agent="test-agent", api_key="secret")
        retriever.http = FakeHttp(
            {
                "articles": [
                    {
                        "id": "gnews-1",
                        "title": "Shared story from GNews",
                        "description": "Shared story snippet",
                        "url": "https://example.com/story?utm_source=x&id=1",
                        "image": "https://example.com/image.jpg",
                        "publishedAt": "2026-07-08T12:00:00Z",
                        "lang": "en",
                        "source": {
                            "name": "Example",
                            "url": "https://example.com/",
                            "country": "US",
                        },
                    }
                ]
            }
        )

        rows, warnings = retriever.search(
            "Shared story",
            timespan_days=7,
            max_records=5,
            source_countries=["US"],
            source_languages=["English"],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(retriever.http.calls[0]["url"], GNEWS_SEARCH_URL)
        self.assertEqual(retriever.http.calls[0]["headers"]["X-Api-Key"], "secret")
        self.assertEqual(retriever.http.calls[0]["params"]["country"], "us")
        self.assertEqual(retriever.http.calls[0]["params"]["lang"], "en")
        self.assertEqual(rows[0]["provider"], "gnews")
        self.assertEqual(rows[0]["canonical_url"], "https://example.com/story?id=1")
        self.assertEqual(rows[0]["source_language"], "English")

    def test_missing_api_key_skips_provider(self) -> None:
        retriever = GNewsRetriever(user_agent="test-agent", api_key="")

        rows, warnings = retriever.search("Shared story", timespan_days=7, max_records=5)

        self.assertEqual(rows, [])
        self.assertTrue(any("missing API key" in warning for warning in warnings))

    def test_rate_limit_honors_retry_after_and_stops_provider_for_run(self) -> None:
        retriever = GNewsRetriever(user_agent="test-agent", api_key="secret")
        http = RateLimitedHttp()
        retriever.http = http

        with patch("mydailynews.retrieval.gnews.time.sleep") as sleep:
            rows, warnings = retriever.search("Shared story", timespan_days=7, max_records=5)
        skipped, skipped_warnings = retriever.search("Another story", timespan_days=7, max_records=5)

        self.assertEqual((rows, skipped), ([], []))
        self.assertEqual(len(http.calls), 2)
        self.assertTrue(any("rate limited" in warning for warning in warnings))
        sleep.assert_any_call(3.0)
        self.assertEqual(skipped_warnings, [])


if __name__ == "__main__":
    unittest.main()
