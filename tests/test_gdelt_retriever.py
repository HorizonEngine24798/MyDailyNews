from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mydailynews.retrieval.gdelt import GDELT_DOC_URL, GdeltDocRetriever, build_gdelt_query


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
        return SimpleNamespace(ok=False, status_code=429, text="", headers={"Retry-After": "2"})


class GdeltRetrieverTests(unittest.TestCase):
    def test_build_query_preserves_short_acronyms_and_scopes(self) -> None:
        query = build_gdelt_query("US AI pact at UN!", source_countries=["US"], source_languages=["English"])

        self.assertEqual(query, "US AI pact UN sourcecountry:us sourcelang:english")

    def test_search_normalizes_article_rows(self) -> None:
        retriever = GdeltDocRetriever(user_agent="test-agent")
        retriever.http = FakeHttp(
            {
                "articles": [
                    {
                        "url": "https://example.com/story?utm_source=x&id=1",
                        "domain": "example.com",
                        "source": "Example",
                        "sourcecountry": "US",
                        "language": "English",
                        "title": "Shared story from GDELT",
                        "snippet": "Shared story snippet",
                        "seendate": "20260708T120000Z",
                    }
                ]
            }
        )

        rows, warnings = retriever.search("Shared story", timespan_days=7, max_records=5, source_countries=["US"])

        self.assertEqual(warnings, [])
        self.assertEqual(retriever.http.calls[0]["url"], GDELT_DOC_URL)
        self.assertIn("sourcecountry:us", retriever.http.calls[0]["params"]["query"])
        self.assertEqual(rows[0]["provider"], "gdelt_doc")
        self.assertEqual(rows[0]["canonical_url"], "https://example.com/story?id=1")
        self.assertEqual(rows[0]["source_country"], "US")

    def test_429_stops_gdelt_for_run(self) -> None:
        retriever = GdeltDocRetriever(user_agent="test-agent")
        http = RateLimitedHttp()
        retriever.http = http

        with patch("mydailynews.retrieval.gdelt.time.sleep") as sleep:
            rows, warnings = retriever.search("Shared story", timespan_days=7, max_records=5)
        skipped_rows, skipped_warnings = retriever.search("Another story", timespan_days=7, max_records=5)

        self.assertEqual(rows, [])
        self.assertEqual(skipped_rows, [])
        self.assertEqual(len(http.calls), 2)
        self.assertTrue(any("status 429" in warning for warning in warnings))
        sleep.assert_any_call(2.0)
        self.assertEqual(skipped_warnings, [])


if __name__ == "__main__":
    unittest.main()
