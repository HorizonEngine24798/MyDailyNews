from __future__ import annotations

import time
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mydailynews.retrieval.gdelt import GdeltDocRetriever
from mydailynews.retrieval.google_news_rss import GoogleNewsRssCoverageRetriever
from mydailynews.perspectives.synthesis import build_framing_comparisons


SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Google News</title>
  <item>
    <title>Global summit reaches historic green energy pact - Reuters</title>
    <link>https://news.google.com/rss/articles/CBMiREF1dGhPcmc?oc=5</link>
    <guid isPermaLink="false">CBMiREF1dGhPcmc</guid>
    <pubDate>Sun, 08 Nov 2026 08:30:00 GMT</pubDate>
    <description>&lt;a href="..."&gt;World leaders agreed on a landmark energy accord today.&lt;/a&gt;</description>
    <source url="https://www.reuters.com">Reuters</source>
  </item>
</channel>
</rss>
"""


class TestPerspectivesTuning(TestCase):
    def test_gdelt_exponential_backoff_and_cooldown(self) -> None:
        retriever = GdeltDocRetriever(
            user_agent="TestAgent/1.0",
            rate_limit_cooldown_seconds=1.0,
        )

        mock_429 = MagicMock()
        mock_429.ok = False
        mock_429.status_code = 429
        mock_429.headers = {}

        # 3 failures -> rate_limited becomes True with cooldown
        with patch.object(retriever.http, "get_text", return_value=mock_429), patch("time.sleep") as mock_sleep:
            results, warnings = retriever.search("diplomatic summit", timespan_days=3, max_records=5)
            self.assertEqual(results, [])
            self.assertTrue(retriever.rate_limited)
            # Verify exponential backoff sleep was called
            self.assertGreaterEqual(mock_sleep.call_count, 2)

        # Immediate next search skipped due to rate_limited
        results2, _ = retriever.search("diplomatic summit", timespan_days=3, max_records=5)
        self.assertEqual(results2, [])

        # After cooldown expires, rate_limited clears automatically
        retriever._rate_limited_until = time.monotonic() - 1.0
        self.assertFalse(retriever.rate_limited)

    def test_google_news_rss_coverage_retriever(self) -> None:
        retriever = GoogleNewsRssCoverageRetriever(user_agent="TestAgent/1.0")

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = SAMPLE_RSS_XML

        with patch.object(retriever.http, "get_text", return_value=mock_resp), patch("time.sleep"):
            articles, warnings = retriever.search(
                "green energy pact",
                timespan_days=5,
                max_records=10,
                source_countries=["FR", "GB"],
            )

        self.assertGreaterEqual(len(articles), 1)
        first = articles[0]
        self.assertEqual(first["provider"], "google_news_rss")
        self.assertIn("energy pact", first["title"].lower())
        self.assertEqual(first["source_name"], "Reuters")
        self.assertIn(first["source_country"], {"FR", "GB"})

    def test_framing_comparison_notes_single_country_coverage(self) -> None:
        inputs = {
            "stories": [
                {
                    "story_id": "s-1",
                    "title": "Local Election Results",
                    "summary": "State elections held.",
                    "claims": [],
                }
            ]
        }
        coverage = {
            "s-1": {
                "coverage_articles": [
                    {
                        "article_id": "art-1",
                        "title": "Election Day",
                        "source_name": "US Local News",
                        "source_country": "US",
                        "context_text": "Polls opened across the state.",
                    }
                ],
                "coverage_quality": {"thin_reasons": []},
            }
        }

        mock_ai = MagicMock()
        mock_ai.config = MagicMock(
            context_window_tokens=16384,
            max_input_tokens=12000,
            max_new_tokens=2048,
        )
        mock_ai.complete_json.return_value = {
            "stories": [
                {
                    "story_id": "s-1",
                    "synthesis": "Single-state election coverage.",
                    "synthesis_article_ids": ["art-1"],
                    "shared_facts": [],
                    "repetition_without_independent_support": [],
                    "verified_or_independently_supported_claims": [],
                    "qualified_disputed_or_unresolved_claims": [],
                    "country_source_comparison": [],
                    "coverage_limitations": [],
                }
            ]
        }

        orchestrator = MagicMock()
        orchestrator.summary_ai_client = mock_ai

        reports = build_framing_comparisons(
            orchestrator,
            inputs=inputs,
            coverage_by_story=coverage,
            plans_by_story={},
            warnings=[],
        )

        self.assertIn("s-1", reports)
        # Check that single-country limitation was passed to payload
        mock_call_args = mock_ai.complete_json.call_args[0]
        prompt_user_text = mock_call_args[1]
        self.assertIn("single national jurisdiction", prompt_user_text)
