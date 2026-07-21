from __future__ import annotations

from types import SimpleNamespace
import unittest

from mydailynews.retrieval.registry_rss import RegistryRssRetriever, parse_feed_articles


class FakeHttp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[str] = []

    def get_text(self, url: str, **kwargs):
        self.calls.append(url)
        return SimpleNamespace(ok=True, status_code=200, text=self.text)


class RegistryRssRetrieverTests(unittest.TestCase):
    def test_feed_parsers_prefer_materially_longer_sanitized_content(self) -> None:
        source = {"source_id": "test"}
        xml_rows = parse_feed_articles(
            """<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><item>
            <title>Story</title><link>https://example.com/story</link><description><![CDATA[<p>Short summary.</p>]]></description>
            <content:encoded><![CDATA[<p>Full report with <b>substantially</b> more reporting detail.</p><p>This second paragraph makes the content materially longer than the summary and useful for analysis.</p>]]></content:encoded>
            </item></channel></rss>""",
            feed_url="https://example.com/feed",
            source=source,
        )
        json_rows = parse_feed_articles(
            '{"items":[{"url":"https://example.com/json","title":"JSON story","summary":"Short summary.","content_html":"<p>Full JSON report with substantially more reporting detail.</p><p>This second paragraph makes the content materially longer than the summary and useful for analysis.</p>"}]}',
            feed_url="https://example.com/feed.json",
            source=source,
        )

        for row in (xml_rows[0], json_rows[0]):
            self.assertIn("second paragraph", row["snippet"])
            self.assertNotIn("<p>", row["snippet"])
            self.assertEqual(row["snippet"], row["feed_content"])
            self.assertEqual(row["feed_summary"], "Short summary.")

    def test_search_fetches_selected_source_once_and_filters_by_query(self) -> None:
        source = {
            "source_id": "gb_test",
            "name": "GB Test",
            "country": "GB",
            "language": "en",
            "feed_urls": ["https://gb.example/feed"],
            "enabled": True,
        }
        retriever = RegistryRssRetriever([source], user_agent="test")
        retriever.http = FakeHttp(
            """<?xml version="1.0"?>
            <rss><channel>
              <item><title>Shared summit reaction</title><link>https://gb.example/story</link><description>Diplomatic summit context</description><pubDate>Wed, 08 Jul 2026 12:00:00 GMT</pubDate></item>
              <item><title>Sports notebook</title><link>https://gb.example/sports</link><description>Unrelated</description></item>
            </channel></rss>"""
        )

        rows, warnings = retriever.search("summit diplomacy", timespan_days=30, max_records=5, source_ids=["gb_test"])
        rows_again, _ = retriever.search("summit", timespan_days=30, max_records=5, source_ids=["gb_test"])

        self.assertEqual(warnings, [])
        self.assertEqual(len(retriever.http.calls), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows_again), 1)
        self.assertEqual(rows[0]["provider"], "registry_rss")
        self.assertEqual(rows[0]["source_id"], "gb_test")


if __name__ == "__main__":
    unittest.main()
