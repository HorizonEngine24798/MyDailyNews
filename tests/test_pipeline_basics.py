from __future__ import annotations

import sys
import time
import types
import unittest
from datetime import timedelta

# Local test environment may not have third-party dependencies installed.
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(RequestException=Exception, get=None, post=None)
if "feedparser" not in sys.modules:
    sys.modules["feedparser"] = types.SimpleNamespace(parse=lambda *_args, **_kwargs: types.SimpleNamespace(entries=[]))
if "trafilatura" not in sys.modules:
    sys.modules["trafilatura"] = types.SimpleNamespace(extract=lambda *_args, **_kwargs: "")

from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.brief import BriefGenerator
from mydailynews.config import _worker_count
from mydailynews.models import (
    GoogleNewsSourceConfig,
    HeadlineDecision,
    NewsCandidate,
    RSSSourceConfig,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from mydailynews.orchestrator import NewsOrchestrator
from mydailynews.retrieval.google_news import GoogleNewsQueryRetriever
from mydailynews.scrapers.rss import RSSScraper
from mydailynews.utils import utc_now


def _candidate(source: str, topic: str = "", published_at=None) -> NewsCandidate:
    return NewsCandidate(
        id=f"{source}:{topic}",
        source=source,
        category="test",
        title=f"title {source} {topic}".strip(),
        url=f"https://example.com/{source}/{topic}",
        snippet="snippet",
        published_at=published_at or utc_now(),
        tags=[],
        metadata={"topic_name": topic} if topic else {},
    )


class PipelineBasicsTests(unittest.TestCase):
    def test_rss_parallel_fetch_preserves_source_order(self) -> None:
        since = utc_now() - timedelta(hours=4)
        sources = [
            RSSSourceConfig(name="source-a", url="https://a.example/rss"),
            RSSSourceConfig(name="source-b", url="https://b.example/rss"),
            RSSSourceConfig(name="source-c", url="https://c.example/rss"),
        ]
        scraper = RSSScraper(sources, "test-agent", max_per_source=1, max_workers=3)

        def fake_fetch(source: RSSSourceConfig, _since):
            delay = {"source-a": 0.03, "source-b": 0.01, "source-c": 0.02}[source.name]
            time.sleep(delay)
            return [_candidate(source.name, published_at=since)]

        scraper._fetch_source = fake_fetch  # type: ignore[method-assign]
        items = scraper.fetch(since)
        self.assertEqual([item.source for item in items], ["source-a", "source-b", "source-c"])

    def test_google_parallel_fetch_preserves_topic_order(self) -> None:
        since = utc_now() - timedelta(hours=4)
        topics = [
            TopicConfig(name="topic-a"),
            TopicConfig(name="topic-b"),
            TopicConfig(name="topic-c"),
        ]
        retriever = GoogleNewsQueryRetriever(
            GoogleNewsSourceConfig(enabled=True),
            "test-agent",
            max_workers=3,
        )

        def fake_fetch(topic: TopicConfig, _since):
            delay = {"topic-a": 0.03, "topic-b": 0.01, "topic-c": 0.02}[topic.name]
            time.sleep(delay)
            return [_candidate("Google News", topic=topic.name, published_at=since)]

        retriever._fetch_topic = fake_fetch  # type: ignore[method-assign]
        items = retriever.fetch(topics, since)
        self.assertEqual([item.metadata.get("topic_name") for item in items], ["topic-a", "topic-b", "topic-c"])

    def test_runtime_worker_config_is_clamped(self) -> None:
        raw_runtime = {
            "max_http_workers": 0,
            "max_article_workers": 99,
            "max_enrichment_workers": -7,
        }
        self.assertEqual(_worker_count(raw_runtime, "max_http_workers", 4), 1)
        self.assertEqual(_worker_count(raw_runtime, "max_article_workers", 4), 32)
        self.assertEqual(_worker_count(raw_runtime, "max_enrichment_workers", 4), 1)
        self.assertEqual(_worker_count({}, "max_http_workers", 4), 4)

    def test_snapshot_window_uses_merged_latest_timestamp(self) -> None:
        now = utc_now()
        candidate = _candidate("source-a", published_at=now - timedelta(days=2))
        candidate.metadata["merged_latest_published_at"] = (now - timedelta(hours=2)).isoformat()
        since = now - timedelta(hours=12)
        self.assertTrue(NewsOrchestrator._candidate_in_window(candidate, since))

    def test_headline_analyzer_assigns_topic_without_llm_topic_field(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy",
                    response_format="json_object",
                )
                self.max_input_tokens = 2048
                self.max_new_tokens = 256

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            @staticmethod
            def complete_json(*_args, **_kwargs):
                return {"decisions": [{"id": "example:ai", "score": 8.5}]}

        analyzer = HeadlineAnalyzer(_DummyClient(), batch_size=4)
        topics = [TopicConfig(name="AI policy", description="regulation and model governance")]
        candidate = NewsCandidate(
            id="example:ai",
            source="Example",
            category="test",
            title="New AI policy proposal advances in Washington",
            url="https://example.com/ai",
            snippet="A new proposal would regulate model releases and safety reporting.",
            published_at=utc_now(),
        )
        decisions = analyzer.analyze([candidate], UserMemory(), topics, "Detailed AI policy brief.")
        self.assertEqual(decisions["example:ai"].topic, "AI policy")

    def test_brief_generator_can_trim_prompt_and_keep_output_shape(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-final",
                    response_format="json_object",
                )
                self.max_input_tokens = 900
                self.max_new_tokens = 256

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 3)

            @staticmethod
            def complete_json(*_args, **_kwargs):
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead summary.",
                    "topic_reports": [],
                    "sections": [],
                }

        generator = BriefGenerator(
            _DummyClient(),
            max_context_chars=900,
            input_token_limit=900,
            max_new_tokens=256,
        )
        selected = []
        for index in range(6):
            candidate = NewsCandidate(
                id=f"item-{index}",
                source="Example",
                category="test",
                title=f"Headline {index}",
                url=f"https://example.com/{index}",
                snippet="snippet",
                published_at=utc_now(),
            )
            decision = HeadlineDecision(candidate_id=candidate.id, score=9 - index, topic="World")
            selected.append(
                SelectedArticle(
                    candidate=candidate,
                    decision=decision,
                    article_text="A" * 3000,
                    extraction_status="ok",
                )
            )

        brief = generator.generate(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General daily news brief.",
            "2026-05-30",
        )
        self.assertIn("major_headlines", brief)
        self.assertIn("selected_articles", brief)
        self.assertLessEqual(len(brief["major_headlines"]), len(selected))
        self.assertIn("snippet", brief["selected_articles"][0])


if __name__ == "__main__":
    unittest.main()
