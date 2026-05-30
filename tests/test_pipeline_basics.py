from __future__ import annotations

from pathlib import Path
import shutil
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
from mydailynews.ai.base import set_ai_artifact_root, write_ai_json_artifact, write_ai_text_artifact
from mydailynews.brief import BriefGenerator
from mydailynews.debug import DebugLogger
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
    def test_ai_invalid_json_artifacts_use_configured_root(self) -> None:
        root = Path("D:/Project/MyDailyNews/.codex_tmp_test/ai_artifacts")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        set_ai_artifact_root(root)
        try:
            text_path = Path(write_ai_text_artifact("ai_invalid_json", "unit", "raw"))
            json_path = Path(write_ai_json_artifact("ai_invalid_json", "unit", {"ok": True}))
        finally:
            set_ai_artifact_root("output")

        self.assertTrue(text_path.exists())
        self.assertTrue(json_path.exists())
        self.assertTrue(str(text_path).startswith(str(root)))
        self.assertTrue(str(json_path).startswith(str(root)))

    def test_debug_analytics_collects_timings_counts_and_artifact(self) -> None:
        debug = DebugLogger(True)
        with debug.span("pipeline.total"):
            with debug.span("snapshot.total"):
                with debug.span("snapshot.rss_fetch"):
                    time.sleep(0.001)
                with debug.span("snapshot.topic_fetch"):
                    time.sleep(0.001)
                with debug.span("snapshot.merge"):
                    time.sleep(0.001)
            with debug.span("headline.shared.total"):
                time.sleep(0.001)
            with debug.span("brief.general.total"):
                with debug.span("brief.general.candidate_prepare"):
                    time.sleep(0.001)
                with debug.span("brief.general.headline_limit"):
                    time.sleep(0.001)
                with debug.span("brief.general.headline_decisions"):
                    time.sleep(0.001)
                with debug.span("brief.general.headline_select"):
                    time.sleep(0.001)
                with debug.span("brief.general.article_fetch"):
                    time.sleep(0.001)
                with debug.span("brief.general.enrichment"):
                    time.sleep(0.001)
                with debug.span("brief.general.final_brief"):
                    time.sleep(0.001)
                with debug.span("brief.general.write_output"):
                    time.sleep(0.001)
        debug.set_metric("pipeline.status", "completed")
        debug.set_metric("pipeline.outputs", 1)
        debug.set_metric("snapshot.raw_candidates", 12)
        debug.set_metric("snapshot.rss_candidates", 7)
        debug.set_metric("snapshot.topic_candidates", 5)
        debug.set_metric("snapshot.unique_candidates", 10)
        debug.set_metric("headline.shared.union_candidates", 8)
        debug.set_metric("headline.shared.decisions", 6)
        debug.set_metric("brief.general.unique_candidates", 10)
        debug.set_metric("brief.general.limited_candidates", 8)
        debug.set_metric("brief.general.decisions", 6)
        debug.set_metric("brief.general.selected", 4)
        debug.set_metric("brief.general.article_fetch.attempted", 4)
        debug.set_metric("brief.general.article_fetch.ok", 3)
        debug.set_metric("brief.general.article_fetch.short_text", 1)
        debug.set_metric("brief.general.article_fetch.failed", 0)
        debug.record_ai(label="headline scoring batch 1/1 (shared)", status="ok", input_tokens=120, output_tokens=42)

        lines = debug.analytics_summary_lines()
        self.assertTrue(any("pipeline total=" in line for line in lines))
        self.assertTrue(any("snapshot timings" in line for line in lines))
        self.assertTrue(any("general timings" in line for line in lines))
        self.assertTrue(any("ai requests=1" in line for line in lines))

        tmpdir = Path("D:/Project/MyDailyNews/.codex_tmp_test/debug_analytics")
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        artifact_path = debug.write_analytics_artifact(tmpdir)
        self.assertTrue(artifact_path.endswith("_debug_analytics.json"))
        with open(artifact_path, "r", encoding="utf-8") as handle:
            payload = handle.read()
        self.assertIn("\"enabled\": true", payload)

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

    def test_shared_snapshot_scoring_unions_candidates_once(self) -> None:
        import mydailynews.orchestrator as orchestrator_module

        shared = _candidate("shared", topic="AI policy")
        general_only = _candidate("general-only", topic="World")
        detailed_only = _candidate("detailed-only", topic="AI policy")

        orchestrator = object.__new__(NewsOrchestrator)
        orchestrator.config = types.SimpleNamespace(
            general_filtering=types.SimpleNamespace(time_window_hours=36, max_headlines_per_ai_batch=6),
            filtering=types.SimpleNamespace(time_window_hours=24, max_headlines_per_ai_batch=4),
            user_memory=UserMemory(),
            cache=types.SimpleNamespace(synth_fresh_seconds=0),
        )
        orchestrator.debug = DebugLogger(False)
        orchestrator.summary_ai_client = object()
        orchestrator.synth_cache = None

        def fake_snapshot_candidates(_snapshot, _since):
            return [], [], [shared, general_only, detailed_only]

        def fake_limit(_candidates, topics, filtering, _since):
            if filtering is orchestrator.config.general_filtering:
                return [shared, general_only]
            self.assertIs(filtering, orchestrator.config.filtering)
            return [shared, detailed_only]

        orchestrator._snapshot_candidates_for_brief = fake_snapshot_candidates  # type: ignore[method-assign]
        orchestrator.limit_candidates_for_ai = fake_limit  # type: ignore[method-assign]

        calls: list[object] = []
        original_analyzer = orchestrator_module.HeadlineAnalyzer

        class _FakeHeadlineAnalyzer:
            def __init__(self, _client, batch_size, *_args, **_kwargs) -> None:
                self.warnings = []
                calls.append(batch_size)

            def analyze(self, candidates, *_args, **_kwargs):
                calls.append([candidate.id for candidate in candidates])
                return {candidate.id: HeadlineDecision(candidate_id=candidate.id, score=7.0) for candidate in candidates}

        orchestrator_module.HeadlineAnalyzer = _FakeHeadlineAnalyzer
        try:
            candidates_by_brief, decisions, warnings = orchestrator._score_snapshot_headlines_once(
                types.SimpleNamespace(merged_candidates=[shared, general_only, detailed_only]),
                utc_now(),
                [TopicConfig(name="World")],
                [TopicConfig(name="AI policy")],
            )
        finally:
            orchestrator_module.HeadlineAnalyzer = original_analyzer

        self.assertEqual(candidates_by_brief["general"], [shared, general_only])
        self.assertEqual(candidates_by_brief["detailed"], [shared, detailed_only])
        self.assertEqual(calls[0], 4)
        self.assertEqual(calls[1], [shared.id, general_only.id, detailed_only.id])
        self.assertEqual(set(decisions.keys()), {shared.id, general_only.id, detailed_only.id})
        self.assertEqual(warnings, [])

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
