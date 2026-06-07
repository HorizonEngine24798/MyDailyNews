from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
import types
import unittest
from datetime import timedelta
from typing import Any, Dict

# Local test environment may not have third-party dependencies installed.
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(RequestException=Exception, get=None, post=None)
if "feedparser" not in sys.modules:
    sys.modules["feedparser"] = types.SimpleNamespace(parse=lambda *_args, **_kwargs: types.SimpleNamespace(entries=[]))
if "trafilatura" not in sys.modules:
    sys.modules["trafilatura"] = types.SimpleNamespace(extract=lambda *_args, **_kwargs: "")

from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.ai.llama_cpp_server_client import LlamaCppServerClient
from mydailynews.ai.base import AITransportError, set_ai_artifact_root, write_ai_json_artifact, write_ai_text_artifact
import mydailynews.ai.factory as ai_factory_module
from mydailynews.analysis_pipeline import DeltaExtractor, EvidenceDistiller
from mydailynews.brief import BriefGenerator
import mydailynews.brief_execution as brief_execution_module
from mydailynews.debug import DebugLogger
from mydailynews.config import _worker_count, load_config
from mydailynews.headline_selection import limit_candidates_for_ai, select_articles
from mydailynews.models import (
    AIConfig,
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
    FilteringConfig,
    GoogleNewsSourceConfig,
    HeadlineDecision,
    NewsCandidate,
    PriorReport,
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
    def test_auto_backend_prefers_llama_cpp_and_lazily_falls_back(self) -> None:
        calls: list[str] = []

        class _FailingLlamaClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(backend="llama_cpp_server")
                self.unloaded = False

            @property
            def max_input_tokens(self) -> int:
                return 512

            @property
            def max_new_tokens(self) -> int:
                return 64

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                raise AITransportError("primary unavailable")

            def unload(self) -> None:
                self.unloaded = True

        class _FallbackClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(backend="transformers")

            @property
            def max_input_tokens(self) -> int:
                return 512

            @property
            def max_new_tokens(self) -> int:
                return 64

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                return {"ok": True, "backend": "transformers"}

            def unload(self) -> None:
                return None

        original_factory = ai_factory_module._create_specific_ai_client

        def fake_factory(config, backend, debug=None):
            _ = config, debug
            calls.append(backend)
            if backend == "llama_cpp_server":
                return _FailingLlamaClient()
            if backend == "transformers":
                return _FallbackClient()
            raise AssertionError(f"unexpected backend: {backend}")

        try:
            ai_factory_module._create_specific_ai_client = fake_factory
            client = ai_factory_module.create_ai_client(AIConfig(backend="auto"), DebugLogger(False))
            result = client.complete_json("system", "user", label="unit.auto_fallback")
        finally:
            ai_factory_module._create_specific_ai_client = original_factory

        self.assertEqual(result, {"ok": True, "backend": "transformers"})
        self.assertEqual(calls, ["llama_cpp_server", "transformers"])

    def test_llama_cpp_server_honors_input_token_limit(self) -> None:
        captured: dict[str, Any] = {}

        class _FakeLlamaClient(LlamaCppServerClient):
            def _post_chat_completion(self, payload: Dict[str, Any]) -> str:  # type: ignore[override]
                captured["payload"] = payload
                return '{"ok": true}'

        client = _FakeLlamaClient(
            AIConfig(
                backend="llama_cpp_server",
                server_model="unit-gguf",
                token_estimation_chars_per_token=2.0,
                json_retries=0,
                max_input_tokens=4096,
                max_new_tokens=128,
            ),
            DebugLogger(False),
        )
        system = "You are a JSON-only assistant."
        user = "Long user content. " * 800
        limit = 120

        result = client.complete_json(system, user, label="unit.llama.limit", input_token_limit=limit)
        self.assertTrue(result.get("ok"))
        payload = captured["payload"]
        sent_system = str(payload["messages"][0]["content"])
        sent_user = str(payload["messages"][1]["content"])
        sent_tokens = client.estimate_tokens(f"System:\n{sent_system}\n\nUser:\n{sent_user}\n\nAssistant:\n")

        self.assertLess(len(sent_user), len(user))
        self.assertLessEqual(sent_tokens, limit)

    def test_llama_cpp_server_keeps_prompt_when_budget_allows(self) -> None:
        captured: dict[str, Any] = {}

        class _FakeLlamaClient(LlamaCppServerClient):
            def _post_chat_completion(self, payload: Dict[str, Any]) -> str:  # type: ignore[override]
                captured["payload"] = payload
                return '{"ok": true}'

        client = _FakeLlamaClient(
            AIConfig(
                backend="llama_cpp_server",
                server_model="unit-gguf",
                token_estimation_chars_per_token=4.0,
                json_retries=0,
                max_input_tokens=4096,
                max_new_tokens=128,
            ),
            DebugLogger(False),
        )
        system = "System rules."
        user = "Short user content."
        limit = 1000

        result = client.complete_json(system, user, label="unit.llama.no_limit", input_token_limit=limit)
        self.assertTrue(result.get("ok"))
        payload = captured["payload"]
        self.assertEqual(str(payload["messages"][0]["content"]), system)
        self.assertEqual(str(payload["messages"][1]["content"]), user)

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

    def test_legacy_single_ai_section_is_rejected(self) -> None:
        base = json.loads(Path("D:/Project/MyDailyNews/config.example.json").read_text(encoding="utf-8"))
        base["ai"] = dict(base["ai_summary"])
        base.pop("ai_summary", None)
        base.pop("ai_final", None)

        tmp_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/config_validation")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "legacy_ai_only.json"
        config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Legacy config key 'ai' is no longer supported"):
            load_config(config_path)

    def test_analysis_defaults_are_applied_when_section_is_missing(self) -> None:
        base = json.loads(Path("D:/Project/MyDailyNews/config.example.json").read_text(encoding="utf-8"))
        base.pop("analysis", None)

        tmp_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/config_analysis_defaults")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "without_analysis.json"
        config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

        loaded = load_config(config_path)
        self.assertFalse(loaded.analysis.evidence_distillation.enabled)
        self.assertFalse(loaded.analysis.delta_extraction.enabled)
        self.assertEqual(loaded.analysis.evidence_distillation.model_role, "summary")
        self.assertEqual(loaded.analysis.delta_extraction.model_role, "summary")

    def test_analysis_section_is_loaded_when_present(self) -> None:
        base = json.loads(Path("D:/Project/MyDailyNews/config.example.json").read_text(encoding="utf-8"))
        base["analysis"] = {
            "evidence_distillation": {
                "enabled": True,
                "model_role": "final",
                "include_reader_qa": False,
                "max_input_tokens": 1800,
                "max_new_tokens": 420,
                "max_articles": 5,
                "max_article_chars": 500,
                "max_context_sources_per_article": 1,
                "max_story_clusters": 3,
                "max_claims_per_cluster": 2,
                "max_questions": 4,
                "cache_ttl_seconds": 3600,
            },
            "delta_extraction": {
                "enabled": True,
                "model_role": "summary",
                "input_source": "evidence_only",
                "require_prior_reports": True,
                "max_input_tokens": 1200,
                "max_new_tokens": 220,
                "max_prior_reports": 2,
                "cache_ttl_seconds": 900,
            },
        }

        tmp_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/config_analysis_present")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "with_analysis.json"
        config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

        loaded = load_config(config_path)
        self.assertTrue(loaded.analysis.evidence_distillation.enabled)
        self.assertEqual(loaded.analysis.evidence_distillation.model_role, "final")
        self.assertFalse(loaded.analysis.evidence_distillation.include_reader_qa)
        self.assertEqual(loaded.analysis.evidence_distillation.max_articles, 5)
        self.assertEqual(loaded.analysis.evidence_distillation.max_questions, 4)

        self.assertTrue(loaded.analysis.delta_extraction.enabled)
        self.assertEqual(loaded.analysis.delta_extraction.model_role, "summary")
        self.assertEqual(loaded.analysis.delta_extraction.input_source, "evidence_only")
        self.assertTrue(loaded.analysis.delta_extraction.require_prior_reports)
        self.assertEqual(loaded.analysis.delta_extraction.max_prior_reports, 2)

    def test_filtering_diversity_settings_are_loaded(self) -> None:
        base = json.loads(Path("D:/Project/MyDailyNews/config.example.json").read_text(encoding="utf-8"))
        base["filtering"]["max_selected_per_source"] = 1
        base["filtering"]["max_selected_per_event_cluster"] = 1
        base["filtering"]["prefer_multi_source_clusters"] = False
        base["filtering"]["multi_source_cluster_bonus"] = 0.8
        base["filtering"]["event_cluster_time_window_hours"] = 10
        base["general_filtering"]["max_selected_per_source"] = 4
        base["general_filtering"]["max_selected_per_event_cluster"] = 3

        tmp_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/config_filtering_diversity")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "with_filtering_diversity.json"
        config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

        loaded = load_config(config_path)
        self.assertEqual(loaded.filtering.max_selected_per_source, 1)
        self.assertEqual(loaded.filtering.max_selected_per_event_cluster, 1)
        self.assertFalse(loaded.filtering.prefer_multi_source_clusters)
        self.assertAlmostEqual(loaded.filtering.multi_source_cluster_bonus, 0.8)
        self.assertEqual(loaded.filtering.event_cluster_time_window_hours, 10)
        self.assertEqual(loaded.general_filtering.max_selected_per_source, 4)
        self.assertEqual(loaded.general_filtering.max_selected_per_event_cluster, 3)

    def test_limit_candidates_for_ai_adds_event_cluster_metadata(self) -> None:
        now = utc_now()
        since = now - timedelta(hours=36)
        candidates = [
            NewsCandidate(
                id="a",
                source="Source A",
                category="test",
                title="US EU tariff talks signal breakthrough for trade ministers",
                url="https://example.com/a",
                snippet="Snippet A",
                published_at=now - timedelta(hours=1),
            ),
            NewsCandidate(
                id="b",
                source="Source B",
                category="test",
                title="Trade ministers signal breakthrough in US EU tariff talks",
                url="https://example.com/b",
                snippet="Snippet B",
                published_at=now - timedelta(hours=2),
            ),
            NewsCandidate(
                id="c",
                source="Source C",
                category="test",
                title="NASA updates timeline for next crewed lunar mission",
                url="https://example.com/c",
                snippet="Snippet C",
                published_at=now - timedelta(hours=3),
            ),
        ]
        filtering = FilteringConfig(
            max_candidates_for_ai=6,
            max_selected_articles=3,
            event_cluster_time_window_hours=24,
        )
        limited = limit_candidates_for_ai(
            candidates,
            [TopicConfig(name="World")],
            filtering,
            since,
            user_memory=UserMemory(),
            debug=DebugLogger(False),
        )

        by_id = {item.id: item for item in limited}
        self.assertIn("a", by_id)
        self.assertIn("b", by_id)
        self.assertIn("c", by_id)
        self.assertTrue(by_id["a"].metadata.get("event_cluster_id"))
        self.assertTrue(by_id["b"].metadata.get("event_cluster_id"))
        self.assertTrue(by_id["c"].metadata.get("event_cluster_id"))
        self.assertEqual(
            by_id["a"].metadata.get("event_cluster_id"),
            by_id["b"].metadata.get("event_cluster_id"),
        )
        self.assertNotEqual(
            by_id["a"].metadata.get("event_cluster_id"),
            by_id["c"].metadata.get("event_cluster_id"),
        )
        self.assertTrue(bool(by_id["a"].metadata.get("event_cluster_multi_source")))

    def test_select_articles_enforces_source_and_cluster_caps(self) -> None:
        now = utc_now()

        def _scored_candidate(
            item_id: str,
            source: str,
            title: str,
            cluster_id: str,
            score: float,
            *,
            multi_source: bool = False,
        ) -> tuple[NewsCandidate, HeadlineDecision]:
            candidate = NewsCandidate(
                id=item_id,
                source=source,
                category="test",
                title=title,
                url=f"https://example.com/{item_id}",
                snippet="snippet",
                published_at=now,
                metadata={
                    "event_cluster_id": cluster_id,
                    "event_cluster_label": title,
                    "event_cluster_size": 2,
                    "event_cluster_source_count": 2 if multi_source else 1,
                    "event_cluster_multi_source": multi_source,
                },
            )
            decision = HeadlineDecision(candidate_id=item_id, score=score, topic="World")
            return candidate, decision

        rows = [
            _scored_candidate("a1", "Source A", "A1", "evt-1", 9.2, multi_source=True),
            _scored_candidate("a2", "Source A", "A2", "evt-1", 9.0, multi_source=True),
            _scored_candidate("b1", "Source B", "B1", "evt-1", 8.8, multi_source=True),
            _scored_candidate("c1", "Source C", "C1", "evt-2", 8.4),
            _scored_candidate("d1", "Source D", "D1", "evt-3", 8.2),
        ]
        candidates = [item[0] for item in rows]
        decisions = {item[1].candidate_id: item[1] for item in rows}
        filtering = FilteringConfig(
            headline_score_cutoff=0.0,
            max_selected_articles=3,
            fill_selected_articles=False,
            max_selected_per_source=1,
            max_selected_per_event_cluster=1,
            prefer_multi_source_clusters=False,
        )

        selected = select_articles(
            candidates,
            decisions,
            [TopicConfig(name="World")],
            filtering,
        )
        selected_ids = [item.candidate.id for item in selected]
        selected_sources = [item.candidate.source for item in selected]
        selected_clusters = [item.candidate.metadata.get("event_cluster_id", "") for item in selected]

        self.assertEqual(len(selected_ids), 3)
        self.assertEqual(len(set(selected_sources)), 3)
        self.assertEqual(len(set(selected_clusters)), 3)

    def test_select_articles_prefers_multi_source_clusters_when_scores_are_close(self) -> None:
        now = utc_now()
        single_source = NewsCandidate(
            id="single",
            source="Source A",
            category="test",
            title="Single-source event",
            url="https://example.com/single",
            snippet="snippet",
            published_at=now,
            metadata={
                "event_cluster_id": "evt-1",
                "event_cluster_multi_source": False,
            },
        )
        multi_source = NewsCandidate(
            id="multi",
            source="Source B",
            category="test",
            title="Multi-source event",
            url="https://example.com/multi",
            snippet="snippet",
            published_at=now,
            metadata={
                "event_cluster_id": "evt-2",
                "event_cluster_multi_source": True,
                "event_cluster_source_count": 3,
            },
        )
        decisions = {
            "single": HeadlineDecision(candidate_id="single", score=8.0, topic="World"),
            "multi": HeadlineDecision(candidate_id="multi", score=7.75, topic="World"),
        }
        filtering = FilteringConfig(
            headline_score_cutoff=0.0,
            max_selected_articles=1,
            fill_selected_articles=False,
            max_selected_per_source=0,
            max_selected_per_event_cluster=0,
            prefer_multi_source_clusters=True,
            multi_source_cluster_bonus=0.4,
        )
        selected = select_articles(
            [single_source, multi_source],
            decisions,
            [TopicConfig(name="World")],
            filtering,
        )
        self.assertEqual(selected[0].candidate.id, "multi")

    def test_evidence_distiller_trims_prompt_and_respects_reader_qa_toggle(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 900
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 3)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "overview": "Overview text.",
                    "story_clusters": [
                        {
                            "cluster_id": "cluster-1",
                            "topic": "World",
                            "label": "Label",
                            "summary": "Summary.",
                            "article_ids": ["item-0"],
                            "key_claims": [
                                {
                                    "claim": "Claim",
                                    "support_article_ids": ["item-0"],
                                    "confidence": "medium",
                                }
                            ],
                            "consensus_points": ["Consensus"],
                            "contested_points": ["Contested"],
                            "known_unknowns": ["Unknown"],
                            "watch_signals": ["Signal"],
                        }
                    ],
                    "global_watch_signals": ["Global signal"],
                    "reader_qa": [
                        {
                            "question": "What changed?",
                            "answer": "Answer.",
                            "article_ids": ["item-0"],
                        }
                    ],
                }

        selected: list[SelectedArticle] = []
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
            selected.append(
                SelectedArticle(
                    candidate=candidate,
                    decision=HeadlineDecision(candidate_id=candidate.id, score=9 - index, topic="World"),
                    article_text=("A" * 2500) + f"#{index}",
                    extraction_status="ok",
                )
            )

        distiller = EvidenceDistiller(
            _DummyClient(),
            EvidenceDistillationConfig(
                enabled=True,
                include_reader_qa=False,
                max_input_tokens=900,
                max_new_tokens=320,
                max_articles=6,
                max_article_chars=1400,
            ),
            debug=DebugLogger(False),
        )
        result = distiller.distill(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General brief.",
            "2026-05-30",
            brief_name="general",
        )

        self.assertIn("overview", result)
        self.assertIn("story_clusters", result)
        self.assertIn("global_watch_signals", result)
        self.assertIn("reader_qa", result)
        self.assertEqual(result["reader_qa"], [])
        self.assertTrue(any("dropped lower-ranked article(s)" in item for item in distiller.warnings))

    def test_evidence_distiller_cache_hit_avoids_repeat_ai_call(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 1800
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "overview": "Overview text.",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        class _MemoryCache:
            def __init__(self) -> None:
                self.store: Dict[str, Dict[str, Any]] = {}

            def get(self, key: str, max_age_seconds: int | None = None):  # noqa: ARG002
                return self.store.get(key)

            def put(self, key: str, value: Dict[str, Any]) -> None:
                self.store[key] = dict(value)

        cache = _MemoryCache()
        client = _DummyClient()
        distiller = EvidenceDistiller(
            client,
            EvidenceDistillationConfig(enabled=True, cache_ttl_seconds=3600),
            debug=DebugLogger(False),
            cache=cache,
        )
        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="A" * 500,
                extraction_status="ok",
            )
        ]

        first = distiller.distill(selected, UserMemory(), [TopicConfig(name="World")], [], "General brief.", "2026-05-30")
        second = distiller.distill(selected, UserMemory(), [TopicConfig(name="World")], [], "General brief.", "2026-05-30")

        self.assertEqual(client.calls, 1)
        self.assertEqual(first, second)

    def test_evidence_distiller_omits_enrichment_context_when_disabled(self) -> None:
        captured: Dict[str, Any] = {}

        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 1800
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            @staticmethod
            def complete_json(_system, user, **_kwargs):
                captured["prompt"] = user
                return {
                    "overview": "Overview text.",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="A" * 500,
                extraction_status="ok",
            )
        ]
        selected[0].context_sources = [
            types.SimpleNamespace(
                kind="wikipedia_summary",
                source="Wikipedia",
                title="Context title",
                summary="Context summary",
                items=[],
            )
        ]

        distiller = EvidenceDistiller(
            _DummyClient(),
            EvidenceDistillationConfig(enabled=True),
            include_enrichment_context=False,
            debug=DebugLogger(False),
        )
        distiller.distill(selected, UserMemory(), [TopicConfig(name="World")], [], "General brief.", "2026-05-30")

        prompt = str(captured.get("prompt", ""))
        self.assertNotIn("context_sources", prompt)

    def test_delta_extractor_skips_when_prior_reports_are_required(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 1600
                self.max_new_tokens = 400

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {}

        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="A" * 500,
                extraction_status="ok",
            )
        ]
        client = _DummyClient()
        extractor = DeltaExtractor(
            client,
            DeltaExtractionConfig(enabled=True, require_prior_reports=True),
            debug=DebugLogger(False),
        )

        result = extractor.extract(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General brief.",
            "2026-05-30",
        )

        self.assertEqual(result, {})
        self.assertEqual(client.calls, 0)
        self.assertTrue(any("require_prior_reports=true" in item for item in extractor.warnings))

    def test_delta_extractor_skips_when_evidence_only_and_no_packet(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 1600
                self.max_new_tokens = 400

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {}

        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="A" * 500,
                extraction_status="ok",
            )
        ]
        client = _DummyClient()
        extractor = DeltaExtractor(
            client,
            DeltaExtractionConfig(enabled=True, input_source="evidence_only"),
            debug=DebugLogger(False),
        )

        result = extractor.extract(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General brief.",
            "2026-05-30",
            evidence_packet={},
        )

        self.assertEqual(result, {})
        self.assertEqual(client.calls, 0)
        self.assertTrue(any("input_source=evidence_only" in item for item in extractor.warnings))

    def test_delta_extractor_parses_schema_and_uses_cache(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 1800
                self.max_new_tokens = 400

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "baseline_coverage_note": "coverage was thin",
                    "new": [{"item": "x", "summary": "y", "article_ids": ["item-1"]}],
                    "escalated": [],
                    "weakened": [],
                    "reframed": [],
                    "unchanged_but_important": [],
                    "evidence_gaps": [{"gap": "g", "why_it_matters": "m"}],
                }

        class _MemoryCache:
            def __init__(self) -> None:
                self.store: Dict[str, Dict[str, Any]] = {}

            def get(self, key: str, max_age_seconds: int | None = None):  # noqa: ARG002
                return self.store.get(key)

            def put(self, key: str, value: Dict[str, Any]) -> None:
                self.store[key] = dict(value)

        cache = _MemoryCache()
        client = _DummyClient()
        extractor = DeltaExtractor(
            client,
            DeltaExtractionConfig(enabled=True, require_prior_reports=False, input_source="evidence_or_articles"),
            debug=DebugLogger(False),
            cache=cache,
        )
        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="A" * 500,
                extraction_status="ok",
            )
        ]
        prior_reports = [
            PriorReport(
                id="report-1",
                date="2026-05-29",
                title="Prior brief",
                path="output/2026-05-29_general_brief.json",
                summary="Prior summary.",
            )
        ]

        first = extractor.extract(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            prior_reports,
            "General brief.",
            "2026-05-30",
            evidence_packet={"overview": "O", "story_clusters": [], "global_watch_signals": [], "reader_qa": []},
        )
        second = extractor.extract(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            prior_reports,
            "General brief.",
            "2026-05-30",
            evidence_packet={"overview": "O", "story_clusters": [], "global_watch_signals": [], "reader_qa": []},
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(first, second)
        self.assertIn("baseline_coverage_note", first)
        self.assertIn("new", first)
        self.assertIn("evidence_gaps", first)

    def test_deterministic_delta_scaffold_derives_overlap_and_new_items(self) -> None:
        prior_reports = [
            PriorReport(
                id="report-1",
                date="2026-05-29",
                title="Prior brief",
                path="output/2026-05-29_general_brief.json",
                summary="Prior summary.",
                major_headlines=[
                    {"headline": "Central bank signals possible rate cuts as inflation cools"},
                    {"headline": "Energy prices fall after regional supply worries ease"},
                ],
            )
        ]
        continuing_candidate = NewsCandidate(
            id="item-continue",
            source="Example",
            category="test",
            title="Central bank signals possible rate cuts amid cooling inflation",
            url="https://example.com/continue",
            snippet="Policy officials signal cuts if inflation continues easing.",
            published_at=utc_now(),
        )
        new_candidate = NewsCandidate(
            id="item-new",
            source="Example",
            category="test",
            title="New satellite launch opens private lunar cargo market",
            url="https://example.com/new",
            snippet="Commercial launch targets new lunar cargo contracts.",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=continuing_candidate,
                decision=HeadlineDecision(candidate_id=continuing_candidate.id, score=8.7, topic="World"),
                article_text="text",
                extraction_status="ok",
            ),
            SelectedArticle(
                candidate=new_candidate,
                decision=HeadlineDecision(candidate_id=new_candidate.id, score=8.5, topic="World"),
                article_text="text",
                extraction_status="ok",
            ),
        ]

        packet = brief_execution_module.build_deterministic_delta_scaffold(
            selected,
            prior_reports,
            max_prior_reports=3,
        )

        self.assertTrue(bool(packet.get("deterministic_scaffold")))
        self.assertIn("baseline_coverage_note", packet)
        self.assertTrue(packet.get("unchanged_but_important") or packet.get("reframed") or packet.get("escalated") or packet.get("weakened"))
        self.assertTrue(packet.get("new"))

    def test_run_brief_uses_deterministic_delta_scaffold_when_delta_stage_disabled(self) -> None:
        class _DummyAIClient:
            def __init__(self, label: str) -> None:
                self.config = types.SimpleNamespace(backend="transformers", effective_model_label=label)
                self.unload_calls = 0

            def unload(self) -> None:
                self.unload_calls += 1

        candidate = NewsCandidate(
            id="item-scaffold",
            source="Example",
            category="test",
            title="Central bank signals possible rate cuts amid cooling inflation",
            url="https://example.com/item-scaffold",
            snippet="Policy officials signal cuts if inflation continues easing.",
            published_at=utc_now(),
        )
        decision = HeadlineDecision(candidate_id=candidate.id, score=8.5, topic="World")
        filtering = types.SimpleNamespace(
            time_window_hours=24,
            max_headlines_per_source=8,
            max_candidates_for_ai=8,
            max_headlines_per_ai_batch=4,
            headline_score_cutoff=6.0,
            max_selected_articles=3,
            fill_selected_articles=True,
            article_text_max_chars=1500,
        )
        output_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/pr8_brief_output")
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = types.SimpleNamespace()
        orchestrator.debug = DebugLogger(False)
        orchestrator.warnings = []
        orchestrator.summary_ai_client = _DummyAIClient("summary")
        orchestrator.final_ai_client = _DummyAIClient("final")
        orchestrator.http_cache = None
        orchestrator.synth_cache = None
        orchestrator.config = types.SimpleNamespace(
            user_agent="test-agent",
            user_memory=UserMemory(),
            output_dir=str(output_dir),
            ai_summary=types.SimpleNamespace(backend="transformers", effective_model_label="summary"),
            ai_final=types.SimpleNamespace(
                backend="transformers",
                effective_model_label="final",
                max_input_tokens=2048,
                max_new_tokens=400,
            ),
            cache=types.SimpleNamespace(http_fresh_seconds=60, synth_fresh_seconds=3600),
            runtime=types.SimpleNamespace(max_enrichment_workers=1),
            enrichment=types.SimpleNamespace(max_context_chars_per_article=450),
            analysis=types.SimpleNamespace(
                evidence_distillation=EvidenceDistillationConfig(enabled=False),
                delta_extraction=DeltaExtractionConfig(
                    enabled=False,
                    max_prior_reports=3,
                ),
            ),
        )

        orchestrator._snapshot_candidates_for_brief = lambda _snapshot, _since: ([], [], [candidate])
        orchestrator._decisions_for_brief = lambda _candidates, shared_decisions, _topics: shared_decisions
        orchestrator.select_articles = lambda candidates, decisions, _topics, _filtering: [
            SelectedArticle(candidate=item, decision=decisions[item.id]) for item in candidates if item.id in decisions
        ]
        orchestrator._populate_article_texts = lambda _name, selected, _retriever, _warnings: [
            setattr(item, "article_text", "full text") or setattr(item, "extraction_status", "ok") for item in selected
        ]
        orchestrator._record_article_fetch_metrics = lambda *_args, **_kwargs: None
        orchestrator._record_enrichment_metrics = lambda *_args, **_kwargs: None

        original_article_retriever = brief_execution_module.ArticleRetriever
        original_enricher = brief_execution_module.SimpleEnricher
        original_brief_generator = brief_execution_module.BriefGenerator
        captured: Dict[str, Any] = {}

        class _FakeArticleRetriever:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            @staticmethod
            def fetch_text(_url):
                return "full text", "ok"

        class _FakeEnricher:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def enrich_many(self, _articles, max_workers=1):  # noqa: ARG002
                return None

        class _FakeBriefGenerator:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def generate(self, *_args, **kwargs):
                captured["delta_packet"] = kwargs.get("delta_packet", {})
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead",
                    "topic_reports": [],
                    "sections": [],
                    "knowns": [],
                    "unknowns": [],
                    "watch_signals": [],
                    "major_headlines": [],
                    "selected_articles": [],
                }

        brief_execution_module.ArticleRetriever = _FakeArticleRetriever
        brief_execution_module.SimpleEnricher = _FakeEnricher
        brief_execution_module.BriefGenerator = _FakeBriefGenerator
        try:
            result = brief_execution_module.run_brief(
                orchestrator,
                name="general",
                output_suffix="general",
                topics=[TopicConfig(name="World")],
                filtering=filtering,
                prior_reports=[
                    PriorReport(
                        id="r1",
                        date="2026-05-29",
                        title="Prior",
                        path="p",
                        summary="s",
                        major_headlines=[{"headline": "Central bank signals possible rate cuts as inflation cools"}],
                    )
                ],
                now=utc_now(),
                date="2026-05-30",
                snapshot=types.SimpleNamespace(fetched_since=utc_now(), metadata={"warnings": []}),
                brief_goal="General brief.",
                limited_candidates_override=[candidate],
                shared_decisions={candidate.id: decision},
            )
        finally:
            brief_execution_module.ArticleRetriever = original_article_retriever
            brief_execution_module.SimpleEnricher = original_enricher
            brief_execution_module.BriefGenerator = original_brief_generator

        self.assertTrue(bool(captured["delta_packet"].get("deterministic_scaffold")))
        self.assertIn("baseline_coverage_note", captured["delta_packet"])
        payload = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["analysis"]["delta_model_role"], "deterministic_scaffold")
        self.assertTrue(bool(payload["analysis"]["delta_packet"].get("deterministic_scaffold")))

    def test_run_brief_includes_evidence_packet_when_enabled(self) -> None:
        class _DummyAIClient:
            def __init__(self, label: str) -> None:
                self.config = types.SimpleNamespace(backend="transformers", effective_model_label=label)
                self.unload_calls = 0

            def unload(self) -> None:
                self.unload_calls += 1

        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-1",
            snippet="snippet",
            published_at=utc_now(),
        )
        decision = HeadlineDecision(candidate_id=candidate.id, score=8.5, topic="World")
        filtering = types.SimpleNamespace(
            time_window_hours=24,
            max_headlines_per_source=8,
            max_candidates_for_ai=8,
            max_headlines_per_ai_batch=4,
            headline_score_cutoff=6.0,
            max_selected_articles=3,
            fill_selected_articles=True,
            article_text_max_chars=1500,
        )
        output_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/pr3_brief_output")
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = types.SimpleNamespace()
        orchestrator.debug = DebugLogger(False)
        orchestrator.warnings = []
        orchestrator.summary_ai_client = _DummyAIClient("summary")
        orchestrator.final_ai_client = _DummyAIClient("final")
        orchestrator.http_cache = None
        orchestrator.synth_cache = None
        orchestrator.config = types.SimpleNamespace(
            user_agent="test-agent",
            user_memory=UserMemory(),
            output_dir=str(output_dir),
            ai_summary=types.SimpleNamespace(backend="transformers", effective_model_label="summary"),
            ai_final=types.SimpleNamespace(
                backend="transformers",
                effective_model_label="final",
                max_input_tokens=2048,
                max_new_tokens=400,
            ),
            cache=types.SimpleNamespace(http_fresh_seconds=60, synth_fresh_seconds=3600),
            runtime=types.SimpleNamespace(max_enrichment_workers=1),
            enrichment=types.SimpleNamespace(max_context_chars_per_article=450),
            analysis=types.SimpleNamespace(
                evidence_distillation=EvidenceDistillationConfig(
                    enabled=True,
                    model_role="summary",
                    cache_ttl_seconds=0,
                ),
                delta_extraction=DeltaExtractionConfig(enabled=False),
            ),
        )

        def _snapshot_candidates_for_brief(_snapshot, _since):
            return [], [], [candidate]

        def _decisions_for_brief(_candidates, shared_decisions, _topics):
            return shared_decisions

        def _select_articles(candidates, decisions, _topics, _filtering):
            selected: list[SelectedArticle] = []
            for item in candidates:
                if item.id in decisions:
                    selected.append(SelectedArticle(candidate=item, decision=decisions[item.id]))
            return selected

        def _populate_article_texts(_name, selected, _article_retriever, _warnings):
            for item in selected:
                item.article_text = "full text"
                item.extraction_status = "ok"

        orchestrator._snapshot_candidates_for_brief = _snapshot_candidates_for_brief
        orchestrator._decisions_for_brief = _decisions_for_brief
        orchestrator.select_articles = _select_articles
        orchestrator._populate_article_texts = _populate_article_texts
        orchestrator._record_article_fetch_metrics = lambda *_args, **_kwargs: None
        orchestrator._record_enrichment_metrics = lambda *_args, **_kwargs: None

        original_article_retriever = brief_execution_module.ArticleRetriever
        original_enricher = brief_execution_module.SimpleEnricher
        original_brief_generator = brief_execution_module.BriefGenerator
        original_evidence_distiller = brief_execution_module.EvidenceDistiller
        captured = {"distill_calls": 0}

        class _FakeArticleRetriever:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            @staticmethod
            def fetch_text(_url):
                return "full text", "ok"

        class _FakeEnricher:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def enrich_many(self, _articles, max_workers=1):  # noqa: ARG002
                return None

        class _FakeEvidenceDistiller:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def distill(self, *_args, **_kwargs):
                captured["distill_calls"] += 1
                return {
                    "overview": "evidence overview",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        class _FakeBriefGenerator:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def generate(self, *_args, **_kwargs):
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead",
                    "topic_reports": [],
                    "sections": [],
                    "major_headlines": [],
                    "selected_articles": [],
                }

        brief_execution_module.ArticleRetriever = _FakeArticleRetriever
        brief_execution_module.SimpleEnricher = _FakeEnricher
        brief_execution_module.BriefGenerator = _FakeBriefGenerator
        brief_execution_module.EvidenceDistiller = _FakeEvidenceDistiller
        try:
            result = brief_execution_module.run_brief(
                orchestrator,
                name="general",
                output_suffix="general",
                topics=[TopicConfig(name="World")],
                filtering=filtering,
                prior_reports=[],
                now=utc_now(),
                date="2026-05-30",
                snapshot=types.SimpleNamespace(fetched_since=utc_now(), metadata={"warnings": []}),
                brief_goal="General brief.",
                limited_candidates_override=[candidate],
                shared_decisions={candidate.id: decision},
            )
        finally:
            brief_execution_module.ArticleRetriever = original_article_retriever
            brief_execution_module.SimpleEnricher = original_enricher
            brief_execution_module.BriefGenerator = original_brief_generator
            brief_execution_module.EvidenceDistiller = original_evidence_distiller

        self.assertEqual(captured["distill_calls"], 1)
        self.assertTrue(Path(result.json_path).exists())
        payload = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        self.assertIn("analysis", payload)
        self.assertIn("evidence_packet", payload["analysis"])
        self.assertEqual(payload["analysis"]["evidence_packet"]["overview"], "evidence overview")

    def test_run_brief_includes_delta_packet_when_enabled(self) -> None:
        class _DummyAIClient:
            def __init__(self, label: str) -> None:
                self.config = types.SimpleNamespace(backend="transformers", effective_model_label=label)
                self.unload_calls = 0

            def unload(self) -> None:
                self.unload_calls += 1

        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-1",
            snippet="snippet",
            published_at=utc_now(),
        )
        decision = HeadlineDecision(candidate_id=candidate.id, score=8.5, topic="World")
        filtering = types.SimpleNamespace(
            time_window_hours=24,
            max_headlines_per_source=8,
            max_candidates_for_ai=8,
            max_headlines_per_ai_batch=4,
            headline_score_cutoff=6.0,
            max_selected_articles=3,
            fill_selected_articles=True,
            article_text_max_chars=1500,
        )
        output_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/pr5_brief_output")
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = types.SimpleNamespace()
        orchestrator.debug = DebugLogger(False)
        orchestrator.warnings = []
        orchestrator.summary_ai_client = _DummyAIClient("summary")
        orchestrator.final_ai_client = _DummyAIClient("final")
        orchestrator.http_cache = None
        orchestrator.synth_cache = None
        orchestrator.config = types.SimpleNamespace(
            user_agent="test-agent",
            user_memory=UserMemory(),
            output_dir=str(output_dir),
            ai_summary=types.SimpleNamespace(backend="transformers", effective_model_label="summary"),
            ai_final=types.SimpleNamespace(
                backend="transformers",
                effective_model_label="final",
                max_input_tokens=2048,
                max_new_tokens=400,
            ),
            cache=types.SimpleNamespace(http_fresh_seconds=60, synth_fresh_seconds=3600),
            runtime=types.SimpleNamespace(max_enrichment_workers=1),
            enrichment=types.SimpleNamespace(max_context_chars_per_article=450),
            analysis=types.SimpleNamespace(
                evidence_distillation=EvidenceDistillationConfig(enabled=False),
                delta_extraction=DeltaExtractionConfig(
                    enabled=True,
                    model_role="summary",
                    cache_ttl_seconds=0,
                ),
            ),
        )

        orchestrator._snapshot_candidates_for_brief = lambda _snapshot, _since: ([], [], [candidate])
        orchestrator._decisions_for_brief = lambda _candidates, shared_decisions, _topics: shared_decisions
        orchestrator.select_articles = lambda candidates, decisions, _topics, _filtering: [
            SelectedArticle(candidate=item, decision=decisions[item.id]) for item in candidates if item.id in decisions
        ]
        orchestrator._populate_article_texts = lambda _name, selected, _retriever, _warnings: [
            setattr(item, "article_text", "full text") or setattr(item, "extraction_status", "ok") for item in selected
        ]
        orchestrator._record_article_fetch_metrics = lambda *_args, **_kwargs: None
        orchestrator._record_enrichment_metrics = lambda *_args, **_kwargs: None

        original_article_retriever = brief_execution_module.ArticleRetriever
        original_enricher = brief_execution_module.SimpleEnricher
        original_brief_generator = brief_execution_module.BriefGenerator
        original_delta_extractor = brief_execution_module.DeltaExtractor
        captured = {"delta_calls": 0}

        class _FakeArticleRetriever:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            @staticmethod
            def fetch_text(_url):
                return "full text", "ok"

        class _FakeEnricher:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def enrich_many(self, _articles, max_workers=1):  # noqa: ARG002
                return None

        class _FakeDeltaExtractor:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def extract(self, *_args, **_kwargs):
                captured["delta_calls"] += 1
                return {
                    "baseline_coverage_note": "note",
                    "new": [],
                    "escalated": [],
                    "weakened": [],
                    "reframed": [],
                    "unchanged_but_important": [],
                    "evidence_gaps": [],
                }

        class _FakeBriefGenerator:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def generate(self, *_args, **_kwargs):
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead",
                    "topic_reports": [],
                    "sections": [],
                    "major_headlines": [],
                    "selected_articles": [],
                }

        brief_execution_module.ArticleRetriever = _FakeArticleRetriever
        brief_execution_module.SimpleEnricher = _FakeEnricher
        brief_execution_module.BriefGenerator = _FakeBriefGenerator
        brief_execution_module.DeltaExtractor = _FakeDeltaExtractor
        try:
            result = brief_execution_module.run_brief(
                orchestrator,
                name="general",
                output_suffix="general",
                topics=[TopicConfig(name="World")],
                filtering=filtering,
                prior_reports=[PriorReport(id="r1", date="2026-05-29", title="Prior", path="p", summary="s")],
                now=utc_now(),
                date="2026-05-30",
                snapshot=types.SimpleNamespace(fetched_since=utc_now(), metadata={"warnings": []}),
                brief_goal="General brief.",
                limited_candidates_override=[candidate],
                shared_decisions={candidate.id: decision},
            )
        finally:
            brief_execution_module.ArticleRetriever = original_article_retriever
            brief_execution_module.SimpleEnricher = original_enricher
            brief_execution_module.BriefGenerator = original_brief_generator
            brief_execution_module.DeltaExtractor = original_delta_extractor

        self.assertEqual(captured["delta_calls"], 1)
        self.assertTrue(Path(result.json_path).exists())
        payload = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        self.assertIn("analysis", payload)
        self.assertIn("delta_packet", payload["analysis"])
        self.assertEqual(payload["analysis"]["delta_packet"]["baseline_coverage_note"], "note")

    def test_run_brief_uses_final_model_role_for_evidence_and_unloads_summary_first(self) -> None:
        class _DummyAIClient:
            def __init__(self, label: str) -> None:
                self.config = types.SimpleNamespace(backend="transformers", effective_model_label=label)
                self.unload_calls = 0

            def unload(self) -> None:
                self.unload_calls += 1

        candidate = NewsCandidate(
            id="item-2",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-2",
            snippet="snippet",
            published_at=utc_now(),
        )
        decision = HeadlineDecision(candidate_id=candidate.id, score=8.5, topic="World")
        filtering = types.SimpleNamespace(
            time_window_hours=24,
            max_headlines_per_source=8,
            max_candidates_for_ai=8,
            max_headlines_per_ai_batch=4,
            headline_score_cutoff=6.0,
            max_selected_articles=3,
            fill_selected_articles=True,
            article_text_max_chars=1500,
        )
        output_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/pr3_brief_output_final_role")
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_client = _DummyAIClient("summary")
        final_client = _DummyAIClient("final")
        orchestrator = types.SimpleNamespace()
        orchestrator.debug = DebugLogger(False)
        orchestrator.warnings = []
        orchestrator.summary_ai_client = summary_client
        orchestrator.final_ai_client = final_client
        orchestrator.http_cache = None
        orchestrator.synth_cache = None
        orchestrator.config = types.SimpleNamespace(
            user_agent="test-agent",
            user_memory=UserMemory(),
            output_dir=str(output_dir),
            ai_summary=types.SimpleNamespace(backend="transformers", effective_model_label="summary"),
            ai_final=types.SimpleNamespace(
                backend="transformers",
                effective_model_label="final",
                max_input_tokens=2048,
                max_new_tokens=400,
            ),
            cache=types.SimpleNamespace(http_fresh_seconds=60, synth_fresh_seconds=3600),
            runtime=types.SimpleNamespace(max_enrichment_workers=1),
            enrichment=types.SimpleNamespace(max_context_chars_per_article=450),
            analysis=types.SimpleNamespace(
                evidence_distillation=EvidenceDistillationConfig(
                    enabled=True,
                    model_role="final",
                    cache_ttl_seconds=0,
                ),
                delta_extraction=DeltaExtractionConfig(enabled=False),
            ),
        )

        orchestrator._snapshot_candidates_for_brief = lambda _snapshot, _since: ([], [], [candidate])
        orchestrator._decisions_for_brief = lambda _candidates, shared_decisions, _topics: shared_decisions
        orchestrator.select_articles = lambda candidates, decisions, _topics, _filtering: [
            SelectedArticle(candidate=item, decision=decisions[item.id]) for item in candidates if item.id in decisions
        ]
        orchestrator._populate_article_texts = lambda _name, selected, _retriever, _warnings: [
            setattr(item, "article_text", "full text") or setattr(item, "extraction_status", "ok") for item in selected
        ]
        orchestrator._record_article_fetch_metrics = lambda *_args, **_kwargs: None
        orchestrator._record_enrichment_metrics = lambda *_args, **_kwargs: None

        original_article_retriever = brief_execution_module.ArticleRetriever
        original_enricher = brief_execution_module.SimpleEnricher
        original_brief_generator = brief_execution_module.BriefGenerator
        original_evidence_distiller = brief_execution_module.EvidenceDistiller
        captured = {"used_final_client": False, "summary_unloaded_before_distill": False}

        class _FakeArticleRetriever:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            @staticmethod
            def fetch_text(_url):
                return "full text", "ok"

        class _FakeEnricher:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def enrich_many(self, _articles, max_workers=1):  # noqa: ARG002
                return None

        class _FakeEvidenceDistiller:
            def __init__(self, client, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []
                captured["used_final_client"] = client is final_client

            def distill(self, *_args, **_kwargs):
                captured["summary_unloaded_before_distill"] = summary_client.unload_calls > 0
                return {
                    "overview": "evidence overview",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        class _FakeBriefGenerator:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def generate(self, *_args, **_kwargs):
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead",
                    "topic_reports": [],
                    "sections": [],
                    "major_headlines": [],
                    "selected_articles": [],
                }

        brief_execution_module.ArticleRetriever = _FakeArticleRetriever
        brief_execution_module.SimpleEnricher = _FakeEnricher
        brief_execution_module.BriefGenerator = _FakeBriefGenerator
        brief_execution_module.EvidenceDistiller = _FakeEvidenceDistiller
        try:
            brief_execution_module.run_brief(
                orchestrator,
                name="general",
                output_suffix="general",
                topics=[TopicConfig(name="World")],
                filtering=filtering,
                prior_reports=[],
                now=utc_now(),
                date="2026-05-30",
                snapshot=types.SimpleNamespace(fetched_since=utc_now(), metadata={"warnings": []}),
                brief_goal="General brief.",
                limited_candidates_override=[candidate],
                shared_decisions={candidate.id: decision},
            )
        finally:
            brief_execution_module.ArticleRetriever = original_article_retriever
            brief_execution_module.SimpleEnricher = original_enricher
            brief_execution_module.BriefGenerator = original_brief_generator
            brief_execution_module.EvidenceDistiller = original_evidence_distiller

        self.assertTrue(captured["used_final_client"])
        self.assertTrue(captured["summary_unloaded_before_distill"])

    def test_run_brief_passes_analysis_packets_to_final_brief_generator(self) -> None:
        class _DummyAIClient:
            def __init__(self, label: str) -> None:
                self.config = types.SimpleNamespace(backend="transformers", effective_model_label=label)
                self.unload_calls = 0

            def unload(self) -> None:
                self.unload_calls += 1

        candidate = NewsCandidate(
            id="item-3",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/item-3",
            snippet="snippet",
            published_at=utc_now(),
        )
        decision = HeadlineDecision(candidate_id=candidate.id, score=8.5, topic="World")
        filtering = types.SimpleNamespace(
            time_window_hours=24,
            max_headlines_per_source=8,
            max_candidates_for_ai=8,
            max_headlines_per_ai_batch=4,
            headline_score_cutoff=6.0,
            max_selected_articles=3,
            fill_selected_articles=True,
            article_text_max_chars=1500,
        )
        output_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/pr6_brief_output")
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = types.SimpleNamespace()
        orchestrator.debug = DebugLogger(False)
        orchestrator.warnings = []
        orchestrator.summary_ai_client = _DummyAIClient("summary")
        orchestrator.final_ai_client = _DummyAIClient("final")
        orchestrator.http_cache = None
        orchestrator.synth_cache = None
        orchestrator.config = types.SimpleNamespace(
            user_agent="test-agent",
            user_memory=UserMemory(),
            output_dir=str(output_dir),
            ai_summary=types.SimpleNamespace(backend="transformers", effective_model_label="summary"),
            ai_final=types.SimpleNamespace(
                backend="transformers",
                effective_model_label="final",
                max_input_tokens=2048,
                max_new_tokens=400,
            ),
            cache=types.SimpleNamespace(http_fresh_seconds=60, synth_fresh_seconds=3600),
            runtime=types.SimpleNamespace(max_enrichment_workers=1),
            enrichment=types.SimpleNamespace(max_context_chars_per_article=450),
            analysis=types.SimpleNamespace(
                evidence_distillation=EvidenceDistillationConfig(enabled=True, model_role="summary", cache_ttl_seconds=0),
                delta_extraction=DeltaExtractionConfig(enabled=True, model_role="summary", cache_ttl_seconds=0),
            ),
        )

        orchestrator._snapshot_candidates_for_brief = lambda _snapshot, _since: ([], [], [candidate])
        orchestrator._decisions_for_brief = lambda _candidates, shared_decisions, _topics: shared_decisions
        orchestrator.select_articles = lambda candidates, decisions, _topics, _filtering: [
            SelectedArticle(candidate=item, decision=decisions[item.id]) for item in candidates if item.id in decisions
        ]
        orchestrator._populate_article_texts = lambda _name, selected, _retriever, _warnings: [
            setattr(item, "article_text", "full text") or setattr(item, "extraction_status", "ok") for item in selected
        ]
        orchestrator._record_article_fetch_metrics = lambda *_args, **_kwargs: None
        orchestrator._record_enrichment_metrics = lambda *_args, **_kwargs: None

        original_article_retriever = brief_execution_module.ArticleRetriever
        original_enricher = brief_execution_module.SimpleEnricher
        original_brief_generator = brief_execution_module.BriefGenerator
        original_evidence_distiller = brief_execution_module.EvidenceDistiller
        original_delta_extractor = brief_execution_module.DeltaExtractor
        captured: Dict[str, Any] = {}

        class _FakeArticleRetriever:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            @staticmethod
            def fetch_text(_url):
                return "full text", "ok"

        class _FakeEnricher:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def enrich_many(self, _articles, max_workers=1):  # noqa: ARG002
                return None

        class _FakeEvidenceDistiller:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def distill(self, *_args, **_kwargs):
                return {
                    "overview": "evidence overview",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        class _FakeDeltaExtractor:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def extract(self, *_args, **_kwargs):
                return {
                    "baseline_coverage_note": "delta note",
                    "new": [],
                    "escalated": [],
                    "weakened": [],
                    "reframed": [],
                    "unchanged_but_important": [],
                    "evidence_gaps": [],
                }

        class _FakeBriefGenerator:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def generate(self, *_args, **kwargs):
                captured["evidence_packet"] = kwargs.get("evidence_packet", {})
                captured["delta_packet"] = kwargs.get("delta_packet", {})
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead",
                    "topic_reports": [],
                    "sections": [],
                    "major_headlines": [],
                    "selected_articles": [],
                }

        brief_execution_module.ArticleRetriever = _FakeArticleRetriever
        brief_execution_module.SimpleEnricher = _FakeEnricher
        brief_execution_module.BriefGenerator = _FakeBriefGenerator
        brief_execution_module.EvidenceDistiller = _FakeEvidenceDistiller
        brief_execution_module.DeltaExtractor = _FakeDeltaExtractor
        try:
            brief_execution_module.run_brief(
                orchestrator,
                name="general",
                output_suffix="general",
                topics=[TopicConfig(name="World")],
                filtering=filtering,
                prior_reports=[PriorReport(id="r1", date="2026-05-29", title="Prior", path="p", summary="s")],
                now=utc_now(),
                date="2026-05-30",
                snapshot=types.SimpleNamespace(fetched_since=utc_now(), metadata={"warnings": []}),
                brief_goal="General brief.",
                limited_candidates_override=[candidate],
                shared_decisions={candidate.id: decision},
            )
        finally:
            brief_execution_module.ArticleRetriever = original_article_retriever
            brief_execution_module.SimpleEnricher = original_enricher
            brief_execution_module.BriefGenerator = original_brief_generator
            brief_execution_module.EvidenceDistiller = original_evidence_distiller
            brief_execution_module.DeltaExtractor = original_delta_extractor

        self.assertEqual(captured["evidence_packet"].get("overview"), "evidence overview")
        self.assertEqual(captured["delta_packet"].get("baseline_coverage_note"), "delta note")

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
        self.assertIn("knowns", brief)
        self.assertIn("unknowns", brief)
        self.assertIn("watch_signals", brief)
        self.assertLessEqual(len(brief["major_headlines"]), len(selected))
        self.assertIn("snippet", brief["selected_articles"][0])

    def test_brief_generator_includes_event_cluster_metadata_in_outputs(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-final",
                    response_format="json_object",
                )
                self.max_input_tokens = 1200
                self.max_new_tokens = 256

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            @staticmethod
            def complete_json(*_args, **_kwargs):
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead summary.",
                    "topic_reports": [],
                    "sections": [],
                }

        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/1",
            snippet="snippet",
            published_at=utc_now(),
            metadata={
                "event_cluster_id": "evt-001",
                "event_cluster_label": "Cluster label",
                "event_cluster_size": 3,
                "event_cluster_source_count": 2,
                "event_cluster_multi_source": True,
                "event_cluster_latest_published_at": "2026-05-30T12:00:00+00:00",
            },
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="full text",
                extraction_status="ok",
            )
        ]
        generator = BriefGenerator(
            _DummyClient(),
            max_context_chars=400,
            input_token_limit=1200,
            max_new_tokens=256,
        )
        brief = generator.generate(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General brief.",
            "2026-05-30",
        )

        selected_cluster = brief["selected_articles"][0].get("event_cluster", {})
        major_cluster = brief["major_headlines"][0].get("event_cluster", {})
        self.assertEqual(selected_cluster.get("id"), "evt-001")
        self.assertEqual(selected_cluster.get("source_count"), 2)
        self.assertTrue(bool(selected_cluster.get("multi_source")))
        self.assertEqual(major_cluster.get("id"), "evt-001")

    def test_brief_generator_includes_analysis_packets_in_prompt(self) -> None:
        captured: Dict[str, Any] = {}

        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-final",
                    response_format="json_object",
                )
                self.max_input_tokens = 2400
                self.max_new_tokens = 300

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            @staticmethod
            def complete_json(_system, user, **_kwargs):
                captured["prompt"] = user
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead summary.",
                    "topic_reports": [],
                    "sections": [],
                }

        generator = BriefGenerator(
            _DummyClient(),
            max_context_chars=600,
            input_token_limit=2400,
            max_new_tokens=300,
        )
        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="full text",
                extraction_status="ok",
            )
        ]

        generator.generate(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General brief.",
            "2026-05-30",
            evidence_packet={
                "overview": "analysis-overview",
                "story_clusters": [],
                "global_watch_signals": [],
                "reader_qa": [],
            },
            delta_packet={
                "baseline_coverage_note": "delta-note",
                "new": [],
                "escalated": [],
                "weakened": [],
                "reframed": [],
                "unchanged_but_important": [],
                "evidence_gaps": [],
            },
        )

        prompt = str(captured.get("prompt", ""))
        self.assertIn("analysis-overview", prompt)
        self.assertIn("delta-note", prompt)

    def test_brief_generator_omits_enrichment_context_when_disabled(self) -> None:
        captured: Dict[str, Any] = {}

        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-final",
                    response_format="json_object",
                )
                self.max_input_tokens = 2400
                self.max_new_tokens = 300

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            @staticmethod
            def complete_json(_system, user, **_kwargs):
                captured["prompt"] = user
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead summary.",
                    "topic_reports": [],
                    "sections": [],
                }

        generator = BriefGenerator(
            _DummyClient(),
            max_context_chars=600,
            input_token_limit=2400,
            max_new_tokens=300,
            include_enrichment_context=False,
        )
        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="full text",
                extraction_status="ok",
                enrichment_reason="Added external context.",
            )
        ]
        selected[0].context_sources = [
            types.SimpleNamespace(
                kind="wikipedia_summary",
                source="Wikipedia",
                title="Context title",
                summary="Context summary",
                items=[],
            )
        ]

        generator.generate(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General brief.",
            "2026-05-30",
        )

        prompt = str(captured.get("prompt", ""))
        self.assertNotIn("context_note", prompt)
        self.assertNotIn("context_sources", prompt)

    def test_brief_generator_compacts_analysis_context_when_over_budget(self) -> None:
        captured: Dict[str, Any] = {}

        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-final",
                    response_format="json_object",
                )
                self.max_input_tokens = 1000
                self.max_new_tokens = 280

            @staticmethod
            def estimate_tokens(text: str) -> int:
                if '"cluster_id":"c5"' in text:
                    return 1400
                if '"cluster_id":"c4"' in text:
                    return 1250
                if '"cluster_id":"c2"' in text:
                    return 860
                return 700

            @staticmethod
            def complete_json(_system, user, **_kwargs):
                captured["prompt"] = user
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead summary.",
                    "topic_reports": [],
                    "sections": [],
                }

        generator = BriefGenerator(
            _DummyClient(),
            max_context_chars=600,
            input_token_limit=1000,
            max_new_tokens=280,
        )
        candidate = NewsCandidate(
            id="item-1",
            source="Example",
            category="test",
            title="Headline",
            url="https://example.com/1",
            snippet="snippet",
            published_at=utc_now(),
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="World"),
                article_text="full text",
                extraction_status="ok",
            )
        ]
        evidence_packet = {
            "overview": "O" * 600,
            "story_clusters": [
                {
                    "cluster_id": f"c{index}",
                    "topic": "World",
                    "label": f"L{index}",
                    "summary": "S" * 300,
                    "article_ids": ["item-1"],
                    "key_claims": [],
                    "consensus_points": [],
                    "contested_points": [],
                    "known_unknowns": [],
                    "watch_signals": [],
                }
                for index in range(6)
            ],
            "global_watch_signals": [],
            "reader_qa": [],
        }

        generator.generate(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            [],
            "General brief.",
            "2026-05-30",
            evidence_packet=evidence_packet,
            delta_packet={},
        )

        prompt = str(captured.get("prompt", ""))
        self.assertNotIn('"cluster_id":"c5"', prompt)
        self.assertTrue(any("compacted analysis context" in item for item in generator.warnings))


if __name__ == "__main__":
    unittest.main()
