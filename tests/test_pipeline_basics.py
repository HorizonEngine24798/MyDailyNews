from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
import types
import unittest
from unittest.mock import patch
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
from mydailynews.ai.base import AIJsonError, AITransportError, set_ai_artifact_root, write_ai_json_artifact, write_ai_text_artifact
import mydailynews.ai.factory as ai_factory_module
from mydailynews.ai.prompts import BRIEF_SYSTEM, BRIEF_USER, HEADLINE_ANALYSIS_SYSTEM, HEADLINE_ANALYSIS_USER
from mydailynews.ai.schemas import FINAL_BRIEF_JSON_SCHEMA, HEADLINE_ANALYSIS_JSON_SCHEMA
from mydailynews.analysis_pipeline import DeltaExtractor, EvidenceDistiller
from mydailynews.article_pipeline import populate_article_texts
from mydailynews.brief import BriefGenerator
import mydailynews.brief_execution as brief_execution_module
from mydailynews.debug import DebugLogger
from mydailynews.config import _worker_count, load_config
from mydailynews.headline_selection import (
    candidate_heuristic_score,
    decisions_for_brief,
    limit_candidates_for_ai,
    select_articles,
    selection_reason_counters,
)
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
from mydailynews.output import render_markdown
import mydailynews.retrieval.article as article_module
from mydailynews.retrieval.article import ArticleRetriever
from mydailynews.retrieval.google_news import GoogleNewsQueryRetriever
from mydailynews.scrapers.rss import RSSScraper
from mydailynews.utils import safe_json_load, utc_now


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


def _base_config_dict() -> dict[str, Any]:
    return {
        "output_dir": "output",
        "user_agent": "MyDailyNews/test",
        "ai_summary": {
            "backend": "auto",
            "preset": "qwen3-1.7b",
            "max_input_tokens": 4096,
            "max_new_tokens": 512,
        },
        "ai_final": {
            "backend": "auto",
            "preset": "qwen3-8b",
            "max_input_tokens": 8192,
            "max_new_tokens": 1024,
        },
        "user_memory": {
            "avoided_topics": [],
            "preferred_sources": [],
            "avoided_sources": [],
            "briefing_style": "Concise.",
            "custom_instructions": "",
        },
        "general_topics": [
            {
                "name": "General",
                "description": "General news",
                "queries": ["general news"],
                "enabled": True,
            }
        ],
        "general_filtering": {
            "time_window_hours": 36,
            "headline_score_cutoff": 5.5,
            "max_headlines_per_source": 16,
            "max_candidates_for_ai": 60,
            "max_headlines_per_ai_batch": 32,
            "max_selected_articles": 12,
            "fill_selected_articles": True,
            "article_text_max_chars": 5000,
        },
        "topics_to_examine": [
            {
                "name": "Detailed",
                "description": "Detailed news",
                "queries": ["detailed news"],
                "enabled": True,
            }
        ],
        "filtering": {
            "time_window_hours": 36,
            "headline_score_cutoff": 6.8,
            "max_headlines_per_source": 16,
            "max_candidates_for_ai": 40,
            "max_headlines_per_ai_batch": 32,
            "max_selected_articles": 8,
            "fill_selected_articles": False,
            "article_text_max_chars": 6000,
        },
        "enrichment": {
            "enabled": True,
            "past_news_days": 30,
            "max_past_news_results": 4,
            "max_wikipedia_results": 3,
            "max_entities": 4,
            "max_context_chars_per_article": 1600,
        },
        "sources": {
            "rss": [
                {
                    "name": "Example",
                    "url": "https://example.com/feed.xml",
                    "category": "test",
                    "tags": [],
                    "enabled": True,
                }
            ],
            "google_news": {"enabled": False},
            "prior_reports": {"enabled": False},
        },
        "analysis": {
            "evidence_distillation": {},
            "delta_extraction": {},
            "rollout": {},
        },
    }


class PipelineBasicsTests(unittest.TestCase):
    def test_safe_json_load_accepts_model_newlines_inside_strings(self) -> None:
        payload = '{\n  "lead": "first line\nsecond line",\n  "sections": []\n}'

        parsed = safe_json_load(payload)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["lead"], "first line\nsecond line")

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

    def test_article_retriever_resolves_google_news_rss_url_before_extraction(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiabc123?oc=5"
        publisher_url = "https://publisher.example/news/full-story"
        wrapper_html = '<div data-n-a-ts="12345" data-n-a-sg="sig123"></div>'
        article_html = "<html><article>full story</article></html>"
        calls: list[str] = []

        class _HTTPResponse:
            def __init__(self, text: str) -> None:
                self.ok = True
                self.status_code = 200
                self.text = text
                self.headers = {}
                self.cache_state = "network"

        class _FakeHttp:
            def get_text(self, url: str, **_kwargs):
                calls.append(url)
                if url == google_url:
                    return _HTTPResponse(wrapper_html)
                if url == publisher_url:
                    return _HTTPResponse(article_html)
                raise AssertionError(f"unexpected URL: {url}")

        class _PostResponse:
            status_code = 200
            text = (
                ")]}'\n\n"
                + json.dumps(
                    [
                        [
                            "wrb.fr",
                            "Fbv4je",
                            json.dumps(["garturlres", publisher_url, 1]),
                            None,
                            None,
                            None,
                            "generic",
                        ]
                    ]
                )
            )

        def fake_post(url: str, **kwargs):
            self.assertEqual(url, "https://news.google.com/_/DotsSplashUi/data/batchexecute")
            self.assertIn("f.req", kwargs.get("data", {}))
            return _PostResponse()

        original_post = article_module.requests.post
        original_extract = article_module.trafilatura.extract
        try:
            article_module.requests.post = fake_post
            article_module.trafilatura.extract = lambda *_args, **_kwargs: "full article text " * 40
            retriever = ArticleRetriever("test-agent", max_chars=5000, debug=DebugLogger(False))
            retriever.http = _FakeHttp()  # type: ignore[assignment]

            text, status, effective_url = retriever.fetch_text_with_url(google_url)
        finally:
            article_module.requests.post = original_post
            article_module.trafilatura.extract = original_extract

        self.assertEqual(status, "ok")
        self.assertEqual(effective_url, publisher_url)
        self.assertEqual(calls, [google_url, publisher_url])
        self.assertIn("full article text", text)

    def test_populate_article_texts_updates_resolved_candidate_url(self) -> None:
        candidate = NewsCandidate(
            id="candidate-1",
            source="Google News",
            category="topic_search",
            title="headline",
            url="https://news.google.com/rss/articles/CBMiabc123?oc=5",
            snippet="snippet",
            published_at=utc_now(),
            tags=[],
            metadata={},
        )
        article = SelectedArticle(candidate=candidate, decision=HeadlineDecision(candidate_id=candidate.id, score=8.0))
        resolved_url = "https://publisher.example/news/full-story"

        class _FakeRetriever:
            def fetch_text_with_url(self, _url: str):
                return "resolved article text", "ok", resolved_url

        populate_article_texts(
            brief_name="unit",
            selected=[article],
            article_retriever=_FakeRetriever(),
            warnings=[],
            max_article_workers=1,
            debug=DebugLogger(False),
        )

        self.assertEqual(article.article_text, "resolved article text")
        self.assertEqual(article.extraction_status, "ok")
        self.assertEqual(article.candidate.url, resolved_url)
        self.assertEqual(article.candidate.metadata["original_url"], "https://news.google.com/rss/articles/CBMiabc123?oc=5")
        self.assertEqual(article.candidate.metadata["resolved_url"], resolved_url)

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
        base = _base_config_dict()
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
        base = _base_config_dict()
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
        base = _base_config_dict()
        base["analysis"] = {
            "evidence_distillation": {
                "enabled": True,
                "model_role": "final",
                "include_reader_qa": False,
                "max_input_tokens": 1800,
                "max_new_tokens": 420,
                "max_articles": 5,
                "max_articles_per_batch": 3,
                "max_articles_dropped_to_avoid_split": 2,
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
                "max_articles": 6,
                "max_articles_per_batch": 3,
                "max_articles_dropped_to_avoid_split": 2,
                "max_article_chars": 360,
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
        self.assertEqual(loaded.analysis.evidence_distillation.max_articles_per_batch, 3)
        self.assertEqual(loaded.analysis.evidence_distillation.max_articles_dropped_to_avoid_split, 2)
        self.assertEqual(loaded.analysis.evidence_distillation.max_questions, 4)

        self.assertTrue(loaded.analysis.delta_extraction.enabled)
        self.assertEqual(loaded.analysis.delta_extraction.model_role, "summary")
        self.assertEqual(loaded.analysis.delta_extraction.input_source, "evidence_only")
        self.assertTrue(loaded.analysis.delta_extraction.require_prior_reports)
        self.assertEqual(loaded.analysis.delta_extraction.max_articles, 6)
        self.assertEqual(loaded.analysis.delta_extraction.max_articles_per_batch, 3)
        self.assertEqual(loaded.analysis.delta_extraction.max_articles_dropped_to_avoid_split, 2)
        self.assertEqual(loaded.analysis.delta_extraction.max_article_chars, 360)
        self.assertEqual(loaded.analysis.delta_extraction.max_prior_reports, 2)

    def test_analysis_rollout_section_is_loaded_when_present(self) -> None:
        base = _base_config_dict()
        base["analysis"]["rollout"] = {
            "enabled": True,
            "profile": "balanced_local",
            "general": {
                "evidence_enabled": False,
                "delta_enabled": False,
            },
            "detailed": {
                "evidence_enabled": True,
                "delta_enabled": True,
                "evidence_max_input_tokens": 1800,
                "evidence_max_new_tokens": 420,
                "evidence_max_articles": 5,
                "evidence_max_articles_per_batch": 3,
                "evidence_max_articles_dropped_to_avoid_split": 2,
                "evidence_max_article_chars": 460,
                "delta_max_input_tokens": 1400,
                "delta_max_new_tokens": 260,
                "delta_max_articles": 6,
                "delta_max_articles_per_batch": 3,
                "delta_max_articles_dropped_to_avoid_split": 2,
                "delta_max_article_chars": 320,
                "delta_max_prior_reports": 2,
            },
        }

        tmp_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/config_analysis_rollout_present")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "with_analysis_rollout.json"
        config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

        loaded = load_config(config_path)
        self.assertTrue(loaded.analysis.rollout.enabled)
        self.assertEqual(loaded.analysis.rollout.profile, "balanced_local")
        self.assertTrue(bool(loaded.analysis.rollout.detailed.evidence_enabled))
        self.assertTrue(bool(loaded.analysis.rollout.detailed.delta_enabled))
        self.assertEqual(int(loaded.analysis.rollout.detailed.evidence_max_articles or 0), 5)
        self.assertEqual(int(loaded.analysis.rollout.detailed.evidence_max_articles_per_batch or 0), 3)
        self.assertEqual(int(loaded.analysis.rollout.detailed.evidence_max_articles_dropped_to_avoid_split or 0), 2)
        self.assertEqual(int(loaded.analysis.rollout.detailed.delta_max_articles or 0), 6)
        self.assertEqual(int(loaded.analysis.rollout.detailed.delta_max_articles_per_batch or 0), 3)
        self.assertEqual(int(loaded.analysis.rollout.detailed.delta_max_articles_dropped_to_avoid_split or 0), 2)
        self.assertEqual(int(loaded.analysis.rollout.detailed.delta_max_article_chars or 0), 320)
        self.assertEqual(int(loaded.analysis.rollout.detailed.delta_max_prior_reports or 0), 2)

    def test_analysis_rollout_profile_drives_mode_specific_stage_enablement(self) -> None:
        analysis = types.SimpleNamespace(
            evidence_distillation=EvidenceDistillationConfig(
                enabled=False,
                max_input_tokens=2300,
                max_new_tokens=700,
                max_articles=8,
                max_articles_per_batch=4,
                max_articles_dropped_to_avoid_split=2,
                max_article_chars=700,
            ),
            delta_extraction=DeltaExtractionConfig(
                enabled=False,
                max_input_tokens=1700,
                max_new_tokens=380,
                max_articles=6,
                max_articles_per_batch=3,
                max_articles_dropped_to_avoid_split=2,
                max_article_chars=360,
                max_prior_reports=3,
            ),
            rollout=types.SimpleNamespace(
                enabled=True,
                profile="safe_local",
                general=types.SimpleNamespace(),
                detailed=types.SimpleNamespace(),
            ),
        )

        evidence_general, delta_general, meta_general = brief_execution_module.resolve_analysis_stage_configs(analysis, "general")
        evidence_detailed, delta_detailed, meta_detailed = brief_execution_module.resolve_analysis_stage_configs(analysis, "detailed")

        self.assertFalse(evidence_general.enabled)
        self.assertFalse(delta_general.enabled)
        self.assertEqual(meta_general["rollout_profile"], "safe_local")

        self.assertTrue(evidence_detailed.enabled)
        self.assertFalse(delta_detailed.enabled)
        self.assertEqual(evidence_detailed.max_input_tokens, 2300)
        self.assertEqual(evidence_detailed.max_new_tokens, 700)
        self.assertEqual(evidence_detailed.max_articles, 8)
        self.assertEqual(evidence_detailed.max_articles_per_batch, 4)
        self.assertEqual(evidence_detailed.max_articles_dropped_to_avoid_split, 2)
        self.assertEqual(evidence_detailed.max_article_chars, 700)
        self.assertEqual(meta_detailed["rollout_profile"], "safe_local")

    def test_filtering_diversity_settings_are_loaded(self) -> None:
        base = _base_config_dict()
        base["filtering"]["max_selected_per_source"] = 1
        base["filtering"]["max_selected_per_event_cluster"] = 1
        base["filtering"]["prefer_multi_source_clusters"] = False
        base["filtering"]["multi_source_cluster_bonus"] = 0.8
        base["filtering"]["event_cluster_time_window_hours"] = 10
        base["filtering"]["use_multifactor_composite_ranking"] = True
        base["filtering"]["min_novelty_for_selection"] = 3.0
        base["filtering"]["source_preference_bonus"] = 0.5
        base["filtering"]["source_avoid_penalty"] = 1.6
        base["filtering"]["headline_max_input_tokens"] = 12345
        base["filtering"]["headline_max_new_tokens"] = 2345
        base["filtering"]["headline_single_replay_max_new_tokens"] = 678
        base["general_filtering"]["max_selected_per_source"] = 4
        base["general_filtering"]["max_selected_per_event_cluster"] = 3
        base["general_filtering"]["headline_max_input_tokens"] = 23456

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
        self.assertTrue(loaded.filtering.use_multifactor_composite_ranking)
        self.assertAlmostEqual(loaded.filtering.min_novelty_for_selection, 3.0)
        self.assertAlmostEqual(loaded.filtering.source_preference_bonus, 0.5)
        self.assertAlmostEqual(loaded.filtering.source_avoid_penalty, 1.6)
        self.assertEqual(loaded.filtering.headline_max_input_tokens, 12345)
        self.assertEqual(loaded.filtering.headline_max_new_tokens, 2345)
        self.assertEqual(loaded.filtering.headline_single_replay_max_new_tokens, 678)
        self.assertEqual(loaded.general_filtering.max_selected_per_source, 4)
        self.assertEqual(loaded.general_filtering.max_selected_per_event_cluster, 3)
        self.assertEqual(loaded.general_filtering.headline_max_input_tokens, 23456)

    def test_filtering_all_limits_are_loaded_as_unbounded(self) -> None:
        base = _base_config_dict()
        base["general_filtering"]["max_candidates_for_ai"] = "all"
        base["general_filtering"]["max_selected_articles"] = "all"
        base["filtering"]["max_candidates_for_ai"] = None
        base["filtering"]["max_selected_articles"] = "unlimited"

        tmp_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/config_filtering_all")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "with_filtering_all.json"
        config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

        loaded = load_config(config_path)
        self.assertIsNone(loaded.general_filtering.max_candidates_for_ai)
        self.assertIsNone(loaded.general_filtering.max_selected_articles)
        self.assertIsNone(loaded.filtering.max_candidates_for_ai)
        self.assertIsNone(loaded.filtering.max_selected_articles)

    def test_user_memory_v2_fields_are_loaded_and_prompted(self) -> None:
        base = _base_config_dict()
        base["user_memory"].update(
            {
                "role": "Policy analyst",
                "geography_focus": ["United States", "European Union"],
                "time_horizon": "strategic",
                "beats": {
                    "AI policy": 1.0,
                    "Semiconductor supply chain": 0.7,
                },
                "wants": ["policy change", "regulatory enforcement"],
                "avoid": ["celebrity gossip", "live sports scores"],
                "portfolio_or_stake_notes": "Direct enterprise exposure to AI compute costs.",
                "preferred_depth": "deep",
            }
        )

        tmp_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/config_user_memory_v2")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "with_user_memory_v2.json"
        config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

        loaded = load_config(config_path)
        self.assertEqual(loaded.user_memory.role, "Policy analyst")
        self.assertEqual(loaded.user_memory.geography_focus, ["United States", "European Union"])
        self.assertEqual(loaded.user_memory.time_horizon, "strategic")
        self.assertAlmostEqual(float(loaded.user_memory.beats.get("AI policy", 0.0)), 1.0, places=4)
        self.assertAlmostEqual(float(loaded.user_memory.beats.get("Semiconductor supply chain", 0.0)), 0.7, places=4)
        self.assertEqual(loaded.user_memory.wants, ["policy change", "regulatory enforcement"])
        self.assertEqual(loaded.user_memory.avoid, ["celebrity gossip", "live sports scores"])
        self.assertEqual(loaded.user_memory.preferred_depth, "deep")

        prompt = loaded.user_memory.to_prompt()
        self.assertIn("Role: Policy analyst", prompt)
        self.assertIn("geography focus: united states, european union", prompt.lower())
        self.assertIn("Time horizon: strategic", prompt)
        self.assertIn("Preferred depth: deep", prompt)
        self.assertIn("Priority beats: AI policy(1.00), Semiconductor supply chain(0.70)", prompt)
        self.assertIn("Wants: policy change, regulatory enforcement", prompt)
        self.assertIn("Avoid classes: celebrity gossip, live sports scores", prompt)

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

    def test_unbounded_candidate_limit_keeps_all_nonnegative_candidates(self) -> None:
        now = utc_now()
        since = now - timedelta(hours=36)
        titles = [
            "Federal agencies publish new AI procurement rules",
            "Chip suppliers expand advanced packaging capacity",
            "Central banks coordinate liquidity guidance",
            "Cybersecurity officials warn about router exploits",
            "Energy regulators approve transmission upgrade plan",
            "Space agency delays lunar cargo mission",
            "Health researchers report vaccine manufacturing shift",
        ]
        candidates = [
            NewsCandidate(
                id=f"item-{index}",
                source=f"Source {index}",
                category="test",
                title=titles[index],
                url=f"https://example.com/{index}",
                snippet="A substantial snippet with enough detail to avoid heuristic penalties.",
                published_at=now - timedelta(minutes=index),
            )
            for index in range(len(titles))
        ]
        filtering = FilteringConfig(max_candidates_for_ai=None)

        limited = limit_candidates_for_ai(
            candidates,
            [TopicConfig(name="AI infrastructure", description="AI infrastructure policy")],
            filtering,
            since,
            user_memory=UserMemory(),
            debug=DebugLogger(False),
        )

        self.assertEqual(len(limited), len(candidates))

    def test_unbounded_selection_uses_cutoff_without_capacity_cap(self) -> None:
        now = utc_now()
        candidates = [
            NewsCandidate(
                id=f"item-{index}",
                source=f"Source {index}",
                category="test",
                title=f"Headline {index}",
                url=f"https://example.com/{index}",
                snippet="Snippet",
                published_at=now - timedelta(minutes=index),
            )
            for index in range(5)
        ]
        decisions = {
            "item-0": HeadlineDecision(candidate_id="item-0", score=8.0, topic="World"),
            "item-1": HeadlineDecision(candidate_id="item-1", score=7.2, topic="World"),
            "item-2": HeadlineDecision(candidate_id="item-2", score=6.9, topic="World"),
            "item-3": HeadlineDecision(candidate_id="item-3", score=4.0, topic="World"),
            "item-4": HeadlineDecision(candidate_id="item-4", score=3.5, topic="World"),
        }
        filtering = FilteringConfig(
            headline_score_cutoff=6.8,
            max_selected_articles=None,
            max_selected_per_source=0,
            max_selected_per_event_cluster=0,
        )

        selected = select_articles(candidates, decisions, [TopicConfig(name="World")], filtering)

        self.assertEqual([item.candidate.id for item in selected], ["item-0", "item-1", "item-2"])
        self.assertEqual(decisions["item-3"].selection_reason_code, "skipped_below_cutoff")
        self.assertNotIn("skipped_capacity", {decision.selection_reason_code for decision in decisions.values()})

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

    def test_candidate_heuristic_score_uses_user_memory_v2_signals(self) -> None:
        now = utc_now()
        since = now - timedelta(hours=24)
        candidate = NewsCandidate(
            id="policy-us",
            source="Example Source",
            category="test",
            title="US Senate advances new AI regulation framework",
            url="https://example.com/policy-us",
            snippet="Regulatory enforcement actions and policy change are expected this quarter.",
            published_at=now,
        )
        topics = [TopicConfig(name="AI policy", queries=["AI regulation", "policy change"])]

        neutral_score = candidate_heuristic_score(
            candidate,
            topics,
            since,
            user_memory=UserMemory(),
        )
        tuned_score = candidate_heuristic_score(
            candidate,
            topics,
            since,
            user_memory=UserMemory(
                geography_focus=["United States"],
                wants=["policy change", "regulatory enforcement"],
                beats={"AI policy": 1.0},
            ),
        )
        penalized_score = candidate_heuristic_score(
            candidate,
            topics,
            since,
            user_memory=UserMemory(
                avoid=["ai regulation"],
                avoided_topics=["policy"],
            ),
        )

        self.assertGreater(tuned_score, neutral_score)
        self.assertLess(penalized_score, neutral_score)

    def test_select_articles_applies_user_memory_v2_rank_adjustments(self) -> None:
        now = utc_now()
        geo_candidate = NewsCandidate(
            id="geo",
            source="Source A",
            category="test",
            title="United States Senate advances AI policy change",
            url="https://example.com/geo",
            snippet="Regulatory enforcement and policy change timeline accelerates.",
            published_at=now,
            metadata={"event_cluster_id": "evt-1"},
        )
        avoid_candidate = NewsCandidate(
            id="avoid",
            source="Source B",
            category="test",
            title="Celebrity gossip recap and sports scores roundup",
            url="https://example.com/avoid",
            snippet="Entertainment chatter with no policy impact.",
            published_at=now,
            metadata={"event_cluster_id": "evt-2"},
        )
        decisions = {
            "geo": HeadlineDecision(candidate_id="geo", score=7.5, topic="World"),
            "avoid": HeadlineDecision(candidate_id="avoid", score=7.6, topic="World"),
        }
        filtering = FilteringConfig(
            headline_score_cutoff=0.0,
            max_selected_articles=1,
            fill_selected_articles=False,
            prefer_multi_source_clusters=False,
            use_multifactor_composite_ranking=False,
        )
        memory = UserMemory(
            geography_focus=["United States"],
            wants=["policy change", "regulatory enforcement"],
            avoid=["celebrity gossip", "sports scores"],
            beats={"AI policy": 1.0},
        )
        selected = select_articles(
            [geo_candidate, avoid_candidate],
            decisions,
            [TopicConfig(name="World")],
            filtering,
            user_memory=memory,
        )

        self.assertEqual(selected[0].candidate.id, "geo")
        self.assertGreater(decisions["geo"].selection_rank_score, decisions["avoid"].selection_rank_score)

    def test_select_articles_composite_ranking_can_override_scalar_score(self) -> None:
        now = utc_now()
        score_led = NewsCandidate(
            id="score-led",
            source="Source A",
            category="test",
            title="Score-led item",
            url="https://example.com/score-led",
            snippet="snippet",
            published_at=now,
            metadata={"event_cluster_id": "evt-1"},
        )
        composite_led = NewsCandidate(
            id="composite-led",
            source="Source B",
            category="test",
            title="Composite-led item",
            url="https://example.com/composite-led",
            snippet="snippet",
            published_at=now,
            metadata={"event_cluster_id": "evt-2"},
        )
        decisions = {
            "score-led": HeadlineDecision(
                candidate_id="score-led",
                score=9.0,
                topic="World",
                personal_relevance=2.0,
                impact=2.0,
                novelty=2.0,
                actionability=2.0,
                urgency=2.0,
                confidence=2.0,
                reason="Low-value despite high scalar score.",
            ),
            "composite-led": HeadlineDecision(
                candidate_id="composite-led",
                score=7.0,
                topic="World",
                personal_relevance=9.0,
                impact=9.0,
                novelty=8.5,
                actionability=8.0,
                urgency=7.0,
                confidence=8.0,
                reason="High multifactor relevance and impact.",
                angle_type="policy_change",
            ),
        }

        composite_filtering = FilteringConfig(
            headline_score_cutoff=0.0,
            max_selected_articles=1,
            fill_selected_articles=False,
            prefer_multi_source_clusters=False,
            use_multifactor_composite_ranking=True,
        )
        selected_composite = select_articles(
            [score_led, composite_led],
            decisions,
            [TopicConfig(name="World")],
            composite_filtering,
        )
        self.assertEqual(selected_composite[0].candidate.id, "composite-led")
        self.assertEqual(selected_composite[0].selection_reason_code, "selected_high_composite")

        scalar_filtering = FilteringConfig(
            headline_score_cutoff=0.0,
            max_selected_articles=1,
            fill_selected_articles=False,
            prefer_multi_source_clusters=False,
            use_multifactor_composite_ranking=False,
        )
        selected_scalar = select_articles(
            [score_led, composite_led],
            decisions,
            [TopicConfig(name="World")],
            scalar_filtering,
        )
        self.assertEqual(selected_scalar[0].candidate.id, "score-led")
        self.assertEqual(selected_scalar[0].selection_reason_code, "selected_high_score")

    def test_select_articles_records_reason_codes_and_counters(self) -> None:
        now = utc_now()
        rows: list[tuple[NewsCandidate, HeadlineDecision]] = []

        def _row(
            item_id: str,
            source: str,
            cluster_id: str,
            score: float,
            *,
            novelty: float = 6.0,
            impact: float = 7.0,
        ) -> None:
            candidate = NewsCandidate(
                id=item_id,
                source=source,
                category="test",
                title=f"Headline {item_id}",
                url=f"https://example.com/{item_id}",
                snippet="snippet",
                published_at=now,
                metadata={"event_cluster_id": cluster_id},
            )
            decision = HeadlineDecision(
                candidate_id=item_id,
                score=score,
                topic="World",
                personal_relevance=8.0,
                impact=impact,
                novelty=novelty,
                actionability=7.0,
                urgency=6.0,
                confidence=7.0,
                reason=f"Reason for {item_id}",
            )
            rows.append((candidate, decision))

        _row("a1", "Source A", "evt-1", 9.2)
        _row("a2", "Source A", "evt-2", 9.0)
        _row("b1", "Source B", "evt-1", 8.8)
        _row("c1", "Source C", "evt-3", 8.4)
        _row("low", "Source D", "evt-4", 6.5, novelty=1.0, impact=4.0)
        _row("cut", "Source E", "evt-5", 4.0)

        candidates = [item[0] for item in rows]
        decisions = {item[1].candidate_id: item[1] for item in rows}
        filtering = FilteringConfig(
            headline_score_cutoff=5.0,
            max_selected_articles=2,
            fill_selected_articles=False,
            max_selected_per_source=1,
            max_selected_per_event_cluster=1,
            prefer_multi_source_clusters=False,
            use_multifactor_composite_ranking=True,
            min_novelty_for_selection=2.5,
        )
        selected = select_articles(
            candidates,
            decisions,
            [TopicConfig(name="World")],
            filtering,
        )
        selected_ids = [item.candidate.id for item in selected]
        self.assertEqual(selected_ids, ["a1", "c1"])
        self.assertTrue(all(item.selection_reason_code.startswith("selected_") for item in selected))

        self.assertEqual(decisions["a2"].selection_reason_code, "skipped_source_cap")
        self.assertEqual(decisions["b1"].selection_reason_code, "skipped_cluster_cap")
        self.assertEqual(decisions["low"].selection_reason_code, "skipped_low_novelty")
        self.assertEqual(decisions["cut"].selection_reason_code, "skipped_below_cutoff")

        reason_counts = selection_reason_counters(decisions)
        self.assertEqual(reason_counts["selected"].get("selected_high_composite"), 2)
        self.assertEqual(reason_counts["skipped"].get("skipped_source_cap"), 1)
        self.assertEqual(reason_counts["skipped"].get("skipped_cluster_cap"), 1)
        self.assertEqual(reason_counts["skipped"].get("skipped_low_novelty"), 1)
        self.assertEqual(reason_counts["skipped"].get("skipped_below_cutoff"), 1)

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
                max_articles_dropped_to_avoid_split=0,
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
        self.assertTrue(any("split selected articles" in item for item in distiller.warnings))

    def test_evidence_distiller_batches_articles_and_merges_results(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 8000
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "overview": f"Overview batch {self.calls}.",
                    "story_clusters": [
                        {
                            "cluster_id": f"cluster-{self.calls}",
                            "topic": "World",
                            "label": f"Label {self.calls}",
                            "summary": "Summary.",
                            "article_ids": [f"item-{self.calls - 1}"],
                            "key_claims": [],
                            "consensus_points": [],
                            "contested_points": [],
                            "known_unknowns": [],
                            "watch_signals": [],
                        }
                    ],
                    "global_watch_signals": [f"Watch {self.calls}"],
                    "reader_qa": [
                        {
                            "question": f"Question {self.calls}?",
                            "answer": "Answer.",
                            "article_ids": [f"item-{self.calls - 1}"],
                        }
                    ],
                }

        selected = []
        for index in range(5):
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
                    decision=HeadlineDecision(candidate_id=candidate.id, score=10 - index, topic="World"),
                    article_text="Article text.",
                    extraction_status="ok",
                )
            )

        client = _DummyClient()
        distiller = EvidenceDistiller(
            client,
            EvidenceDistillationConfig(
                enabled=True,
                max_articles=5,
                max_articles_per_batch=2,
                max_articles_dropped_to_avoid_split=0,
                max_article_chars=500,
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
        )

        self.assertEqual(client.calls, 3)
        self.assertIn("Overview batch 1", result["overview"])
        self.assertEqual(len(result["story_clusters"]), 3)
        self.assertEqual(len(result["global_watch_signals"]), 3)
        self.assertEqual(len(result["reader_qa"]), 3)

    def test_evidence_distiller_drops_small_tail_to_avoid_split(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 8000
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "overview": "Overview.",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        selected = []
        for index in range(5):
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
                    decision=HeadlineDecision(candidate_id=candidate.id, score=10 - index, topic="World"),
                    article_text="Article text.",
                    extraction_status="ok",
                )
            )

        client = _DummyClient()
        distiller = EvidenceDistiller(
            client,
            EvidenceDistillationConfig(
                enabled=True,
                max_articles=5,
                max_articles_per_batch=4,
                max_articles_dropped_to_avoid_split=1,
                max_article_chars=500,
            ),
            debug=DebugLogger(False),
        )
        distiller.distill(selected, UserMemory(), [TopicConfig(name="World")], [], "General brief.", "2026-05-30")

        self.assertEqual(client.calls, 1)
        self.assertTrue(any("instead of splitting" in item for item in distiller.warnings))
        self.assertFalse(any("split selected articles" in item for item in distiller.warnings))

    def test_evidence_distiller_batches_by_event_cluster_with_headline_awareness(self) -> None:
        captured_prompts: list[str] = []

        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 8000
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, _system, user, **_kwargs):
                self.calls += 1
                captured_prompts.append(user)
                return {
                    "overview": f"Overview {self.calls}.",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        selected = []
        for index, cluster_id in enumerate(["evt-a", "evt-a", "evt-b", "evt-b"]):
            candidate = NewsCandidate(
                id=f"item-{index}",
                source="Example",
                category="test",
                title=f"Headline {index}",
                url=f"https://example.com/{index}",
                snippet="snippet",
                published_at=utc_now(),
                metadata={
                    "event_cluster_id": cluster_id,
                    "event_cluster_label": f"Cluster {cluster_id}",
                    "event_cluster_size": 2,
                    "event_cluster_source_count": 2,
                },
            )
            selected.append(
                SelectedArticle(
                    candidate=candidate,
                    decision=HeadlineDecision(candidate_id=candidate.id, score=10 - index, topic="World"),
                    article_text="Article text.",
                    extraction_status="ok",
                )
            )

        distiller = EvidenceDistiller(
            _DummyClient(),
            EvidenceDistillationConfig(
                enabled=True,
                max_articles=4,
                max_articles_per_batch=2,
                max_articles_dropped_to_avoid_split=0,
                max_article_chars=500,
            ),
            debug=DebugLogger(False),
        )
        distiller.distill(selected, UserMemory(), [TopicConfig(name="World")], [], "General brief.", "2026-05-30")

        self.assertEqual(len(captured_prompts), 2)
        self.assertIn('"id":"item-0"', captured_prompts[0])
        self.assertIn('"id":"item-1"', captured_prompts[0])
        self.assertIn("Headline-only awareness", captured_prompts[0])
        self.assertIn('"id":"item-2"', captured_prompts[0])
        self.assertIn('"id":"item-3"', captured_prompts[0])
        self.assertIn('"id":"item-2"', captured_prompts[1])
        self.assertIn('"id":"item-3"', captured_prompts[1])

    def test_evidence_distiller_records_prompt_pressure_compaction_counters(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 1200
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(_text: str) -> int:
                return 999999

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "overview": "Overview text.",
                    "story_clusters": [],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }

        debug = DebugLogger(True)
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
                    article_text=("A" * 2400) + f"#{index}",
                    extraction_status="ok",
                )
            )
        prior_reports = [
            PriorReport(id=f"r-{index}", date="2026-05-29", title=f"Prior {index}", path="p", summary="s")
            for index in range(3)
        ]

        distiller = EvidenceDistiller(
            _DummyClient(),
            EvidenceDistillationConfig(
                enabled=True,
                max_input_tokens=1000,
                max_new_tokens=320,
                max_articles=6,
                max_articles_dropped_to_avoid_split=0,
                max_article_chars=1200,
            ),
            debug=debug,
        )
        result = distiller.distill(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            prior_reports,
            "General brief.",
            "2026-05-30",
            brief_name="general",
        )

        self.assertIn("overview", result)
        self.assertTrue(any("split selected articles" in item for item in distiller.warnings))
        analytics = debug.analytics_payload()
        counts = analytics.get("counts", {})
        metrics = analytics.get("metrics", {})
        self.assertGreater(int(metrics.get("analysis.evidence.batches", 0)), 1)
        self.assertGreater(int(counts.get("analysis.evidence.prompt_pressure_checks", 0)), 0)
        self.assertGreater(int(counts.get("analysis.evidence.prompt_compaction.drop_prior_report", 0)), 0)

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

    def test_delta_extractor_batches_articles_and_merges_results(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 8000
                self.max_new_tokens = 512

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "baseline_coverage_note": f"coverage note {self.calls}",
                    "new": [
                        {
                            "item": f"new item {self.calls}",
                            "summary": "summary",
                            "article_ids": [f"item-{self.calls - 1}"],
                        }
                    ],
                    "escalated": [],
                    "weakened": [],
                    "reframed": [],
                    "unchanged_but_important": [],
                    "evidence_gaps": [
                        {
                            "gap": f"gap {self.calls}",
                            "why_it_matters": "matters",
                        }
                    ],
                }

        selected = []
        for index in range(5):
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
                    decision=HeadlineDecision(candidate_id=candidate.id, score=10 - index, topic="World"),
                    article_text="Article text.",
                    extraction_status="ok",
                )
            )

        client = _DummyClient()
        extractor = DeltaExtractor(
            client,
            DeltaExtractionConfig(
                enabled=True,
                input_source="articles_only",
                max_articles=5,
                max_articles_per_batch=2,
                max_articles_dropped_to_avoid_split=0,
                max_article_chars=300,
            ),
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

        self.assertEqual(client.calls, 3)
        self.assertIn("coverage note 1", result["baseline_coverage_note"])
        self.assertEqual(len(result["new"]), 3)
        self.assertEqual(len(result["evidence_gaps"]), 3)

    def test_delta_extractor_records_prompt_pressure_compaction_counters(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-summary",
                    response_format="json_object",
                )
                self.max_input_tokens = 1200
                self.max_new_tokens = 400

            @staticmethod
            def estimate_tokens(_text: str) -> int:
                return 999999

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

        debug = DebugLogger(True)
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
                    article_text=("A" * 2200) + f"#{index}",
                    extraction_status="ok",
                )
            )
        prior_reports = [
            PriorReport(id=f"r-{index}", date="2026-05-29", title=f"Prior {index}", path="p", summary="s")
            for index in range(3)
        ]
        evidence_packet = {
            "overview": "Overview",
            "story_clusters": [
                {
                    "cluster_id": "cluster-1",
                    "topic": "World",
                    "label": "Label",
                    "summary": "S" * 600,
                    "article_ids": ["item-0", "item-1"],
                    "watch_signals": ["Watch 1", "Watch 2"],
                }
            ],
            "global_watch_signals": ["Global 1", "Global 2"],
            "reader_qa": [{"question": "Q", "answer": "A", "article_ids": ["item-0"]}],
        }

        extractor = DeltaExtractor(
            _DummyClient(),
            DeltaExtractionConfig(
                enabled=True,
                input_source="evidence_or_articles",
                max_input_tokens=1000,
                max_new_tokens=320,
                max_articles_dropped_to_avoid_split=0,
                max_prior_reports=3,
            ),
            debug=debug,
        )
        result = extractor.extract(
            selected,
            UserMemory(),
            [TopicConfig(name="World")],
            prior_reports,
            "General brief.",
            "2026-05-30",
            evidence_packet=evidence_packet,
            brief_name="general",
        )

        self.assertIn("baseline_coverage_note", result)
        self.assertTrue(any("split selected articles" in item for item in extractor.warnings))
        analytics = debug.analytics_payload()
        counts = analytics.get("counts", {})
        metrics = analytics.get("metrics", {})
        self.assertGreater(int(metrics.get("analysis.delta.batches", 0)), 1)
        self.assertGreater(int(counts.get("analysis.delta.prompt_pressure_checks", 0)), 0)
        self.assertGreater(int(counts.get("analysis.delta.prompt_compaction.drop_prior_report", 0)), 0)

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

    def test_run_brief_rollout_safe_local_disables_optional_stages_for_general(self) -> None:
        class _DummyAIClient:
            def __init__(self, label: str) -> None:
                self.config = types.SimpleNamespace(backend="transformers", effective_model_label=label)
                self.unload_calls = 0

            def unload(self) -> None:
                self.unload_calls += 1

        candidate = NewsCandidate(
            id="item-rollout-general",
            source="Example",
            category="test",
            title="Central bank signals possible rate cuts amid cooling inflation",
            url="https://example.com/item-rollout-general",
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
        output_dir = Path("D:/Project/MyDailyNews/.codex_tmp_test/pr7_rollout_general_output")
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = types.SimpleNamespace()
        orchestrator.debug = DebugLogger(True)
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
                evidence_distillation=EvidenceDistillationConfig(enabled=True),
                delta_extraction=DeltaExtractionConfig(
                    enabled=True,
                    max_prior_reports=3,
                ),
                rollout=types.SimpleNamespace(
                    enabled=True,
                    profile="safe_local",
                    general=types.SimpleNamespace(),
                    detailed=types.SimpleNamespace(),
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
        original_evidence_distiller = brief_execution_module.EvidenceDistiller
        original_delta_extractor = brief_execution_module.DeltaExtractor
        captured = {"distill_calls": 0, "delta_calls": 0}

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

        class _FakeDeltaExtractor:
            def __init__(self, *_args, **_kwargs) -> None:
                self.warnings: list[str] = []

            def extract(self, *_args, **_kwargs):
                captured["delta_calls"] += 1
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

            def generate(self, *_args, **_kwargs):
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
        brief_execution_module.EvidenceDistiller = _FakeEvidenceDistiller
        brief_execution_module.DeltaExtractor = _FakeDeltaExtractor
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
            brief_execution_module.EvidenceDistiller = original_evidence_distiller
            brief_execution_module.DeltaExtractor = original_delta_extractor

        self.assertEqual(captured["distill_calls"], 0)
        self.assertEqual(captured["delta_calls"], 0)
        payload = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        rollout_meta = payload["metadata"]["analysis_rollout"]
        self.assertTrue(bool(rollout_meta.get("enabled")))
        self.assertEqual(rollout_meta.get("profile"), "safe_local")
        self.assertTrue(bool(rollout_meta.get("evidence_requested_enabled")))
        self.assertFalse(bool(rollout_meta.get("evidence_enabled")))
        self.assertTrue(bool(rollout_meta.get("delta_requested_enabled")))
        self.assertFalse(bool(rollout_meta.get("delta_enabled")))
        self.assertEqual(payload["analysis"]["delta_model_role"], "deterministic_scaffold")
        self.assertTrue(bool(payload["analysis"]["delta_packet"].get("deterministic_scaffold")))

        metrics = orchestrator.debug.analytics_payload().get("metrics", {})
        self.assertTrue(bool(metrics.get("brief.general.analysis.rollout.enabled")))
        self.assertEqual(metrics.get("brief.general.analysis.rollout.profile"), "safe_local")
        self.assertEqual(metrics.get("brief.general.analysis.evidence.skipped_reason.rollout_disabled"), 1)
        self.assertEqual(metrics.get("brief.general.analysis.delta.skipped_reason.rollout_disabled"), 1)
        self.assertEqual(metrics.get("brief.general.analysis.delta.scaffold_reason.disabled"), 1)

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

    def test_headline_prompt_rubric_refresh_contains_pr2_anchors(self) -> None:
        system_text = HEADLINE_ANALYSIS_SYSTEM.lower()
        user_text = HEADLINE_ANALYSIS_USER.lower()

        self.assertIn("editorial triage scorer", system_text)
        self.assertIn("regret", user_text)
        self.assertIn("personal relevance", user_text)
        self.assertIn("impact", user_text)
        self.assertIn("novelty", user_text)
        self.assertIn("actionability", user_text)
        self.assertIn("urgency", user_text)
        self.assertIn("topic keyword match with low impact", user_text)
        self.assertIn("high-value must-know", user_text)
        self.assertIn("low-value noise", user_text)

    def test_headline_prompt_contract_supports_multifactor_shape_and_compactness(self) -> None:
        normalized = " ".join((HEADLINE_ANALYSIS_USER + "\n" + HEADLINE_ANALYSIS_SYSTEM).split())
        self.assertIn('"decisions"', HEADLINE_ANALYSIS_USER)
        self.assertIn('"id"', HEADLINE_ANALYSIS_USER)
        self.assertIn('"score"', HEADLINE_ANALYSIS_USER)
        self.assertIn('"reason"', HEADLINE_ANALYSIS_USER)
        self.assertIn('"skip_reason"', HEADLINE_ANALYSIS_USER)
        self.assertIn('"angle_type"', HEADLINE_ANALYSIS_USER)
        self.assertLessEqual(len(normalized), 4400)

    def test_headline_analysis_schema_supports_multifactor_optional_fields(self) -> None:
        schema = HEADLINE_ANALYSIS_JSON_SCHEMA.schema
        decision_props = schema["properties"]["decisions"]["items"]["properties"]
        required = schema["properties"]["decisions"]["items"]["required"]

        self.assertIn("id", required)
        self.assertIn("score", required)
        self.assertIn("personal_relevance", decision_props)
        self.assertIn("impact", decision_props)
        self.assertIn("novelty", decision_props)
        self.assertIn("urgency", decision_props)
        self.assertIn("actionability", decision_props)
        self.assertIn("confidence", decision_props)
        self.assertIn("reason", decision_props)
        self.assertIn("skip_reason", decision_props)
        self.assertIn("angle_type", decision_props)

    def test_brief_prompt_structured_voice_contains_pr6_anchors(self) -> None:
        system_text = BRIEF_SYSTEM.lower()
        user_text = BRIEF_USER.lower()

        self.assertIn("structured briefing writer", system_text)
        self.assertIn("generic summarizer", system_text)
        self.assertIn("reject generic phrasing", user_text)
        self.assertIn("why_it_matters", user_text)
        self.assertIn("what_changed", user_text)
        self.assertIn("who_is_affected", user_text)
        self.assertIn("what_to_watch", user_text)
        self.assertIn("do not invent facts", system_text)

    def test_final_brief_schema_supports_structured_topic_fields(self) -> None:
        schema = FINAL_BRIEF_JSON_SCHEMA.schema
        topic_props = schema["properties"]["topic_reports"]["items"]["properties"]

        self.assertIn("topic", topic_props)
        self.assertIn("why_it_matters", topic_props)
        self.assertIn("what_changed", topic_props)
        self.assertIn("who_is_affected", topic_props)
        self.assertIn("narrative_changes", topic_props)
        self.assertIn("what_to_watch", topic_props)

    def test_headline_analyzer_parses_multifactor_fields_with_clamping(self) -> None:
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
                return {
                    "decisions": [
                        {
                            "id": "example:ai",
                            "score": 12.2,
                            "personal_relevance": -2,
                            "impact": 8.4,
                            "novelty": 11,
                            "urgency": -0.1,
                            "actionability": 7.1,
                            "confidence": 100,
                            "reason": "A" * 220,
                            "skip_reason": None,
                            "angle_type": "policy_change",
                        }
                    ]
                }

        debug = DebugLogger(True)
        analyzer = HeadlineAnalyzer(_DummyClient(), batch_size=4, debug=debug)
        topics = [TopicConfig(name="AI policy")]
        candidate = NewsCandidate(
            id="example:ai",
            source="Example",
            category="test",
            title="AI regulation proposal",
            url="https://example.com/ai",
            snippet="snippet",
            published_at=utc_now(),
        )

        decisions = analyzer.analyze([candidate], UserMemory(), topics, "Detailed AI policy brief.")
        decision = decisions["example:ai"]

        self.assertEqual(decision.score, 10.0)
        self.assertEqual(decision.personal_relevance, 0.0)
        self.assertEqual(decision.impact, 8.4)
        self.assertEqual(decision.novelty, 10.0)
        self.assertEqual(decision.urgency, 0.0)
        self.assertEqual(decision.actionability, 7.1)
        self.assertEqual(decision.confidence, 10.0)
        self.assertTrue(decision.reason)
        self.assertLessEqual(len(decision.reason), 180)
        self.assertIsNone(decision.skip_reason)
        self.assertEqual(decision.angle_type, "policy_change")

        analytics = debug.analytics_payload()
        self.assertEqual(analytics["metrics"]["headline.multifactor.decisions"], 1)
        self.assertEqual(analytics["metrics"]["headline.multifactor.present_ratio.personal_relevance"], 1.0)
        self.assertEqual(analytics["metrics"]["headline.multifactor.present_ratio.reason"], 1.0)

    def test_headline_analyzer_defaults_multifactor_for_legacy_id_score_payload(self) -> None:
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
                return {"decisions": [{"id": "example:legacy", "score": 7.8}]}

        analyzer = HeadlineAnalyzer(_DummyClient(), batch_size=4)
        candidate = NewsCandidate(
            id="example:legacy",
            source="Example",
            category="test",
            title="Legacy output shape",
            url="https://example.com/legacy",
            snippet="snippet",
            published_at=utc_now(),
        )
        decisions = analyzer.analyze([candidate], UserMemory(), [TopicConfig(name="World")], "General brief.")
        decision = decisions["example:legacy"]

        self.assertEqual(decision.score, 7.8)
        self.assertEqual(decision.personal_relevance, 5.0)
        self.assertEqual(decision.impact, 5.0)
        self.assertEqual(decision.novelty, 5.0)
        self.assertEqual(decision.urgency, 5.0)
        self.assertEqual(decision.actionability, 5.0)
        self.assertEqual(decision.confidence, 5.0)
        self.assertEqual(decision.reason, "")
        self.assertIsNone(decision.skip_reason)
        self.assertEqual(decision.angle_type, "")

    def test_headline_analyzer_uses_llama_sized_dynamic_budgets(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="llama_cpp_server",
                    effective_model_label="unit-gguf",
                    response_format="json_object",
                )
                self.max_input_tokens = 40960
                self.max_new_tokens = 8192

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

        analyzer = HeadlineAnalyzer(
            _DummyClient(),
            batch_size=32,
            input_token_limit=32000,
            max_new_tokens=6000,
            single_replay_max_new_tokens=1500,
        )

        self.assertEqual(analyzer._headline_input_token_limit(), 32000)
        self.assertEqual(analyzer._headline_batch_max_new_tokens(1), 6000)
        self.assertEqual(analyzer._headline_batch_max_new_tokens(16), 6000)
        self.assertEqual(analyzer._headline_batch_max_new_tokens(32), 6000)
        self.assertEqual(analyzer._headline_single_max_new_tokens(), 1500)

    def test_headline_analyzer_recovers_invalid_batch_with_single_item_replay(self) -> None:
        candidates = [
            NewsCandidate(
                id="example:a",
                source="Example",
                category="test",
                title="AI policy change",
                url="https://example.com/a",
                snippet="snippet",
                published_at=utc_now(),
            ),
            NewsCandidate(
                id="example:b",
                source="Example",
                category="test",
                title="Semiconductor export controls",
                url="https://example.com/b",
                snippet="snippet",
                published_at=utc_now(),
            ),
        ]

        class _RecoveringClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="llama_cpp_server",
                    effective_model_label="unit-gguf",
                    response_format="json_object",
                )
                self.max_input_tokens = 40960
                self.max_new_tokens = 8192
                self.labels: list[str] = []

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, _system, _user, label="ai.complete_json", **_kwargs):
                self.labels.append(label)
                if label.startswith("headline scoring batch"):
                    raise AIJsonError("simulated truncated batch JSON")
                candidate_id = label.rsplit("[", 1)[-1].rstrip("]")
                return {"decisions": [{"id": candidate_id, "score": 8.5, "reason": "Recovered replay."}]}

        client = _RecoveringClient()
        analyzer = HeadlineAnalyzer(client, batch_size=2, debug=DebugLogger(False))
        decisions = analyzer.analyze(candidates, UserMemory(), [TopicConfig(name="Policy")], "General brief.")

        self.assertEqual(set(decisions.keys()), {"example:a", "example:b"})
        self.assertEqual(decisions["example:a"].score, 8.5)
        self.assertEqual(len([label for label in client.labels if "single replay" in label]), 2)
        self.assertTrue(any("recovered 2/2" in warning for warning in analyzer.warnings))

    def test_decisions_for_brief_preserves_multifactor_fields(self) -> None:
        candidate = NewsCandidate(
            id="example:shared",
            source="Example",
            category="test",
            title="Shared candidate",
            url="https://example.com/shared",
            snippet="snippet",
            published_at=utc_now(),
            metadata={},
        )
        shared = HeadlineDecision(
            candidate_id=candidate.id,
            score=8.0,
            topic="",
            personal_relevance=9.0,
            impact=8.0,
            novelty=7.0,
            urgency=6.0,
            actionability=5.0,
            confidence=8.5,
            reason="High-impact policy change.",
            skip_reason=None,
            angle_type="policy_change",
        )
        scoped = decisions_for_brief(
            [candidate],
            {candidate.id: shared},
            [TopicConfig(name="Policy", queries=["policy"])],
        )
        decision = scoped[candidate.id]
        self.assertEqual(decision.personal_relevance, 9.0)
        self.assertEqual(decision.impact, 8.0)
        self.assertEqual(decision.novelty, 7.0)
        self.assertEqual(decision.urgency, 6.0)
        self.assertEqual(decision.actionability, 5.0)
        self.assertEqual(decision.confidence, 8.5)
        self.assertEqual(decision.reason, "High-impact policy change.")
        self.assertEqual(decision.angle_type, "policy_change")

    def test_headline_analyzer_cache_fingerprint_version_for_multifactor_schema(self) -> None:
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
                return {"decisions": []}

        analyzer = HeadlineAnalyzer(_DummyClient(), batch_size=2)
        payload = [{"id": "c1", "title": "Title"}]
        with patch("mydailynews.ai.headline_analyzer.JSONCache.make_key", side_effect=lambda raw: raw):
            raw_key = analyzer._batch_cache_key(payload, UserMemory(), [TopicConfig(name="World")], "General brief.")
        fingerprint = json.loads(raw_key)
        self.assertEqual(fingerprint["v"], 8)

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
            input_token_limit=1600,
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

    def test_brief_generator_drops_lowest_scored_articles_until_prompt_fits(self) -> None:
        class _BudgetClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="llama_cpp_server",
                    effective_model_label="dummy-final",
                    response_format="json_object",
                )
                self.max_input_tokens = 1200
                self.max_new_tokens = 256
                self.final_prompt_tokens = 0

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, _system, user, **_kwargs):
                self.final_prompt_tokens = self.estimate_tokens(user)
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead summary.",
                    "topic_reports": [],
                    "sections": [],
                }

        client = _BudgetClient()
        generator = BriefGenerator(
            client,
            max_context_chars=1200,
            input_token_limit=1200,
            max_new_tokens=256,
        )
        selected = []
        for index in range(10):
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
                    decision=HeadlineDecision(candidate_id=candidate.id, score=10.0 - index, topic="World"),
                    article_text=("Long article text. " * 180),
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

        used_scores = [item["score"] for item in brief["selected_articles"]]
        self.assertLess(len(used_scores), len(selected))
        self.assertEqual(used_scores, sorted(used_scores, reverse=True))
        self.assertGreaterEqual(min(used_scores), 10.0 - len(used_scores) + 1)
        self.assertTrue(any("effective final score floor" in warning for warning in generator.warnings))

    def test_selection_budget_prune_marks_lowest_scored_articles(self) -> None:
        class _BudgetClient:
            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

        config = types.SimpleNamespace(
            ai_final=types.SimpleNamespace(max_input_tokens=1800, max_new_tokens=256),
            enrichment=types.SimpleNamespace(max_context_chars_per_article=900),
            user_memory=UserMemory(),
        )
        orchestrator = types.SimpleNamespace(
            config=config,
            final_ai_client=_BudgetClient(),
            debug=DebugLogger(False),
        )
        selected = []
        for index in range(8):
            candidate = NewsCandidate(
                id=f"item-{index}",
                source="Example",
                category="test",
                title=f"Headline {index}",
                url=f"https://example.com/{index}",
                snippet="snippet",
                published_at=utc_now(),
            )
            decision = HeadlineDecision(candidate_id=candidate.id, score=10.0 - index, topic="World")
            selected.append(SelectedArticle(candidate=candidate, decision=decision))

        warnings: list[str] = []
        pruned = brief_execution_module._prune_selected_for_final_token_budget(
            orchestrator,
            brief_name="general",
            selected=selected,
            filtering=FilteringConfig(article_text_max_chars=900, headline_score_cutoff=7.0),
            topics=[TopicConfig(name="World")],
            prior_reports=[],
            brief_goal="General daily news brief.",
            date="2026-05-30",
            include_enrichment_context=False,
            run_warnings=warnings,
        )

        kept_scores = [item.decision.score for item in pruned]
        dropped_scores = [item.decision.score for item in selected if item not in pruned]
        self.assertLess(len(pruned), len(selected))
        self.assertEqual(kept_scores, sorted(kept_scores, reverse=True))
        self.assertTrue(dropped_scores)
        self.assertGreater(min(kept_scores), max(dropped_scores))
        self.assertTrue(all(item.decision.selection_reason_code == "skipped_final_budget" for item in selected if item not in pruned))
        self.assertTrue(any("dynamic final-context budget raised effective headline score floor" in item for item in warnings))

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

    def test_brief_generator_normalizes_topic_report_structured_fields(self) -> None:
        class _DummyClient:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    backend="transformers",
                    effective_model_label="dummy-final",
                    response_format="json_object",
                )
                self.max_input_tokens = 1200
                self.max_new_tokens = 240

            @staticmethod
            def estimate_tokens(text: str) -> int:
                return max(1, len(text) // 4)

            @staticmethod
            def complete_json(*_args, **_kwargs):
                return {
                    "title": "Daily Brief - 2026-05-30",
                    "lead": "Lead summary.",
                    "topic_reports": [
                        {
                            "topic": "Policy",
                            "narrative_summary": "Legacy topic summary.",
                            "narrative_changes": [
                                {
                                    "narrative": "Export controls",
                                    "status": "escalating",
                                    "summary": "Rules tightened across advanced chips.",
                                }
                            ],
                            "what_to_watch": ["Agency implementation timeline"],
                        }
                    ],
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
        )
        selected = [
            SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id=candidate.id, score=8.0, topic="Policy"),
                article_text="full text",
                extraction_status="ok",
            )
        ]
        generator = BriefGenerator(
            _DummyClient(),
            max_context_chars=480,
            input_token_limit=1200,
            max_new_tokens=240,
        )
        brief = generator.generate(
            selected,
            UserMemory(),
            [TopicConfig(name="Policy")],
            [],
            "General brief.",
            "2026-05-30",
        )

        report = brief["topic_reports"][0]
        self.assertEqual(report.get("topic"), "Policy")
        self.assertEqual(report.get("why_it_matters"), "Legacy topic summary.")
        self.assertEqual(report.get("what_changed"), "Rules tightened across advanced chips.")
        self.assertEqual(report.get("who_is_affected"), [])
        self.assertIn("Agency implementation timeline", report.get("what_to_watch", []))

    def test_render_markdown_includes_structured_topic_fields(self) -> None:
        markdown = render_markdown(
            {
                "title": "Daily Brief - 2026-05-30",
                "lead": "Lead summary.",
                "knowns": ["Known point"],
                "unknowns": ["Unknown point"],
                "watch_signals": ["Watch point"],
                "topic_reports": [
                    {
                        "topic": "AI policy",
                        "why_it_matters": "Policy timing now affects deployment plans.",
                        "what_changed": "Draft guidance shifted from principles to enforceable controls.",
                        "who_is_affected": ["Enterprise AI teams", "Chip suppliers"],
                        "narrative_changes": [
                            {
                                "narrative": "Compliance burden",
                                "status": "escalating",
                                "summary": "Reporting requirements widened across model classes.",
                            }
                        ],
                        "what_to_watch": ["Final regulator publication date"],
                    }
                ],
                "sections": [],
                "references": [],
            }
        )

        self.assertIn("Why it matters: Policy timing now affects deployment plans.", markdown)
        self.assertIn("What changed: Draft guidance shifted from principles to enforceable controls.", markdown)
        self.assertIn("Who is affected:", markdown)
        self.assertIn("- Enterprise AI teams", markdown)
        self.assertIn("What to watch:", markdown)
        self.assertIn("- Final regulator publication date", markdown)

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
