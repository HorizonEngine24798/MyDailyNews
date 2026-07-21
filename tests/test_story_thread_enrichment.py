from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
import unittest
import uuid

from mydailynews.app.models import AppConfig, EnrichmentConfig, HeadlineDecision, NewsCandidate, SelectedArticle
from mydailynews.common.cache import HTTPFetchResult, JSONCache
from mydailynews.enrichment.models import StoryEnrichment
from mydailynews.enrichment.runner import StoryThreadEnricher
from mydailynews.story_grouping.models import ResearchQuestion, StoryGroup
from mydailynews.retrieval.ddg import DuckDuckGoSearchRetriever


TEMP_ROOT = Path(__file__).resolve().parents[1] / ".codex_tmp_test" / "story_thread_enrichment"
PUBLISHED_AT = datetime(2099, 1, 1, tzinfo=timezone.utc)


@dataclass
class FakeAIConfig:
    backend: str = "fake"
    response_format: str = "json_object"

    @property
    def effective_model_label(self) -> str:
        return "fake-story-model"


class FakeAIClient:
    config = FakeAIConfig()

    def __init__(self, responses: list[dict], *, max_input_tokens: int = 12000, max_new_tokens: int = 1200) -> None:
        self.responses = list(responses)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.calls: list[dict] = []

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def complete_json(self, system: str, user: str, label: str = "ai.complete_json", **kwargs) -> dict:
        self.calls.append({"system": system, "user": user, "label": label, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected AI call: {label}")
        return self.responses.pop(0)

    def unload(self) -> None:
        return None


class FakeHttp:
    def get_text(self, *args, **kwargs) -> HTTPFetchResult:
        return HTTPFetchResult(ok=False, status_code=0, text="", headers={}, cache_state="network")


def _candidate(candidate_id: str, title: str, *, source: str = "Source") -> NewsCandidate:
    return NewsCandidate(
        id=candidate_id,
        source=source,
        category="general",
        title=title,
        url=f"https://example.test/{candidate_id}",
        snippet=f"Snippet for {title} with policy and supply-chain context.",
        published_at=PUBLISHED_AT,
        metadata={"topic_name": "Technology policy"},
    )


def _selected(candidate_id: str, title: str, *, source: str = "Source", text: str = "") -> SelectedArticle:
    return SelectedArticle(
        candidate=_candidate(candidate_id, title, source=source),
        decision=HeadlineDecision(candidate_id, score=8.0, topic="Technology policy"),
        article_text=text or (f"Full article text about {title}. " * 40),
        extraction_status="ok",
    )


def _config(**overrides) -> AppConfig:
    enrichment = EnrichmentConfig(
        enabled=True,
        mode="story_llm",
        search_results_per_query=0,
        max_fetched_research_pages_per_story=0,
        cache_ttl_seconds=604800,
    )
    for key, value in overrides.items():
        setattr(enrichment, key, value)
    return AppConfig(enrichment=enrichment)


def _synthesis_response(story_id: str = "story-001") -> dict:
    return {
        "story_id": story_id,
        "story_title": "Chip supply-chain scrutiny",
        "internal_articles": [
            {
                "title": "Why the chip supply-chain story matters",
                "summary": "The selected articles point to tighter scrutiny affecting semiconductor supply chains.",
                "what_it_adds": "Connects the policy signal to operational supply-chain risk.",
                "source_ids": ["selected-a"],
                "confidence": "medium",
            }
        ],
        "confirmed_facts": [{"fact": "Scrutiny increased.", "source_ids": ["selected-a"]}],
        "conflicting_claims": [{"claim": "The licensing timeline remains disputed.", "source_ids": ["selected-a"]}],
        "open_questions": [{"question": "How quickly licensing changes bite.", "source_ids": ["selected-a"]}],
    }


class StoryThreadEnrichmentTests(unittest.TestCase):
    def test_planner_grouping_attaches_story_context_to_all_thread_articles(self) -> None:
        articles = [
            _selected("a", "Chip export scrutiny expands", source="Source A"),
            _selected("b", "AI chip supply-chain pressure rises", source="Source B"),
        ]
        ai = FakeAIClient(
            [
                {
                    "story_threads": [
                        {
                            "story_id": "story-001",
                            "story_title": "Chip supply-chain scrutiny",
                            "article_ids": ["a", "b"],
                            "research_questions": [
                                {
                                    "question": "What changed in export scrutiny?",
                                    "queries": ["chip export scrutiny supply chain"],
                                }
                            ],
                        }
                    ]
                },
                _synthesis_response("story-001"),
            ]
        )
        enricher = StoryThreadEnricher(_config(), ai_client=ai, brief_name="general", date="2099-01-01")

        enricher.enrich_many(articles)

        self.assertEqual(enricher.story_threads_created, 1)
        self.assertEqual(enricher.story_threads_enriched, 1)
        thread = enricher.artifact["story_threads"][0]
        self.assertEqual(thread["confirmed_facts"][0]["fact"], "Scrutiny increased.")
        self.assertEqual(thread["conflicting_claims"][0]["claim"], "The licensing timeline remains disputed.")
        self.assertEqual(thread["open_questions"][0]["question"], "How quickly licensing changes bite.")
        self.assertEqual([len(article.context_sources) for article in articles], [1, 1])
        for article in articles:
            source = article.context_sources[0]
            self.assertEqual(source.kind, "story_llm_research_context")
            self.assertEqual(source.items[0]["story_id"], "story-001")

    def test_omitted_article_skip_policy_does_not_create_fallback_thread(self) -> None:
        articles = [
            _selected("a", "Chip export scrutiny expands"),
            _selected("b", "Separate antitrust case advances"),
        ]
        ai = FakeAIClient(
            [
                {
                    "story_threads": [
                        {
                            "story_id": "story-001",
                            "story_title": "Chip supply-chain scrutiny",
                            "article_ids": ["a"],
                            "research_questions": [],
                        }
                    ]
                },
                _synthesis_response("story-001"),
            ]
        )
        enricher = StoryThreadEnricher(_config(max_story_threads=3), ai_client=ai)

        enricher.enrich_many(articles)

        self.assertEqual(enricher.story_threads_created, 1)
        self.assertTrue(any("skipped enrichment" in warning for warning in enricher.warnings))
        self.assertEqual(enricher.artifact["planner"]["omitted_article_ids"], ["b"])
        self.assertFalse(articles[1].context_sources)
        self.assertIn("omitted_article_policy=skip", articles[1].enrichment_reason)

    def test_omitted_article_fallback_policy_still_enriches_singleton(self) -> None:
        articles = [
            _selected("a", "Chip export scrutiny expands"),
            _selected("b", "Separate antitrust case advances"),
        ]
        ai = FakeAIClient(
            [
                {
                    "story_threads": [
                        {
                            "story_id": "story-001",
                            "story_title": "Chip supply-chain scrutiny",
                            "article_ids": ["a"],
                            "research_questions": [],
                        }
                    ]
                },
                _synthesis_response("story-001"),
                _synthesis_response("story-002"),
            ]
        )
        enricher = StoryThreadEnricher(
            _config(max_story_threads=3, omitted_article_policy="fallback_singleton"),
            ai_client=ai,
        )

        enricher.enrich_many(articles)

        self.assertEqual(enricher.story_threads_created, 2)
        self.assertTrue(articles[1].context_sources)
        self.assertEqual(articles[1].context_sources[0].items[0]["story_id"], "story-002")

    def test_misc_story_is_artifact_only_by_default(self) -> None:
        article = _selected("a", "Minor unrelated policy mention")
        shared_groups = [StoryGroup("misc", "Miscellaneous", ["a"], [], disposition="misc")]
        ai = FakeAIClient([])
        enricher = StoryThreadEnricher(_config(), ai_client=ai)

        enricher.enrich_many([article], story_groups=shared_groups)

        self.assertEqual(ai.calls, [])
        self.assertEqual(enricher.artifact["story_threads"][0]["status"], "skipped_misc")
        self.assertFalse(article.context_sources)
        self.assertIn("MISC", article.enrichment_reason)

    def test_planner_chosen_singleton_gets_full_enrichment(self) -> None:
        article = _selected("a", "Standalone antitrust case advances")
        shared_groups = [StoryGroup("story-001", "Standalone antitrust case", ["a"], [], disposition="singleton")]
        ai = FakeAIClient([_synthesis_response("story-001")])
        enricher = StoryThreadEnricher(_config(), ai_client=ai)

        enricher.enrich_many([article], story_groups=shared_groups)

        self.assertEqual(len(ai.calls), 1)
        self.assertTrue(article.context_sources)
        self.assertEqual(enricher.artifact["story_threads"][0]["status"], "enriched")

    def test_precomputed_story_groups_skip_planner_and_attach_context(self) -> None:
        articles = [
            _selected("a", "Chip export scrutiny expands", source="Source A"),
            _selected("b", "AI chip supply-chain pressure rises", source="Source B"),
        ]
        ai = FakeAIClient([_synthesis_response("shared-001")])
        shared_groups = [
            StoryGroup(
                story_id="shared-001",
                story_title="Shared chip grouping",
                article_ids=["a", "b"],
                research_questions=[
                    ResearchQuestion(
                        question="What changed in export scrutiny?",
                        queries=["chip export scrutiny"],
                    )
                ],
            )
        ]
        enricher = StoryThreadEnricher(_config(), ai_client=ai)

        enricher.enrich_many(articles, story_groups=shared_groups)

        self.assertEqual(len(ai.calls), 1)
        self.assertIn("synthesis", ai.calls[0]["label"])
        self.assertEqual(enricher.artifact["planner"]["status"], "shared_story_grouping")
        self.assertEqual(enricher.story_threads_created, 1)
        self.assertEqual([len(article.context_sources) for article in articles], [1, 1])
        self.assertEqual(articles[0].context_sources[0].items[0]["story_id"], "shared-001")

    def test_progress_sink_reports_story_enrichment_steps(self) -> None:
        article = _selected("a", "Chip export scrutiny expands", source="Source A")
        ai = FakeAIClient([_synthesis_response("shared-001")])
        shared_groups = [StoryGroup("shared-001", "Shared chip grouping", ["a"], [])]
        progress: list[str] = []
        enricher = StoryThreadEnricher(_config(), ai_client=ai, progress_sink=progress.append)

        enricher.enrich_many([article], story_groups=shared_groups)

        joined = "\n".join(progress)
        self.assertIn("Using shared story grouping", joined)
        self.assertIn("Story enrichment planned 1 thread", joined)
        self.assertIn("Story 1/1: enriching", joined)
        self.assertIn("retrieved", joined)
        self.assertIn("finished status=enriched", joined)

    def test_max_queries_per_story_caps_progress_and_retrieval_query_count(self) -> None:
        article = _selected("a", "Chip export scrutiny expands", source="Source A")
        ai = FakeAIClient([_synthesis_response("story-001")])
        shared_groups = [
            StoryGroup(
                "story-001",
                "Chip supply-chain scrutiny",
                ["a"],
                [
                    ResearchQuestion(
                        "What changed?",
                        ["query one", "query two", "query three"],
                    )
                ],
            )
        ]
        progress: list[str] = []
        enricher = StoryThreadEnricher(_config(max_queries_per_story=2), ai_client=ai, progress_sink=progress.append)

        enricher.enrich_many([article], story_groups=shared_groups)

        entry = enricher.artifact["story_threads"][0]
        self.assertEqual(entry["queries"], ["query one", "query two"])
        self.assertEqual(entry["query_count_before_cap"], 3)
        self.assertEqual(entry["query_count_after_cap"], 2)
        self.assertIn("queries=2", "\n".join(progress))

    def test_empty_precomputed_story_groups_do_not_replan(self) -> None:
        articles = [
            _selected("a", "Chip export scrutiny expands", source="Source A"),
            _selected("b", "AI chip supply-chain pressure rises", source="Source B"),
        ]
        ai = FakeAIClient([])
        enricher = StoryThreadEnricher(_config(), ai_client=ai)

        enricher.enrich_many(articles, story_groups=[])

        self.assertEqual(ai.calls, [])
        self.assertEqual(enricher.artifact["planner"]["status"], "shared_story_grouping")
        self.assertEqual(enricher.story_threads_created, 0)
        self.assertEqual([article.enrichment_needed for article in articles], [False, False])
        self.assertEqual(enricher.artifact["planner"]["omitted_article_ids"], ["a", "b"])
        self.assertTrue(all("omitted_article_policy=skip" in article.enrichment_reason for article in articles))

    def test_capped_story_threads_mark_skipped_articles(self) -> None:
        articles = [
            _selected("a", "Chip export scrutiny expands", source="Source A"),
            _selected("b", "Separate antitrust case advances", source="Source B"),
        ]
        shared_groups = [
            StoryGroup("story-001", "Chip supply-chain scrutiny", ["a"], []),
            StoryGroup("story-002", "Antitrust case", ["b"], []),
        ]
        ai = FakeAIClient([_synthesis_response("story-001")])
        enricher = StoryThreadEnricher(_config(max_story_threads=1), ai_client=ai)

        enricher.enrich_many(articles, story_groups=shared_groups)

        self.assertTrue(articles[0].enrichment_needed)
        self.assertFalse(articles[1].enrichment_needed)
        self.assertIn("thread cap", articles[1].enrichment_reason)
        self.assertEqual(enricher.story_threads_skipped, 1)
        self.assertEqual(
            [entry["status"] for entry in enricher.artifact["story_threads"]],
            ["enriched", "skipped_thread_cap"],
        )

    def test_retrieval_runs_one_story_ahead_and_keeps_failure_order(self) -> None:
        articles = [_selected(str(index), f"Story {index}") for index in range(1, 4)]
        groups = [StoryGroup(f"story-{index}", f"Story {index}", [str(index)], []) for index in range(1, 4)]
        enricher = StoryThreadEnricher(_config(max_story_threads=3), ai_client=FakeAIClient([]))
        second_retrieval_started = Event()
        activity_lock = Lock()
        events: list[str] = []
        active_retrievals = 0
        max_active_retrievals = 0

        def retrieve(story, story_articles, queries, *, warning_sink=None):
            nonlocal active_retrievals, max_active_retrievals
            with activity_lock:
                active_retrievals += 1
                max_active_retrievals = max(max_active_retrievals, active_retrievals)
                events.append(f"retrieve:{story.story_id}")
            try:
                if story.story_id == "story-2":
                    if warning_sink:
                        warning_sink("research warning story-2")
                    second_retrieval_started.set()
                    raise RuntimeError("research boom")
                return []
            finally:
                with activity_lock:
                    active_retrievals -= 1

        def synthesize(story, story_articles, research_results):
            events.append(f"synthesize:{story.story_id}:start")
            if story.story_id == "story-1":
                self.assertTrue(second_retrieval_started.wait(timeout=2))
                enricher._push_warning("synthesis warning story-1")
            events.append(f"synthesize:{story.story_id}:end")
            return (
                StoryEnrichment(
                    story_id=story.story_id,
                    story_title=story.story_title,
                    internal_articles=[
                        {
                            "title": story.story_title,
                            "summary": "Summary",
                            "source_ids": list(story.article_ids),
                        }
                    ],
                    confirmed_facts=[],
                    conflicting_claims=[],
                    open_questions=[],
                ),
                {"status": "ok"},
            )

        enricher._retrieve_research_results = retrieve
        enricher._synthesize_story = synthesize

        enricher.enrich_many(articles, story_groups=groups)

        self.assertLess(events.index("retrieve:story-2"), events.index("synthesize:story-1:end"))
        self.assertEqual(max_active_retrievals, 1)
        self.assertEqual(active_retrievals, 0)
        self.assertNotIn("synthesize:story-2:start", events)
        self.assertIn("synthesize:story-3:end", events)
        self.assertEqual(
            [(entry["story_id"], entry["status"]) for entry in enricher.artifact["story_threads"]],
            [("story-1", "enriched"), ("story-3", "enriched"), ("story-2", "failed")],
        )
        self.assertEqual(
            enricher.warnings,
            [
                "synthesis warning story-1",
                "research warning story-2",
                "story enrichment story-2: RuntimeError: research boom",
            ],
        )

    def test_failed_synthesis_marks_articles_as_not_enriched(self) -> None:
        article = _selected("a", "Chip export scrutiny expands")
        ai = FakeAIClient([])
        shared_groups = [StoryGroup("story-001", "Chip supply-chain scrutiny", ["a"], [])]
        enricher = StoryThreadEnricher(_config(), ai_client=ai)

        enricher.enrich_many([article], story_groups=shared_groups)

        self.assertFalse(article.enrichment_needed)
        self.assertIn("synthesis failed", article.enrichment_reason)
        self.assertEqual(enricher.artifact["story_threads"][0]["status"], "skipped_synthesis")

    def test_no_internal_articles_marks_articles_as_not_enriched(self) -> None:
        article = _selected("a", "Chip export scrutiny expands")
        ai = FakeAIClient(
            [
                {
                    "story_id": "story-001",
                    "story_title": "Chip supply-chain scrutiny",
                    "internal_articles": [],
                    "confirmed_facts": [],
                    "conflicting_claims": [],
                    "open_questions": [],
                }
            ]
        )
        shared_groups = [StoryGroup("story-001", "Chip supply-chain scrutiny", ["a"], [])]
        enricher = StoryThreadEnricher(_config(), ai_client=ai)

        enricher.enrich_many([article], story_groups=shared_groups)

        self.assertFalse(article.enrichment_needed)
        self.assertIn("no internal articles", article.enrichment_reason)
        self.assertEqual(enricher.artifact["story_threads"][0]["status"], "no_internal_articles")

    def test_synthesis_preserves_full_model_text_and_lists(self) -> None:
        article = _selected("a", "Chip export scrutiny expands")
        long_summary = "Full enrichment summary " + ("x" * 1200)
        facts = [{"fact": f"Fact {index} " + ("y" * 350), "source_ids": ["selected-a"]} for index in range(13)]
        response = {
            "story_id": "story-001",
            "story_title": "Full chip supply-chain story " + ("t" * 250),
            "internal_articles": [
                {
                    "title": f"Internal article {index}",
                    "summary": long_summary,
                    "what_it_adds": "Additional context " + ("z" * 400),
                    "source_ids": ["selected-a"],
                    "confidence": "high",
                }
                for index in range(4)
            ],
            "confirmed_facts": facts,
            "conflicting_claims": [],
            "open_questions": [],
        }
        ai = FakeAIClient([response])
        shared_groups = [StoryGroup("story-001", "Chip supply-chain scrutiny", ["a"], [])]
        enricher = StoryThreadEnricher(_config(), ai_client=ai)

        enricher.enrich_many([article], story_groups=shared_groups)

        entry = enricher.artifact["story_threads"][0]
        self.assertEqual(len(entry["internal_articles"]), 4)
        self.assertEqual(entry["internal_articles"][-1]["summary"], long_summary)
        self.assertEqual(len(article.context_sources), 4)
        self.assertEqual(len(article.context_sources[0].items[0]["confirmed_facts"]), 13)
        self.assertEqual(article.context_sources[0].items[0]["confirmed_facts"][-1]["fact"], facts[-1]["fact"])

    def test_planner_budget_skips_when_configured_excerpt_does_not_fit(self) -> None:
        article = _selected("a", "Very long story", text="Long context. " * 2000)
        ai = FakeAIClient([], max_input_tokens=800)
        config = _config(max_selected_article_excerpt_chars=2000)
        enricher = StoryThreadEnricher(config, ai_client=ai)
        self.assertIsNotNone(enricher.planner)

        request = enricher.planner.fit_request([article])

        self.assertIsNone(request)

    def test_planner_budget_uses_stage_limit_when_configured(self) -> None:
        article = _selected("a", "Very long story", text="Long context. " * 2000)
        ai = FakeAIClient([], max_input_tokens=12000, max_new_tokens=8192)
        config = _config(max_selected_article_excerpt_chars=2000, planner_max_input_tokens=300)
        enricher = StoryThreadEnricher(config, ai_client=ai)
        self.assertIsNotNone(enricher.planner)

        request = enricher.planner.fit_request([article])

        self.assertIsNone(request)

    def test_planner_request_uses_stage_output_limit_when_configured(self) -> None:
        article = _selected("a", "Chip export scrutiny expands")
        ai = FakeAIClient(
            [
                {
                    "story_threads": [
                        {
                            "story_id": "story-001",
                            "story_title": "Chip supply-chain scrutiny",
                            "article_ids": ["a"],
                            "research_questions": [],
                        }
                    ]
                },
                _synthesis_response("story-001"),
            ],
            max_new_tokens=8192,
        )
        config = _config(planner_max_new_tokens=256, synthesis_max_new_tokens=512)
        enricher = StoryThreadEnricher(config, ai_client=ai)

        enricher.enrich_many([article])

        self.assertEqual(ai.calls[0]["kwargs"]["max_new_tokens"], 256)
        self.assertEqual(ai.calls[1]["kwargs"]["max_new_tokens"], 512)

    def test_synthesis_budget_uses_stage_limit_when_configured(self) -> None:
        article = _selected("a", "Very long story", text="Long context. " * 2000)
        ai = FakeAIClient([_synthesis_response("story-001")], max_input_tokens=12000, max_new_tokens=8192)
        config = _config(max_selected_article_excerpt_chars=2000, synthesis_max_input_tokens=300)
        shared_groups = [StoryGroup("story-001", "Very long story", ["a"], [])]
        enricher = StoryThreadEnricher(config, ai_client=ai)

        enricher.enrich_many([article], story_groups=shared_groups)

        self.assertEqual(ai.calls, [])
        self.assertEqual(enricher.artifact["story_threads"][0]["status"], "skipped_synthesis")

    def test_disabled_mode_skips_without_story_ai(self) -> None:
        article = _selected("a", "Context-rich article", text=("Full sentence. " * 120))
        enricher = StoryThreadEnricher(_config(mode="disabled"), ai_client=FakeAIClient([]))

        enricher.enrich_many([article])

        self.assertFalse(article.enrichment_needed)
        self.assertIn("enrichment is disabled", article.enrichment_reason)
        self.assertEqual(enricher.story_threads_created, 0)
        self.assertEqual(enricher.warnings, [])

    def test_ddg_no_network_returns_empty_results_with_warning(self) -> None:
        retriever = DuckDuckGoSearchRetriever("test-agent")
        retriever.http = FakeHttp()

        results = retriever.search("chip export scrutiny", 10)

        self.assertEqual(results, [])
        self.assertTrue(any("DDG search failed" in warning for warning in retriever.errors))

    def test_planner_cache_key_changes_when_article_text_changes(self) -> None:
        ai = FakeAIClient([])
        enricher = StoryThreadEnricher(_config(), ai_client=ai)
        self.assertIsNotNone(enricher.planner)
        article = _selected("a", "Chip export scrutiny expands", text="First article body.")
        first = enricher.planner.fit_request([article])
        self.assertIsNotNone(first)

        article.article_text = "Changed article body with materially different context."
        second = enricher.planner.fit_request([article])
        self.assertIsNotNone(second)

        self.assertNotEqual(enricher.planner.cache_key(first), enricher.planner.cache_key(second))

    def test_synthesis_cache_reuses_prior_result(self) -> None:
        cache = JSONCache(str(TEMP_ROOT / uuid.uuid4().hex), "synth", enabled=True)
        articles = [_selected("a", "Chip export scrutiny expands")]
        planner = {
            "story_threads": [
                {
                    "story_id": "story-001",
                    "story_title": "Chip supply-chain scrutiny",
                    "article_ids": ["a"],
                    "research_questions": [],
                }
            ]
        }
        ai_first = FakeAIClient([planner, _synthesis_response("story-001")])
        first = StoryThreadEnricher(_config(), ai_client=ai_first, cache=cache)
        first.enrich_many(articles)
        self.assertEqual(len(ai_first.calls), 2)

        fresh_articles = [_selected("a", "Chip export scrutiny expands")]
        ai_second = FakeAIClient([])
        second = StoryThreadEnricher(_config(), ai_client=ai_second, cache=cache)
        second.enrich_many(fresh_articles)

        self.assertEqual(len(ai_second.calls), 0)
        self.assertEqual(second.story_threads_enriched, 1)
        self.assertTrue(fresh_articles[0].context_sources)


if __name__ == "__main__":
    unittest.main()
