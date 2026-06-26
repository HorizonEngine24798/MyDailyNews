from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import unittest
import uuid

from mydailynews.app.models import AppConfig, EnrichmentConfig, HeadlineDecision, NewsCandidate, SelectedArticle
from mydailynews.common.cache import HTTPFetchResult, JSONCache
from mydailynews.pipeline.enrichment import StoryThreadEnricher
from mydailynews.pipeline.story_grouping_models import ResearchQuestion, StoryGroup
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
        "conflicting_claims": [],
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
        self.assertEqual([len(article.context_sources) for article in articles], [1, 1])
        for article in articles:
            source = article.context_sources[0]
            self.assertEqual(source.kind, "story_llm_research_context")
            self.assertEqual(source.items[0]["story_id"], "story-001")

    def test_omitted_article_gets_singleton_fallback_thread(self) -> None:
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
        enricher = StoryThreadEnricher(_config(max_story_threads=3), ai_client=ai)

        enricher.enrich_many(articles)

        self.assertEqual(enricher.story_threads_created, 2)
        self.assertTrue(any("omitted selected article" in warning for warning in enricher.warnings))
        self.assertTrue(articles[1].context_sources)
        self.assertEqual(articles[1].context_sources[0].items[0]["story_id"], "story-002")

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
        self.assertTrue(all("no usable story threads" in article.enrichment_reason for article in articles))

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

    def test_planner_budget_skips_when_configured_excerpt_does_not_fit(self) -> None:
        article = _selected("a", "Very long story", text="Long context. " * 2000)
        ai = FakeAIClient([], max_input_tokens=800)
        config = _config(max_selected_article_excerpt_chars=2000)
        enricher = StoryThreadEnricher(config, ai_client=ai)
        self.assertIsNotNone(enricher.planner)

        request = enricher.planner.fit_request([article])

        self.assertIsNone(request)

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
