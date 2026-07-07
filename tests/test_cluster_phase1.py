from __future__ import annotations

from datetime import datetime, timezone
import unittest

from mydailynews.analysis.shared import article_cache_payload, story_thread_payloads
from mydailynews.app.models import ContextSource, HeadlineDecision, NewsCandidate, SelectedArticle
from mydailynews.briefing.generator import BriefGenerator
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.pipeline.article_pipeline import record_enrichment_metrics, story_thread_enrichment_counts


PUBLISHED_AT = datetime(2099, 1, 1, tzinfo=timezone.utc)


def _candidate(candidate_id: str, title: str, *, source: str = "Example News") -> NewsCandidate:
    return NewsCandidate(
        id=candidate_id,
        source=source,
        category="general",
        title=title,
        url=f"https://example.test/{candidate_id}",
        snippet=(
            "A detailed report about semiconductor policy, supply chains, and market effects "
            "with enough wording to exercise heuristic scoring."
        ),
        published_at=PUBLISHED_AT,
        metadata={"topic_name": "Semiconductors"},
    )


def _selected_article(candidate_id: str = "a") -> SelectedArticle:
    return SelectedArticle(
        candidate=_candidate(candidate_id, "Chip export scrutiny expands across Asian supply chains"),
        decision=HeadlineDecision(candidate_id, score=8.0),
        article_text="Article text about the chip supply chain.",
        extraction_status="ok",
    )


class FakeAIClient:
    max_input_tokens = 12000

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class StoryThreadArtifactTests(unittest.TestCase):
    def test_article_payloads_use_story_threads_from_context_sources(self) -> None:
        article = _selected_article()
        article.context_sources.append(
            ContextSource(
                id="story-source-1",
                parent_article_id=article.candidate.id,
                kind="story_llm_research_context",
                title="Why indium matters to AI chips",
                source="LLM research",
                url="",
                summary="Explains the supply-chain relevance.",
                items=[{"story_id": "story-001", "story_title": "Indium export scrutiny"}],
            )
        )

        threads = story_thread_payloads(article)
        cache_payload = article_cache_payload(article)
        brief_payload = BriefGenerator(FakeAIClient(), max_context_chars=800)._article_payload(article, 400)

        self.assertEqual(threads[0]["story_id"], "story-001")
        self.assertIn("story_threads", cache_payload)
        self.assertIn("story_threads", brief_payload)

    def test_enrichment_metrics_report_story_thread_counts(self) -> None:
        article = _selected_article()
        article.context_sources.append(
            ContextSource(
                id="story-source-1",
                parent_article_id=article.candidate.id,
                kind="story_llm_research_context",
                title="Why indium matters to AI chips",
                source="LLM research",
                url="",
                summary="Explains the supply-chain relevance.",
                items=[{"story_id": "story-001", "story_title": "Indium export scrutiny"}],
            )
        )
        debug = DebugLogger(True)

        record_enrichment_metrics(brief_name="general", selected=[article], debug=debug)

        metrics = debug.analytics_payload()["metrics"]
        self.assertEqual(story_thread_enrichment_counts([article]), (1, 1, 0))
        self.assertEqual(metrics["brief.general.enrichment.story_threads_created"], 1)
        self.assertEqual(metrics["brief.general.enrichment.story_threads_enriched"], 1)
        self.assertEqual(metrics["brief.general.enrichment.story_threads_skipped"], 0)


if __name__ == "__main__":
    unittest.main()
