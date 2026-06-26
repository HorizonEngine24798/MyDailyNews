from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
import uuid

from mydailynews.analysis.shared import article_cache_payload, story_thread_payloads
from mydailynews.app.config import load_config
from mydailynews.app.models import ContextSource, FilteringConfig, HeadlineDecision, NewsCandidate, SelectedArticle, TopicConfig, UserMemory
from mydailynews.briefing.generator import BriefGenerator
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.domain.candidate_annotations import candidate_annotations
from mydailynews.domain.headline_selection import limit_candidates_for_ai, select_articles
from mydailynews.pipeline.article_pipeline import record_enrichment_metrics, story_thread_enrichment_counts


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "cluster_phase2"
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


class ClusterPhaseTwoTests(unittest.TestCase):
    def test_filtering_config_no_longer_exposes_cluster_knobs(self) -> None:
        filtering = FilteringConfig()

        self.assertFalse(hasattr(filtering, "max_selected_per_event_cluster"))
        self.assertFalse(hasattr(filtering, "prefer_multi_source_clusters"))
        self.assertFalse(hasattr(filtering, "multi_source_cluster_bonus"))
        self.assertFalse(hasattr(filtering, "event_cluster_time_window_hours"))

    def test_removed_cluster_config_knobs_are_rejected(self) -> None:
        payload = json.loads((REPO_ROOT / "config.example.json").read_text(encoding="utf-8-sig"))
        payload["filtering"]["max_selected_per_event_cluster"] = 1

        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEMP_ROOT / f"{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(deepcopy(payload), ensure_ascii=False, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, r"Config section filtering has unrecognized key\(s\): max_selected_per_event_cluster"):
            load_config(path)

    def test_stale_event_cluster_metadata_is_ignored_by_selection(self) -> None:
        candidates = [
            _candidate("a", "Chip export scrutiny expands across Asian supply chains", source="Source A"),
            _candidate("b", "AI chip supply chain pressure rises after export review", source="Source B"),
        ]
        for candidate in candidates:
            candidate.metadata.update(
                {
                    "event_cluster_id": "evt-stale",
                    "event_cluster_label": "Stale heuristic cluster",
                    "event_cluster_multi_source": True,
                }
            )

        limited = limit_candidates_for_ai(
            candidates,
            [TopicConfig(name="Semiconductors", queries=["chip supply chain"])],
            FilteringConfig(max_candidates_for_ai=10),
            PUBLISHED_AT,
            user_memory=UserMemory(),
            debug=DebugLogger(False),
        )
        decisions = {
            "a": HeadlineDecision("a", score=9.0),
            "b": HeadlineDecision("b", score=8.0),
        }
        selected = select_articles(
            limited,
            decisions,
            [TopicConfig(name="Semiconductors")],
            FilteringConfig(headline_score_cutoff=0.0, max_selected_articles=2, max_selected_per_source=0),
            user_memory=UserMemory(),
        )

        self.assertEqual([article.candidate.id for article in selected], ["a", "b"])
        self.assertEqual({article.selection_reason_code for article in selected}, {"selected_high_score"})
        self.assertFalse(hasattr(candidate_annotations(candidates[0]), "event_cluster"))

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
        self.assertNotIn("event_cluster", cache_payload)
        self.assertNotIn("event_cluster", brief_payload)

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
