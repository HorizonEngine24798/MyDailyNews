from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import uuid

from mydailynews.app.models import (
    BriefOutput,
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
)
from mydailynews.pipeline.enrichment_module import collect_enrichment_inputs, run_enrichment
from mydailynews.pipeline.handoff import load_brief_handoff, write_brief_handoff
from mydailynews.story_grouping.models import ResearchQuestion, StoryGroup


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "handoff_enrichment"
PUBLISHED_AT = datetime(2026, 6, 14, tzinfo=timezone.utc)


def _selected(candidate_id: str, url: str, *, text: str = "Full article text.") -> SelectedArticle:
    return SelectedArticle(
        candidate=NewsCandidate(
            id=candidate_id,
            source="Example News",
            category="technology",
            title=f"Headline {candidate_id}",
            url=url,
            snippet=f"Snippet {candidate_id}",
            published_at=PUBLISHED_AT,
            tags=["technology"],
            metadata={"topic_name": "Technology policy"},
        ),
        decision=HeadlineDecision(
            candidate_id=candidate_id,
            score=8.4,
            topic="Technology policy",
            reason="Worth briefing.",
            selection_reason_code="score_cutoff",
            selection_rank_score=8.4,
            selection_rank_mode="composite",
        ),
        selection_reason_code="score_cutoff",
        selection_rank_score=8.4,
        selection_rank_mode="composite",
        article_text=text,
        extraction_status="ok",
    )


def _group(story_id: str, article_ids: list[str]) -> StoryGroup:
    return StoryGroup(
        story_id=story_id,
        story_title=f"Story {story_id}",
        article_ids=article_ids,
        research_questions=[ResearchQuestion(question="What changed?", queries=["story update"])],
        topic="Technology policy",
    )


class HandoffAndEnrichmentModuleTests(unittest.TestCase):
    def test_handoff_round_trip_preserves_article_text_and_decision(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        article = _selected("a", "https://example.test/a", text="Full article text with enough context.")
        story_group = _group("story-001", ["a"])

        path = write_brief_handoff(
            output_dir=output_dir,
            date="2026-06-14",
            brief_name="general",
            json_path=output_dir / "2026-06-14_general_brief.json",
            markdown_path=output_dir / "2026-06-14_general_brief.md",
            topics=[TopicConfig(name="Technology policy")],
            prior_reports=[],
            brief_goal="Brief goal",
            filtering=FilteringConfig(),
            selected_articles=[article],
            story_groups=[story_group],
        )
        loaded = load_brief_handoff(path)

        self.assertEqual(loaded.payload["schema_version"], "brief_handoff.v1")
        self.assertEqual(len(loaded.selected_articles), 1)
        round_tripped = loaded.selected_articles[0]
        self.assertEqual(round_tripped.article_text, article.article_text)
        self.assertEqual(round_tripped.extraction_status, "ok")
        self.assertEqual(round_tripped.decision.score, 8.4)
        self.assertEqual(round_tripped.selection_rank_mode, "composite")
        self.assertEqual(round_tripped.candidate.metadata["source_briefs"], ["general"])
        self.assertEqual(loaded.story_groups[0]["article_ids"], ["a"])
        self.assertEqual(loaded.story_groups[0]["research_questions"][0]["queries"], ["story update"])

    def test_legacy_handoff_without_story_groups_still_loads_and_replans(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        date = "2026-06-14"
        path = write_brief_handoff(
            output_dir=output_dir,
            date=date,
            brief_name="general",
            json_path=output_dir / f"{date}_general_brief.json",
            markdown_path=output_dir / f"{date}_general_brief.md",
            topics=[],
            prior_reports=[],
            brief_goal="Brief goal",
            filtering=FilteringConfig(),
            selected_articles=[_selected("a", "https://example.test/a")],
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("story_groups")
        path.write_text(json.dumps(payload), encoding="utf-8")

        self.assertEqual(load_brief_handoff(path).story_groups, [])
        self.assertIsNone(collect_enrichment_inputs(output_dir=output_dir, date=date).story_groups)

    def test_enrichment_input_prefers_handoff_and_dedupes_rehydrated_brief_by_url(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        date = "2026-06-14"
        shared_url = "https://example.test/shared"
        handoff_article = _selected("a", shared_url, text="Long handoff article text.")
        write_brief_handoff(
            output_dir=output_dir,
            date=date,
            brief_name="general",
            json_path=output_dir / f"{date}_general_brief.json",
            markdown_path=output_dir / f"{date}_general_brief.md",
            topics=[],
            prior_reports=[],
            brief_goal="Brief goal",
            filtering=FilteringConfig(),
            selected_articles=[handoff_article],
            story_groups=[_group("story-001", ["a"])],
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{date}_detailed_brief.json").write_text(
            json.dumps(
                {
                    "selected_articles": [
                        {
                            "id": "b",
                            "headline": "Duplicate headline",
                            "source": "Example News",
                            "url": shared_url,
                            "score": 7.2,
                            "topic": "Technology policy",
                            "snippet": "Short rehydrated snippet.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        inputs = collect_enrichment_inputs(output_dir=output_dir, date=date)

        self.assertEqual(inputs.input_mode["general"], "handoff")
        self.assertEqual(inputs.input_mode["detailed"], "rehydrated_brief")
        self.assertEqual(inputs.source_briefs, ["general", "detailed"])
        self.assertEqual(len(inputs.selected_articles), 1)
        self.assertEqual(inputs.selected_articles[0].candidate.id, "a")
        self.assertEqual(inputs.selected_articles[0].article_text, "Long handoff article text.")
        self.assertEqual(inputs.selected_articles[0].candidate.metadata["source_briefs"], ["general", "detailed"])
        self.assertIsNone(inputs.story_groups)

    def test_enrichment_input_reuses_and_normalizes_groups_from_all_handoffs(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        date = "2026-06-14"
        for brief_name, article, group in (
            (
                "general",
                _selected("a", "https://example.test/a"),
                _group("story-001", ["a", "unknown"]),
            ),
            (
                "detailed",
                _selected("b", "https://example.test/b"),
                _group("story-001", ["b"]),
            ),
        ):
            write_brief_handoff(
                output_dir=output_dir,
                date=date,
                brief_name=brief_name,
                json_path=output_dir / f"{date}_{brief_name}_brief.json",
                markdown_path=output_dir / f"{date}_{brief_name}_brief.md",
                topics=[],
                prior_reports=[],
                brief_goal="Brief goal",
                filtering=FilteringConfig(),
                selected_articles=[article],
                story_groups=[group],
            )

        inputs = collect_enrichment_inputs(output_dir=output_dir, date=date)

        self.assertEqual([article.candidate.id for article in inputs.selected_articles], ["a", "b"])
        self.assertIsNotNone(inputs.story_groups)
        self.assertEqual([group.story_id for group in inputs.story_groups or []], ["story-001", "story-002"])
        self.assertEqual([group.article_ids for group in inputs.story_groups or []], [["a"], ["b"]])
        self.assertTrue(any("unknown article id unknown" in warning for warning in inputs.warnings))

    def test_run_enrichment_passes_reusable_groups_to_enricher(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        date = "2026-06-14"
        write_brief_handoff(
            output_dir=output_dir,
            date=date,
            brief_name="general",
            json_path=output_dir / f"{date}_general_brief.json",
            markdown_path=output_dir / f"{date}_general_brief.md",
            topics=[],
            prior_reports=[],
            brief_goal="Brief goal",
            filtering=FilteringConfig(),
            selected_articles=[_selected("a", "https://example.test/a")],
            story_groups=[_group("story-001", ["a"])],
        )
        orchestrator = SimpleNamespace(
            config=SimpleNamespace(
                output_dir=str(output_dir),
                enrichment=SimpleNamespace(enabled=True, mode="story_llm"),
            ),
            warnings=[],
            debug=MagicMock(),
            reporter=SimpleNamespace(phase=lambda _message: None),
            summary_ai_client=object(),
        )
        enricher = MagicMock()
        enricher.warnings = []
        enricher.story_threads_created = 1
        enricher.story_threads_enriched = 0
        enricher.story_threads_skipped = 1
        enricher.artifact = {"story_threads": []}

        with patch("mydailynews.pipeline.enrichment_module.StoryThreadEnricher", return_value=enricher):
            result = run_enrichment(orchestrator, date=date)

        self.assertIsNotNone(result)
        groups = enricher.enrich_many.call_args.kwargs["story_groups"]
        self.assertEqual([group.article_ids for group in groups], [["a"]])

    def test_enrichment_input_consumes_single_existing_brief(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        date = "2026-06-14"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{date}_general_brief.json").write_text(
            json.dumps(
                {
                    "selected_articles": [
                        {
                            "id": "a",
                            "headline": "Only available article",
                            "source": "Example News",
                            "url": "https://example.test/only",
                            "score": 6.5,
                            "topic": "Technology policy",
                            "snippet": "Fallback text.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        inputs = collect_enrichment_inputs(output_dir=output_dir, date=date)

        self.assertEqual(inputs.input_mode["general"], "rehydrated_brief")
        self.assertEqual(inputs.input_mode["detailed"], "missing")
        self.assertEqual(inputs.source_briefs, ["general"])
        self.assertEqual(len(inputs.selected_articles), 1)
        self.assertEqual(inputs.selected_articles[0].article_text, "Fallback text.")
        self.assertEqual(inputs.selected_articles[0].extraction_status, "degraded_brief_json")

    def test_enrichment_input_can_ignore_stale_disk_fallbacks_for_current_run(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        date = "2026-06-14"
        output_dir.mkdir(parents=True, exist_ok=True)
        current_general = output_dir / "current_general_brief.json"
        current_general.write_text(
            json.dumps(
                {
                    "selected_articles": [
                        {
                            "id": "fresh-general",
                            "headline": "Fresh current-run article",
                            "source": "Example News",
                            "url": "https://example.test/fresh",
                            "score": 8.1,
                            "topic": "Technology policy",
                            "snippet": "Fresh current-run text.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (output_dir / f"{date}_detailed_brief.json").write_text(
            json.dumps(
                {
                    "selected_articles": [
                        {
                            "id": "stale-detailed",
                            "headline": "Stale detailed article",
                            "source": "Example News",
                            "url": "https://example.test/stale",
                            "score": 7.1,
                            "topic": "Technology policy",
                            "snippet": "Stale same-day text.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        inputs = collect_enrichment_inputs(
            output_dir=output_dir,
            date=date,
            source_outputs=[
                BriefOutput(
                    name="general",
                    markdown_path=str(output_dir / "current_general_brief.md"),
                    json_path=str(current_general),
                    candidate_count=1,
                    selected_count=1,
                )
            ],
            allow_disk_fallback=False,
        )

        self.assertEqual(inputs.input_mode["general"], "rehydrated_brief")
        self.assertEqual(inputs.input_mode["detailed"], "missing")
        self.assertEqual(inputs.source_briefs, ["general"])
        self.assertEqual([article.candidate.id for article in inputs.selected_articles], ["fresh-general"])


if __name__ == "__main__":
    unittest.main()
