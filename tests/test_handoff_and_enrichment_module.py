from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
import uuid

from mydailynews.app.models import (
    BriefOutput,
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
)
from mydailynews.pipeline.enrichment_module import collect_enrichment_inputs
from mydailynews.pipeline.handoff import load_brief_handoff, write_brief_handoff


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


class HandoffAndEnrichmentModuleTests(unittest.TestCase):
    def test_handoff_round_trip_preserves_article_text_and_decision(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        article = _selected("a", "https://example.test/a", text="Full article text with enough context.")

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
