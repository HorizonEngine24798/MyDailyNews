from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
import uuid

from mydailynews.app.models import (
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
)
from mydailynews.pipeline.handoff import load_brief_handoff, write_brief_handoff
from mydailynews.story_grouping.models import ResearchQuestion, StoryGroup


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "handoff"
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


class HandoffTests(unittest.TestCase):
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

    def test_legacy_handoff_without_story_groups_still_loads(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
