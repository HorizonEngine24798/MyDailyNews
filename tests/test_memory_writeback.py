from __future__ import annotations

import unittest
from datetime import datetime, timezone

from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, NewsCandidate, SelectedArticle
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.memory.coverage import coverage_records_for_selected
from mydailynews.memory.story_store import _updated_thread_events


class MemoryWritebackTests(unittest.TestCase):
    def test_grouped_story_day_creates_one_thread_event_for_all_articles(self) -> None:
        decision = {
            "article_ids": ["article-a", "article-b"],
            "relationship": "same_story",
            "change_type": "incremental",
            "summary": "A supported fact was added.",
            "operations": [],
        }

        first = _updated_thread_events(
            [],
            decision,
            article_id="article-a",
            observed_at="2026-09-10",
        )
        second = _updated_thread_events(
            first,
            decision,
            article_id="article-b",
            observed_at="2026-09-10",
        )

        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].article_ids, ["article-a", "article-b"])

    def test_continuing_bullet_does_not_consume_lead_prominence(self) -> None:
        def article(candidate_id: str, policy: str) -> SelectedArticle:
            candidate = NewsCandidate(
                id=candidate_id,
                source="Example",
                category="world",
                title=candidate_id,
                url=f"https://example.test/{candidate_id}",
                snippet="",
                published_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
            )
            set_memory_annotation(
                candidate,
                MemoryAnnotation(
                    story_key=f"story:{candidate_id}",
                    story_family_key="family",
                    story_title=candidate_id,
                    today_policy=policy,
                    change_type="incremental",
                ),
            )
            return SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id, score=7.0),
                selection_rank_score=7.0,
            )

        records = coverage_records_for_selected(
            date="2026-09-10",
            brief_name="general",
            selected=[article("capsule", "capsule"), article("main", "normal")],
        )

        self.assertEqual([record.prominence for record in records], ["capsule", "lead"])


if __name__ == "__main__":
    unittest.main()
