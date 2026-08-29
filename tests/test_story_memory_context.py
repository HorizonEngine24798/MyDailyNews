from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest
import uuid

from mydailynews.app.models import DeltaExtractionConfig, HeadlineDecision, MemoryAnnotation, NewsCandidate, PriorReport, SelectedArticle, TopicConfig, UserMemory
from mydailynews.analysis.delta import DeltaExtractor
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.memory.context import build_story_memory_context
from mydailynews.memory.coverage import CoverageMemoryStore, CoverageRecord
from mydailynews.memory.story_store import StoryRecord, StoryStore


class StoryMemoryContextTests(unittest.TestCase):
    def test_delta_prompt_contains_story_memory_and_normalizes_decisions(self) -> None:
        extractor = DeltaExtractor(object(), DeltaExtractionConfig())
        prompt = extractor._render_prompt(
            articles=[],
            excerpt_chars=200,
            memory=UserMemory(),
            topics=[TopicConfig(name="World")],
            prior_reports=[],
            brief_goal="daily brief",
            date="2026-06-27",
            evidence_packet={},
            story_memory={"stories": [{"story_key": "iran-ceasefire"}]},
        )
        self.assertIn('"iran-ceasefire"', prompt)
        result = extractor._normalize_result(
            {
                "baseline_coverage_note": "one baseline",
                "new": [],
                "escalated": [],
                "weakened": [],
                "reframed": [],
                "unchanged_but_important": [],
                "evidence_gaps": [],
                "story_decisions": [
                    {
                        "story_key": "iran-ceasefire",
                        "article_ids": ["current"],
                        "relationship": "same_story",
                        "change_type": "unchanged",
                        "materiality": 2,
                        "confidence": -1,
                        "disposition": "continuing_bullet",
                    }
                ],
            }
        )
        self.assertEqual(result["story_decisions"][0]["materiality"], 1.0)
        self.assertEqual(result["story_decisions"][0]["confidence"], 0.0)

    def test_context_combines_annotation_baseline_and_coverage(self) -> None:
        root = Path(__file__).resolve().parents[1] / ".codex_tmp_test" / f"story_memory_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            store = StoryStore(root / "story_store.json")
            store.replace_records(
                [StoryRecord(
                    story_key="iran-ceasefire",
                    story_family_key="iran-israel",
                    title="Iran-Israel ceasefire",
                    topic="World",
                    tokens=["iran", "israel", "ceasefire"],
                    first_seen="2026-06-20",
                    last_seen="2026-06-26",
                    last_material_change_date="2026-06-25",
                    last_change_type="escalated",
                    last_delta_summary="The ceasefire expanded to include a second border crossing.",
                    last_knowns=["A ceasefire is in effect."],
                    last_unknowns=["Whether it will hold."],
                    last_watch_signals=["New violations or formal talks."],
                    last_disposition="full_report",
                )]
            )
            coverage = CoverageMemoryStore.from_state_dir(root)
            coverage.write_records(
                [
                    CoverageRecord(
                        schema_version=1,
                        date="2026-06-26",
                        brief_name="general",
                        story_key="iran-ceasefire",
                        story_family_key="iran-israel",
                        title="Iran-Israel ceasefire",
                        prominence="lead",
                        article_ids=["old"],
                    )
                ]
            )

            candidate = NewsCandidate(
                id="current",
                source="Example News",
                category="world",
                title="Iran and Israel extend ceasefire",
                url="https://example.test/current",
                snippet="The ceasefire remains in place.",
                published_at=datetime(2026, 6, 27, tzinfo=timezone.utc),
            )
            set_memory_annotation(
                candidate,
                MemoryAnnotation(
                    story_key="iran-ceasefire",
                    story_family_key="iran-israel",
                    story_title="Iran-Israel ceasefire",
                    match_confidence=0.9,
                    recent_coverage_count=1,
                    recent_lead_count=1,
                    covered_yesterday=True,
                    today_policy="capsule_unless_material_update",
                ),
            )
            article = SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision("current", score=8.0),
                article_text="The ceasefire remains in place.",
            )
            context = build_story_memory_context(
                selected=[article],
                story_groups=[],
                story_store=store,
                coverage_store=coverage,
                prior_reports=[PriorReport("r1", "2026-06-26", "Yesterday", "", "", story_baselines=[])],
                date="2026-06-27",
            )

            story = context["stories"][0]
            self.assertEqual(story["current_memory"]["story_key"], "iran-ceasefire")
            self.assertEqual(story["prior_baselines"][0]["last_change_type"], "escalated")
            self.assertEqual(story["prior_baselines"][0]["recent_coverage"][0]["prominence"], "lead")
        finally:
            for path in root.glob("*"):
                path.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
