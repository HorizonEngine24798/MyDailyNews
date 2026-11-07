from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, NewsCandidate, PriorReport, SelectedArticle
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.memory.context import build_story_memory_context
from mydailynews.memory.coverage import CoverageMemoryStore, CoverageRecord
from mydailynews.memory.story_store import StoryRecord, StoryStore


class StoryMemoryContextTests(unittest.TestCase):
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

    def test_overbroad_topic_group_splits_unrelated_stories_while_preserving_cohesive_group(self) -> None:
        root = Path(__file__).resolve().parents[1] / ".codex_tmp_test" / f"story_memory_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            store = StoryStore(root / "story_store.json")
            store.replace_records(
                [
                    StoryRecord(
                        story_key="iran-war-hormuz-crisis",
                        story_family_key="middle-east",
                        title="Iran war Hormuz shipping crisis",
                        topic="Geopolitics",
                        tokens=["iran", "war", "hormuz", "strait", "blockade", "tankers", "middle", "east"],
                        entity_tokens=["iran", "hormuz", "us", "persian", "gulf"],
                        event_tokens=["war", "blockade", "strait", "tankers", "shipping"],
                        first_seen="2026-11-01",
                        last_seen="2026-11-03",
                    ),
                    StoryRecord(
                        story_key="trump-administration-is-interfering-in-2026",
                        story_family_key="us-elections",
                        title="Trump Administration Is Interfering in the 2026 Midterm Elections",
                        topic="US Politics",
                        tokens=["trump", "administration", "interfering", "2026", "midterm", "elections", "ballots"],
                        entity_tokens=["trump", "congress", "justice"],
                        event_tokens=["interfering", "midterm", "elections", "voting"],
                        number_tokens=["2026"],
                        first_seen="2026-11-02",
                        last_seen="2026-11-03",
                    ),
                ]
            )

            def make_article(cid: str, key: str, title: str, text: str, topic: str) -> SelectedArticle:
                cand = NewsCandidate(
                    id=cid,
                    source="Example Wire",
                    category=topic,
                    title=title,
                    url=f"https://example.test/{cid}",
                    snippet=text,
                    published_at=datetime(2026, 11, 4, tzinfo=timezone.utc),
                )
                set_memory_annotation(
                    cand,
                    MemoryAnnotation(
                        story_key=key,
                        story_family_key=topic.lower().replace(" ", "-"),
                        story_title=title,
                    ),
                )
                return SelectedArticle(
                    candidate=cand,
                    decision=HeadlineDecision(cid, score=8.2, topic=topic),
                    article_text=text,
                )

            iran_a = make_article(
                "iran-a",
                "iran-war-hormuz-crisis-escalates",
                "Iran war Hormuz crisis escalates as tankers halt",
                "The Iran war and Strait of Hormuz shipping crisis intensified as US naval forces tightened the Persian Gulf blockade.",
                "Geopolitics",
            )
            iran_b = make_article(
                "iran-b",
                "us-blockade-middle-east-tightens",
                "US blockade in Middle East tightens around Persian Gulf",
                "US naval forces in the Middle East expanded the Strait of Hormuz blockade amid the ongoing Iran war.",
                "Geopolitics",
            )
            midterms = make_article(
                "midterms-c",
                "trump-administration-interfering-2026-midterm-elections",
                "Trump Administration Is Interfering in the 2026 Midterm Elections",
                "Critics warn the Trump administration is interfering in the 2026 midterm elections through federal ballot rules.",
                "Geopolitics",
            )
            ai_openai = make_article(
                "ai-1",
                "openai-abu-dhabi-stargate-data-center",
                "OpenAI and Abu Dhabi G42 build massive UAE data center",
                "OpenAI and Abu Dhabi investment firm G42 announced a five-gigawatt Stargate UAE data center campus.",
                "Artificial Intelligence",
            )
            ai_tinder = make_article(
                "ai-2",
                "tinder-world-id-age-verification-japan",
                "Tinder rolls out World ID iris verification in Japan",
                "Dating app Tinder partnered with Sam Altman's World ID to verify user ages in Tokyo.",
                "Artificial Intelligence",
            )

            groups = [
                SimpleNamespace(
                    story_id="story-001",
                    topic="Geopolitics",
                    story_title="US-Iran War and Blockade",
                    article_ids=["iran-a", "iran-b", "midterms-c"],
                ),
                SimpleNamespace(
                    story_id="story-002",
                    topic="Artificial Intelligence",
                    story_title="Artificial Intelligence",
                    article_ids=["ai-1", "ai-2"],
                ),
            ]

            context = build_story_memory_context(
                selected=[iran_a, iran_b, midterms, ai_openai, ai_tinder],
                story_groups=groups,
                story_store=store,
                coverage_store=None,
                prior_reports=[],
                date="2026-11-04",
            )

            by_articles = {
                tuple(story["current_article_ids"]): story
                for story in context["stories"]
            }
            self.assertIn(("iran-a", "iran-b"), by_articles)
            self.assertIn(("midterms-c",), by_articles)
            self.assertIn(("ai-1",), by_articles)
            self.assertIn(("ai-2",), by_articles)
            self.assertEqual(
                by_articles[("midterms-c",)]["prior_baselines"][0]["story_key"],
                "trump-administration-is-interfering-in-2026",
            )
            self.assertEqual(
                by_articles[("ai-1",)]["current_title"],
                "OpenAI and Abu Dhabi G42 build massive UAE data center",
            )
            self.assertEqual(
                by_articles[("ai-2",)]["current_title"],
                "Tinder rolls out World ID iris verification in Japan",
            )
        finally:
            for path in root.glob("*"):
                path.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
