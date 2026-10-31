from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from mydailynews.app.models import (
    AppConfig,
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
    HeadlineDecision,
    MemoryAnnotation,
    NewsCandidate,
    PriorReport,
    SelectedArticle,
    UserMemory,
)
from mydailynews.briefing.generator import BriefGenerator
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.pipeline.brief_analysis_stages import _run_delta_stage


def _article(candidate_id: str, score: float) -> SelectedArticle:
    return SelectedArticle(
        candidate=NewsCandidate(
            id=candidate_id,
            source="Example",
            category="world",
            title=f"Headline {candidate_id}",
            url=f"https://example.test/{candidate_id}",
            snippet="A concise source-backed report.",
            published_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        ),
        decision=HeadlineDecision(candidate_id, score=score),
        article_text="A concise source-backed report.",
        extraction_status="ok",
    )


class _PressureClient:
    max_input_tokens = 1000
    max_new_tokens = 256
    config = SimpleNamespace(context_window_tokens=2048)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        both_articles = '"id":"full"' in text and '"id":"continuing"' in text
        return 2000 if both_articles else 100

    def complete_json(self, _system: str, user: str, **_kwargs):
        self.last_user = user
        return {
            "title": "Test brief",
            "lead": "Supported lead.",
            "topic_reports": [],
            "sections": [],
        }


class BriefDeltaIntegrationTests(unittest.TestCase):
    def test_disabled_delta_does_not_invoke_a_fallback_analyzer(self) -> None:
        orchestrator = SimpleNamespace(config=AppConfig(), debug=DebugLogger(False))

        result = _run_delta_stage(
            orchestrator,
            brief_name="general",
            selected=[_article("current", 8.0)],
            prior_reports=[],
            brief_goal="test",
            date="2026-09-10",
            evidence_config=EvidenceDistillationConfig(enabled=False),
            delta_config=DeltaExtractionConfig(enabled=False),
        )

        self.assertEqual(result.delta_packet, {})
        self.assertEqual(result.warnings, [])

    def test_compaction_preserves_every_writer_disposition(self) -> None:
        packet = {
            "story_decisions": [
                {
                    "story_key": f"story-{index}",
                    "article_ids": [f"article-{index}"],
                    "disposition": "continuing_bullet",
                }
                for index in range(7)
            ]
        }

        compact = BriefGenerator._compact_delta_packet(packet, mode="minimal")

        self.assertEqual(len(compact["story_decisions"]), 7)

    def test_tightest_payload_tier_keeps_editorial_control(self) -> None:
        packet = {
            "baseline_coverage_note": "Explanatory prose that may be dropped.",
            "story_decisions": [
                {
                    "story_key": "story-one",
                    "article_ids": ["article-one"],
                    "disposition": "omit",
                }
            ],
        }
        generator = BriefGenerator(_PressureClient(), max_context_chars=400)

        label, evidence, delta = generator._analysis_payload_options({}, packet)[-1]

        self.assertEqual(label, "control")
        self.assertEqual(evidence, {})
        self.assertEqual(delta["story_decisions"][0]["disposition"], "omit")
        self.assertNotIn("baseline_coverage_note", delta)

    def test_prompt_pressure_keeps_full_report_before_higher_scored_continuation(self) -> None:
        full = _article("full", 4.0)
        continuing = _article("continuing", 9.0)
        set_memory_annotation(full.candidate, MemoryAnnotation(story_key="story-full"))
        set_memory_annotation(
            continuing.candidate,
            MemoryAnnotation(story_key="story-continuing"),
        )
        delta_packet = {
            "story_decisions": [
                {"story_key": "story-full", "article_ids": ["full"], "disposition": "full_report"},
                {
                    "story_key": "story-continuing",
                    "article_ids": ["continuing"],
                    "disposition": "continuing_bullet",
                },
            ]
        }
        generator = BriefGenerator(_PressureClient(), max_context_chars=400)

        prompt, used = generator._build_prompt(
            [continuing, full],
            UserMemory(),
            [],
            [],
            "test",
            "2026-09-10",
            evidence_packet={},
            delta_packet=delta_packet,
        )

        self.assertEqual([article.candidate.id for article in used], ["full"])
        self.assertIn('"disposition":"full_report"', prompt)
        self.assertNotIn('"disposition":"continuing_bullet"', prompt)

    def test_prompt_pressure_drops_mixed_story_analysis_atomically(self) -> None:
        full = _article("full", 4.0)
        continuing = _article("continuing", 9.0)
        evidence_packet = {
            "overview": "Overview built from both articles.",
            "story_clusters": [
                {
                    "cluster_id": "mixed",
                    "article_ids": ["full", "continuing"],
                    "summary": "DROPPED_MEMBER_SECRET",
                    "key_claims": [],
                }
            ],
        }
        delta_packet = {
            "new": [
                {
                    "item": "Mixed story",
                    "summary": "DROPPED_DELTA_SECRET",
                    "article_ids": ["full", "continuing"],
                }
            ],
            "story_decisions": [
                {"story_key": "story-full", "article_ids": ["full"], "disposition": "full_report"},
                {
                    "story_key": "story-continuing",
                    "article_ids": ["continuing"],
                    "disposition": "continuing_bullet",
                },
            ],
        }
        client = _PressureClient()
        generator = BriefGenerator(client, max_context_chars=400)

        result = generator.generate(
            [continuing, full],
            UserMemory(),
            [],
            [
                PriorReport(
                    id="prior",
                    date="2026-09-09",
                    title="PRIOR_DROPPED_SECRET",
                    path="prior.json",
                    summary="PRIOR_DROPPED_SECRET",
                    major_headlines=[
                        {
                            "story_key": "story-continuing",
                            "headline": "PRIOR_DROPPED_SECRET",
                        }
                    ],
                    story_baselines=[
                        {
                            "story_key": "story-continuing",
                            "summary": "PRIOR_DROPPED_SECRET",
                        }
                    ],
                )
            ],
            "test",
            "2026-09-10",
            evidence_packet=evidence_packet,
            delta_packet=delta_packet,
            recall_packet={
                "coverage_guidance": [
                    {
                        "story_key": "story-continuing",
                        "reason": "RECALL_DROPPED_SECRET",
                    }
                ]
            },
        )

        self.assertEqual([row["id"] for row in result["selected_articles"]], ["full"])
        self.assertNotIn("DROPPED_MEMBER_SECRET", client.last_user)
        self.assertNotIn("DROPPED_DELTA_SECRET", client.last_user)
        self.assertNotIn("PRIOR_DROPPED_SECRET", client.last_user)
        self.assertNotIn("RECALL_DROPPED_SECRET", client.last_user)
        self.assertNotIn("DROPPED_MEMBER_SECRET", str(result))
        self.assertNotIn("DROPPED_DELTA_SECRET", str(result))


if __name__ == "__main__":
    unittest.main()
