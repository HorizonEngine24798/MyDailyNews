from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import unittest

from mydailynews.app.models import AppConfig, EnrichmentConfig, EvidenceDistillationConfig, HeadlineDecision, NewsCandidate, SelectedArticle
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.pipeline.brief_stages import _story_grouping_stage
from mydailynews.pipeline.stage_artifacts import next_stage_after
from mydailynews.pipeline.stage_results import StoryGroupingStageResult
from mydailynews.pipeline.stages import normalize_stage_name
from mydailynews.pipeline.story_grouping_models import ResearchQuestion, StoryGroup
from mydailynews.story_grouping.normalization import normalize_story_groups


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


class FakeOrchestrator:
    def __init__(self, config: AppConfig, ai_client: FakeAIClient | None) -> None:
        self.config = config
        self.summary_ai_client = ai_client
        self.synth_cache = None
        self.debug = DebugLogger(False)
        self.reporter = None


def _selected(candidate_id: str, title: str) -> SelectedArticle:
    return SelectedArticle(
        candidate=NewsCandidate(
            id=candidate_id,
            source="Source",
            category="general",
            title=title,
            url=f"https://example.test/{candidate_id}",
            snippet=f"Snippet for {title}.",
            published_at=PUBLISHED_AT,
            metadata={"topic_name": "Technology policy"},
        ),
        decision=HeadlineDecision(candidate_id, score=8.0, topic="Technology policy"),
        article_text=f"Full article text about {title}.",
        extraction_status="ok",
    )


class StoryGroupingStageResultTests(unittest.TestCase):
    def test_planned_artifact_records_groups_and_inputs(self) -> None:
        selected = [
            _selected("a", "Chip export scrutiny expands"),
            _selected("b", "AI chip supply-chain pressure rises"),
        ]
        groups = [
            StoryGroup(
                story_id="story-001",
                story_title="Chip supply-chain scrutiny",
                article_ids=["a", "b"],
                research_questions=[
                    ResearchQuestion(
                        question="What changed in export scrutiny?",
                        queries=["chip export scrutiny supply chain"],
                    )
                ],
                topic="Technology policy",
            )
        ]
        result = StoryGroupingStageResult.planned(
            selected=selected,
            story_groups=groups,
            planner_artifact={"requests": [{"request": 1, "status": "cache_hit"}]},
            warnings=["cached grouping reused"],
            cache_hit=True,
        )

        self.assertTrue(result.artifact["enabled"])
        self.assertEqual(result.artifact["status"], "ok")
        self.assertEqual(result.artifact["selected_article_ids"], ["a", "b"])
        self.assertEqual(result.artifact["story_groups"][0]["story_id"], "story-001")
        self.assertEqual(result.artifact["story_groups"][0]["topic"], "Technology policy")
        self.assertEqual(result.artifact["fallback_groups"], [])
        self.assertTrue(result.artifact["cache_hit"])
        self.assertEqual(result.artifact["requests"], [{"request": 1, "status": "cache_hit"}])
        self.assertEqual(result.warnings, ["cached grouping reused"])
        self.assertIs(result.story_threads[0], groups[0])

    def test_planned_artifact_records_fallback_groups(self) -> None:
        selected = [_selected("a", "Standalone antitrust case advances")]
        group = StoryGroup(
            story_id="story-001",
            story_title="Standalone antitrust case advances",
            article_ids=["a"],
            research_questions=[],
            fallback=True,
        )

        result = StoryGroupingStageResult.planned(selected=selected, story_groups=[group])

        self.assertEqual(result.artifact["fallback_groups"], [result.artifact["story_groups"][0]])

    def test_skipped_artifact_records_reason_without_groups(self) -> None:
        selected = [_selected("a", "Context-rich article")]

        result = StoryGroupingStageResult.skipped(selected=selected, reason="evidence_disabled")

        self.assertFalse(result.artifact["enabled"])
        self.assertEqual(result.artifact["status"], "skipped")
        self.assertEqual(result.artifact["skipped_reason"], "evidence_disabled")
        self.assertEqual(result.artifact["selected_article_ids"], ["a"])
        self.assertEqual(result.story_groups, [])
        self.assertEqual(result.artifact["requests"], [])

    def test_stage_order_includes_story_grouping_after_article_fetch(self) -> None:
        self.assertEqual(normalize_stage_name("story-grouping"), "story_grouping")
        self.assertEqual(next_stage_after("article_fetch"), "story_grouping")
        self.assertEqual(next_stage_after("story_grouping"), "evidence_distillation")

    def test_stage_runs_once_when_enrichment_and_evidence_enabled(self) -> None:
        selected = [
            _selected("a", "Chip export scrutiny expands"),
            _selected("b", "AI chip supply-chain pressure rises"),
        ]
        ai = FakeAIClient(
            [
                {
                    "story_groups": [
                        {
                            "story_id": "story-001",
                            "story_title": "Chip supply-chain scrutiny",
                            "article_ids": ["a", "b"],
                            "research_questions": [],
                        }
                    ]
                }
            ]
        )
        config = AppConfig(
            enrichment=EnrichmentConfig(enabled=True, mode="story_llm"),
        )
        result = _story_grouping_stage(
            FakeOrchestrator(config, ai),
            brief_name="general",
            selected=selected,
            include_enrichment_context=True,
            evidence_config=EvidenceDistillationConfig(enabled=True),
            date="2099-01-01",
        )

        self.assertEqual(len(ai.calls), 1)
        self.assertEqual(result.artifact["status"], "ok")
        self.assertEqual([group.story_id for group in result.story_groups], ["story-001"])
        self.assertEqual(result.artifact["requests"][0]["status"], "ok")

    def test_stage_runs_when_only_evidence_enabled(self) -> None:
        selected = [
            _selected("a", "Chip export scrutiny expands"),
            _selected("b", "AI chip supply-chain pressure rises"),
        ]
        ai = FakeAIClient(
            [
                {
                    "story_groups": [
                        {
                            "story_id": "story-001",
                            "story_title": "Chip supply-chain scrutiny",
                            "article_ids": ["a", "b"],
                            "research_questions": [],
                        }
                    ]
                }
            ]
        )
        config = AppConfig(enrichment=EnrichmentConfig(enabled=False))

        result = _story_grouping_stage(
            FakeOrchestrator(config, ai),
            brief_name="general",
            selected=selected,
            include_enrichment_context=False,
            evidence_config=EvidenceDistillationConfig(enabled=True),
        )

        self.assertEqual(len(ai.calls), 1)
        self.assertEqual(result.artifact["status"], "ok")
        self.assertEqual([group.story_id for group in result.story_groups], ["story-001"])

    def test_shared_normalizer_cleans_common_group_shape(self) -> None:
        selected = [
            _selected("a", "Chip export scrutiny expands"),
            _selected("b", "AI chip supply-chain pressure rises"),
            _selected("c", "Separate antitrust case advances"),
        ]

        result = normalize_story_groups(
            selected=selected,
            raw_groups=[
                {
                    "story_id": "story-001",
                    "story_title": "Chip supply-chain scrutiny",
                    "article_ids": ["a", "z"],
                    "research_questions": [
                        {"question": "What changed?", "queries": ["chip export scrutiny"]}
                    ],
                    "topic": "Technology policy",
                },
                {
                    "story_id": "story-001",
                    "story_title": "Duplicate story id",
                    "article_ids": ["a", "b"],
                    "research_questions": [],
                },
            ],
            caller="story grouping",
            allow_singleton_fallback=True,
            fallback_questions=lambda title: [ResearchQuestion(f"What explains {title}?", [title])],
        )

        self.assertEqual(result.unknown_article_ids, 1)
        self.assertEqual(result.duplicate_article_ids, 1)
        self.assertEqual(result.duplicate_story_ids, 1)
        self.assertEqual(result.fallback_groups, 1)
        self.assertEqual([group.article_ids for group in result.groups], [["a"], ["b"], ["c"]])
        self.assertEqual(result.groups[0].research_questions[0].queries, ["chip export scrutiny"])
        self.assertEqual(result.groups[0].topic, "Technology policy")
        self.assertTrue(result.groups[2].fallback)


if __name__ == "__main__":
    unittest.main()
