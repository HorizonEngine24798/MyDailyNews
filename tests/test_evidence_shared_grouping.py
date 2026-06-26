from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import unittest

from mydailynews.analysis.evidence import EvidenceDistiller
from mydailynews.app.models import EvidenceDistillationConfig, HeadlineDecision, NewsCandidate, SelectedArticle, UserMemory
from mydailynews.pipeline.story_grouping_models import StoryGroup


PUBLISHED_AT = datetime(2099, 1, 1, tzinfo=timezone.utc)


@dataclass
class FakeAIConfig:
    backend: str = "fake"
    response_format: str = "json_object"

    @property
    def effective_model_label(self) -> str:
        return "fake-evidence-model"


class FakeAIClient:
    config = FakeAIConfig()

    def __init__(self, responses: list[dict], *, max_input_tokens: int = 60000) -> None:
        self.responses = list(responses)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = 1200
        self.calls: list[dict] = []

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def complete_json(self, system: str, user: str, label: str = "ai.complete_json", **kwargs) -> dict:
        self.calls.append({"system": system, "user": user, "label": label, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected AI call: {label}")
        return self.responses.pop(0)


def _selected(candidate_id: str, title: str, *, score: float = 8.0) -> SelectedArticle:
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
        decision=HeadlineDecision(candidate_id, score=score, topic="Technology policy"),
        article_text=f"Full article text about {title}. " * 20,
        extraction_status="ok",
    )


def _config(**overrides) -> EvidenceDistillationConfig:
    config = EvidenceDistillationConfig(
        enabled=True,
        max_articles=10,
        max_articles_per_batch=5,
        max_article_chars=900,
        max_story_clusters=10,
        max_claims_per_cluster=6,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class EvidenceSharedGroupingTests(unittest.TestCase):
    def test_prompt_uses_grouping_plan_and_parser_trims_cross_boundary_ids(self) -> None:
        articles = [
            _selected("a", "Chip export scrutiny expands", score=9.0),
            _selected("b", "AI chip supply-chain pressure rises", score=8.5),
            _selected("c", "Separate antitrust case advances", score=8.0),
        ]
        groups = [
            StoryGroup(
                story_id="story-001",
                story_title="Chip supply-chain scrutiny",
                article_ids=["a", "b"],
                research_questions=[],
                topic="Technology policy",
            )
        ]
        ai = FakeAIClient(
            [
                {
                    "overview": "The selected articles point to several policy risks.",
                    "story_clusters": [
                        {
                            "cluster_id": "invented",
                            "topic": "Technology policy",
                            "label": "Mixed policy cluster",
                            "summary": "The model crossed story boundaries.",
                            "article_ids": ["a", "c"],
                            "key_claims": [
                                {
                                    "claim": "Scrutiny increased.",
                                    "support_article_ids": ["a", "c"],
                                    "confidence": "medium",
                                }
                            ],
                            "consensus_points": [],
                            "contested_points": [],
                            "known_unknowns": [],
                            "watch_signals": [],
                        }
                    ],
                    "global_watch_signals": [],
                    "reader_qa": [
                        {
                            "question": "What should I watch?",
                            "answer": "Licensing signals.",
                            "article_ids": ["a", "z"],
                        }
                    ],
                }
            ]
        )
        distiller = EvidenceDistiller(ai, _config())

        result = distiller.distill(
            articles,
            UserMemory(),
            [],
            [],
            "Brief goal",
            "2099-01-01",
            brief_name="general",
            story_groups=groups,
        )

        self.assertIn("Story grouping plan", ai.calls[0]["user"])
        self.assertIn("story-001", ai.calls[0]["user"])
        cluster = result["story_clusters"][0]
        self.assertEqual(cluster["cluster_id"], "story-001")
        self.assertEqual(cluster["article_ids"], ["a"])
        self.assertEqual(cluster["key_claims"][0]["support_article_ids"], ["a"])
        self.assertEqual(result["reader_qa"][0]["article_ids"], ["a"])
        self.assertGreater(distiller.group_boundary_warning_count, 0)

    def test_cache_key_differs_between_free_and_shared_grouping(self) -> None:
        article = _selected("a", "Chip export scrutiny expands")
        distiller = EvidenceDistiller(FakeAIClient([]), _config())
        free_key = distiller._cache_key(
            used_articles=[article],
            used_reports=[],
            memory=UserMemory(),
            topics=[],
            brief_goal="Brief goal",
            date="2099-01-01",
            headline_context_articles=[],
            story_groups=None,
        )
        shared_key = distiller._cache_key(
            used_articles=[article],
            used_reports=[],
            memory=UserMemory(),
            topics=[],
            brief_goal="Brief goal",
            date="2099-01-01",
            headline_context_articles=[],
            story_groups=[
                StoryGroup(
                    story_id="story-001",
                    story_title="Chip supply-chain scrutiny",
                    article_ids=["a"],
                    research_questions=[],
                )
            ],
        )

        self.assertNotEqual(free_key, shared_key)

    def test_empty_shared_grouping_stays_in_shared_mode(self) -> None:
        article = _selected("a", "Chip export scrutiny expands")
        ai = FakeAIClient(
            [
                {
                    "overview": "The article has policy implications.",
                    "story_clusters": [
                        {
                            "cluster_id": "invented",
                            "topic": "Technology policy",
                            "label": "Invented cluster",
                            "summary": "The model invented a story group.",
                            "article_ids": ["a"],
                            "key_claims": [],
                            "consensus_points": [],
                            "contested_points": [],
                            "known_unknowns": [],
                            "watch_signals": [],
                        }
                    ],
                    "global_watch_signals": [],
                    "reader_qa": [],
                }
            ]
        )
        distiller = EvidenceDistiller(ai, _config())

        result = distiller.distill(
            [article],
            UserMemory(),
            [],
            [],
            "Brief goal",
            "2099-01-01",
            brief_name="general",
            story_groups=[],
        )

        self.assertIn("Story grouping plan", ai.calls[0]["user"])
        self.assertIn("[]", ai.calls[0]["user"])
        self.assertEqual(result["story_clusters"], [])
        self.assertGreater(distiller.group_boundary_warning_count, 0)
        shared_key = distiller._cache_key(
            used_articles=[article],
            used_reports=[],
            memory=UserMemory(),
            topics=[],
            brief_goal="Brief goal",
            date="2099-01-01",
            headline_context_articles=[],
            story_groups=[],
        )
        free_key = distiller._cache_key(
            used_articles=[article],
            used_reports=[],
            memory=UserMemory(),
            topics=[],
            brief_goal="Brief goal",
            date="2099-01-01",
            headline_context_articles=[],
            story_groups=None,
        )
        self.assertNotEqual(shared_key, free_key)

    def test_batching_preserves_supplied_group_boundaries_when_possible(self) -> None:
        articles = [
            _selected("a", "A", score=9.0),
            _selected("b", "B", score=8.9),
            _selected("c", "C", score=8.8),
            _selected("d", "D", score=8.7),
        ]
        groups = [
            StoryGroup("story-001", "First story", ["a", "b"], []),
            StoryGroup("story-002", "Second story", ["c", "d"], []),
        ]
        distiller = EvidenceDistiller(FakeAIClient([]), _config(max_articles_per_batch=2))

        batches = distiller._build_article_batches(
            articles,
            memory=UserMemory(),
            topics=[],
            prior_reports=[],
            brief_goal="Brief goal",
            date="2099-01-01",
            story_groups=groups,
        )

        self.assertEqual([[article.candidate.id for article in batch] for batch in batches], [["a", "b"], ["c", "d"]])


if __name__ == "__main__":
    unittest.main()
