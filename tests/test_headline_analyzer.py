from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.ai.prompts import HEADLINE_ANALYSIS_USER
from mydailynews.ai.schemas import HEADLINE_ANALYSIS_JSON_SCHEMA
from mydailynews.app.models import NewsCandidate


class _Client:
    def __init__(self, max_new_tokens: int = 2048) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = 4096
        self.config = SimpleNamespace(backend="test", effective_model_label="test", response_format="json_schema")


class HeadlineAnalyzerTests(unittest.TestCase):
    def test_compact_decision_keeps_every_consumed_score(self) -> None:
        properties = HEADLINE_ANALYSIS_JSON_SCHEMA.schema["properties"]["decisions"]["items"]["properties"]
        self.assertEqual(
            set(properties),
            {
                "id",
                "score",
                "personal_relevance",
                "impact",
                "novelty",
                "urgency",
                "actionability",
                "confidence",
                "angle_type",
            },
        )
        self.assertNotIn("`reason`", HEADLINE_ANALYSIS_USER)
        self.assertNotIn("`skip_reason`", HEADLINE_ANALYSIS_USER)

        analyzer = HeadlineAnalyzer(_Client(), batch_size=10)
        analyzer._reset_multifactor_stats()
        candidate = NewsCandidate(
            id="candidate-1",
            source="Example",
            category="world",
            title="Material policy change",
            url="https://example.test/1",
            snippet="",
            published_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        )
        decision = analyzer._parse_batch_result(
            {
                "decisions": [
                    {
                        "id": candidate.id,
                        "score": 8.0,
                        "personal_relevance": 8.1,
                        "impact": 8.2,
                        "novelty": 8.3,
                        "urgency": 8.4,
                        "actionability": 8.5,
                        "confidence": 8.6,
                        "angle_type": "policy_change",
                        "reason": "legacy prose must not survive",
                        "skip_reason": "legacy prose must not survive",
                    }
                ]
            },
            [candidate],
            [],
            "test",
            1,
            1,
        )[candidate.id]

        self.assertEqual(
            (
                decision.score,
                decision.personal_relevance,
                decision.impact,
                decision.novelty,
                decision.urgency,
                decision.actionability,
                decision.confidence,
                decision.angle_type,
            ),
            (8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, "policy_change"),
        )
        self.assertEqual(decision.reason, "")
        self.assertIsNone(decision.skip_reason)

    def test_output_budget_scales_to_batch_and_respects_both_ceilings(self) -> None:
        analyzer = HeadlineAnalyzer(_Client(2048), batch_size=20, max_new_tokens=1600)

        self.assertEqual(analyzer._headline_batch_max_new_tokens(1), 192)
        self.assertEqual(analyzer._headline_batch_max_new_tokens(10), 1344)
        self.assertEqual(analyzer._headline_batch_max_new_tokens(20), 1600)

        client_limited = HeadlineAnalyzer(
            _Client(1000),
            batch_size=20,
            max_new_tokens=3000,
            single_replay_max_new_tokens=3000,
        )
        self.assertEqual(client_limited._headline_batch_max_new_tokens(20), 1000)
        self.assertEqual(client_limited._headline_single_max_new_tokens(), 1000)


if __name__ == "__main__":
    unittest.main()
