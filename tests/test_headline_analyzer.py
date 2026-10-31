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
    def test_compact_decision_uses_three_independent_axes_and_code_priority(self) -> None:
        properties = HEADLINE_ANALYSIS_JSON_SCHEMA.schema["properties"]["decisions"]["items"]["properties"]
        self.assertEqual(
            set(properties),
            {
                "id",
                "novelty",
                "novelty_basis",
                "impact",
                "impact_basis",
                "urgency",
                "urgency_basis",
            },
        )
        self.assertIn("nominal common sense", HEADLINE_ANALYSIS_USER)
        self.assertIn("school shooting", HEADLINE_ANALYSIS_USER)
        self.assertNotIn('"score"', HEADLINE_ANALYSIS_USER)

        analyzer = HeadlineAnalyzer(_Client(), batch_size=10)
        analyzer._reset_axis_stats()
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
                        "novelty": 3,
                        "novelty_basis": "reverses established knowledge",
                        "impact": 2,
                        "impact_basis": "meaningful sector consequences",
                        "urgency": 1,
                        "urgency_basis": "safe to wait a day",
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
                decision.novelty,
                decision.impact,
                decision.urgency,
                decision.novelty_basis,
                decision.impact_basis,
                decision.urgency_basis,
            ),
            (
                6.5,
                10.0,
                6.6667,
                3.3333,
                "reverses established knowledge",
                "meaningful sector consequences",
                "safe to wait a day",
            ),
        )

    def test_priority_calculation_does_not_copy_one_axis(self) -> None:
        self.assertEqual(HeadlineAnalyzer.priority_score(novelty=3, impact=0, urgency=0), 2.5)
        self.assertEqual(HeadlineAnalyzer.priority_score(novelty=0, impact=3, urgency=0), 4.5)
        self.assertEqual(HeadlineAnalyzer.priority_score(novelty=0, impact=0, urgency=3), 3.0)

    def test_axis_metrics_expose_all_equal_model_behavior_without_forcing_difference(self) -> None:
        analyzer = HeadlineAnalyzer(_Client(), batch_size=10)
        analyzer._reset_axis_stats()
        analyzer._record_axis_row(raw={}, novelty=3.3333, impact=3.3333, urgency=3.3333)
        analyzer._record_axis_row(raw={}, novelty=0.0, impact=6.6667, urgency=10.0)
        analyzer._emit_axis_metrics()

        self.assertEqual(
            analyzer.debug.analytics_payload()["metrics"]["headline.axes.all_equal_ratio"],
            0.5,
        )

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
