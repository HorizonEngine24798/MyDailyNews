from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from types import SimpleNamespace
import unittest

from mydailynews.analysis.story_delta import (
    MAX_EDITOR_FULL_REPORTS,
    StoryDeltaAnalyzer,
    _validated_editor_decisions,
)
from mydailynews.app.models import (
    DeltaExtractionConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    UserMemory,
)


def _card(index: int) -> dict:
    return {
        "card_id": f"story-card-{index:03d}",
        "title": f"Story {index}",
        "operation_validation": {"safe_to_apply": True},
        "editorial_signals": [{
            "priority": float(index),
            "novelty": float(index),
            "impact": float(index),
            "urgency": float(index),
        }],
    }


def _article(index: int) -> SelectedArticle:
    candidate_id = f"article-{index}"
    body = (f"Source-backed detail for story {index}. " * 80).strip()
    return SelectedArticle(
        candidate=NewsCandidate(
            id=candidate_id,
            source="Example",
            category="world",
            title=f"Story {index}: " + ("important development " * 20),
            url=f"https://example.test/{candidate_id}",
            snippet=body,
            published_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        ),
        decision=HeadlineDecision(
            candidate_id,
            score=float(index),
            novelty=float(index),
            impact=float(index),
            urgency=float(index),
        ),
        article_text=body,
        extraction_status="ok",
    )


class _BudgetClient:
    max_input_tokens = 700
    max_new_tokens = 256
    config = SimpleNamespace(
        backend="test",
        effective_model_label="test",
        context_window_tokens=4096,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )

    def __init__(self) -> None:
        self.calls = 0
        self.last_user = ""
        self.last_request_tokens = 0
        self.submitted_card_ids: list[str] = []

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def complete_json(self, system: str, user: str, **kwargs):
        self.calls += 1
        request = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
        self.last_request_tokens = self.estimate_tokens(request)
        if self.last_request_tokens > int(kwargs["input_token_limit"]):
            raise AssertionError("editor prompt reached the client over budget")
        self.last_user = user
        card_json = user.split("Validated story cards:\n", 1)[1].split(
            "\n\nReturn exactly one decision",
            1,
        )[0]
        self.submitted_card_ids = [str(card["card_id"]) for card in json.loads(card_json)]
        return {
            "decisions": [
                {
                    "card_id": card_id,
                    "disposition": "full_report",
                    "materiality": 3,
                    "summary": f"Supported delta for {card_id}.",
                    "basis": "High comparative importance.",
                }
                for card_id in self.submitted_card_ids
            ]
        }


class StoryDeltaEditorTests(unittest.TestCase):
    def test_more_than_five_full_reports_are_downgraded_by_priority(self) -> None:
        cards = [_card(index) for index in range(1, 8)]
        raw = {
            "decisions": [
                {
                    "card_id": card["card_id"],
                    "disposition": "full_report",
                    "materiality": 3,
                    "summary": card["title"],
                    "basis": "Model selected full coverage.",
                }
                for card in cards
            ]
        }

        decisions = _validated_editor_decisions(raw, cards)

        full_ids = {
            decision["card_id"]
            for decision in decisions
            if decision["disposition"] == "full_report"
        }
        self.assertEqual(len(full_ids), MAX_EDITOR_FULL_REPORTS)
        self.assertEqual(full_ids, {f"story-card-{index:03d}" for index in range(3, 8)})
        self.assertTrue(all(
            decision["disposition"] == "continuing_bullet"
            for decision in decisions[:2]
        ))

    def test_long_cards_are_bounded_before_client_and_all_receive_decisions(self) -> None:
        articles = [_article(index) for index in range(1, 9)]
        story_memory = {
            "stories": [
                {
                    "story_key": f"story-{index}",
                    "current_title": article.candidate.title,
                    "current_article_ids": [article.candidate.id],
                    "prior_baselines": [],
                }
                for index, article in enumerate(articles, start=1)
            ]
        }
        client = _BudgetClient()
        analyzer = StoryDeltaAnalyzer(
            client,
            DeltaExtractionConfig(
                enabled=True,
                max_input_tokens=client.max_input_tokens,
                max_new_tokens=client.max_new_tokens,
                max_articles=len(articles),
                max_article_chars=120,
            ),
        )

        packet = analyzer.extract(
            articles,
            UserMemory(),
            "Rank the most important current developments.",
            story_memory,
        )

        self.assertEqual(client.calls, 1)
        self.assertGreater(len(client.submitted_card_ids), 0)
        self.assertLess(len(client.submitted_card_ids), len(articles))
        self.assertLessEqual(client.last_request_tokens, int(client.max_input_tokens * 0.95))
        self.assertIn("Use full_report for at most five cards", client.last_user)
        self.assertIn("basis explains the coverage choice", client.last_user)
        self.assertEqual(len(packet["story_decisions"]), len(articles))
        self.assertEqual(
            sum(row["disposition"] == "full_report" for row in packet["story_decisions"]),
            MAX_EDITOR_FULL_REPORTS,
        )
        self.assertEqual(
            {
                row["article_ids"][0]
                for row in packet["story_decisions"]
                if row["disposition"] == "full_report"
            },
            {f"article-{index}" for index in range(4, 9)},
        )
        self.assertEqual(
            {row["article_ids"][0] for row in packet["story_decisions"]},
            {article.candidate.id for article in articles},
        )
        self.assertTrue(any("omitted" in warning for warning in analyzer.warnings))
        self.assertEqual(
            set(client.submitted_card_ids),
            set(re.findall(r"story-card-\d{3}", client.last_user)),
        )

    def test_multi_article_cards_do_not_starve_subsequent_cards_from_editor_prompt(self) -> None:
        articles = [_article(index) for index in range(1, 16)]
        story_memory = {
            "stories": [
                {
                    "story_key": "story-1",
                    "current_title": "Story 1",
                    "current_article_ids": [f"article-{i}" for i in range(1, 8)],
                    "prior_baselines": [],
                },
                {
                    "story_key": "story-2",
                    "current_title": "Story 2",
                    "current_article_ids": [f"article-{i}" for i in range(8, 13)],
                    "prior_baselines": [],
                },
                {
                    "story_key": "story-3",
                    "current_title": "Story 3",
                    "current_article_ids": ["article-13"],
                    "prior_baselines": [],
                },
                {
                    "story_key": "story-4",
                    "current_title": "Story 4",
                    "current_article_ids": ["article-14"],
                    "prior_baselines": [],
                },
                {
                    "story_key": "story-5",
                    "current_title": "Story 5",
                    "current_article_ids": ["article-15"],
                    "prior_baselines": [],
                },
            ]
        }
        client = _BudgetClient()
        client.max_input_tokens = 14000
        client.config.max_input_tokens = 14000
        analyzer = StoryDeltaAnalyzer(
            client,
            DeltaExtractionConfig(
                enabled=True,
                max_input_tokens=14000,
                max_new_tokens=512,
                max_articles=12,
                max_article_chars=400,
            ),
        )

        packet = analyzer.extract(
            articles,
            UserMemory(),
            "Rank the most important current developments.",
            story_memory,
        )

        self.assertEqual(len(client.submitted_card_ids), 5)
        self.assertEqual(analyzer.warnings, [])
        self.assertFalse(
            any("Fail-open coverage" in row["reason"] for row in packet["story_decisions"])
        )

    def test_verified_unchanged_card_is_programmatically_omitted_even_if_editor_promotes_it(self) -> None:
        cards = [
            {
                **_card(1),
                "change_type": "unchanged",
                "operation_validation": {"safe_to_apply": True},
            },
            {
                **_card(2),
                "change_type": "unchanged",
                "operation_validation": {"safe_to_apply": False},
            },
        ]
        raw = {
            "decisions": [
                {
                    "card_id": "story-card-001",
                    "disposition": "continuing_bullet",
                    "materiality": 1,
                    "summary": "Repeated coverage.",
                    "basis": "Editor kept it as a bullet despite zero new facts.",
                },
                {
                    "card_id": "story-card-002",
                    "disposition": "omit",
                    "materiality": 0,
                    "summary": "Unverified repeat.",
                    "basis": "Editor tried to omit an unvalidated card.",
                },
            ]
        }

        decisions = _validated_editor_decisions(raw, cards)

        self.assertEqual(decisions[0]["disposition"], "omit")
        self.assertEqual(decisions[0]["materiality"], 0)
        self.assertIn("Programmatically omitted", decisions[0]["basis"])
        self.assertEqual(decisions[1]["disposition"], "full_report")

    def test_high_confidence_baseline_links_continuing_story_when_add_operations_omit_prior_fact_ids(self) -> None:
        article = _article(1)
        story_memory = {
            "stories": [
                {
                    "story_key": "us-blockade-middle-east-tightens",
                    "current_title": "US blockade in Middle East tightens",
                    "current_article_ids": [article.candidate.id],
                    "prior_baselines": [
                        {
                            "story_key": "iran-war-hormuz-crisis",
                            "title": "Iran war Hormuz shipping crisis",
                            "candidate_score": 0.65,
                            "candidate_signals": {"numeric_conflict": False},
                            "source_facts": [
                                {
                                    "fact_id": "fact:hormuz-1",
                                    "text": "Tanker traffic halted in the Strait of Hormuz.",
                                    "source_id": "prior-1",
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        class _AddWithoutPriorIdsClient:
            config = SimpleNamespace(backend="test", effective_model_label="test")

            def complete_json(self, system, user, **kwargs):
                if kwargs["json_schema"].name == "fact_operations":
                    comparison = json.loads(
                        user.split("Story comparison:\n", 1)[1].split("\n\nReturn", 1)[0]
                    )
                    return {
                        "story_key": comparison["story_key"],
                        "operations": [
                            {
                                "operation": "add",
                                "current_evidence_id": claim["claim_id"],
                                "prior_fact_ids": [],
                            }
                            for claim in comparison["current_evidence"]
                        ],
                    }
                return {
                    "decisions": [
                        {
                            "card_id": "story-card-001",
                            "disposition": "full_report",
                            "materiality": 2,
                            "summary": "US blockade tightened around the Strait of Hormuz.",
                            "basis": "Substantive escalation of the continuing crisis.",
                        }
                    ]
                }

        packet = StoryDeltaAnalyzer(
            _AddWithoutPriorIdsClient(),
            DeltaExtractionConfig(enabled=True),
        ).extract(
            [article],
            UserMemory(),
            "Track continuing geopolitical developments.",
            story_memory,
        )

        decision = packet["story_decisions"][0]
        self.assertEqual(decision["story_key"], "iran-war-hormuz-crisis")
        self.assertEqual(decision["change_type"], "incremental")


if __name__ == "__main__":
    unittest.main()
