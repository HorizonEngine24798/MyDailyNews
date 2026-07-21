from __future__ import annotations

import unittest
from types import SimpleNamespace

from mydailynews.pipeline.perspectives_report import (
    _coverage_relevance_decision,
    collect_verification_documents,
    _normalize_framing_response,
    _retrieval_requests,
)


class PerspectivesRelevanceDiagnosticsTests(unittest.TestCase):
    def test_verification_empty_states_are_truthful(self) -> None:
        no_claims = collect_verification_documents(
            SimpleNamespace(),
            inputs={"stories": [{"story_id": "story-1", "claims": []}]},
            config=SimpleNamespace(verification_enabled=True),
            plans_by_story={},
            warnings=[],
        )[1]
        disabled = collect_verification_documents(
            SimpleNamespace(),
            inputs={"stories": [{"story_id": "story-1", "claims": [{"claim_id": "c1"}]}]},
            config=SimpleNamespace(verification_enabled=False),
            plans_by_story={},
            warnings=[],
        )[1]

        self.assertEqual(no_claims["status"], "no_claims")
        self.assertEqual(disabled["status"], "verification_not_requested")
        self.assertEqual(disabled["reason"], "disabled")

    def test_relevance_decision_explains_accept_and_reject(self) -> None:
        story = {"story_title": "Iran Hormuz blockade", "summary": "Shipping disruption in the Strait of Hormuz"}
        accepted = {
            "retrieval_query": "Iran Hormuz blockade",
            "title": "Iran extends Hormuz blockade",
            "snippet": "Shipping remains disrupted",
        }
        rejected = {
            "retrieval_query": "Iran Hormuz blockade",
            "title": "Election officials debate voter files",
            "snippet": "A domestic election security dispute",
        }

        self.assertEqual(_coverage_relevance_decision(accepted, story), (True, "query_title_overlap"))
        self.assertEqual(_coverage_relevance_decision(rejected, story), (False, "insufficient_event_overlap"))

    def test_query_rejection_and_reader_prose_cleanup_keep_structured_ids(self) -> None:
        plan = {
            "queries": ["Iran Hormuz blockade", "Election voter files"],
            "anchor_groups": [],
            "diagnostics": [],
        }
        requests = _retrieval_requests(
            plan,
            [{"country": "US", "source_id": "source"}],
            story={"story_title": "Iran Hormuz blockade", "summary": "Shipping disruption"},
        )
        warnings: list[str] = []
        framing = _normalize_framing_response(
            {
                "stories": [
                    {
                        "story_id": "story-1",
                        "synthesis": "Accounts agree [article_id:a1, article_id:a2].",
                        "synthesis_article_ids": ["a1", "a2"],
                        "shared_facts": [{"text": "The event occurred article_id:a1.", "article_ids": ["a1"]}],
                    }
                ]
            },
            known_story_ids=["story-1"],
            known_article_ids={"story-1": ["a1", "a2"]},
            warnings=warnings,
        )["story-1"]

        self.assertEqual([request["query"] for request in requests], ["Iran Hormuz blockade"])
        self.assertTrue(any("Election voter files" in item for item in plan["diagnostics"]))
        self.assertNotIn("article_id", framing["synthesis"])
        self.assertNotIn("a1", framing["shared_facts"][0]["text"])
        self.assertEqual(framing["synthesis_article_ids"], ["a1", "a2"])


if __name__ == "__main__":
    unittest.main()
