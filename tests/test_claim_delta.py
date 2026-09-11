from __future__ import annotations

import unittest

from mydailynews.analysis.claim_delta import (
    ClaimEvidence,
    FactOperationRequest,
    publication_policy_for_operations,
    validate_fact_operations,
)


class FactOperationContractTests(unittest.TestCase):
    def request(self, current: str, prior: str = "The bridge remains closed pending inspection."):
        return FactOperationRequest(
            story_key="bridge-thread",
            current_claims=(ClaimEvidence("current-1", current, "current", source_id="today"),),
            prior_claims=(ClaimEvidence("prior-1", prior, "prior", story_key="bridge-thread", source_id="yesterday"),),
            article_ids=("today",),
        )

    def validate(self, request, operation, prior_ids=()):
        return validate_fact_operations(
            {
                "story_key": request.story_key,
                "operations": [{
                    "operation": operation,
                    "current_evidence_id": "current-1",
                    "prior_fact_ids": list(prior_ids),
                }],
            },
            request,
        )

    def test_exact_or_weaker_restatement_is_a_repeat(self) -> None:
        for text in (
            "The bridge remains closed pending inspection.",
            "The bridge remains closed.",
        ):
            request = self.request(text)
            validation = self.validate(request, "repeat", ("prior-1",))
            self.assertTrue(validation.safe_to_apply)
            self.assertEqual(publication_policy_for_operations(validation, request), "no_change")

    def test_additive_detail_keeps_prior_state(self) -> None:
        request = self.request("Inspectors published a new structural analysis.")
        validation = self.validate(request, "add")
        self.assertTrue(validation.safe_to_apply)
        self.assertEqual(validation.operations[0].current_evidence_id, "current-1")
        self.assertEqual(publication_policy_for_operations(validation, request), "incremental")

    def test_structurally_valid_replace_cites_its_target(self) -> None:
        request = self.request(
            "The agency corrected the notice: the bridge reopened Tuesday.",
            "The agency said the bridge would remain closed through Tuesday.",
        )
        validation = self.validate(request, "replace", ("prior-1",))
        self.assertTrue(validation.safe_to_apply)
        self.assertEqual(validation.operations[0].prior_fact_ids, ("prior-1",))
        self.assertEqual(publication_policy_for_operations(validation, request), "materiality_candidate")

    def test_validator_does_not_reinterpret_model_semantics(self) -> None:
        request = self.request(
            "The council enacted the proposal on Tuesday.",
            "The council proposed the measure on Monday.",
        )
        validation = self.validate(request, "replace", ("prior-1",))
        self.assertTrue(validation.safe_to_apply)
        self.assertEqual(validation.operations[0].prior_fact_ids, ("prior-1",))

    def test_semantic_resolution_does_not_require_a_cue_word(self) -> None:
        request = self.request(
            "The injunction was lifted.",
            "The injunction remains in force pending appeal.",
        )
        validation = self.validate(request, "resolve", ("prior-1",))
        self.assertTrue(validation.safe_to_apply)
        self.assertEqual(validation.operations[0].prior_fact_ids, ("prior-1",))

    def test_unknown_cross_story_ids_and_conflicts_fail_open(self) -> None:
        request = self.request("The bridge remains closed.")
        unknown = validate_fact_operations({
            "story_key": "bridge-thread",
            "operations": [{
                "operation": "repeat",
                "current_evidence_id": "current-1",
                "prior_fact_ids": ["another-story-fact"],
            }],
        }, request)
        self.assertFalse(unknown.safe_to_apply)
        self.assertFalse(unknown.structurally_valid_references)

        cross_story = FactOperationRequest(
            story_key=request.story_key,
            current_claims=request.current_claims,
            prior_claims=(ClaimEvidence("prior-1", "A different bridge closed.", "prior", story_key="other-thread", source_id="other"),),
        )
        crossed = self.validate(cross_story, "repeat", ("prior-1",))
        self.assertFalse(crossed.safe_to_apply)
        self.assertFalse(crossed.structurally_valid_references)

        conflicting = validate_fact_operations({
            "story_key": "bridge-thread",
            "operations": [
                {"operation": "repeat", "current_evidence_id": "current-1", "prior_fact_ids": ["prior-1"]},
                {"operation": "add", "current_evidence_id": "current-1", "prior_fact_ids": []},
            ],
        }, request)
        self.assertFalse(conflicting.safe_to_apply)
        self.assertEqual(conflicting.operations, ())

    def test_malformed_or_uncertain_output_fails_open(self) -> None:
        request = self.request("The bridge status is unclear.")
        self.assertFalse(validate_fact_operations({}, request).safe_to_apply)
        uncertain = self.validate(request, "uncertain")
        self.assertFalse(uncertain.safe_to_apply)
        self.assertEqual(publication_policy_for_operations(uncertain, request), "visible_fail_open")


if __name__ == "__main__":
    unittest.main()
