from __future__ import annotations

import unittest

from mydailynews.analysis.claim_delta import (
    ClaimComparisonRequest,
    ClaimEvidence,
    SemanticClaimRelation,
    SemanticDeltaDecision,
    assess_claim_comparison,
    assess_claim_delta,
    build_claim_comparison,
    current_claim_evidence,
    prior_claim_evidence,
    validate_semantic_decision,
)
from mydailynews.analysis.deterministic_delta import merge_claim_delta_with_model


def _request():
    current = current_claim_evidence(
        article_id="today-observatory",
        title="Observatory changes calibration interval",
        text="The North Ridge Observatory will calibrate its primary sensor every six hours.",
        source_name="North Ridge Bulletin",
        source_url="https://example.test/current",
        published_at="2031-02-04T10:00:00Z",
    )
    prior = prior_claim_evidence(
        {
            "story_key": "north-ridge-sensor",
            "source_facts": [
                {
                    "fact_id": "fact:prior-calibration",
                    "text": "The North Ridge Observatory calibrated its primary sensor once per day.",
                    "kind": "source_sentence",
                    "source_id": "yesterday-observatory",
                    "source": "North Ridge Bulletin",
                    "url": "https://example.test/prior",
                    "published_at": "2031-02-03T10:00:00Z",
                    "observed_at": "2031-02-03",
                }
            ],
        }
    )
    return build_claim_comparison(current, prior)


def _scaffold_packet(request):
    assessment = assess_claim_comparison(request)
    return {
        "story_decisions": [
            {
                "story_key": "current-provisional-key",
                "article_ids": ["today-observatory"],
                "prior_story_key": "",
                "relationship": assessment.relationship,
                "change_type": assessment.change_type,
                "materiality": assessment.materiality,
                "confidence": assessment.confidence,
                "disposition": assessment.disposition,
                "summary": assessment.summary,
                "bullet": "Observatory changes calibration interval",
                "reason": assessment.summary,
                "knowns": assessment.added_claims,
                "unknowns": [],
                "watch_signals": [],
                "claim_delta": {
                    **request.payload(),
                    "decision_basis": assessment.decision_basis,
                    "requires_semantic_inference": assessment.requires_semantic_inference,
                    "added_claims": assessment.added_claims,
                    "repeated_claims": assessment.repeated_claims,
                    "superseded_claims": [],
                },
            }
        ],
        "evidence_gaps": [],
    }


class ClaimDeltaEvidenceContractTests(unittest.TestCase):
    def test_weaker_restatement_can_ground_unchanged(self) -> None:
        request = _request()
        current_id = next(
            claim.claim_id for claim in request.current_claims
            if claim.kind != "headline"
        )
        prior_id = request.prior_claims[0].claim_id
        decision = SemanticDeltaDecision(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.8,
            disposition="omit",
            summary="The current report makes a weaker version of the prior claim.",
            prior_story_key="north-ridge-sensor",
            current_evidence_ids=(current_id,),
            prior_evidence_ids=(prior_id,),
            claim_relations=(
                SemanticClaimRelation(
                    current_claim_id=current_id,
                    prior_claim_id=prior_id,
                    relation="weaker_restatement",
                    current_entails_prior="no",
                    prior_entails_current="yes",
                ),
            ),
        )

        validated, errors = validate_semantic_decision(decision, request)

        self.assertIs(validated, decision)
        self.assertEqual(errors, [])

    def test_weaker_restatement_requires_one_way_entailment(self) -> None:
        request = _request()
        current_id = next(
            claim.claim_id for claim in request.current_claims
            if claim.kind != "headline"
        )
        prior_id = request.prior_claims[0].claim_id
        decision = SemanticDeltaDecision(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.8,
            disposition="omit",
            summary="This edge uses the wrong entailment direction.",
            prior_story_key="north-ridge-sensor",
            current_evidence_ids=(current_id,),
            prior_evidence_ids=(prior_id,),
            claim_relations=(
                SemanticClaimRelation(
                    current_claim_id=current_id,
                    prior_claim_id=prior_id,
                    relation="weaker_restatement",
                    current_entails_prior="yes",
                    prior_entails_current="yes",
                ),
            ),
        )

        validated, errors = validate_semantic_decision(decision, request)

        self.assertIsNone(validated)
        self.assertIn(
            "weaker restatement requires prior-to-current entailment only",
            errors,
        )

    def test_unchanged_requires_coverage_of_every_body_claim(self) -> None:
        current = current_claim_evidence(
            article_id="current",
            title="Observatory publishes an update",
            text=(
                "The observatory continues its nightly survey of the northern sky. "
                "The observatory also released a newly calibrated image archive."
            ),
        )
        prior = prior_claim_evidence(
            {
                "story_key": "observatory-survey",
                "source_facts": [
                    {
                        "fact_id": "fact:prior-survey",
                        "text": "The observatory conducts a nightly northern-sky survey.",
                    }
                ],
            }
        )
        request = build_claim_comparison(current, prior)
        body_ids = [
            claim.claim_id for claim in request.current_claims
            if claim.kind != "headline"
        ]
        prior_id = request.prior_claims[0].claim_id
        decision = SemanticDeltaDecision(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.8,
            disposition="omit",
            summary="Only one of two current source claims was considered.",
            prior_story_key="observatory-survey",
            current_evidence_ids=(body_ids[0],),
            prior_evidence_ids=(prior_id,),
            claim_relations=(
                SemanticClaimRelation(
                    current_claim_id=body_ids[0],
                    prior_claim_id=prior_id,
                    relation="equivalent",
                    current_entails_prior="yes",
                    prior_entails_current="yes",
                ),
            ),
        )

        validated, errors = validate_semantic_decision(decision, request)

        self.assertIsNone(validated)
        self.assertIn("unchanged must cite every current source claim", errors)

    def test_unchanged_allows_non_substantive_edges_beside_equivalence(self) -> None:
        current = current_claim_evidence(
            article_id="current",
            title="Observatory publishes an update",
            text=(
                "The observatory continues its nightly survey of the northern sky. "
                "The notice repeats the publication date in its footer."
            ),
        )
        prior = prior_claim_evidence(
            {
                "story_key": "observatory-survey",
                "source_facts": [
                    {
                        "fact_id": "fact:prior-survey",
                        "text": "The observatory conducts a nightly northern-sky survey.",
                    }
                ],
            }
        )
        request = build_claim_comparison(current, prior)
        body_ids = [
            claim.claim_id for claim in request.current_claims
            if claim.kind != "headline"
        ]
        prior_id = request.prior_claims[0].claim_id
        decision = SemanticDeltaDecision(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.8,
            disposition="omit",
            summary="All substantive content is unchanged.",
            prior_story_key="observatory-survey",
            current_evidence_ids=tuple(body_ids),
            prior_evidence_ids=(prior_id,),
            claim_relations=(
                SemanticClaimRelation(
                    current_claim_id=body_ids[0],
                    prior_claim_id=prior_id,
                    relation="equivalent",
                    current_entails_prior="yes",
                    prior_entails_current="yes",
                ),
                SemanticClaimRelation(
                    current_claim_id=body_ids[1],
                    prior_claim_id=prior_id,
                    relation="non_substantive",
                ),
            ),
        )

        validated, errors = validate_semantic_decision(decision, request)

        self.assertIs(validated, decision)
        self.assertEqual(errors, [])

    def test_unchanged_rejects_new_fact_in_story_edge(self) -> None:
        request = _request()
        current_id = next(
            claim.claim_id for claim in request.current_claims
            if claim.kind != "headline"
        )
        prior_id = request.prior_claims[0].claim_id
        decision = SemanticDeltaDecision(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.8,
            disposition="omit",
            summary="A new fact cannot be classified as unchanged.",
            prior_story_key="north-ridge-sensor",
            current_evidence_ids=(current_id,),
            prior_evidence_ids=(prior_id,),
            claim_relations=(
                SemanticClaimRelation(
                    current_claim_id=current_id,
                    prior_claim_id=prior_id,
                    relation="new_fact_in_story",
                ),
            ),
        )

        validated, errors = validate_semantic_decision(decision, request)

        self.assertIsNone(validated)
        self.assertIn("unchanged contains a relation that denotes change", errors)

    def test_new_fact_in_story_can_ground_a_material_update(self) -> None:
        request = _request()
        current_id = next(
            claim.claim_id for claim in request.current_claims
            if claim.kind != "headline"
        )
        prior_id = request.prior_claims[0].claim_id
        decision = SemanticDeltaDecision(
            relationship="same_story",
            change_type="material_update",
            materiality=0.8,
            confidence=0.8,
            disposition="full_report",
            summary="The report adds a new fact within the selected story.",
            prior_story_key="north-ridge-sensor",
            current_evidence_ids=(current_id,),
            prior_evidence_ids=(prior_id,),
            claim_relations=(
                SemanticClaimRelation(
                    current_claim_id=current_id,
                    prior_claim_id=prior_id,
                    relation="new_fact_in_story",
                ),
            ),
        )

        validated, errors = validate_semantic_decision(decision, request)

        self.assertIs(validated, decision)
        self.assertEqual(errors, [])

    def test_unchanged_requires_all_current_claims_when_only_headlines_exist(self) -> None:
        request = ClaimComparisonRequest(
            current_claims=(
                ClaimEvidence(
                    claim_id="current-headline-one",
                    text="First current headline.",
                    side="current",
                    kind="headline",
                ),
                ClaimEvidence(
                    claim_id="current-headline-two",
                    text="Second current headline.",
                    side="current",
                    kind="headline",
                ),
            ),
            prior_claims=(
                ClaimEvidence(
                    claim_id="prior-headline",
                    text="Prior headline.",
                    side="prior",
                    kind="headline",
                    story_key="headline-story",
                ),
            ),
        )
        decision = SemanticDeltaDecision(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.8,
            disposition="omit",
            summary="Only one of the current headlines was assessed.",
            prior_story_key="headline-story",
            current_evidence_ids=("current-headline-one",),
            prior_evidence_ids=("prior-headline",),
            claim_relations=(
                SemanticClaimRelation(
                    current_claim_id="current-headline-one",
                    prior_claim_id="prior-headline",
                    relation="equivalent",
                    current_entails_prior="yes",
                    prior_entails_current="yes",
                ),
            ),
        )

        validated, errors = validate_semantic_decision(decision, request)

        self.assertIsNone(validated)
        self.assertIn("unchanged must cite every current source claim", errors)

    def test_action_words_do_not_create_transition_labels(self) -> None:
        result = assess_claim_delta(
            "Institute approves a new archive label",
            "The institute approved a blue archive label for its specimen drawers.",
            ["The institute catalogued specimen drawers using numeric archive labels."],
        )

        self.assertEqual(result.relationship, "uncertain")
        self.assertEqual(result.change_type, "uncertain")
        self.assertEqual(result.disposition, "full_report")
        self.assertTrue(result.requires_semantic_inference)

    def test_exact_source_proposition_is_the_only_structural_unchanged_case(self) -> None:
        first = "The ferry inspection is scheduled for Tuesday at 09:00."
        second = "The inspection notice names Pier Seven as the meeting point."
        result = assess_claim_delta(
            "Ferry inspection schedule",
            f"{first} {second}",
            [first, second],
        )

        self.assertEqual(result.relationship, "same_story")
        self.assertEqual(result.change_type, "unchanged")
        self.assertEqual(result.disposition, "omit")
        self.assertEqual(result.decision_basis, "exact_repetition")
        self.assertTrue(result.current_evidence_ids)
        self.assertTrue(result.prior_evidence_ids)

    def test_one_generic_repeated_sentence_is_not_enough_to_suppress(self) -> None:
        generic = "The department said that further information will be released later."
        result = assess_claim_delta(
            "Department issues an afternoon notice",
            generic,
            [generic],
        )

        self.assertEqual(result.change_type, "uncertain")
        self.assertEqual(result.disposition, "full_report")

    def test_injected_semantic_backend_is_model_agnostic_and_grounded(self) -> None:
        request = _request()
        current_id = next(claim.claim_id for claim in request.current_claims if claim.kind != "headline")
        prior_id = request.prior_claims[0].claim_id

        class StaticInferencer:
            def compare(self, supplied):
                self.assert_request = supplied
                return SemanticDeltaDecision(
                    relationship="same_story",
                    change_type="status_change",
                    materiality=0.8,
                    confidence=0.81,
                    disposition="full_report",
                    summary="The calibration interval changed from daily to six-hourly.",
                    prior_story_key="north-ridge-sensor",
                    current_evidence_ids=(current_id,),
                    prior_evidence_ids=(prior_id,),
                    superseded_prior_evidence_ids=(prior_id,),
                    claim_relations=(
                        SemanticClaimRelation(
                            current_claim_id=current_id,
                            prior_claim_id=prior_id,
                            relation="supersedes",
                            current_entails_prior="no",
                            prior_entails_current="no",
                        ),
                    ),
                )

        inferencer = StaticInferencer()
        result = assess_claim_comparison(request, inferencer=inferencer)

        self.assertIs(inferencer.assert_request, request)
        self.assertEqual(result.change_type, "status_change")
        self.assertEqual(result.decision_basis, "semantic_inference")
        self.assertEqual(result.superseded_claims, [request.prior_claims[0].text])
        self.assertEqual(result.claim_relations[0]["relation"], "supersedes")

    def test_unknown_evidence_reference_forces_abstention(self) -> None:
        request = _request()

        class HallucinatingInferencer:
            def compare(self, supplied):
                return SemanticDeltaDecision(
                    relationship="same_story",
                    change_type="resolved",
                    materiality=1.0,
                    confidence=0.99,
                    disposition="omit",
                    summary="Unsupported conclusion.",
                    prior_story_key="north-ridge-sensor",
                    current_evidence_ids=("invented-current-id",),
                    prior_evidence_ids=("invented-prior-id",),
                )

        result = assess_claim_comparison(request, inferencer=HallucinatingInferencer())

        self.assertEqual(result.change_type, "uncertain")
        self.assertEqual(result.disposition, "full_report")
        self.assertTrue(result.requires_semantic_inference)

    def test_evidence_from_one_candidate_cannot_link_another_candidate(self) -> None:
        request = _request()
        current_id = next(claim.claim_id for claim in request.current_claims if claim.kind != "headline")
        prior_id = request.prior_claims[0].claim_id

        class CrossCandidateInferencer:
            def compare(self, supplied):
                return SemanticDeltaDecision(
                    relationship="same_story",
                    change_type="material_update",
                    materiality=0.8,
                    confidence=0.9,
                    disposition="full_report",
                    summary="Claims were linked to the wrong candidate key.",
                    prior_story_key="different-candidate",
                    current_evidence_ids=(current_id,),
                    prior_evidence_ids=(prior_id,),
                    claim_relations=(
                        SemanticClaimRelation(
                            current_claim_id=current_id,
                            prior_claim_id=prior_id,
                            relation="adds_detail",
                            current_entails_prior="yes",
                            prior_entails_current="no",
                        ),
                    ),
                )

        result = assess_claim_comparison(request, inferencer=CrossCandidateInferencer())

        self.assertEqual(result.relationship, "uncertain")
        self.assertEqual(result.change_type, "uncertain")

    def test_context_only_edge_cannot_establish_a_same_story_change(self) -> None:
        request = _request()
        current_id = next(claim.claim_id for claim in request.current_claims if claim.kind != "headline")
        prior_id = request.prior_claims[0].claim_id

        class ContextOnlyInferencer:
            def compare(self, supplied):
                return SemanticDeltaDecision(
                    relationship="same_story",
                    change_type="status_change",
                    materiality=0.8,
                    confidence=0.9,
                    disposition="full_report",
                    summary="A label is not supported by the cited relation.",
                    prior_story_key="north-ridge-sensor",
                    current_evidence_ids=(current_id,),
                    prior_evidence_ids=(prior_id,),
                    claim_relations=(
                        SemanticClaimRelation(
                            current_claim_id=current_id,
                            prior_claim_id=prior_id,
                            relation="context_only",
                        ),
                    ),
                )

        result = assess_claim_comparison(request, inferencer=ContextOnlyInferencer())

        self.assertEqual(result.relationship, "uncertain")
        self.assertTrue(result.requires_semantic_inference)

    def test_every_cited_claim_must_participate_in_an_edge(self) -> None:
        request = _request()
        current_ids = tuple(claim.claim_id for claim in request.current_claims)
        prior_id = request.prior_claims[0].claim_id

        class PartlyGroundedInferencer:
            def compare(self, supplied):
                return SemanticDeltaDecision(
                    relationship="same_story",
                    change_type="material_update",
                    materiality=0.8,
                    confidence=0.8,
                    disposition="full_report",
                    summary="One cited claim has no semantic edge.",
                    prior_story_key="north-ridge-sensor",
                    current_evidence_ids=current_ids,
                    prior_evidence_ids=(prior_id,),
                    claim_relations=(
                        SemanticClaimRelation(
                            current_claim_id=current_ids[0],
                            prior_claim_id=prior_id,
                            relation="supports",
                            current_entails_prior="yes",
                            prior_entails_current="uncertain",
                        ),
                    ),
                )

        result = assess_claim_comparison(request, inferencer=PartlyGroundedInferencer())

        self.assertEqual(result.relationship, "uncertain")
        self.assertTrue(result.requires_semantic_inference)

    def test_merge_accepts_only_evidence_validated_model_decision(self) -> None:
        request = _request()
        current_id = next(claim.claim_id for claim in request.current_claims if claim.kind != "headline")
        prior_id = request.prior_claims[0].claim_id
        model_packet = {
            "story_decisions": [
                {
                    "article_ids": ["today-observatory"],
                    "prior_story_key": "north-ridge-sensor",
                    "relationship": "same_story",
                    "change_type": "status_change",
                    "materiality": 0.8,
                    "confidence": 0.82,
                    "disposition": "full_report",
                    "summary": "The calibration interval changed.",
                    "current_evidence_ids": [current_id],
                    "prior_evidence_ids": [prior_id],
                    "superseded_prior_evidence_ids": [prior_id],
                    "claim_relations": [
                        {
                            "current_claim_id": current_id,
                            "prior_claim_id": prior_id,
                            "relation": "supersedes",
                            "current_entails_prior": "no",
                            "prior_entails_current": "no",
                        }
                    ],
                }
            ]
        }

        merged = merge_claim_delta_with_model(_scaffold_packet(request), model_packet)

        decision = merged["story_decisions"][0]
        self.assertEqual(decision["change_type"], "status_change")
        self.assertEqual(decision["semantic_validation"]["status"], "accepted")
        self.assertEqual(merged["semantic_engine"]["model_used_for_uncertain"], 1)

    def test_merge_rejects_label_only_model_output(self) -> None:
        request = _request()
        model_packet = {
            "story_decisions": [
                {
                    "article_ids": ["today-observatory"],
                    "prior_story_key": "north-ridge-sensor",
                    "relationship": "same_story",
                    "change_type": "status_change",
                    "materiality": 1.0,
                    "confidence": 0.99,
                    "disposition": "omit",
                    "summary": "A label without evidence.",
                }
            ]
        }

        merged = merge_claim_delta_with_model(_scaffold_packet(request), model_packet)

        decision = merged["story_decisions"][0]
        self.assertEqual(decision["change_type"], "uncertain")
        self.assertEqual(decision["disposition"], "full_report")
        self.assertEqual(decision["semantic_validation"]["status"], "rejected")
        self.assertEqual(merged["semantic_engine"]["model_decisions_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
