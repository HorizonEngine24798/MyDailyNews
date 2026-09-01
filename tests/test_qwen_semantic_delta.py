from __future__ import annotations

import unittest

from mydailynews.ai.qwen_semantic_delta import QwenSemanticDeltaInferencer
from mydailynews.analysis.claim_delta import (
    assess_claim_comparison,
    build_claim_comparison,
    current_claim_evidence,
    prior_claim_evidence,
)


class _StaticClient:
    def __init__(self, pair_label="adds_specificity") -> None:
        self.prompts = []
        self.pair_label = pair_label

    def complete_json(self, system, user, **kwargs):
        self.prompts.append((system, user, kwargs))
        if kwargs["json_schema"].name == "compact_story_identity":
            return {
                "candidate": "s0",
                "relation": "same_event",
                "current_evidence": "c1",
                "prior_evidence": "p0",
            }
        return {
            key: self.pair_label
            for key in kwargs["json_schema"].schema["required"]
        }


class QwenSemanticDeltaAdapterTests(unittest.TestCase):
    def test_short_aliases_resolve_to_durable_evidence_ids(self) -> None:
        current = current_claim_evidence(
            article_id="current-report",
            title="Observatory adds a second sensor",
            text="The North Ridge Observatory added a second atmospheric sensor.",
        )
        prior = prior_claim_evidence(
            {
                "story_key": "north-ridge-observatory",
                "source_facts": [
                    {
                        "fact_id": "fact:prior-sensor",
                        "text": "The North Ridge Observatory operates an atmospheric sensor.",
                        "kind": "source_sentence",
                        "source_id": "prior-report",
                    }
                ],
            }
        )
        request = build_claim_comparison(current, prior)
        client = _StaticClient()
        inferencer = QwenSemanticDeltaInferencer(client)

        result = assess_claim_comparison(request, inferencer=inferencer)

        self.assertEqual(result.relationship, "same_story")
        self.assertEqual(result.change_type, "incremental")
        self.assertEqual(result.current_evidence_ids, [current[1].claim_id])
        self.assertEqual(result.prior_evidence_ids, [prior[0].claim_id])
        self.assertNotIn(current[1].claim_id, client.prompts[0][1])
        self.assertIn('"i":"c1"', client.prompts[0][1])
        self.assertIn('"i":"p0"', client.prompts[0][1])
        self.assertEqual(len(client.prompts), 2)
        schema = client.prompts[0][2]["json_schema"].schema
        self.assertEqual(schema["properties"]["current_evidence"]["enum"], ["c0", "c1"])
        self.assertEqual(schema["properties"]["prior_evidence"]["enum"], ["p0"])
        self.assertEqual(schema["properties"]["candidate"]["enum"], ["s0"])
        self.assertEqual(client.prompts[1][2]["json_schema"].name, "compact_pair_relations_1")

    def test_unchanged_requires_every_body_claim_and_excludes_headline(self) -> None:
        current = current_claim_evidence(
            article_id="current-report",
            title="Observatory operations update",
            text=(
                "The North Ridge Observatory operates an atmospheric sensor. "
                "The observatory publishes readings each morning."
            ),
        )
        prior = prior_claim_evidence(
            {
                "story_key": "north-ridge-observatory",
                "source_facts": [
                    {
                        "fact_id": "fact:prior-operations",
                        "text": "The North Ridge Observatory operates an atmospheric sensor.",
                        "kind": "source_sentence",
                        "source_id": "prior-report",
                    }
                ],
            }
        )
        request = build_claim_comparison(current, prior)
        client = _StaticClient(pair_label="equivalent")
        inferencer = QwenSemanticDeltaInferencer(client, max_pairs_per_current=1)

        result = assess_claim_comparison(request, inferencer=inferencer)

        self.assertEqual(result.relationship, "same_story")
        self.assertEqual(result.change_type, "unchanged")
        self.assertEqual(set(result.current_evidence_ids), {item.claim_id for item in current[1:]})
        self.assertNotIn(current[0].text, client.prompts[1][1])


if __name__ == "__main__":
    unittest.main()
