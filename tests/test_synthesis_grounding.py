from __future__ import annotations

from unittest import TestCase

from mydailynews.perspectives.grounding import verify_synthesis_grounding
from mydailynews.perspectives.synthesis import build_framing_comparisons


class FakeAIClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.max_input_tokens = 8000
        self.max_new_tokens = 1800
        self.calls: list[dict] = []

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def complete_json(self, system: str, user: str, label: str = "ai.complete_json", **kwargs) -> dict:
        self.calls.append({"system": system, "user": user, "label": label})
        if not self.responses:
            raise AssertionError(f"unexpected AI call: {label}")
        return self.responses.pop(0)


class TestSynthesisGrounding(TestCase):
    def setUp(self) -> None:
        self.articles_by_id = {
            "art-us-01": {
                "article_id": "art-us-01",
                "source_name": "US Daily",
                "source_country": "US",
                "title": "US signs historic green energy pact at global summit",
                "context_text": "The United States agreed to a landmark renewable energy treaty today.",
            },
            "art-fr-01": {
                "article_id": "art-fr-01",
                "source_name": "Le Figaro",
                "source_country": "FR",
                "title": "Accord historique sur le climat adopté à Paris",
                "context_text": "La France et ses partenaires ont adopté l'accord sur les énergies renouvelables.",
            },
        }
        self.story = {
            "story_id": "story-001",
            "story_title": "Global green energy treaty",
            "summary": "Nations agreed to an energy transition pact.",
            "articles": list(self.articles_by_id.values()),
        }

    def test_prunes_hallucinated_article_ids(self) -> None:
        raw_report = {
            "synthesis": "Broad agreement was reached on the climate pact.",
            "synthesis_article_ids": ["art-us-01", "ghost-article-999"],
            "shared_facts": [
                {"text": "Delegates voted in favor", "article_ids": ["art-us-01", "phantom-02"]}
            ],
            "country_source_comparison": [],
            "coverage_limitations": [],
        }
        warnings: list[str] = []
        grounded = verify_synthesis_grounding(
            raw_report,
            story=self.story,
            articles_by_id=self.articles_by_id,
            warnings=warnings,
        )

        # Hallucinated IDs must be dropped
        self.assertEqual(grounded["synthesis_article_ids"], ["art-us-01"])
        self.assertEqual(grounded["shared_facts"][0]["article_ids"], ["art-us-01"])
        self.assertTrue(any("ghost-article-999" in w for w in warnings))

        # Check diagnostics
        diag = grounded["grounding_diagnostics"]
        self.assertLess(diag["citation_validity_rate"], 1.0)
        self.assertTrue(any("Pruned ungrounded" in c for c in diag["corrections"]))

    def test_detects_ungrounded_country_attribution(self) -> None:
        # LLM asserts German perspectives, but only US and French articles exist
        raw_report = {
            "synthesis": "Differences emerged over timeline.",
            "synthesis_article_ids": ["art-us-01"],
            "shared_facts": [],
            "country_source_comparison": [
                {
                    "text": "German newspapers criticized the pact's transition speed.",
                    "article_ids": ["art-us-01"],  # US article cited for German reaction!
                }
            ],
            "coverage_limitations": [],
        }
        warnings: list[str] = []
        grounded = verify_synthesis_grounding(
            raw_report,
            story=self.story,
            articles_by_id=self.articles_by_id,
            warnings=warnings,
        )

        diag = grounded["grounding_diagnostics"]
        self.assertFalse(diag["country_grounding_passed"])
        self.assertTrue(any("German" in c or "DE" in c for c in diag["corrections"]))
        # Limitation added to report
        self.assertTrue(any("DE" in lim or "German" in lim for lim in grounded["coverage_limitations"]))

    def test_demotes_dependent_evidence_to_reports_or_quotes(self) -> None:
        raw_report = {
            "synthesis": "Summary of claims.",
            "synthesis_article_ids": ["art-us-01"],
            "shared_facts": [],
            "country_source_comparison": [],
            "claim_perspectives": [
                {
                    "claim_id": "claim-001",
                    "claimants": [{"name": "White House"}],
                    "coverage": [
                        {
                            "article_id": "art-us-01",
                            "relationship": "independently_supports",
                            "evidence_basis": "White House spokesperson said the treaty will create jobs.",
                            "explanation": "Spokesperson confirmed projected employment growth.",
                        }
                    ],
                }
            ],
            "coverage_limitations": [],
        }
        warnings: list[str] = []
        grounded = verify_synthesis_grounding(
            raw_report,
            story=self.story,
            articles_by_id=self.articles_by_id,
            warnings=warnings,
        )

        coverage = grounded["claim_perspectives"][0]["coverage"]
        self.assertEqual(coverage[0]["relationship"], "reports_or_quotes")
        self.assertFalse(grounded["grounding_diagnostics"]["independent_support_grounded"])

    def test_enforces_single_country_coverage_limitation(self) -> None:
        single_country_articles = {
            "art-us-01": self.articles_by_id["art-us-01"],
        }
        story = {**self.story, "articles": list(single_country_articles.values())}
        raw_report = {
            "synthesis": "General narrative.",
            "synthesis_article_ids": ["art-us-01"],
            "shared_facts": [],
            "country_source_comparison": [],
            "coverage_limitations": [],
        }
        warnings: list[str] = []
        grounded = verify_synthesis_grounding(
            raw_report,
            story=story,
            articles_by_id=single_country_articles,
            warnings=warnings,
        )
        self.assertTrue(any("Multi-country perspectives are limited" in item for item in grounded["coverage_limitations"]))

    def test_build_framing_comparisons_with_emulated_llm(self) -> None:
        emulated_response = {
            "story_id": "story-001",
            "framing_report": {
                "synthesis": "US and French outlets shared core treaty terms while French reports highlighted Paris host role.",
                "synthesis_article_ids": ["art-us-01", "art-fr-01"],
                "shared_facts": [
                    {"text": "Energy pact approved by consensus", "article_ids": ["art-us-01", "art-fr-01"]}
                ],
                "country_source_comparison": [
                    {"text": "French coverage emphasized the Paris leadership initiative.", "article_ids": ["art-fr-01"]}
                ],
                "repetition_without_independent_support": [],
                "verified_or_independently_supported_claims": [],
                "qualified_disputed_or_unresolved_claims": [],
                "coverage_limitations": [],
            },
            "claim_perspectives": [
                {
                    "claim_id": "claim-001",
                    "coverage": [
                        {
                            "article_id": "art-us-01",
                            "relationship": "reports_or_quotes",
                            "evidence_basis": "Energy minister speech",
                            "explanation": "Outlined goals",
                        }
                    ],
                    "synthesis": "Ministerial claims were quoted across press outlets.",
                    "coverage_limitations": [],
                }
            ],
        }

        mock_orchestrator = type("MockOrch", (), {
            "summary_ai_client": FakeAIClient([emulated_response]),
        })()

        inputs = {
            "stories": [
                {
                    "story_id": "story-001",
                    "title": "Global green energy treaty",
                    "summary": "Nations agreed to an energy transition pact.",
                    "claims": [{"claim_id": "claim-001", "claim": "Treaty targets 50% clean energy by 2030"}],
                }
            ],
            "articles": list(self.articles_by_id.values()),
        }

        coverage_by_story = {
            "story-001": {
                "coverage_status": "ok",
                "coverage_articles": list(self.articles_by_id.values()),
                "coverage_quality": {"thin_reasons": []},
            }
        }

        plans_by_story = {
            "story-001": {
                "status": "ok",
                "verification_targets": [
                    {"claim_id": "claim-001", "importance_reason": "Key numerical target"}
                ],
            }
        }

        warnings: list[str] = []
        results = build_framing_comparisons(
            mock_orchestrator,
            inputs=inputs,
            coverage_by_story=coverage_by_story,
            plans_by_story=plans_by_story,
            verification_by_claim={"claim-001": {"status": "checked", "verdict": "supported"}},
            verification_documents={},
            warnings=warnings,
        )

        self.assertIn("story-001", results)
        report = results["story-001"]
        self.assertEqual(report["synthesis_article_ids"], ["art-us-01", "art-fr-01"])
        self.assertIn("grounding_diagnostics", report)
        diag = report["grounding_diagnostics"]
        self.assertEqual(diag["citation_validity_rate"], 1.0)
        self.assertTrue(diag["country_grounding_passed"])
        self.assertGreaterEqual(diag["grounding_score"], 0.9)
