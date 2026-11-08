from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock

from mydailynews.perspectives.coverage import _coverage_relevance_decision, _retrieval_requests
from mydailynews.perspectives.query_alignment import (
    MULTILINGUAL_ASSIST_LOOKUP,
    align_queries_with_llm,
    align_query_with_assist,
    detect_text_language,
    expand_multilingual_anchor_groups,
    normalize_language_code,
    resolve_country_language,
)


class FakeAIClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_json(self, system: str, user: str, label: str = "ai.complete_json", **kwargs) -> dict:
        self.calls.append({"system": system, "user": user, "label": label, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected AI call: {label}")
        return self.responses.pop(0)


class TestCrossLanguageQueryAlignment(TestCase):
    def test_normalize_language_code(self) -> None:
        self.assertEqual(normalize_language_code("french"), "fr")
        self.assertEqual(normalize_language_code("GERMAN"), "de")
        self.assertEqual(normalize_language_code("es"), "es")
        self.assertEqual(normalize_language_code("unknown"), "un")
        self.assertEqual(normalize_language_code(""), "en")

    def test_resolve_country_language(self) -> None:
        self.assertEqual(resolve_country_language("FR"), "fr")
        self.assertEqual(resolve_country_language("DE"), "de")
        self.assertEqual(resolve_country_language("JP"), "ja")
        self.assertEqual(resolve_country_language("ES"), "es")
        self.assertEqual(resolve_country_language("US"), "en")
        self.assertEqual(resolve_country_language("GB"), "en")
        self.assertEqual(resolve_country_language("XX", default="en"), "en")

    def test_detect_text_language(self) -> None:
        self.assertEqual(detect_text_language("Le président français a signé le traité hier avec ses partenaires."), "fr")
        self.assertEqual(detect_text_language("Der Bundeskanzler hat in Berlin über die Energiewende gesprochen."), "de")
        self.assertEqual(detect_text_language("El presidente confirmó el acuerdo en la conferencia de prensa."), "es")
        self.assertEqual(detect_text_language("日本の首相が東京で新しい経済政策を発表した。"), "ja")
        self.assertEqual(detect_text_language("中国代表团出席了联合国气候大会。"), "zh")
        self.assertEqual(detect_text_language("Президент подписал указ о новых экономических мерах."), "ru")
        self.assertEqual(detect_text_language("The prime minister announced the new economic accord today."), "en")

    def test_assist_lookup_replaces_standard_entities(self) -> None:
        # Assist lookup translates official international bodies without hardcoding general words
        query = "UN and EU investigate WHO response"
        fr_aligned = align_query_with_assist(query, "fr")
        self.assertIn("ONU", fr_aligned)
        self.assertIn("UE", fr_aligned)
        self.assertIn("OMS", fr_aligned)
        # General English words remain untouched in assist fallback
        self.assertIn("investigate", fr_aligned)

        es_aligned = align_query_with_assist(query, "es")
        self.assertIn("ONU", es_aligned)
        self.assertIn("UE", es_aligned)
        self.assertIn("OMS", es_aligned)

    def test_align_queries_with_llm_for_generic_news(self) -> None:
        # 1. Generic Biotech News
        biotech_response = {
            "aligned_queries": {
                "fr": ["autorisation FDA thérapie génique CRISPR drépanocytose"],
                "de": ["FDA Zulassung CRISPR Gentherapie Sichelzellkrankheit"],
            }
        }
        fake_ai = FakeAIClient([biotech_response])
        result = align_queries_with_llm(
            fake_ai,
            queries=["FDA approves CRISPR gene therapy for sickle cell disease"],
            target_languages=["fr", "de"],
            story_context="Biotechnology breakthrough in medical treatments.",
        )
        self.assertIn("fr", result)
        self.assertIn("de", result)
        self.assertEqual(result["fr"], ["autorisation FDA thérapie génique CRISPR drépanocytose"])
        self.assertEqual(result["de"], ["FDA Zulassung CRISPR Gentherapie Sichelzellkrankheit"])

        # Verify assist lookup was passed to the model prompt
        call = fake_ai.calls[0]
        self.assertIn("assist_lookup", call["user"])
        self.assertIn("UN", call["user"])

        # 2. Generic Aviation / Disaster News
        aviation_response = {
            "aligned_queries": {
                "es": ["Boeing 737 aterrizaje de emergencia incendio de motor Denver"],
            }
        }
        fake_ai_2 = FakeAIClient([aviation_response])
        result_2 = align_queries_with_llm(
            fake_ai_2,
            queries=["Boeing 737 engine fire emergency landing Denver"],
            target_languages=["es"],
        )
        self.assertEqual(result_2["es"], ["Boeing 737 aterrizaje de emergencia incendio de motor Denver"])

    def test_align_queries_with_llm_graceful_fallback(self) -> None:
        # When ai_client is None, falls back to assist entity replacement
        result = align_queries_with_llm(
            None,
            queries=["UN peacekeeper mission in region"],
            target_languages=["fr"],
        )
        self.assertIn("fr", result)
        self.assertIn("ONU", result["fr"][0])

    def test_expand_multilingual_anchor_groups_with_assist(self) -> None:
        anchors = [
            {"label": "orgs", "terms": ["UN", "NATO"]},
        ]
        expanded = expand_multilingual_anchor_groups(anchors, ["fr", "es"])
        terms = expanded[0]["terms"]
        self.assertIn("ONU", terms)
        self.assertIn("OTAN", terms)

    def test_retrieval_requests_uses_model_aligned_queries(self) -> None:
        # When the planner model returns aligned_queries, coverage dispatches them directly
        plan = {
            "queries": ["CRISPR gene therapy sickle cell"],
            "aligned_queries": {
                "fr": ["CRISPR thérapie génique drépanocytose"],
                "de": ["CRISPR Gentherapie Sichelzellkrankheit"],
            },
            "anchor_groups": [],
            "align_queries": True,
        }
        selected_sources = [
            {"source_id": "us_news", "country": "US", "language": "en"},
            {"source_id": "fr_sante", "country": "FR", "language": "fr"},
            {"source_id": "de_aerzte", "country": "DE", "language": "de"},
        ]
        story = {
            "story_title": "CRISPR gene therapy approved for sickle cell",
            "summary": "Health authorities approve genetic treatment.",
        }

        requests = _retrieval_requests(plan, selected_sources, story=story)
        query_types = [r["query_type"] for r in requests]
        self.assertIn("canonical", query_types)
        self.assertIn("aligned_canonical", query_types)

        fr_req = next(r for r in requests if r.get("target_language") == "fr")
        self.assertEqual(fr_req["query"], "CRISPR thérapie génique drépanocytose")
        self.assertIn("FR", fr_req["source_countries"])

    def test_coverage_relevance_decision_generic_news_semantic(self) -> None:
        # Cross-lingual relevance works for generic news (tech/biotech) via multilingual embeddings
        story = {
            "story_title": "FDA approves landmark CRISPR gene therapy for sickle cell patients",
            "summary": "The US health regulator cleared the first gene-editing cure for hereditary blood disease.",
        }
        # French article on generic biotech topic
        fr_article = {
            "title": "La FDA autorise une première thérapie génique CRISPR contre la drépanocytose",
            "snippet": "L'agence sanitaire américaine a donné son feu vert à un traitement innovant d'édition du génome.",
            "retrieval_query": "CRISPR thérapie génique drépanocytose",
            "retrieval_anchor_groups": [],
        }
        accepted, reason = _coverage_relevance_decision(fr_article, story)
        self.assertTrue(accepted)
        self.assertIn(reason, {"multilingual_semantic_match", "query_title_overlap", "seed_story_overlap"})

        # Unrelated sports news
        sports_article = {
            "title": "Local football team wins championship match in overtime",
            "snippet": "The striker scored in the final minutes to secure the national trophy.",
            "retrieval_query": "football sports league",
            "retrieval_anchor_groups": [],
        }
        rej_accepted, rej_reason = _coverage_relevance_decision(sports_article, story)
        self.assertFalse(rej_accepted)
        self.assertEqual(rej_reason, "insufficient_event_overlap")
