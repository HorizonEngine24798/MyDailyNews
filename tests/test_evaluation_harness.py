from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from mydailynews.analysis.deterministic_delta import assess_lexical_change
from mydailynews.analysis.delta import DeltaExtractor
from mydailynews.ai.base import AIJsonError
from mydailynews.analysis.policy_filter import (
    filter_delta_packet_for_articles,
    filter_evidence_packet_for_articles,
    filter_prior_reports_for_articles,
)
from mydailynews.app.models import (
    DeltaExtractionConfig,
    HeadlineDecision,
    MemoryAnnotation,
    NewsCandidate,
    PriorReport,
    SelectedArticle,
    UserMemory,
)
from mydailynews.briefing.generator import no_material_changes_brief
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.domain.headline_selection import profile_match_signals
from mydailynews.domain.text_similarity import word_tokens
from mydailynews.evaluation.adapters import (
    CandidateMetadataReplayAdapter,
    FaultInjectionAdapter,
    LocalDeltaModelAdapter,
    PredictionFileAdapter,
    ProductionHeuristicAdapter,
    ScriptedOracleAdapter,
)
from mydailynews.evaluation.providers import FixtureNewsProvider
from mydailynews.evaluation.investigations import build_investigation
from mydailynews.evaluation.runner import evaluate_adapter
from mydailynews.evaluation.schema import EvalExpectation, EvalPrediction, load_corpus
from mydailynews.memory.coverage import coverage_records_for_selected
from mydailynews.memory.recall import partition_selected_for_brief, recall_packet_for_selected
from mydailynews.memory.story_index import StoryIndexRecord, StoryIndexStore
from mydailynews.memory.story_keys import story_identity_for_candidate


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json"


def _candidate(candidate_id: str, title: str, snippet: str = "") -> NewsCandidate:
    return NewsCandidate(
        id=candidate_id,
        source="Fixture Wire",
        category="synthetic",
        title=title,
        url=f"https://fixture.test/{candidate_id}",
        snippet=snippet,
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _selected(candidate_id: str, title: str) -> SelectedArticle:
    return SelectedArticle(
        candidate=_candidate(candidate_id, title),
        decision=HeadlineDecision(candidate_id=candidate_id, score=8.0),
        article_text=title,
    )


class _RecordingDeltaClient:
    max_input_tokens = 4000
    max_new_tokens = 1000
    config = SimpleNamespace(
        backend="fake",
        effective_model_label="fake",
        response_format="json_schema",
        context_window_tokens=4096,
    )

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def complete_json(self, system, user, **kwargs):
        self.prompts.append(f"{system}\n{user}")
        return {"story_decisions": []}


class EvaluationHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(CORPUS_PATH)

    def test_corpus_is_multi_day_diverse_and_public_input_hides_gold(self) -> None:
        self.assertGreaterEqual(len(self.corpus.arcs), 15)
        case_count = sum(len(day.documents) for arc in self.corpus.arcs for day in arc.days)
        self.assertGreaterEqual(case_count, 74)
        self.assertGreaterEqual(len(self.corpus.quiet_days()), 4)
        self.assertTrue(
            any(
                len(day.documents) >= 3 and all(not expected.should_select for expected in day.expectations)
                for arc in self.corpus.arcs
                for day in arc.days
            )
        )
        self.assertTrue(any(arc.split == "holdout" for arc in self.corpus.arcs))

        public = self.corpus.arcs[0].public_input()
        self.assertFalse(hasattr(public, "fact_catalog"))
        self.assertFalse(hasattr(public, "tags"))
        document = public.days[0].documents[0]
        self.assertFalse(hasattr(document, "canonical_story_id"))
        candidate = document.to_candidate()
        self.assertNotIn("expected", candidate.metadata)
        self.assertNotIn("fact", candidate.metadata)

    def test_existing_corpus_supports_oracle_candidate_and_ledger_interventions(self) -> None:
        candidate_investigation = build_investigation(self.corpus, "oracle_candidate")
        ledger_investigation = build_investigation(self.corpus, "oracle_ledger")
        continuation_keys = [
            (arc.id, day.date, expected.document_id)
            for arc in self.corpus.arcs
            for day in arc.days
            for expected in day.expectations
            if expected.relationship == "same_story"
        ]

        self.assertEqual(len(continuation_keys), 26)
        self.assertTrue(all(candidate_investigation.case_for(*key).has_prior_story for key in continuation_keys))
        self.assertTrue(all(not candidate_investigation.case_for(*key).prior_facts for key in continuation_keys))
        self.assertTrue(all(ledger_investigation.case_for(*key).prior_facts for key in continuation_keys))
        self.assertEqual(
            sum(bool(ledger_investigation.case_for(*key).current_facts) for key in continuation_keys),
            18,
        )
        disclosure = ledger_investigation.disclosure()
        self.assertTrue(disclosure["uses_private_gold"])
        self.assertFalse(disclosure["production_comparable"])

    def test_oracle_proves_scoring_contract_can_reach_perfect_scores(self) -> None:
        result = evaluate_adapter(self.corpus, ScriptedOracleAdapter(self.corpus))
        overall = result["overall"]
        self.assertEqual(result["prediction_counts"]["missing"], 0)
        self.assertEqual(overall["story_identity_pairwise_f1"], 1.0)
        self.assertEqual(overall["delta_type_accuracy"], 1.0)
        self.assertEqual(overall["novelty_detection_f1"], 1.0)
        self.assertEqual(overall["display_policy_accuracy"], 1.0)
        self.assertEqual(overall["required_fact_recall"], 1.0)
        self.assertEqual(overall["faithfulness_pass_rate"], 1.0)
        self.assertEqual(overall["false_suppression_rate"], 0.0)
        self.assertEqual(overall["quiet_days"], 4)
        self.assertEqual(overall["quiet_day_outputs"], 0)
        self.assertEqual(overall["quiet_day_abstention_rate"], 1.0)
        self.assertEqual(overall["candidate_recall_at_1"], 1.0)
        self.assertEqual(overall["relationship_accuracy_given_candidate"], 1.0)
        self.assertEqual(overall["continuation_delta_accuracy_given_correct_identity"], 1.0)
        self.assertEqual(overall["display_accuracy_given_correct_semantics"], 1.0)
        self.assertEqual(overall["stage_diagnostics"]["policy"]["cases_with_correct_semantics"], 74)
        self.assertTrue(result["run"]["investigation"]["uses_private_gold"])
        self.assertTrue(any("capability ceiling" in item for item in result["validation"]["warnings"]))
        rescored = evaluate_adapter(
            self.corpus,
            PredictionFileAdapter(
                [EvalPrediction.from_dict(item) for item in result["predictions"]],
                name="oracle-roundtrip",
            ),
        )
        self.assertTrue(rescored["run"]["investigation"]["uses_private_gold"])

    def test_corpus_validation_rejects_bad_source_metadata_and_impossible_history(self) -> None:
        arc = self.corpus.arcs[0]
        first_day = arc.days[0]
        bad_document = replace(
            first_day.documents[0],
            url="fixture-without-a-scheme",
            published_at=f"{first_day.date}T08:00:00",
        )
        bad_expectation = replace(first_day.expectations[0], relationship="same_story")
        bad_day = replace(
            first_day,
            documents=[bad_document, *first_day.documents[1:]],
            expectations=[bad_expectation, *first_day.expectations[1:]],
        )
        invalid = replace(self.corpus, arcs=[replace(arc, days=[bad_day, *arc.days[1:]])])

        with self.assertRaises(ValueError) as raised:
            invalid.validate()
        self.assertIn("absolute HTTP(S) URL", str(raised.exception))
        self.assertIn("same_story requires", str(raised.exception))

    def test_schema_does_not_coerce_string_booleans_or_malformed_claim_lists(self) -> None:
        with self.assertRaisesRegex(ValueError, "material must be a JSON boolean"):
            EvalExpectation.from_dict({"material": "false", "should_select": False})
        with self.assertRaisesRegex(ValueError, "reported_fact_ids must be an array or null"):
            EvalPrediction.from_dict(
                {
                    "material": False,
                    "selected": False,
                    "reported_fact_ids": "fact-1",
                }
            )

    def test_candidate_stage_metadata_cannot_reference_unknown_or_future_documents(self) -> None:
        predictions = [
            item
            for arc in self.corpus.arcs
            for item in ScriptedOracleAdapter(self.corpus).predict(arc.public_input())
        ]
        predictions[0].metadata["candidate_prior_stories"] = [
            {
                "story_key": "invalid",
                "document_ids": [predictions[0].document_id, "not-in-corpus"],
                "score": 1.0,
            }
        ]
        valid_prior_candidate = next(
            item
            for item in predictions[1:]
            if item.metadata.get("candidate_prior_stories")
        )
        valid_prior_candidate.metadata["candidate_prior_stories"][0]["story_key"] = "fabricated-link"
        result = evaluate_adapter(
            self.corpus,
            PredictionFileAdapter(predictions, name="invalid-candidate-metadata"),
        )

        errors = "\n".join(result["validation"]["errors"])
        self.assertIn("is not earlier in corpus chronology", errors)
        self.assertIn("references unknown document", errors)
        self.assertIn("does not match prior prediction", errors)

    def test_candidate_recall_penalizes_selectively_missing_stage_metadata(self) -> None:
        expected_by_key = self.corpus.expectations_by_key()
        predictions = [
            item
            for arc in self.corpus.arcs
            for item in ScriptedOracleAdapter(self.corpus).predict(arc.public_input())
        ]
        kept_continuation = False
        for prediction in predictions:
            key = (prediction.arc_id, prediction.date, prediction.document_id)
            if expected_by_key[key].relationship != "same_story":
                continue
            if not kept_continuation:
                kept_continuation = True
                continue
            prediction.metadata.pop("candidate_prior_stories")

        result = evaluate_adapter(
            self.corpus,
            PredictionFileAdapter(predictions, name="selective-candidate-metadata"),
        )
        stage = result["overall"]["stage_diagnostics"]["candidate_retrieval"]

        self.assertEqual(stage["continuation_cases"], 26)
        self.assertEqual(stage["continuation_cases_with_metadata"], 1)
        self.assertEqual(stage["recall_at_1"], round(1 / 26, 4))

    def test_correct_identity_requires_linking_the_correct_supplied_candidate(self) -> None:
        predictions = [
            item
            for arc in self.corpus.arcs
            for item in ScriptedOracleAdapter(self.corpus).predict(arc.public_input())
        ]
        expected_by_key = self.corpus.expectations_by_key()
        broken = next(
            prediction
            for prediction in predictions
            if expected_by_key[(prediction.arc_id, prediction.date, prediction.document_id)].relationship
            == "same_story"
        )
        broken.predicted_story_id = "not-the-supplied-prior-story"
        broken.delta_type = "uncertain"

        result = evaluate_adapter(
            self.corpus,
            PredictionFileAdapter(predictions, name="unlinked-same-story"),
        )
        stages = result["overall"]["stage_diagnostics"]

        self.assertEqual(stages["candidate_retrieval"]["recall_at_1"], 1.0)
        self.assertEqual(stages["identity"]["same_story_recall_given_correct_candidate"], 1.0)
        self.assertEqual(
            stages["identity"]["same_story_link_accuracy_given_correct_candidate"],
            round(25 / 26, 4),
        )
        self.assertEqual(stages["delta"]["continuation_cases_with_correct_identity"], 25)
        self.assertEqual(stages["delta"]["continuation_accuracy_given_correct_identity"], 1.0)
        self.assertLess(result["overall"]["continuation_delta_type_accuracy"], 1.0)

    def test_historical_predictions_can_replay_broad_candidate_metadata_without_gold(self) -> None:
        corpus = replace(self.corpus, arcs=[self.corpus.arcs[0]])
        predictions = ScriptedOracleAdapter(corpus).predict(corpus.arcs[0].public_input())
        for prediction in predictions:
            prediction.metadata = {}
        result = evaluate_adapter(
            corpus,
            CandidateMetadataReplayAdapter(
                PredictionFileAdapter(predictions, name="historical-local-delta")
            ),
        )

        self.assertEqual(result["overall"]["candidate_metadata_coverage"], 1.0)
        self.assertEqual(result["overall"]["candidate_recall_at_3"], 1.0)
        self.assertTrue(result["run"]["investigation"]["candidate_metadata_replayed"])
        self.assertFalse(result["run"]["investigation"]["uses_private_gold"])

    def test_negative_controls_are_detected(self) -> None:
        oracle = ScriptedOracleAdapter(self.corpus)
        merged = evaluate_adapter(self.corpus, FaultInjectionAdapter(oracle, "merge_all_stories"))
        omitted = evaluate_adapter(self.corpus, FaultInjectionAdapter(oracle, "omit_everything"))
        hallucinated = evaluate_adapter(self.corpus, FaultInjectionAdapter(oracle, "unsupported_claims"))
        quiet_hallucination = evaluate_adapter(
            self.corpus,
            FaultInjectionAdapter(oracle, "hallucinate_quiet_days"),
        )
        dropped = evaluate_adapter(self.corpus, FaultInjectionAdapter(oracle, "drop_every_other"))
        everything_new = evaluate_adapter(self.corpus, FaultInjectionAdapter(oracle, "call_everything_new"))

        self.assertLess(merged["overall"]["story_identity_pairwise_f1"], 0.9)
        self.assertLess(merged["overall"]["story_identity_pairwise_precision"], 0.7)
        self.assertGreater(omitted["overall"]["false_suppression_rate"], 0.5)
        self.assertEqual(hallucinated["overall"]["faithfulness_pass_rate"], 0.0)
        self.assertEqual(quiet_hallucination["overall"]["quiet_day_false_output_rate"], 1.0)
        self.assertEqual(quiet_hallucination["overall"]["quiet_day_abstention_rate"], 0.0)
        self.assertEqual(quiet_hallucination["prediction_counts"]["quiet_day_outputs"], 4)
        self.assertEqual(quiet_hallucination["prediction_counts"]["extra"], 4)
        self.assertTrue(quiet_hallucination["validation"]["errors"])
        self.assertGreater(dropped["prediction_counts"]["missing"], 0)
        self.assertLess(everything_new["overall"]["novelty_detection_precision"], 1.0)

    def test_production_adapter_runs_without_gold_or_network(self) -> None:
        result = evaluate_adapter(self.corpus, ProductionHeuristicAdapter())
        self.assertEqual(result["prediction_counts"]["missing"], 0)
        self.assertEqual(result["overall"]["claim_evaluation_coverage"], 0.0)
        self.assertIsNone(result["overall"]["faithfulness_pass_rate"])
        self.assertGreater(result["overall"]["display_policy_accuracy"], 0.0)

    def test_production_adapter_can_retrieve_every_case_through_fake_rss(self) -> None:
        result = evaluate_adapter(
            self.corpus,
            ProductionHeuristicAdapter(fixture_mode="rss"),
        )
        expected_cases = sum(len(day.documents) for arc in self.corpus.arcs for day in arc.days)
        self.assertEqual(result["prediction_counts"]["received"], expected_cases)
        self.assertEqual(result["prediction_counts"]["missing"], 0)
        self.assertEqual(result["prediction_counts"]["extra"], 0)
        self.assertEqual(result["run"]["adapter"], "production_heuristic:rss")

    def test_local_model_adapter_uses_existing_delta_contract(self) -> None:
        class EmptyDeltaClient:
            max_input_tokens = 4000
            max_new_tokens = 1000
            config = SimpleNamespace(
                backend="fake",
                effective_model_label="fake",
                response_format="json_schema",
            )

            def __init__(self) -> None:
                self.calls = 0

            def estimate_tokens(self, text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *args, **kwargs):
                self.calls += 1
                return {
                    "baseline_coverage_note": "fake model returned no decisions",
                    "new": [],
                    "escalated": [],
                    "weakened": [],
                    "reframed": [],
                    "unchanged_but_important": [],
                    "story_decisions": [],
                    "evidence_gaps": [],
                }

        client = EmptyDeltaClient()
        arc = self.corpus.arcs[0].public_input()
        adapter = LocalDeltaModelAdapter(
            client,
            DeltaExtractionConfig(
                enabled=True,
                max_input_tokens=4000,
                max_new_tokens=1000,
                max_articles=8,
                max_articles_per_batch=4,
            ),
        )
        predictions = adapter.predict(arc)

        self.assertEqual(len(predictions), sum(len(day.documents) for day in arc.days))
        self.assertGreater(client.calls, 0)
        self.assertTrue(all(item.metadata["model_fallback_used"] for item in predictions))

    def test_retrieved_top3_mode_is_gold_blind_and_emits_candidate_stage_metadata(self) -> None:
        corpus = replace(self.corpus, arcs=[self.corpus.arcs[0]])
        client = _RecordingDeltaClient()
        result = evaluate_adapter(
            corpus,
            LocalDeltaModelAdapter(
                client,
                DeltaExtractionConfig(
                    enabled=True,
                    output_mode="decision_only",
                    max_input_tokens=4000,
                    max_new_tokens=1000,
                    max_articles_per_batch=4,
                ),
                investigation=build_investigation(corpus, "retrieved_top3"),
                candidate_limit=3,
            ),
        )

        self.assertFalse(result["run"]["investigation"]["uses_private_gold"])
        self.assertTrue(result["run"]["investigation"]["production_comparable"])
        self.assertEqual(result["overall"]["candidate_metadata_coverage"], 1.0)
        self.assertTrue(
            all(len(item["metadata"]["candidate_prior_stories"]) <= 3 for item in result["predictions"])
        )
        self.assertFalse(any("private evaluation intervention" in prompt for prompt in client.prompts))

    def test_oracle_ledger_mode_supplies_existing_fact_packets_and_discloses_contamination(self) -> None:
        corpus = replace(self.corpus, arcs=[self.corpus.arcs[0]])
        investigation = build_investigation(corpus, "oracle_ledger")
        client = _RecordingDeltaClient()
        result = evaluate_adapter(
            corpus,
            LocalDeltaModelAdapter(
                client,
                DeltaExtractionConfig(
                    enabled=True,
                    output_mode="decision_only",
                    max_input_tokens=4000,
                    max_new_tokens=1000,
                    max_articles_per_batch=4,
                ),
                investigation=investigation,
            ),
        )
        fact_texts = [
            fact
            for packet in investigation.cases.values()
            for fact in [*packet.prior_facts, *packet.current_facts]
        ]

        self.assertTrue(result["run"]["investigation"]["uses_private_gold"])
        self.assertFalse(result["run"]["investigation"]["production_comparable"])
        self.assertTrue(any("capability ceiling" in item for item in result["validation"]["warnings"]))
        self.assertTrue(any("private evaluation intervention" in prompt for prompt in client.prompts))
        self.assertTrue(any(fact in prompt for fact in fact_texts for prompt in client.prompts))
        self.assertEqual(
            result["overall"]["stage_diagnostics"]["oracle_fact_packet"]["continuation_prior_fact_coverage"],
            1.0,
        )

    def test_local_model_adapter_preserves_results_when_one_model_call_fails(self) -> None:
        class FailingDeltaClient:
            max_input_tokens = 1400
            max_new_tokens = 256
            config = SimpleNamespace(
                backend="fake",
                effective_model_label="fake",
                response_format="json_schema",
                context_window_tokens=2048,
            )

            def estimate_tokens(self, text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, *args, **kwargs):
                raise AIJsonError("injected malformed response")

        arc = self.corpus.arcs[0].public_input()
        predictions = LocalDeltaModelAdapter(
            FailingDeltaClient(),
            DeltaExtractionConfig(
                enabled=True,
                output_mode="decision_only",
                max_input_tokens=1400,
                max_new_tokens=256,
                max_articles_per_batch=2,
            ),
        ).predict(arc)

        self.assertEqual(len(predictions), sum(len(day.documents) for day in arc.days))
        self.assertTrue(all(item.metadata["model_fallback_used"] for item in predictions))
        self.assertTrue(all("injected malformed response" in item.metadata["model_error"] for item in predictions))

    def test_fixture_source_exercises_the_real_rss_parser_without_network(self) -> None:
        public_arc = self.corpus.arcs[0].public_input()
        provider = FixtureNewsProvider(public_arc)
        day = public_arc.days[0]
        scraper = provider.rss_scraper(day.date)
        candidates = scraper.fetch(datetime(2025, 12, 31, tzinfo=timezone.utc))

        self.assertEqual(len(candidates), len(day.documents))
        self.assertEqual(candidates[0].title, day.documents[0].title)
        self.assertEqual(candidates[0].metadata["entry_id"], day.documents[0].id)
        self.assertEqual(candidates[0].snippet, day.documents[0].snippet)
        self.assertEqual(len(scraper.http.calls), 1)

    def test_fixture_source_preserves_source_empty_days_without_network_calls(self) -> None:
        quiet_arc = next(
            arc.public_input()
            for arc in self.corpus.arcs
            if any(not day.documents for day in arc.days)
        )
        quiet_day = next(day for day in quiet_arc.days if not day.documents)
        provider = FixtureNewsProvider(quiet_arc)

        direct = provider.fetch(quiet_day.date)
        via_rss = provider.fetch_via_rss(quiet_day.date)

        self.assertEqual(direct.candidates, [])
        self.assertEqual(via_rss.candidates, [])

    def test_unicode_and_numbers_are_identity_evidence(self) -> None:
        tokens = word_tokens("Želva Model 3 meets Модель 4")
        self.assertIn("želva", tokens)
        self.assertIn("модель", tokens)
        self.assertIn("3", tokens)
        self.assertIn("4", tokens)

        model_3 = story_identity_for_candidate(_candidate("m3", "Oriole Model 3 battery fault"))
        model_4 = story_identity_for_candidate(_candidate("m4", "Oriole Model 4 battery fault"))
        self.assertNotEqual(model_3.story_key, model_4.story_key)
        self.assertIn("3", model_3.tokens)
        self.assertIn("4", model_4.tokens)

        store = StoryIndexStore(Path("unused-story-index.json"))
        store._records = [
            StoryIndexRecord(
                story_key=model_3.story_key,
                story_family_key=model_3.story_family_key,
                title=model_3.story_title,
                topic="Arbitrary devices",
                tokens=model_3.tokens,
                first_seen="2026-01-01",
                last_seen="2026-01-01",
            )
        ]
        matched = store.match_candidate(_candidate("m4", "Oriole Model 4 battery fault"))
        self.assertNotEqual(matched.story_key, model_3.story_key)

    def test_deterministic_delta_suppresses_duplicates_but_not_unknown_semantics(self) -> None:
        duplicate = assess_lexical_change(
            "Toenail magic becomes the new fashion",
            "Toenail magic becomes the new fashion",
            story_key_match=True,
        )
        unfamiliar_update = assess_lexical_change(
            "Royal guild adopts toenail charms for winter uniforms",
            "Toenail magic becomes the new fashion",
            story_key_match=True,
        )

        self.assertEqual(duplicate.change_type, "unchanged")
        self.assertEqual(duplicate.disposition, "omit")
        self.assertEqual(unfamiliar_update.change_type, "uncertain")
        self.assertNotEqual(unfamiliar_update.disposition, "omit")

    def test_delta_schema_keeps_domain_neutral_change_labels(self) -> None:
        normalized = DeltaExtractor(object(), DeltaExtractionConfig())._normalize_result(
            {
                "baseline_coverage_note": "baseline",
                "new": [],
                "escalated": [],
                "weakened": [],
                "reframed": [],
                "unchanged_but_important": [],
                "evidence_gaps": [],
                "story_decisions": [{
                    "story_key": "elf-fashion",
                    "article_ids": ["elf"],
                    "relationship": "same_story",
                    "change_type": "status_change",
                    "materiality": 0.9,
                    "confidence": 0.8,
                    "disposition": "full_report",
                }],
            }
        )
        self.assertEqual(normalized["story_decisions"][0]["change_type"], "status_change")

    def test_decision_only_delta_contract_avoids_editorial_schema_overhead(self) -> None:
        class DecisionClient:
            max_input_tokens = 1400
            max_new_tokens = 256
            config = SimpleNamespace(
                backend="fake",
                effective_model_label="fake",
                response_format="json_schema",
                context_window_tokens=2048,
            )

            def __init__(self) -> None:
                self.system = ""
                self.user = ""
                self.schema_name = ""

            def estimate_tokens(self, text: str) -> int:
                return max(1, len(text) // 4)

            def complete_json(self, system, user, **kwargs):
                self.system = system
                self.user = user
                self.schema_name = kwargs["json_schema"].name
                return {
                    "story_decisions": [{
                        "story_key": "elf-fashion",
                        "article_ids": ["elf"],
                        "prior_story_key": "",
                        "relationship": "distinct_story",
                        "change_type": "new",
                        "materiality": 0.9,
                        "confidence": 0.8,
                        "disposition": "full_report",
                        "summary": "Toenail magic is newly fashionable.",
                    }]
                }

        client = DecisionClient()
        result = DeltaExtractor(
            client,
            DeltaExtractionConfig(
                enabled=True,
                input_source="articles_only",
                output_mode="decision_only",
                max_input_tokens=1400,
                max_new_tokens=256,
            ),
        ).extract(
            [_selected("elf", "Faraway fashion embraces toenail magic")],
            UserMemory(role="Elf", wants=["toenail magic"]),
            [],
            [],
            "Report material changes.",
            "2026-01-01",
            story_memory={
                "stories": [{
                    "current_article_ids": ["elf"],
                    "prior_baselines": [],
                }]
            },
        )

        self.assertEqual(client.schema_name, "delta_decisions")
        self.assertIn("Current source evidence", client.user)
        self.assertNotIn("Fallback selected article evidence", client.user)
        self.assertNotIn('"unchanged_but_important"', client.user)
        self.assertEqual(result["new"], [])
        self.assertEqual(result["story_decisions"][0]["change_type"], "new")

        evidence_client = DecisionClient()
        DeltaExtractor(
            evidence_client,
            DeltaExtractionConfig(
                enabled=True,
                input_source="evidence_only",
                output_mode="decision_only",
                max_input_tokens=1400,
                max_new_tokens=256,
            ),
        ).extract(
            [],
            UserMemory(role="Elf", wants=["toenail magic"]),
            [],
            [],
            "Report material changes.",
            "2026-01-01",
            evidence_packet={
                "story_clusters": [{
                    "cluster_id": "cluster-elf",
                    "label": "Toenail magic",
                    "summary": "A sourced magical-fashion change.",
                    "article_ids": ["elf"],
                }]
            },
        )
        self.assertIn('"evidence_packet"', evidence_client.user)
        self.assertIn("cluster-elf", evidence_client.user)

    def test_arbitrary_profile_terms_drive_relevance_without_domain_rules(self) -> None:
        candidate = _candidate(
            "elf",
            "Faraway fashion embraces toenail magic",
            "Moon-crystal charms appear in tailor demonstrations.",
        )
        signals = profile_match_signals(
            candidate,
            UserMemory(
                role="Elf from a faraway land",
                wants=["toenail magic"],
                beats={"faraway fashion": 2.5},
            ),
        )
        self.assertEqual(signals["wants_matches"], ["toenail magic"])
        self.assertEqual(signals["beat_matches"], ["faraway fashion"])

    def test_omit_policy_fails_open_for_low_confidence_and_conflicts(self) -> None:
        article = _selected("story-1", "Toenail magic becomes the new fashion")
        valid = {
            "story_decisions": [{
                "article_ids": ["story-1"],
                "relationship": "same_story",
                "change_type": "unchanged",
                "confidence": 0.95,
                "disposition": "omit",
            }]
        }
        unsafe = {
            "story_decisions": [{
                "article_ids": ["story-1"],
                "relationship": "distinct_story",
                "change_type": "new",
                "confidence": 0.99,
                "disposition": "omit",
            }]
        }
        conflicting = {
            "story_decisions": [
                {"article_ids": ["story-1"], "relationship": "same_story", "change_type": "unchanged", "confidence": 0.95, "disposition": "omit"},
                {"article_ids": ["story-1"], "relationship": "same_story", "change_type": "material_update", "confidence": 0.9, "disposition": "full_report"},
            ]
        }

        included, omitted = partition_selected_for_brief(selected=[article], delta_packet=valid)
        self.assertEqual(included, [])
        self.assertEqual(omitted, [article])
        self.assertEqual(partition_selected_for_brief(selected=[article], delta_packet=unsafe)[0], [article])
        self.assertEqual(partition_selected_for_brief(selected=[article], delta_packet=conflicting)[0], [article])

    def test_coverage_never_records_an_omitted_article(self) -> None:
        omitted = _selected("omitted", "Repeated story")
        included = _selected("included", "Material story")
        set_memory_annotation(
            omitted.candidate,
            MemoryAnnotation(story_key="repeat", story_family_key="repeat", today_policy="omit"),
        )
        set_memory_annotation(
            included.candidate,
            MemoryAnnotation(story_key="material", story_family_key="material"),
        )
        records = coverage_records_for_selected(
            date="2026-01-01",
            brief_name="general",
            selected=[omitted, included],
        )
        self.assertEqual([record.article_ids for record in records], [["included"]])
        self.assertEqual(records[0].prominence, "lead")

    def test_writer_context_removes_every_suppressed_story_side_channel(self) -> None:
        included = _selected("included", "Crown feathers become formal courtwear")
        omitted = _selected("omitted", "Toenail magic remains yesterday's fashion")
        set_memory_annotation(
            included.candidate,
            MemoryAnnotation(story_key="crown-feathers", story_family_key="court-fashion"),
        )
        set_memory_annotation(
            omitted.candidate,
            MemoryAnnotation(story_key="toenail-magic", story_family_key="court-fashion"),
        )

        delta = {
            "baseline_coverage_note": "Yesterday's toenail magic report is unchanged.",
            "new": [{"article_ids": ["included"], "summary": "Crown feathers are new."}],
            "reframed": [{
                "article_ids": ["included", "omitted"],
                "summary": "Mixed prose mentions toenail magic.",
            }],
            "unchanged_but_important": [{
                "article_ids": ["omitted"],
                "summary": "Toenail magic is repeated.",
            }],
            "story_decisions": [
                {"article_ids": ["included"], "story_key": "crown-feathers"},
                {"article_ids": ["omitted"], "story_key": "toenail-magic"},
            ],
            "evidence_gaps": ["Does toenail magic persist?"],
        }
        evidence = {
            "overview": "Crown feathers and toenail magic are both discussed.",
            "story_clusters": [
                {
                    "article_ids": ["included"],
                    "title": "Crown feathers",
                    "key_claims": [{"claim": "Crown feathers debuted.", "support_article_ids": ["included"]}],
                },
                {
                    "article_ids": ["omitted"],
                    "title": "Toenail magic",
                    "key_claims": [{"claim": "Toenail magic persists.", "support_article_ids": ["omitted"]}],
                },
                {
                    "article_ids": ["included", "omitted"],
                    "title": "Mixed fashion",
                    "key_claims": [],
                },
            ],
            "reader_qa": [
                {"question": "What are crown feathers?", "article_ids": ["included"]},
                {"question": "What is toenail magic?", "article_ids": ["omitted"]},
            ],
            "global_watch_signals": ["Watch toenail magic."],
        }
        prior_reports = [
            PriorReport(
                id="prior",
                date="2025-12-31",
                title="Toenail magic daily",
                path="prior.json",
                summary="Toenail magic dominated yesterday's report.",
                major_headlines=[
                    {
                        "story_key": "crown-feathers",
                        "headline": "Crown feather precursor",
                        "story_threads": [{"summary": "Cross-linked to toenail magic."}],
                    },
                    {"story_key": "toenail-magic", "headline": "Toenail magic repeats"},
                ],
                story_baselines=[
                    {"story_key": "crown-feathers", "summary": "Crown feather baseline"},
                    {"story_key": "toenail-magic", "summary": "Toenail magic baseline"},
                ],
            )
        ]
        recall = {
            "coverage_guidance": [
                {"story_key": "crown-feathers", "title": "Crown feathers"},
                {"story_key": "toenail-magic", "title": "Toenail magic"},
            ],
            "selection_summary": {"story_count": 2},
        }

        writer_delta = filter_delta_packet_for_articles(
            delta,
            allowed_article_ids=["included"],
            omitted_count=1,
        )
        writer_evidence = filter_evidence_packet_for_articles(
            evidence,
            allowed_article_ids=["included"],
            omitted_count=1,
        )
        writer_reports = filter_prior_reports_for_articles(
            prior_reports,
            selected=[included],
            omitted_count=1,
        )
        writer_recall = recall_packet_for_selected(recall, [included])
        writer_context = json.dumps(
            [writer_delta, writer_evidence, [report.__dict__ for report in writer_reports], writer_recall],
            ensure_ascii=False,
        ).lower()

        self.assertNotIn("toenail magic", writer_context)
        self.assertNotIn("omitted", writer_context)
        self.assertIn("crown feathers", writer_context)
        self.assertEqual(writer_delta["reframed"], [])
        self.assertEqual(len(writer_evidence["story_clusters"]), 1)
        self.assertEqual(writer_reports[0].summary, "")
        self.assertEqual(writer_reports[0].path, "")
        self.assertEqual(writer_reports[0].topics, [])
        self.assertEqual(len(writer_recall["coverage_guidance"]), 1)

    def test_all_repeats_produce_a_deterministic_source_empty_brief(self) -> None:
        brief = no_material_changes_brief("2026-01-02")

        self.assertIn("No material changes", brief["lead"])
        self.assertEqual(brief["selected_articles"], [])
        self.assertEqual(brief["major_headlines"], [])
        self.assertEqual(brief["references"], [])
        self.assertEqual(brief["topic_reports"], [])


if __name__ == "__main__":
    unittest.main()
