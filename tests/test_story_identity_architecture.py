from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mydailynews.analysis.delta import _compact_decision_baselines
from mydailynews.analysis.identity_gate import enforce_candidate_identity_gate
from mydailynews.app.models import (
    FilteringConfig,
    HeadlineDecision,
    MemoryAnnotation,
    MemoryConfig,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.domain.headline_selection import select_articles
from mydailynews.memory.context import build_story_memory_context
from mydailynews.memory.coverage import CoverageMemoryStore, CoverageRecord
from mydailynews.memory.recall import apply_delta_signals_to_selected
from mydailynews.memory.story_store import StoryStore
from mydailynews.evaluation.retrieval_diagnostics import evaluate_story_store_retrieval
from mydailynews.evaluation.schema import load_corpus


def _candidate(
    document_id: str,
    title: str,
    body: str,
    *,
    url: str | None = None,
    category: str = "invented",
) -> NewsCandidate:
    return NewsCandidate(
        id=document_id,
        source="Faraway Chronicle",
        category=category,
        title=title,
        url=url or f"https://fixture.test/{document_id}",
        snippet=body,
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _article(candidate: NewsCandidate, body: str | None = None) -> SelectedArticle:
    return SelectedArticle(
        candidate=candidate,
        decision=HeadlineDecision(candidate.id, score=8.0),
        article_text=body or candidate.snippet,
    )


def _annotate(candidate: NewsCandidate, story_key: str) -> None:
    set_memory_annotation(
        candidate,
        MemoryAnnotation(
            story_key=story_key,
            story_family_key="invented-events",
            story_title=candidate.title,
            match_confidence=1.0,
        ),
    )


def _story_memory(*, candidates: list[str]) -> dict:
    return {
        "stories": [
            {
                "story_key": "current-provisional",
                "current_title": "Current observation",
                "current_article_ids": ["current"],
                "prior_baselines": [
                    {"story_key": story_key, "title": story_key}
                    for story_key in candidates
                ],
            }
        ]
    }


class CandidateIdentityGateTests(unittest.TestCase):
    def test_no_candidate_always_becomes_new_even_when_model_claims_same_story(self) -> None:
        packet = enforce_candidate_identity_gate(
            {
                "story_decisions": [
                    {
                        "article_ids": ["current"],
                        "relationship": "same_story",
                        "prior_story_key": "invented-by-model",
                        "change_type": "unchanged",
                        "disposition": "omit",
                        "confidence": 0.99,
                    }
                ]
            },
            _story_memory(candidates=[]),
        )

        decision = packet["story_decisions"][0]
        self.assertEqual(decision["story_key"], "current-provisional")
        self.assertEqual(decision["relationship"], "distinct_story")
        self.assertEqual(decision["change_type"], "new")
        self.assertEqual(decision["disposition"], "full_report")
        self.assertEqual(decision["prior_story_key"], "")
        self.assertEqual(packet["identity_gate"]["forced_new_without_candidate"], 1)

    def test_model_cannot_link_to_key_outside_candidate_set(self) -> None:
        packet = enforce_candidate_identity_gate(
            {
                "story_decisions": [
                    {
                        "article_ids": ["current"],
                        "relationship": "same_story",
                        "prior_story_key": "not-supplied",
                        "change_type": "unchanged",
                        "disposition": "omit",
                        "confidence": 0.95,
                    }
                ]
            },
            _story_memory(candidates=["allowed-prior"]),
        )

        decision = packet["story_decisions"][0]
        self.assertEqual(decision["story_key"], "current-provisional")
        self.assertEqual(decision["relationship"], "uncertain")
        self.assertEqual(decision["disposition"], "full_report")
        self.assertEqual(packet["identity_gate"]["rejected_links"], 1)

    def test_only_an_allowed_candidate_link_reuses_prior_story_key(self) -> None:
        candidate = _candidate("current", "Current observation", "A source-backed current fact is reported.")
        _annotate(candidate, "current-provisional")
        selected = [_article(candidate)]
        packet = enforce_candidate_identity_gate(
            {
                "story_decisions": [
                    {
                        "article_ids": ["current"],
                        "relationship": "same_story",
                        "prior_story_key": "allowed-prior",
                        "change_type": "status_change",
                        "materiality": 0.9,
                        "disposition": "full_report",
                        "confidence": 0.82,
                    }
                ]
            },
            _story_memory(candidates=["allowed-prior", "other-prior"]),
        )

        apply_delta_signals_to_selected(selected=selected, delta_packet=packet)

        self.assertEqual(candidate_memory_annotation(candidate).story_key, "allowed-prior")
        self.assertEqual(packet["identity_gate"]["accepted_links"], 1)

    def test_missing_model_decision_is_synthesized_and_remains_visible(self) -> None:
        packet = enforce_candidate_identity_gate({}, _story_memory(candidates=[]))

        self.assertEqual(len(packet["story_decisions"]), 1)
        self.assertEqual(packet["story_decisions"][0]["disposition"], "full_report")
        self.assertEqual(packet["identity_gate"]["synthesized_decisions"], 1)

    def test_distinct_decision_reconciles_contradictory_editorial_lists(self) -> None:
        packet = enforce_candidate_identity_gate(
            {
                "story_decisions": [
                    {
                        "article_ids": ["current"],
                        "relationship": "distinct_story",
                        "change_type": "unchanged",
                        "disposition": "omit",
                        "confidence": 0.9,
                        "bullet": "Current observation",
                    }
                ],
                "unchanged_but_important": [
                    {"item": "Contradictory model row", "article_ids": ["current"]}
                ],
            },
            _story_memory(candidates=["plausible-but-distinct"]),
        )

        self.assertEqual(packet["unchanged_but_important"], [])
        self.assertEqual(packet["story_decisions"][0]["change_type"], "new")
        self.assertEqual(packet["story_decisions"][0]["disposition"], "full_report")
        self.assertEqual(packet["new"][0]["article_ids"], ["current"])


class StoryStoreTests(unittest.TestCase):
    def test_legacy_index_and_ledger_merge_once_into_canonical_store(self) -> None:
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "story_index.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "stories": [
                            {
                                "story_key": "elf-toenail-trend",
                                "story_family_key": "faraway-fashion",
                                "title": "Moon-crystal toenail fashion",
                                "topic": "Faraway fashion",
                                "tokens": ["moon", "crystal", "toenail", "fashion"],
                                "first_seen": "2026-01-01",
                                "last_seen": "2026-01-03",
                                "status": "active",
                                "last_change_type": "escalated",
                                "last_delta_summary": "The guild adopted the style.",
                                "last_knowns": ["The style is now official."],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "story_ledger.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stories": [
                            {
                                "story_key": "elf-toenail-trend",
                                "story_family_key": "faraway-fashion",
                                "title": "Moon-crystal charms reach Lyr Vale",
                                "aliases": ["Moon-crystal charms reach Lyr Vale"],
                                "entity_tokens": ["moon", "crystal", "lyr", "vale"],
                                "event_tokens": ["charms", "fashion"],
                                "first_seen": "2026-01-01",
                                "last_seen": "2026-01-02",
                                "source_document_ids": ["elf-01"],
                                "facts": [
                                    {
                                        "fact_id": "fact:elf-01",
                                        "text": "Moon-crystal toenail charms debuted at Lyr Vale market.",
                                        "kind": "source_sentence",
                                        "source_id": "elf-01",
                                        "source_name": "Faraway Chronicle",
                                        "source_url": "https://fixture.test/elf-01",
                                        "published_at": "2026-01-01T00:00:00+00:00",
                                        "observed_at": "2026-01-01",
                                        "tokens": ["moon", "crystal", "toenail", "charms"],
                                        "user_visible": True,
                                    }
                                ],
                                "last_user_visible_fact_ids": ["fact:elf-01"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = StoryStore.from_state_dir(root)
            self.assertTrue(store.using_legacy_migration)
            records = store.records()

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.title, "Moon-crystal toenail fashion")
            self.assertEqual(record.last_seen, "2026-01-03")
            self.assertEqual(record.last_change_type, "escalated")
            self.assertEqual(record.source_document_ids, ["elf-01"])
            self.assertEqual(record.facts[0].source_url, "https://fixture.test/elf-01")

            store.replace_records(records)
            self.assertTrue((root / "story_store.json").exists())
            self.assertTrue((root / "story_index.json").exists())
            self.assertTrue((root / "story_ledger.json").exists())
            self.assertFalse(StoryStore.from_state_dir(root).using_legacy_migration)

    def test_unvalidated_retrieval_candidate_cannot_hard_suppress_selection(self) -> None:
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            store = StoryStore.from_state_dir(root)
            previous = _candidate(
                "prior",
                "Glass Orchard northern gate closes for repair",
                "The Glass Orchard northern gate closed for a month-long repair.",
            )
            _annotate(previous, "glass-orchard-gate")
            store.update_selected(selected=[_article(previous)], date="2026-03-01")
            coverage = CoverageMemoryStore.from_state_dir(root)
            coverage.write_records(
                [
                    CoverageRecord(
                        schema_version=1,
                        date="2026-03-01",
                        brief_name="general",
                        story_key="glass-orchard-gate",
                        story_family_key="invented-events",
                        title=previous.title,
                        prominence="lead",
                        article_ids=["prior"],
                    )
                ]
            )
            current = _candidate(
                "current",
                "Inspectors revisit Glass Orchard northern entrance",
                "Inspectors revisited the Glass Orchard northern gate after the repair.",
            )
            decision = HeadlineDecision(
                "current",
                score=8.0,
                novelty=2.0,
                impact=4.0,
                urgency=2.0,
            )

            selected = select_articles(
                [current],
                {"current": decision},
                [TopicConfig(name="Invented")],
                FilteringConfig(
                    headline_score_cutoff=0.0,
                    max_selected_articles=1,
                    max_selected_per_source=0,
                ),
                user_memory=UserMemory(),
                memory_config=MemoryConfig(enabled=True),
                coverage_store=coverage,
                story_store=store,
                date="2026-03-02",
            )

            self.assertEqual([article.candidate.id for article in selected], ["current"])
            annotation = candidate_memory_annotation(current)
            self.assertNotEqual(annotation.story_key, "glass-orchard-gate")
            self.assertGreaterEqual(annotation.score_adjustment, -0.35)
            self.assertEqual(current.metadata["memory_identity_state"], "provisional")

    def test_store_persists_exact_source_facts_and_provenance(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = StoryStore.from_state_dir(Path(raw_dir))
            body = "Moon-crystal toenail charms debuted at Lyr Vale market. The guild has not adopted them yet."
            candidate = _candidate("elf-01", "Moon-crystal toenail charms reach Lyr Vale", body)
            _annotate(candidate, "elf-toenail-trend")

            store.update_selected(
                selected=[_article(candidate)],
                date="2026-01-01",
                visible_article_ids=["elf-01"],
                delta_packet={},
            )
            reloaded = StoryStore.from_state_dir(Path(raw_dir)).records()[0]

            self.assertEqual(reloaded.story_key, "elf-toenail-trend")
            self.assertIn("elf-01", reloaded.source_document_ids)
            source_fact = next(
                fact
                for fact in reloaded.facts
                if fact.text == "Moon-crystal toenail charms debuted at Lyr Vale market."
            )
            self.assertEqual(source_fact.text, "Moon-crystal toenail charms debuted at Lyr Vale market.")
            self.assertEqual(source_fact.source_url, "https://fixture.test/elf-01")
            self.assertTrue(source_fact.user_visible)
            self.assertIn(source_fact.fact_id, reloaded.last_user_visible_fact_ids)

            store.update_selected(
                selected=[_article(candidate)],
                date="2026-01-02",
                visible_article_ids=[],
                delta_packet={},
            )
            hidden_pass_record = StoryStore.from_state_dir(Path(raw_dir)).records()[0]
            hidden_pass_fact = next(
                fact for fact in hidden_pass_record.facts if fact.fact_id == source_fact.fact_id
            )
            self.assertTrue(hidden_pass_fact.user_visible)
            self.assertIn(source_fact.fact_id, hidden_pass_record.last_user_visible_fact_ids)

    def test_heuristic_retrieval_handles_invented_domain_and_changed_headline(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = StoryStore.from_state_dir(Path(raw_dir))
            first_body = (
                "Artisans demonstrated moon-crystal toenail charms at Lyr Vale market. "
                "The Royal Tailors Guild called the style voluntary."
            )
            first = _candidate(
                "elf-01",
                "Moon-crystal toenail charms arrive at Lyr Vale market",
                first_body,
                category="faraway fashion",
            )
            _annotate(first, "elf-toenail-trend")
            store.update_selected(selected=[_article(first)], date="2026-01-01")

            update_body = (
                "The Royal Tailors Guild adopted moon-crystal toenail charms for winter uniforms, "
                "moving the style from market demonstrations into official dress."
            )
            update = _candidate(
                "elf-03",
                "Royal Tailors Guild adopts enchanted pedicure for winter uniforms",
                update_body,
                category="faraway fashion",
            )
            matches = store.candidate_stories(update, source_text=update_body)

            self.assertTrue(matches)
            self.assertEqual(matches[0].record.story_key, "elf-toenail-trend")
            self.assertGreaterEqual(matches[0].score, 0.25)

            unrelated = _candidate(
                "noise-01",
                "Submarine cable auction closes in Pelagic Republic",
                "The communications ministry selected a bidder for an undersea cable concession.",
                category="infrastructure",
            )
            self.assertEqual(store.candidate_stories(unrelated, source_text=unrelated.snippet), [])

    def test_numeric_signals_rank_model_four_above_model_three(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = StoryStore.from_state_dir(Path(raw_dir))
            model_three = _candidate(
                "beacon-3",
                "Aster launches Model 3 rescue beacon",
                "Aster launched its Model 3 rescue beacon for coastal crews.",
                category="devices",
            )
            model_four = _candidate(
                "beacon-4",
                "Aster launches Model 4 rescue beacon",
                "Aster launched its Model 4 rescue beacon for mountain crews.",
                category="devices",
            )
            _annotate(model_three, "aster-beacon-model-3")
            _annotate(model_four, "aster-beacon-model-4")
            store.update_selected(selected=[_article(model_three), _article(model_four)], date="2026-02-01")

            query = _candidate(
                "beacon-4-review",
                "Safety agency reviews Aster Model 4 beacon",
                "The safety agency opened a review of Aster's Model 4 rescue beacon.",
                category="devices",
            )
            matches = store.candidate_stories(query, source_text=query.snippet)

            self.assertTrue(matches)
            self.assertEqual(matches[0].record.story_key, "aster-beacon-model-4")
            self.assertNotIn("aster-beacon-model-3", [match.record.story_key for match in matches])
            permissive_matches = store.candidate_stories(query, source_text=query.snippet, min_score=0.0)
            model_three_match = next(
                match for match in permissive_matches if match.record.story_key == "aster-beacon-model-3"
            )
            self.assertTrue(model_three_match.numeric_conflict)

    def test_story_context_contains_only_retrieved_source_backed_candidates(self) -> None:
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            store = StoryStore.from_state_dir(root)
            first = _candidate(
                "ledger-old",
                "Glass Orchard opens its northern gate",
                "The Glass Orchard opened its northern gate after a month-long repair.",
            )
            _annotate(first, "glass-orchard-gate")
            store.update_selected(selected=[_article(first)], date="2026-03-01", visible_article_ids=["ledger-old"])

            current = _candidate(
                "ledger-new",
                "Inspectors revisit Glass Orchard northern entrance",
                "Inspectors revisited the Glass Orchard northern gate after the repair.",
            )
            _annotate(current, "current-glass-observation")
            context = build_story_memory_context(
                selected=[_article(current)],
                story_groups=[],
                story_store=store,
                coverage_store=None,
                prior_reports=[],
                date="2026-03-02",
            )

            baselines = context["stories"][0]["prior_baselines"]
            self.assertEqual(len(baselines), 1)
            self.assertEqual(baselines[0]["story_key"], "glass-orchard-gate")
            self.assertTrue(baselines[0]["source_facts"])
            self.assertEqual(baselines[0]["source_facts"][0]["source_id"], "ledger-old")
            self.assertLessEqual(len(baselines), 3)

            compact = _compact_decision_baselines(context, [], [_article(current)])
            self.assertTrue(compact[0]["source_facts"])
            self.assertEqual(compact[0]["source_facts"][0]["source_id"], "ledger-old")

    def test_full_corpus_candidate_recall_regression(self) -> None:
        corpus_path = Path(__file__).resolve().parents[1] / "evals" / "cases" / "change_monitoring.v1.json"
        payload = evaluate_story_store_retrieval(load_corpus(corpus_path)).payload()

        self.assertEqual(payload["documents"], 74)
        self.assertEqual(payload["historical_continuations"], 25)
        self.assertGreaterEqual(payload["recall_at_3"], 0.95)
        self.assertGreaterEqual(payload["new_story_without_candidate_rate"], 0.97)
        self.assertEqual(payload["same_day_only_continuations_excluded"], 1)
        self.assertTrue(payload["uses_private_gold_for_historical_writeback"])


if __name__ == "__main__":
    unittest.main()
