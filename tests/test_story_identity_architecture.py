from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from mydailynews.analysis.story_delta import StoryDeltaAnalyzer
from mydailynews.app.models import (
    FilteringConfig,
    DeltaExtractionConfig,
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
from mydailynews.memory.context import _story_matches, build_story_memory_context
from mydailynews.memory.coverage import CoverageMemoryStore, CoverageRecord
from mydailynews.memory.recall import partition_selected_for_brief
from mydailynews.memory.story_store import (
    StoryStore,
    source_facts_for_article,
    story_baseline_payload,
)
from mydailynews.memory.story_retrieval import StoryCandidateMatch
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
            self.assertTrue((root / "memory.sqlite3").exists())
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

    def test_suppressed_story_still_updates_source_evidence(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = StoryStore.from_state_dir(Path(raw_dir))
            candidate = _candidate(
                "hidden-repeat",
                "Harbor bridge remains closed",
                "Inspectors said the Harbor bridge remains closed.",
            )
            _annotate(candidate, "harbor-bridge")
            store.update_selected(
                selected=[_article(candidate)],
                date="2026-04-01",
                visible_article_ids=[],
                delta_packet={"story_decisions": [{
                    "article_ids": [candidate.id],
                    "relationship": "same_story",
                    "change_type": "unchanged",
                    "disposition": "omit",
                    "summary": "Repeated source evidence.",
                }]},
            )
            record = store.records()[0]
            self.assertIn(candidate.id, record.source_document_ids)
            self.assertTrue(any(fact.source_id == candidate.id for fact in record.facts))
            self.assertEqual(record.last_seen, "2026-04-01")

    def test_production_delta_selects_one_prior_then_edits_all_cards(self) -> None:
        current = _candidate(
            "current",
            "Bridge remains closed",
            "The bridge remains closed pending inspection.",
        )
        selected = [_article(current)]
        story_memory = {
            "stories": [{
                "story_key": "current-provisional",
                "current_title": "Bridge remains closed",
                "current_article_ids": ["current"],
                "current_articles": [{"id": "current", "headline": current.title}],
                "prior_baselines": [
                    {
                        "story_key": "wrong-bridge",
                        "title": "A different bridge opened",
                        "source_facts": [{
                            "fact_id": "fact:wrong",
                            "text": "A different bridge opened.",
                            "source_id": "old-wrong",
                        }],
                    },
                    {
                        "story_key": "right-bridge",
                        "title": "Bridge remains closed",
                        "source_facts": [{
                            "fact_id": "fact:right",
                            "text": "The bridge remains closed pending inspection.",
                            "source_id": "old-right",
                        }],
                    },
                ],
            }],
        }

        class Client:
            def __init__(self):
                self.config = SimpleNamespace(
                    backend="test",
                    effective_model_label="test",
                )
                self.calls = []

            def complete_json(self, system, user, **kwargs):
                schema_name = kwargs["json_schema"].name
                self.calls.append(schema_name)
                if schema_name == "story_identity_selection":
                    return {
                        "relationship": "same_story",
                        "prior_story_key": "right-bridge",
                        "confidence": 0.9,
                        "basis": "Same bridge and unresolved closure.",
                    }
                if schema_name == "fact_operations":
                    comparison = json.loads(user.split("Story comparison:\n", 1)[1].split("\n\nReturn", 1)[0])
                    return {
                        "story_key": "right-bridge",
                        "operations": [
                            {
                                "operation": "repeat",
                                "current_evidence_id": claim["claim_id"],
                                "prior_fact_ids": ["fact:right"],
                            }
                            for claim in comparison["current_evidence"]
                        ],
                    }
                return {
                    "decisions": [{
                        "card_id": "story-card-001",
                        "disposition": "omit",
                        "materiality": 0,
                        "summary": "Bridge closure is unchanged.",
                        "basis": "No information gain today.",
                    }],
                }

        client = Client()
        packet = StoryDeltaAnalyzer(client, DeltaExtractionConfig(enabled=True)).extract(
            selected,
            UserMemory(),
            "daily brief",
            story_memory,
        )

        self.assertEqual(
            client.calls,
            ["story_identity_selection", "fact_operations", "story_editor_selection"],
        )
        self.assertEqual(packet["story_cards"][0]["prior_story_key"], "right-bridge")
        self.assertNotIn("fact:wrong", json.dumps(packet["story_cards"][0]))
        self.assertEqual(packet["story_decisions"][0]["change_type"], "unchanged")
        self.assertEqual(packet["story_decisions"][0]["disposition"], "omit")

        with TemporaryDirectory() as raw_dir:
            _annotate(current, "right-bridge")
            store = StoryStore.from_state_dir(Path(raw_dir))
            store.update_selected(
                selected=selected,
                date="2026-09-10",
                delta_packet=packet,
            )
            event = store.records()[0].thread_events[-1]
            self.assertEqual(event.operations[0]["operation"], "repeat")
            self.assertEqual(
                event.current_evidence_ids,
                packet["story_decisions"][0]["current_evidence_ids"],
            )
            self.assertEqual(event.prior_evidence_ids, ["fact:right"])

    def test_stashed_replacement_is_the_next_days_active_baseline(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = StoryStore.from_state_dir(Path(raw_dir))
            first = _article(_candidate(
                "state-a",
                "Harbor gate schedule",
                "Officials said the harbor gate will reopen on Monday.",
            ))
            _annotate(first.candidate, "harbor-gate")
            store.update_selected(
                selected=[first],
                date="2026-09-08",
                visible_article_ids=["state-a"],
            )
            first_record = store.records()[0]
            fact_a = next(
                fact
                for fact in first_record.facts
                if fact.kind == "source_sentence"
            )

            replacement = _article(_candidate(
                "state-b",
                "Harbor gate schedule",
                "Officials corrected the schedule: the harbor gate will reopen on Tuesday rather than Monday.",
            ))
            _annotate(replacement.candidate, "harbor-gate")
            fact_b = next(
                fact
                for fact in source_facts_for_article(
                    replacement,
                    observed_at="2026-09-09",
                    user_visible=False,
                )
                if fact.kind == "source_sentence"
            )
            store.update_selected(
                selected=[replacement],
                date="2026-09-09",
                visible_article_ids=[],
                delta_packet={
                    "story_delta_version": "story-cards.v1",
                    "story_decisions": [{
                        "story_key": "harbor-gate",
                        "article_ids": ["state-b"],
                        "relationship": "same_story",
                        "change_type": "correction",
                        "materiality": 1.0,
                        "disposition": "omit",
                        "editor_safe_to_defer": True,
                        "current_evidence_ids": [fact_b.fact_id],
                        "prior_evidence_ids": [fact_a.fact_id],
                        "superseded_prior_evidence_ids": [fact_a.fact_id],
                        "operations": [{
                            "operation": "replace",
                            "current_evidence_id": fact_b.fact_id,
                            "prior_fact_ids": [fact_a.fact_id],
                        }],
                    }],
                },
            )

            next_day_record = StoryStore.from_state_dir(Path(raw_dir)).records()[0]
            baseline = story_baseline_payload(StoryCandidateMatch(
                score=1.0,
                record=next_day_record,
                lexical_score=1.0,
                alias_score=0.0,
                entity_score=0.0,
                event_score=0.0,
                fact_score=1.0,
                numeric_conflict=False,
            ))

        baseline_ids = {fact["fact_id"] for fact in baseline["source_facts"]}
        self.assertEqual(next_day_record.last_shown, "2026-09-08")
        self.assertIn(fact_b.fact_id, next_day_record.active_fact_ids)
        self.assertNotIn(fact_a.fact_id, next_day_record.active_fact_ids)
        self.assertIn(fact_b.fact_id, baseline_ids)
        self.assertNotIn(fact_a.fact_id, baseline_ids)

    def test_only_validated_story_editor_can_defer_a_material_story(self) -> None:
        article = _article(_candidate("deferred", "Rare event", "A rare event occurred."))
        decision = {
            "story_key": "rare-event",
            "article_ids": ["deferred"],
            "relationship": "distinct_story",
            "change_type": "new",
            "confidence": 0.9,
            "disposition": "omit",
            "editor_safe_to_defer": True,
        }
        included, omitted = partition_selected_for_brief(
            selected=[article],
            delta_packet={
                "story_delta_version": "story-cards.v1",
                "story_decisions": [decision],
            },
        )
        self.assertEqual(included, [])
        self.assertEqual(omitted, [article])

        for packet in (
            {"story_delta_version": "story-cards.v1", "story_decisions": [{**decision, "editor_safe_to_defer": False}]},
            {"story_decisions": [decision]},
        ):
            included, omitted = partition_selected_for_brief(selected=[article], delta_packet=packet)
            self.assertEqual(included, [article])
            self.assertEqual(omitted, [])

    def test_candidate_context_excludes_same_day_history(self) -> None:
        article = _article(_candidate("current", "Bridge update", "The bridge remains closed."))

        def match(story_key: str, last_seen: str, score: float) -> StoryCandidateMatch:
            return StoryCandidateMatch(
                score=score,
                record=SimpleNamespace(
                    story_key=story_key,
                    title=story_key,
                    last_seen=last_seen,
                    source_document_ids=[],
                ),
                lexical_score=score,
                alias_score=0.0,
                entity_score=0.0,
                event_score=0.0,
                fact_score=0.0,
                numeric_conflict=False,
            )

        class Store:
            def candidate_stories(self, *args, **kwargs):
                return [
                    match("written-by-general", "2026-09-10", 0.9),
                    match("yesterdays-baseline", "2026-09-09", 0.8),
                ]

        matches = _story_matches(
            [article],
            story_store=Store(),
            limit=3,
            as_of_date="2026-09-10",
        )

        self.assertEqual(
            [item.record.story_key for item in matches],
            ["yesterdays-baseline"],
        )
        self.assertEqual(
            [item["story_key"] for item in article.candidate.metadata["memory_prior_story_candidates"]],
            ["yesterdays-baseline"],
        )

    def test_store_compacts_repeated_and_old_evidence_but_keeps_visible_baseline(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = StoryStore.from_state_dir(Path(raw_dir))
            first = _candidate(
                "repeat-1",
                "Harbor bridge review begins",
                "Inspectors began a review of the Harbor bridge on Monday.",
            )
            _annotate(first, "harbor-bridge-review")
            store.update_selected(
                selected=[_article(first)],
                date="2026-01-01",
                visible_article_ids=["repeat-1"],
            )
            visible_ids = store.records()[0].last_user_visible_fact_ids

            # Syndicated copies of identical source claims must not create a
            # new evidence history entry for each article ID.
            repeat = _candidate(
                "repeat-2",
                "Harbor bridge review begins",
                "Inspectors began a review of the Harbor bridge on Monday.",
            )
            _annotate(repeat, "harbor-bridge-review")
            store.update_selected(selected=[_article(repeat)], date="2026-01-02")
            self.assertEqual(len(store.records()[0].facts), 2)

            for index in range(40):
                candidate = _candidate(
                    f"bridge-{index}",
                    f"Harbor bridge review update {index}",
                    f"Inspectors published Harbor bridge review update number {index}.",
                )
                _annotate(candidate, "harbor-bridge-review")
                store.update_selected(selected=[_article(candidate)], date=f"2026-02-{(index % 27) + 1:02d}")

            record = StoryStore.from_state_dir(Path(raw_dir)).records()[0]
            self.assertLessEqual(len(record.facts), 32)
            self.assertTrue(set(visible_ids).issubset({fact.fact_id for fact in record.facts}))
            self.assertEqual(record.last_user_visible_fact_ids, visible_ids)

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

            strict_context = build_story_memory_context(
                selected=[_article(current)],
                story_groups=[],
                story_store=store,
                coverage_store=None,
                prior_reports=[],
                date="2026-03-02",
                candidate_threshold=1.0,
            )
            self.assertEqual(strict_context["stories"][0]["prior_baselines"], [])

            # A general brief may already have written today's record before a
            # detailed brief starts. It must not become the detailed baseline.
            same_day_context = build_story_memory_context(
                selected=[_article(current)],
                story_groups=[],
                story_store=store,
                coverage_store=None,
                prior_reports=[],
                date="2026-03-01",
            )
            self.assertEqual(same_day_context["stories"][0]["prior_baselines"], [])

    def test_full_corpus_candidate_recall_regression(self) -> None:
        corpus_path = Path(__file__).resolve().parents[1] / "evals" / "cases" / "change_monitoring.v1.json"
        payload = evaluate_story_store_retrieval(load_corpus(corpus_path)).payload()

        self.assertEqual(payload["documents"], 74)
        self.assertEqual(payload["historical_continuations"], 25)
        self.assertGreaterEqual(payload["recall_at_3"], 0.95)
        self.assertGreaterEqual(payload["new_story_without_candidate_rate"], 0.97)
        self.assertEqual(payload["same_day_only_continuations_excluded"], 1)
        self.assertTrue(payload["uses_private_gold_for_historical_writeback"])

    def test_story_store_persists_bounded_claim_thread_events(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = StoryStore.from_state_dir(Path(raw_dir))
            for index in range(30):
                candidate = _candidate(
                    f"thread-{index}",
                    f"Glass Orchard gate update {index}",
                    f"The Glass Orchard gate entered operational state {index}.",
                )
                _annotate(candidate, "glass-orchard-thread")
                packet = {"story_delta_version": "story-cards.v1", "story_decisions": [{
                    "article_ids": [candidate.id],
                    "relationship": "same_story" if index else "distinct_story",
                    "change_type": "status_change" if index else "new",
                    "materiality": 0.9,
                    "disposition": "full_report",
                    "summary": f"Operational state {index} was source-confirmed.",
                    "current_evidence_ids": [f"fact:state-{index}"],
                    "prior_evidence_ids": [f"fact:state-{index - 1}"] if index else [],
                    "superseded_prior_evidence_ids": [f"fact:state-{index - 1}"] if index else [],
                    "operations": [{
                        "operation": "replace" if index else "add",
                        "current_evidence_id": f"fact:state-{index}",
                        "prior_fact_ids": [f"fact:state-{index - 1}"] if index else [],
                    }],
                }]}
                store.update_selected(
                    selected=[_article(candidate)], date=f"2026-04-{index + 1:02d}", delta_packet=packet,
                )
            record = StoryStore.from_state_dir(Path(raw_dir)).records()[0]

        self.assertEqual(len(record.thread_events), 24)
        self.assertEqual(record.thread_events[-1].current_evidence_ids, ["fact:state-29"])
        self.assertEqual(record.thread_events[-1].superseded_prior_evidence_ids, ["fact:state-28"])
        self.assertNotIn("fact:state-0", [fact_id for event in record.thread_events for fact_id in event.current_evidence_ids])


if __name__ == "__main__":
    unittest.main()
