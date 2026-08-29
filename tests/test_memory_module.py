from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
import uuid

from mydailynews.app.config import load_config
from mydailynews.app.models import (
    FilteringConfig,
    HeadlineDecision,
    MemoryAnnotation,
    MemoryConfig,
    NewsCandidate,
    PriorReportsSourceConfig,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.domain.headline_selection import select_articles, selection_rationale_rows
from mydailynews.memory.coverage import CoverageMemoryStore, CoverageRecord
from mydailynews.memory.feedback import FEEDBACK_ACTIONS, FeedbackEvent, FeedbackStore
from mydailynews.memory.health import memory_health_checks
from mydailynews.memory.learned_preferences import LearnedPreferences, LearnedPreferencesStore
from mydailynews.memory.preference_learning import apply_feedback_event, preference_delta_for_event
from mydailynews.memory.recall import build_recall_packet
from mydailynews.memory.repair import (
    coverage_row_id,
    feedback_row_id,
    merge_stories,
    repair_coverage_rows,
    repair_feedback_events,
    split_story,
)
from mydailynews.memory.story_store import StoryStore
from mydailynews.memory.story_keys import story_identity_for_candidate
from mydailynews.pipeline.handoff import load_brief_handoff, write_brief_handoff
from mydailynews.retrieval.reports import PriorReportRetriever


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "memory_module"
PUBLISHED_AT = datetime(2026, 6, 27, tzinfo=timezone.utc)


def _candidate(candidate_id: str, title: str, *, source: str = "Example News") -> NewsCandidate:
    return NewsCandidate(
        id=candidate_id,
        source=source,
        category="world",
        title=title,
        url=f"https://example.test/{candidate_id}",
        snippet="A detailed report with enough context to score and select.",
        published_at=PUBLISHED_AT,
        metadata={"topic_name": "Major world events"},
    )


def _memory_config(**overrides) -> MemoryConfig:
    values = {
        "enabled": True,
        "coverage_window_days": 10,
        "coverage_retention_days": 30,
        "story_stale_after_days": 7,
        "story_retention_days": 30,
        "recent_story_penalty": 0.6,
        "recent_lead_penalty": 1.1,
        "material_update_boost": 0.9,
        "max_selected_per_story": 1,
        "max_selected_per_story_family": 0,
        "recall_prompt_enabled": True,
        "save_recall_packets": True,
        "feedback_enabled": False,
    }
    values.update(overrides)
    return MemoryConfig(**values)


class MemoryModuleTests(unittest.TestCase):
    def _temp_dir(self) -> Path:
        path = TEMP_ROOT / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_config_accepts_memory_section_and_rejects_unknown_keys(self) -> None:
        config = load_config(REPO_ROOT / "config.example.json")

        self.assertTrue(config.memory.enabled)
        self.assertEqual(config.memory.state_dir, "state/memory")
        self.assertTrue(config.memory.recall_prompt_enabled)
        self.assertTrue(config.memory.save_recall_packets)
        self.assertEqual(config.memory.coverage_retention_days, 30)
        self.assertEqual(config.memory.story_stale_after_days, 7)
        self.assertEqual(config.memory.story_retention_days, 30)
        self.assertTrue(config.memory.feedback_enabled)

        payload = json.loads((REPO_ROOT / "config.example.json").read_text(encoding="utf-8-sig"))
        payload["memory"]["unknown_knob"] = True
        path = self._temp_dir() / "bad_memory.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, r"Config section memory has unrecognized key\(s\): unknown_knob"):
            load_config(path)

        missing_memory = json.loads((REPO_ROOT / "config.example.json").read_text(encoding="utf-8-sig"))
        missing_memory.pop("memory", None)
        missing_path = self._temp_dir() / "missing_memory.json"
        missing_path.write_text(json.dumps(missing_memory), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, r"Config missing required section\(s\): memory"):
            load_config(missing_path)

    def test_story_key_normalization_ignores_weak_update_terms(self) -> None:
        first = story_identity_for_candidate(_candidate("a", "Breaking: Iran Israel ceasefire updates"))
        second = story_identity_for_candidate(_candidate("b", "Iran Israel ceasefire"))
        unrelated = story_identity_for_candidate(_candidate("c", "Central bank cuts interest rates"))

        self.assertEqual(first.story_key, second.story_key)
        self.assertNotEqual(first.story_key, unrelated.story_key)

    def test_recent_lead_penalty_lowers_rank_and_marks_skip_reason(self) -> None:
        temp_dir = self._temp_dir()
        repeated = _candidate("repeat", "Iran Israel ceasefire")
        fresh = _candidate("fresh", "Central bank cuts interest rates", source="Finance Wire")
        story_key = story_identity_for_candidate(repeated).story_key
        store = CoverageMemoryStore.from_state_dir(temp_dir)
        store.write_records(
            [
                CoverageRecord(
                    schema_version=1,
                    date="2026-06-26",
                    brief_name="general",
                    story_key=story_key,
                    story_family_key="iran-israel",
                    title="Iran Israel ceasefire",
                    prominence="lead",
                    article_ids=["old"],
                    rank_score=8.8,
                )
            ]
        )
        decisions = {
            "repeat": HeadlineDecision("repeat", score=8.5, novelty=4.0, impact=7.0),
            "fresh": HeadlineDecision("fresh", score=8.0, novelty=8.0, impact=8.0),
        }

        selected = select_articles(
            [repeated, fresh],
            decisions,
            [TopicConfig(name="Major world events")],
            FilteringConfig(headline_score_cutoff=0.0, max_selected_articles=1, max_selected_per_source=0),
            user_memory=UserMemory(),
            memory_config=_memory_config(max_selected_per_story=0),
            coverage_store=store,
            date="2026-06-27",
        )

        self.assertEqual([article.candidate.id for article in selected], ["fresh"])
        annotation = candidate_memory_annotation(repeated)
        self.assertIsNotNone(annotation)
        self.assertLess(annotation.score_adjustment, 0)
        self.assertEqual(decisions["repeat"].selection_reason_code, "skipped_recent_coverage")
        rationale = selection_rationale_rows([repeated, fresh], decisions)
        self.assertTrue(any(row["memory"] and row["memory"]["story_key"] == story_key for row in rationale))

    def test_material_update_boost_can_offset_recent_body_coverage(self) -> None:
        temp_dir = self._temp_dir()
        candidate = _candidate("update", "Iran Israel ceasefire")
        story_key = story_identity_for_candidate(candidate).story_key
        store = CoverageMemoryStore.from_state_dir(temp_dir)
        store.write_records(
            [
                CoverageRecord(
                    schema_version=1,
                    date="2026-06-26",
                    brief_name="general",
                    story_key=story_key,
                    story_family_key="iran-israel",
                    title="Iran Israel ceasefire",
                    prominence="body",
                    article_ids=["old"],
                )
            ]
        )
        decisions = {
            "update": HeadlineDecision(
                "update",
                score=8.0,
                novelty=9.0,
                impact=9.0,
                urgency=8.0,
                angle_type="policy_change",
            )
        }

        selected = select_articles(
            [candidate],
            decisions,
            [TopicConfig(name="Major world events")],
            FilteringConfig(headline_score_cutoff=0.0, max_selected_articles=1, max_selected_per_source=0),
            user_memory=UserMemory(),
            memory_config=_memory_config(recent_story_penalty=0.4, max_selected_per_story=0),
            coverage_store=store,
            date="2026-06-27",
        )

        self.assertEqual([article.candidate.id for article in selected], ["update"])
        annotation = candidate_memory_annotation(candidate)
        self.assertIsNotNone(annotation)
        self.assertGreater(annotation.score_adjustment, 0)
        self.assertEqual(selected[0].selection_reason_code, "selected_material_update_override")

    def test_non_material_recent_story_is_ineligible_even_when_capacity_remains(self) -> None:
        temp_dir = self._temp_dir()
        repeated = _candidate("repeat", "Iran Israel ceasefire")
        fresh = _candidate("fresh", "Central bank cuts interest rates", source="Finance Wire")
        story_key = story_identity_for_candidate(repeated).story_key
        store = CoverageMemoryStore.from_state_dir(temp_dir)
        store.write_records(
            [
                CoverageRecord(
                    schema_version=1,
                    date="2026-06-26",
                    brief_name="general",
                    story_key=story_key,
                    story_family_key="iran-israel",
                    title="Iran Israel ceasefire",
                    prominence="body",
                    article_ids=["old"],
                )
            ]
        )
        decisions = {
            "repeat": HeadlineDecision("repeat", score=9.0, novelty=3.0, impact=5.0, urgency=3.0),
            "fresh": HeadlineDecision("fresh", score=8.0, novelty=8.0, impact=8.0),
        }

        selected = select_articles(
            [repeated, fresh],
            decisions,
            [TopicConfig(name="Major world events")],
            FilteringConfig(headline_score_cutoff=0.0, max_selected_articles=2, max_selected_per_source=0),
            user_memory=UserMemory(),
            memory_config=_memory_config(max_selected_per_story=0),
            coverage_store=store,
            date="2026-06-27",
        )

        self.assertEqual([article.candidate.id for article in selected], ["fresh"])
        self.assertEqual(decisions["repeat"].selection_reason_code, "skipped_recent_coverage")

    def test_story_store_matches_same_event_with_changed_sequence_words(self) -> None:
        temp_dir = self._temp_dir()
        store = StoryStore.from_state_dir(temp_dir)
        store.path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stories": [
                        {
                            "story_key": "strikes-hit-iran-seventh-consecutive",
                            "story_family_key": "iran-conflict",
                            "title": "US-Iran conflict",
                            "topic": "World",
                            "tokens": ["strikes", "hit", "iran", "seventh", "consecutive"],
                            "first_seen": "2026-06-20",
                            "last_seen": "2026-06-26",
                            "status": "active",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        identity = store.match_candidate(_candidate("next", "US launches Iran strikes for ninth night"))

        self.assertEqual(identity.story_key, "strikes-hit-iran-seventh-consecutive")
        self.assertGreaterEqual(identity.match_confidence, 0.58)

    def test_prior_reports_use_one_structured_brief_per_date(self) -> None:
        temp_dir = self._temp_dir()
        for name, title in (
            ("2026-06-26_narrative_brief.json", "Narrative"),
            ("2026-06-26_general_brief.json", "General"),
            ("2026-06-26_detailed_brief.json", "Detailed"),
            ("2026-06-25_general_brief.json", "Older general"),
        ):
            (temp_dir / name).write_text(json.dumps({"title": title, "lead": title}), encoding="utf-8")
        retriever = PriorReportRetriever(
            PriorReportsSourceConfig(enabled=True, days=7, max_reports=5, output_dir=str(temp_dir)),
            str(temp_dir),
        )

        reports = retriever.fetch(datetime(2026, 6, 28, tzinfo=timezone.utc).date())

        self.assertEqual([report.title for report in reports], ["Detailed", "Older general"])

    def test_story_caps_are_disabled_when_memory_is_disabled(self) -> None:
        first = _candidate("a", "Iran Israel ceasefire", source="Source A")
        second = _candidate("b", "Iran Israel ceasefire", source="Source B")
        decisions = {
            "a": HeadlineDecision("a", score=8.0),
            "b": HeadlineDecision("b", score=7.9),
        }
        filtering = FilteringConfig(headline_score_cutoff=0.0, max_selected_articles=2, max_selected_per_source=0)

        disabled = select_articles(
            [first, second],
            decisions,
            [TopicConfig(name="Major world events")],
            filtering,
            user_memory=UserMemory(),
            memory_config=MemoryConfig(enabled=False),
        )

        self.assertEqual([article.candidate.id for article in disabled], ["a", "b"])

        first_enabled = _candidate("a", "Iran Israel ceasefire", source="Source A")
        second_enabled = _candidate("b", "Iran Israel ceasefire", source="Source B")
        enabled_decisions = {
            "a": HeadlineDecision("a", score=8.0),
            "b": HeadlineDecision("b", score=7.9),
        }
        enabled = select_articles(
            [first_enabled, second_enabled],
            enabled_decisions,
            [TopicConfig(name="Major world events")],
            filtering,
            user_memory=UserMemory(),
            memory_config=_memory_config(max_selected_per_story=1),
            date="2026-06-27",
        )

        self.assertEqual([article.candidate.id for article in enabled], ["a"])
        self.assertEqual(enabled_decisions["b"].selection_reason_code, "skipped_story_cap")

    def test_coverage_writeback_is_idempotent_for_same_brief_story(self) -> None:
        temp_dir = self._temp_dir()
        store = CoverageMemoryStore.from_state_dir(temp_dir)
        article = SelectedArticle(
            candidate=_candidate("a", "Iran Israel ceasefire"),
            decision=HeadlineDecision("a", score=8.0, selection_rank_score=8.0),
            selection_rank_score=8.0,
        )
        set_memory_annotation(
            article.candidate,
            MemoryAnnotation(
                story_key="iran-israel-ceasefire",
                story_family_key="iran-israel",
                story_title="Iran Israel ceasefire",
            ),
        )

        store.write_selected(date="2026-06-27", brief_name="general", selected=[article])
        store.write_selected(date="2026-06-27", brief_name="general", selected=[article])

        records = store.read_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].story_key, "iran-israel-ceasefire")

    def test_coverage_prune_applies_retention_days(self) -> None:
        temp_dir = self._temp_dir()
        store = CoverageMemoryStore.from_state_dir(temp_dir)
        store.write_records(
            [
                CoverageRecord(
                    schema_version=1,
                    date="2026-05-20",
                    brief_name="general",
                    story_key="old-story",
                    story_family_key="old",
                    title="Old story",
                    prominence="body",
                    article_ids=["old"],
                ),
                CoverageRecord(
                    schema_version=1,
                    date="2026-06-20",
                    brief_name="general",
                    story_key="recent-story",
                    story_family_key="recent",
                    title="Recent story",
                    prominence="body",
                    article_ids=["recent"],
                ),
            ]
        )

        removed = store.prune(as_of_date="2026-06-28", retention_days=30)

        self.assertEqual(removed, 1)
        self.assertEqual([record.story_key for record in store.read_records()], ["recent-story"])

    def test_story_store_marks_stale_after_week_and_prunes_old_records(self) -> None:
        temp_dir = self._temp_dir()
        store = StoryStore.from_state_dir(temp_dir)
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stories": [
                        {
                            "story_key": "active-story",
                            "story_family_key": "active",
                            "title": "Active story",
                            "topic": "World",
                            "tokens": ["active", "story"],
                            "first_seen": "2026-06-25",
                            "last_seen": "2026-06-25",
                            "status": "active",
                        },
                        {
                            "story_key": "stale-story",
                            "story_family_key": "stale",
                            "title": "Stale story",
                            "topic": "World",
                            "tokens": ["stale", "story"],
                            "first_seen": "2026-06-01",
                            "last_seen": "2026-06-19",
                            "status": "active",
                        },
                        {
                            "story_key": "old-story",
                            "story_family_key": "old",
                            "title": "Old story",
                            "topic": "World",
                            "tokens": ["old", "story"],
                            "first_seen": "2026-05-01",
                            "last_seen": "2026-05-20",
                            "status": "active",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        records = store.refresh_lifecycle(
            as_of_date="2026-06-28",
            stale_after_days=7,
            retention_days=30,
            prune=True,
        )

        self.assertEqual([record.story_key for record in records], ["active-story", "stale-story"])
        status_by_key = {record.story_key: record.status for record in records}
        self.assertEqual(status_by_key["active-story"], "active")
        self.assertEqual(status_by_key["stale-story"], "stale")

    def test_feedback_store_records_mvp_actions(self) -> None:
        temp_dir = self._temp_dir()
        store = FeedbackStore.from_state_dir(temp_dir)

        for action in FEEDBACK_ACTIONS:
            store.record(
                action=action,
                report_date="2026-06-28",
                brief_name="general",
                article_id=f"article-{action}",
                story_key="story",
                title="A story",
            )

        self.assertEqual(len(store.read_events()), 4)
        self.assertEqual(store.counts_by_action()["too_repetitive"], 1)
        with self.assertRaisesRegex(ValueError, "Unsupported feedback action"):
            store.record(action="bad_source")

    def test_preference_learning_maps_feedback_actions(self) -> None:
        event = FeedbackEvent(
            schema_version=1,
            created_at="2026-06-28T12:00:00+00:00",
            action="more_like_this",
            topic="AI policy",
            source="Example News",
        )
        delta = preference_delta_for_event(event)

        self.assertEqual(delta.topic_weights, {"AI policy": 0.35})
        self.assertEqual(delta.source_weights, {"Example News": 0.2})

        updated, _ = apply_feedback_event(
            LearnedPreferences(topic_weights={"AI policy": 2.8}, source_weights={"Example News": 2.9}),
            event,
        )
        self.assertEqual(updated.topic_weights["AI policy"], 3.0)
        self.assertEqual(updated.source_weights["Example News"], 3.0)

        not_interested = preference_delta_for_event(
            FeedbackEvent(
                schema_version=1,
                created_at="2026-06-28T12:00:00+00:00",
                action="not_interested_in_topic",
                topic="AI policy",
                source="Example News",
            )
        )
        self.assertEqual(not_interested.topic_weights, {"AI policy": -0.7})
        self.assertEqual(not_interested.source_weights, {})

        not_relevant = preference_delta_for_event(
            FeedbackEvent(
                schema_version=1,
                created_at="2026-06-28T12:00:00+00:00",
                action="not_relevant",
                topic="AI policy",
                source="Example News",
            )
        )
        self.assertEqual(not_relevant.topic_weights, {"AI policy": -0.3})
        self.assertEqual(not_relevant.source_weights, {"Example News": -0.15})

        repetitive = preference_delta_for_event(
            FeedbackEvent(
                schema_version=1,
                created_at="2026-06-28T12:00:00+00:00",
                action="too_repetitive",
                topic="AI policy",
                source="Example News",
            )
        )
        self.assertFalse(repetitive.changed)

    def test_learned_preferences_adjust_selection_without_mutating_user_memory(self) -> None:
        preferred = _candidate("preferred", "AI policy breakthrough", source="Example News")
        preferred.metadata["topic_name"] = "AI policy"
        other = _candidate("other", "Market update", source="Other News")
        other.metadata["topic_name"] = "Markets"
        decisions = {
            "preferred": HeadlineDecision("preferred", score=7.9, topic="AI policy"),
            "other": HeadlineDecision("other", score=8.0, topic="Markets"),
        }
        user_memory = UserMemory(role="Analyst", wants=["supply chains"])
        before_user_memory = asdict(user_memory)

        selected = select_articles(
            [preferred, other],
            decisions,
            [TopicConfig(name="AI policy"), TopicConfig(name="Markets")],
            FilteringConfig(headline_score_cutoff=0.0, max_selected_articles=1, max_selected_per_source=0),
            user_memory=user_memory,
            memory_config=MemoryConfig(enabled=False),
            learned_preferences=LearnedPreferences(
                topic_weights={"AI policy": 3.0},
                source_weights={"Example News": 1.0},
            ),
        )

        self.assertEqual([article.candidate.id for article in selected], ["preferred"])
        self.assertEqual(asdict(user_memory), before_user_memory)
        self.assertGreater(decisions["preferred"].selection_rank_score, decisions["other"].selection_rank_score)
        self.assertEqual(decisions["preferred"].selection_rank_mode, "score_learned")
        rationale = selection_rationale_rows([preferred, other], decisions)
        preferred_row = next(row for row in rationale if row["candidate_id"] == "preferred")
        self.assertGreater(preferred_row["learned_preferences"]["score_adjustment"], 0)
        self.assertEqual(preferred_row["learned_preferences"]["matched_topics"], ["AI policy"])
        self.assertEqual(preferred_row["learned_preferences"]["matched_sources"], ["Example News"])

    def test_learned_preferences_store_is_separate_from_user_memory(self) -> None:
        temp_dir = self._temp_dir()
        store = LearnedPreferencesStore.from_state_dir(temp_dir)

        written = store.write(
            LearnedPreferences(
                preferred_topics=["semiconductors"],
                avoided_topics=["routine earnings"],
                topic_weights={"AI policy": 1.5, "celebrity": -2.0},
                source_weights={"Example News": 0.75},
                notes="User-editable evolving profile.",
            )
        )
        loaded = store.read()

        self.assertTrue(written.updated_at)
        self.assertEqual(loaded.preferred_topics, ["semiconductors"])
        self.assertEqual(loaded.avoided_topics, ["routine earnings"])
        self.assertEqual(loaded.topic_weights["AI policy"], 1.5)
        self.assertEqual(loaded.source_weights["Example News"], 0.75)

    def test_memory_health_checks_find_user_repairable_problems(self) -> None:
        temp_dir = self._temp_dir()
        feedback_path = temp_dir / "feedback_events.jsonl"
        feedback_path.write_text(
            "\n".join(
                [
                    json.dumps({"schema_version": 1, "action": "more_like_this", "created_at": "2026-06-28"}),
                    "{bad-json",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        health = memory_health_checks(
            state_dir=temp_dir,
            story_store=[
                {"story_key": "duplicate-story", "last_seen": "2026-06-28"},
                {"story_key": "duplicate-story", "last_seen": ""},
            ],
            coverage_records=[
                CoverageRecord(
                    schema_version=1,
                    date="2026-06-28",
                    brief_name="general",
                    story_key="missing-story",
                    story_family_key="missing",
                    title="Missing story",
                    prominence="body",
                    article_ids=["missing"],
                )
            ],
            feedback_events=[{"schema_version": 1, "action": "more_like_this", "created_at": "2026-06-28"}],
        )

        self.assertFalse(health["ok"])
        self.assertEqual(health["counts"]["invalid_feedback_rows"], 1)
        self.assertEqual(health["counts"]["duplicate_story_keys"], 1)
        self.assertEqual(health["counts"]["story_records_missing_last_seen"], 1)
        self.assertEqual(health["counts"]["coverage_story_key_missing_from_store"], 1)
        self.assertEqual(health["counts"]["feedback_rows_missing_identity"], 1)

    def test_story_merge_rewrites_store_coverage_feedback_and_creates_backup(self) -> None:
        temp_dir = self._temp_dir()
        self._write_repair_memory(
            temp_dir,
            stories=[
                {"story_key": "story-a", "story_family_key": "family-a", "title": "Story A", "tokens": ["alpha"], "first_seen": "2026-06-20", "last_seen": "2026-06-27"},
                {"story_key": "story-b", "story_family_key": "family-b", "title": "Story B", "tokens": ["beta"], "first_seen": "2026-06-22", "last_seen": "2026-06-28"},
                {"story_key": "story-c", "story_family_key": "family-c", "title": "Story C", "tokens": ["gamma"], "first_seen": "2026-06-25", "last_seen": "2026-06-28"},
            ],
            coverage=[
                {"schema_version": 1, "date": "2026-06-27", "brief_name": "general", "story_key": "story-a", "story_family_key": "family-a", "title": "Story A", "prominence": "lead", "article_ids": ["a"]},
                {"schema_version": 1, "date": "2026-06-28", "brief_name": "general", "story_key": "story-b", "story_family_key": "family-b", "title": "Story B", "prominence": "body", "article_ids": ["b"]},
                {"schema_version": 1, "date": "2026-06-28", "brief_name": "general", "story_key": "story-c", "story_family_key": "family-c", "title": "Story C", "prominence": "body", "article_ids": ["c"]},
            ],
            feedback=[
                {"schema_version": 1, "created_at": "2026-06-28T12:00:00+00:00", "action": "more_like_this", "story_key": "story-a", "story_family_key": "family-a", "title": "Story A"},
                {"schema_version": 1, "created_at": "2026-06-28T13:00:00+00:00", "action": "not_relevant", "story_key": "story-b", "story_family_key": "family-b", "title": "Story B"},
            ],
        )

        result = merge_stories(
            temp_dir,
            source_story_keys=["story-a", "story-b"],
            canonical_story={
                "story_key": "story-ab",
                "story_family_key": "family-ab",
                "title": "Story AB",
                "topic": "AI policy",
                "tokens": ["alpha", "beta"],
            },
            confirm=True,
        )

        stories = {record.story_key: record for record in StoryStore.from_state_dir(temp_dir).records()}
        self.assertEqual(sorted(stories), ["story-ab", "story-c"])
        self.assertEqual(stories["story-ab"].first_seen, "2026-06-20")
        self.assertEqual(stories["story-ab"].last_seen, "2026-06-28")
        coverage_keys = [record.story_key for record in CoverageMemoryStore.from_state_dir(temp_dir).read_records()]
        self.assertEqual(coverage_keys, ["story-ab", "story-ab", "story-c"])
        feedback_keys = [event.story_key for event in FeedbackStore.from_state_dir(temp_dir).read_events()]
        self.assertEqual(feedback_keys, ["story-ab", "story-ab"])
        backup = Path(result["backup"]["path"])
        self.assertTrue((backup / "story_store.json").exists())
        self.assertTrue((backup / "coverage_log.jsonl").exists())
        self.assertTrue((backup / "feedback_events.jsonl").exists())

    def test_story_merge_preserves_source_evidence_and_semantic_state(self) -> None:
        temp_dir = self._temp_dir()
        self._write_repair_memory(
            temp_dir,
            stories=[
                {
                    "story_key": "story-a",
                    "title": "Story A",
                    "tokens": ["alpha"],
                    "last_seen": "2026-06-27",
                    "source_document_ids": ["doc-a"],
                    "facts": [
                        {
                            "fact_id": "fact:a",
                            "text": "Alpha was announced.",
                            "source_id": "doc-a",
                            "tokens": ["alpha", "announced"],
                            "user_visible": True,
                        }
                    ],
                    "last_user_visible_fact_ids": ["fact:a"],
                },
                {
                    "story_key": "story-b",
                    "title": "Story B",
                    "tokens": ["beta"],
                    "last_seen": "2026-06-28",
                    "source_document_ids": ["doc-b"],
                    "facts": [
                        {
                            "fact_id": "fact:b",
                            "text": "Beta was confirmed.",
                            "source_id": "doc-b",
                            "tokens": ["beta", "confirmed"],
                        }
                    ],
                    "last_change_type": "confirmed",
                    "last_delta_summary": "Beta is now confirmed.",
                },
            ],
            coverage=[],
            feedback=[],
        )

        merge_stories(
            temp_dir,
            source_story_keys=["story-a", "story-b"],
            canonical_story={
                "story_key": "story-ab",
                "story_family_key": "",
                "title": "",
                "topic": "",
                "tokens": ["canonical", "token"],
            },
            confirm=True,
        )

        record = StoryStore.from_state_dir(temp_dir).records()[0]
        self.assertEqual(record.title, "Story B")
        self.assertEqual(record.tokens, ["canonical", "token"])
        self.assertEqual(record.source_document_ids, ["doc-a", "doc-b"])
        self.assertEqual({fact.fact_id for fact in record.facts}, {"fact:a", "fact:b"})
        self.assertEqual(record.last_user_visible_fact_ids, ["fact:a"])
        self.assertEqual(record.last_change_type, "confirmed")
        self.assertEqual(record.last_delta_summary, "Beta is now confirmed.")

    def test_story_repair_invalid_payload_fails_without_backup_or_rewrite(self) -> None:
        temp_dir = self._temp_dir()
        self._write_repair_memory(
            temp_dir,
            stories=[
                {"story_key": "story-a", "story_family_key": "family-a", "title": "Story A", "tokens": ["alpha"]},
                {"story_key": "story-b", "story_family_key": "family-b", "title": "Story B", "tokens": ["beta"]},
            ],
            coverage=[],
            feedback=[],
        )
        story_path = temp_dir / "story_store.json"
        before = story_path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Story key not found"):
            merge_stories(
                temp_dir,
                source_story_keys=["story-a", "missing-story"],
                canonical_story={"story_key": "merged"},
                confirm=True,
            )

        self.assertEqual(story_path.read_text(encoding="utf-8"), before)
        self.assertFalse((temp_dir / "backups").exists())

    def test_story_split_moves_selected_rows_and_creates_backup(self) -> None:
        temp_dir = self._temp_dir()
        self._write_repair_memory(
            temp_dir,
            stories=[
                {"story_key": "story-a", "story_family_key": "family-a", "title": "Story A", "tokens": ["alpha"], "first_seen": "2026-06-20", "last_seen": "2026-06-28"},
            ],
            coverage=[
                {"schema_version": 1, "date": "2026-06-27", "brief_name": "general", "story_key": "story-a", "story_family_key": "family-a", "title": "Story A", "prominence": "lead", "article_ids": ["a"]},
                {"schema_version": 1, "date": "2026-06-28", "brief_name": "general", "story_key": "story-a", "story_family_key": "family-a", "title": "Story A", "prominence": "body", "article_ids": ["b"]},
            ],
            feedback=[
                {"schema_version": 1, "created_at": "2026-06-28T12:00:00+00:00", "action": "more_like_this", "story_key": "story-a", "story_family_key": "family-a", "title": "Story A"},
            ],
        )
        coverage_records = CoverageMemoryStore.from_state_dir(temp_dir).read_records()
        feedback_events = FeedbackStore.from_state_dir(temp_dir).read_events()

        result = split_story(
            temp_dir,
            source_story_key="story-a",
            new_story={
                "story_key": "story-new",
                "story_family_key": "family-new",
                "title": "Story New",
                "topic": "Markets",
                "tokens": ["new"],
            },
            coverage_row_ids=[coverage_row_id(0, coverage_records[0])],
            feedback_row_ids=[feedback_row_id(0, feedback_events[0])],
            confirm=True,
        )

        self.assertEqual(result["coverage_rows_rewritten"], 1)
        stories = {record.story_key: record for record in StoryStore.from_state_dir(temp_dir).records()}
        self.assertEqual(sorted(stories), ["story-a", "story-new"])
        coverage_after = CoverageMemoryStore.from_state_dir(temp_dir).read_records()
        self.assertEqual([record.story_key for record in coverage_after], ["story-new", "story-a"])
        feedback_after = FeedbackStore.from_state_dir(temp_dir).read_events()
        self.assertEqual(feedback_after[0].story_key, "story-new")
        self.assertTrue((Path(result["backup"]["path"]) / "story_store.json").exists())

    def test_coverage_archive_and_feedback_edit_delete_create_backups(self) -> None:
        temp_dir = self._temp_dir()
        self._write_repair_memory(
            temp_dir,
            stories=[],
            coverage=[
                {"schema_version": 1, "date": "2026-06-27", "brief_name": "general", "story_key": "story-a", "story_family_key": "family-a", "title": "Story A", "prominence": "lead", "article_ids": ["a"]},
                {"schema_version": 1, "date": "2026-06-28", "brief_name": "general", "story_key": "story-b", "story_family_key": "family-b", "title": "Story B", "prominence": "body", "article_ids": ["b"]},
            ],
            feedback=[
                {"schema_version": 1, "created_at": "2026-06-28T12:00:00+00:00", "action": "more_like_this", "story_key": "story-a", "story_family_key": "family-a", "title": "Story A"},
                {"schema_version": 1, "created_at": "2026-06-28T13:00:00+00:00", "action": "not_relevant", "story_key": "story-b", "story_family_key": "family-b", "title": "Story B"},
            ],
        )
        coverage_records = CoverageMemoryStore.from_state_dir(temp_dir).read_records()

        archive_result = repair_coverage_rows(
            temp_dir,
            row_ids=[coverage_row_id(0, coverage_records[0])],
            action="archive",
            confirm=True,
        )

        self.assertEqual(archive_result["coverage_rows_archived"], 1)
        self.assertEqual([record.story_key for record in CoverageMemoryStore.from_state_dir(temp_dir).read_records()], ["story-b"])
        archive_text = (temp_dir / "coverage_log.archive.jsonl").read_text(encoding="utf-8")
        self.assertIn("story-a", archive_text)
        self.assertTrue((Path(archive_result["backup"]["path"]) / "coverage_log.jsonl").exists())

        feedback_events = FeedbackStore.from_state_dir(temp_dir).read_events()
        edit_result = repair_feedback_events(
            temp_dir,
            action="edit",
            row_ids=[feedback_row_id(0, feedback_events[0])],
            event_patch={"action": "not_interested_in_topic", "topic": "AI policy", "notes": "Updated."},
            confirm=True,
        )

        edited = FeedbackStore.from_state_dir(temp_dir).read_events()
        self.assertEqual(edit_result["feedback_events_edited"], 1)
        self.assertEqual(edited[0].action, "not_interested_in_topic")
        self.assertEqual(edited[0].topic, "AI policy")

        delete_result = repair_feedback_events(
            temp_dir,
            action="delete",
            row_ids=[feedback_row_id(1, edited[1])],
            confirm=True,
        )

        self.assertEqual(delete_result["feedback_events_deleted"], 1)
        self.assertEqual(len(FeedbackStore.from_state_dir(temp_dir).read_events()), 1)

    def test_recall_packet_is_compact_guidance(self) -> None:
        candidate = _candidate("a", "Iran Israel ceasefire")
        set_memory_annotation(
            candidate,
            MemoryAnnotation(
                story_key="iran-israel-ceasefire",
                story_family_key="iran-israel",
                story_title="Iran Israel ceasefire",
                recent_coverage_count=2,
                recent_lead_count=1,
                covered_yesterday=True,
                score_adjustment=-1.2,
                today_policy="capsule_unless_material_update",
                reason="Recently prominent.",
            ),
        )
        decisions = {"a": HeadlineDecision("a", score=8.0, selection_reason_code="skipped_recent_coverage")}

        packet = build_recall_packet(
            date="2026-06-27",
            brief_name="general",
            candidates=[candidate],
            decisions=decisions,
        )

        rendered = json.dumps(packet)
        self.assertEqual(len(packet["coverage_guidance"]), 1)
        self.assertIn("capsule_unless_material_update", rendered)
        self.assertNotIn("article_text", rendered)

    def test_handoff_round_trip_preserves_memory_annotation(self) -> None:
        temp_dir = self._temp_dir()
        article = SelectedArticle(
            candidate=_candidate("a", "Iran Israel ceasefire"),
            decision=HeadlineDecision("a", score=8.0),
            article_text="Full article text.",
            extraction_status="ok",
        )
        set_memory_annotation(
            article.candidate,
            MemoryAnnotation(
                story_key="iran-israel-ceasefire",
                story_family_key="iran-israel",
                story_title="Iran Israel ceasefire",
                recent_coverage_count=1,
                score_adjustment=-0.6,
                reason="Recently covered.",
            ),
        )

        path = write_brief_handoff(
            output_dir=temp_dir,
            date="2026-06-27",
            brief_name="general",
            json_path=temp_dir / "brief.json",
            markdown_path=temp_dir / "brief.md",
            topics=[TopicConfig(name="Major world events")],
            prior_reports=[],
            brief_goal="Brief goal",
            filtering=FilteringConfig(),
            selected_articles=[article],
        )
        loaded = load_brief_handoff(path)
        annotation = candidate_memory_annotation(loaded.selected_articles[0].candidate)

        self.assertIsNotNone(annotation)
        self.assertEqual(annotation.story_key, "iran-israel-ceasefire")
        self.assertEqual(annotation.score_adjustment, -0.6)

    def _write_repair_memory(
        self,
        state_dir: Path,
        *,
        stories: List[dict],
        coverage: List[dict],
        feedback: List[dict],
    ) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "story_store.json").write_text(
            json.dumps({"schema_version": 1, "stories": stories}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (state_dir / "coverage_log.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in coverage),
            encoding="utf-8",
        )
        (state_dir / "feedback_events.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in feedback),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
