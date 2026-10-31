from __future__ import annotations

import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, NewsCandidate, SelectedArticle
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.memory.coverage import coverage_records_for_selected
from mydailynews.memory.story_store import _updated_thread_events
from mydailynews.pipeline.brief_execution import (
    capture_pending_memory_write,
    write_pending_memory,
)


class _Debug:
    def set_metric(self, *_args, **_kwargs) -> None:
        pass


class _CoverageStore:
    database: set[str] = set()

    def __init__(self, path=Path("coverage_log.jsonl"), *, database_path=Path("memory.sqlite3")) -> None:
        self.path = Path(path)
        self.database_path = Path(database_path)
        self.written_ids: list[str] = []

    def read_records(self):
        return []

    def write_selected(self, *, date, brief_name, selected, database=None):
        self.written_ids.extend(article.candidate.id for article in selected)
        type(self).database.update(article.candidate.id for article in selected)
        return list(selected)

    def prune(self, *, as_of_date, retention_days, database=None):
        return 0


class _FreshStoryStore:
    database: set[str] = set()

    def __init__(self, path, **_kwargs) -> None:
        self.path = Path(path)
        self.legacy_index_path = _kwargs.get("legacy_index_path")
        self.legacy_ledger_path = _kwargs.get("legacy_ledger_path")
        self.database_path = Path(_kwargs.get("database_path", "memory.sqlite3"))
        self.snapshot = set(self.database)

    def records(self):
        return []

    def update_selected(self, *, selected, **_kwargs):
        self.snapshot.update(article.candidate.id for article in selected)
        type(self).database = set(self.snapshot)
        return [SimpleNamespace(status="active", facts=[]) for _ in self.snapshot]


def _stale_story_store() -> SimpleNamespace:
    return SimpleNamespace(
        path=Path("story_store.json"),
        legacy_index_path=None,
        legacy_ledger_path=None,
        database_path=Path("memory.sqlite3"),
    )


class DeferredMemoryWriteTests(unittest.TestCase):
    def test_grouped_story_day_creates_one_thread_event_for_all_articles(self) -> None:
        decision = {
            "article_ids": ["article-a", "article-b"],
            "relationship": "same_story",
            "change_type": "incremental",
            "summary": "A supported fact was added.",
            "operations": [],
        }

        first = _updated_thread_events(
            [],
            decision,
            article_id="article-a",
            observed_at="2026-09-10",
        )
        second = _updated_thread_events(
            first,
            decision,
            article_id="article-b",
            observed_at="2026-09-10",
        )

        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].article_ids, ["article-a", "article-b"])

    def test_queued_briefs_keep_captured_state_and_merge_against_latest_commit(self) -> None:
        _FreshStoryStore.database = set()
        _CoverageStore.database = set()
        shared = SimpleNamespace(candidate=SimpleNamespace(id="general"))
        packet = {"brief": "general"}
        general_coverage = _CoverageStore()
        detailed_coverage = _CoverageStore()
        config = SimpleNamespace(
            story_stale_after_days=7,
            story_retention_days=30,
            coverage_retention_days=30,
        )

        general = capture_pending_memory_write(
            brief_name="general",
            date="2026-09-10",
            memory_config=config,
            coverage_store=general_coverage,
            story_store=_stale_story_store(),
            selected=[shared],
            rendered_selected=[shared],
            story_groups=[],
            delta_packet=packet,
            warnings=[],
        )
        shared.candidate.id = "detailed"
        packet["brief"] = "detailed"
        detailed = capture_pending_memory_write(
            brief_name="detailed",
            date="2026-09-10",
            memory_config=config,
            coverage_store=detailed_coverage,
            story_store=_stale_story_store(),
            selected=[shared],
            rendered_selected=[shared],
            story_groups=[],
            delta_packet=packet,
            warnings=[],
        )

        self.assertEqual(general.selected[0].candidate.id, "general")
        self.assertEqual(general.delta_packet["brief"], "general")
        with (
            patch("mydailynews.pipeline.brief_execution.StoryStore", _FreshStoryStore),
            patch("mydailynews.pipeline.brief_execution.CoverageMemoryStore", _CoverageStore),
            patch(
                "mydailynews.pipeline.brief_execution.open_database",
                return_value=nullcontext(object()),
            ),
        ):
            write_pending_memory(general, _Debug())
            write_pending_memory(detailed, _Debug())

        self.assertEqual(_FreshStoryStore.database, {"general", "detailed"})
        self.assertEqual(_CoverageStore.database, {"general", "detailed"})

    def test_continuing_bullet_does_not_consume_lead_prominence(self) -> None:
        def article(candidate_id: str, policy: str) -> SelectedArticle:
            candidate = NewsCandidate(
                id=candidate_id,
                source="Example",
                category="world",
                title=candidate_id,
                url=f"https://example.test/{candidate_id}",
                snippet="",
                published_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
            )
            set_memory_annotation(
                candidate,
                MemoryAnnotation(
                    story_key=f"story:{candidate_id}",
                    story_family_key="family",
                    story_title=candidate_id,
                    today_policy=policy,
                    change_type="incremental",
                ),
            )
            return SelectedArticle(
                candidate=candidate,
                decision=HeadlineDecision(candidate_id, score=7.0),
                selection_rank_score=7.0,
            )

        records = coverage_records_for_selected(
            date="2026-09-10",
            brief_name="general",
            selected=[article("capsule", "capsule"), article("main", "normal")],
        )

        self.assertEqual([record.prominence for record in records], ["capsule", "lead"])


if __name__ == "__main__":
    unittest.main()
