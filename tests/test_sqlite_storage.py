from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from mydailynews.common.cache import JSONCache
from mydailynews.memory.coverage import CoverageMemoryStore, CoverageRecord
from mydailynews.memory.feedback import FeedbackStore
from mydailynews.memory.story_store import StoryStore


class SQLiteStorageTests(unittest.TestCase):
    def test_memory_stores_import_legacy_files_once_into_one_database(self) -> None:
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            story_path = root / "story_store.json"
            coverage_path = root / "coverage_log.jsonl"
            feedback_path = root / "feedback_events.jsonl"
            story_path.write_text(
                json.dumps({"stories": [{"story_key": "story-a", "title": "Story A"}]}),
                encoding="utf-8",
            )
            coverage_path.write_text(
                json.dumps({
                    "date": "2026-09-09",
                    "brief_name": "general",
                    "story_key": "story-a",
                    "article_ids": ["article-a"],
                }) + "\n",
                encoding="utf-8",
            )
            feedback_path.write_text(
                json.dumps({"created_at": "2026-09-09T00:00:00+00:00", "action": "more_like_this"}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual([row.story_key for row in StoryStore.from_state_dir(root).records()], ["story-a"])
            self.assertEqual([row.story_key for row in CoverageMemoryStore.from_state_dir(root).read_records()], ["story-a"])
            self.assertEqual([row.action for row in FeedbackStore.from_state_dir(root).read_events()], ["more_like_this"])
            self.assertTrue((root / "memory.sqlite3").exists())

            story_path.write_text(json.dumps({"stories": [{"story_key": "late-edit"}]}), encoding="utf-8")
            self.assertEqual([row.story_key for row in StoryStore.from_state_dir(root).records()], ["story-a"])

            # sqlite3.Connection's context manager commits but does not close;
            # close explicitly so Windows can remove the temporary database.
            with closing(sqlite3.connect(root / "memory.sqlite3")) as database:
                self.assertEqual(database.execute("SELECT COUNT(*) FROM stories").fetchone()[0], 1)
                self.assertEqual(database.execute("SELECT COUNT(*) FROM coverage").fetchone()[0], 1)
                self.assertEqual(database.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 1)

    def test_json_cache_imports_legacy_namespace_once(self) -> None:
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            legacy = root / "json" / "synth"
            legacy.mkdir(parents=True)
            record_path = legacy / "answer.json"
            record_path.write_text(
                json.dumps({"cached_at": "2026-09-10T00:00:00+00:00", "value": {"answer": 42}}),
                encoding="utf-8",
            )

            self.assertEqual(JSONCache(str(root), "synth").get("answer"), {"answer": 42})
            record_path.write_text(
                json.dumps({"cached_at": "2026-09-10T00:00:00+00:00", "value": {"answer": 0}}),
                encoding="utf-8",
            )
            self.assertEqual(JSONCache(str(root), "synth").get("answer"), {"answer": 42})
            self.assertTrue((root / "cache.sqlite3").exists())

    def test_coverage_upsert_unions_articles_and_preserves_best_fields(self) -> None:
        with TemporaryDirectory() as raw_dir:
            store = CoverageMemoryStore.from_state_dir(raw_dir)
            shared = {
                "schema_version": 1,
                "date": "2026-09-10",
                "brief_name": "general",
                "story_key": "story-a",
                "story_family_key": "family-a",
                "angle": "update",
            }
            store.write_records([
                CoverageRecord(**shared, title="Highest score", prominence="body", article_ids=["a"], rank_score=9.0),
                CoverageRecord(**shared, title="Lead", prominence="lead", article_ids=["b"], rank_score=7.0),
            ])
            store.write_records([
                CoverageRecord(**shared, title="Later", prominence="capsule", article_ids=["c"], rank_score=5.0),
            ])

            records = store.read_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].article_ids, ["a", "b", "c"])
            self.assertEqual(records[0].prominence, "lead")
            self.assertEqual(records[0].rank_score, 9.0)
            self.assertEqual(records[0].title, "Highest score")

    def test_malformed_story_migration_rolls_back_and_retries_after_repair(self) -> None:
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            legacy = root / "story_store.json"
            legacy.write_text("{broken", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                StoryStore.from_state_dir(root).records()

            legacy.write_text(
                json.dumps({"stories": [{"story_key": "story-repaired"}]}),
                encoding="utf-8",
            )
            records = StoryStore.from_state_dir(root).records()
            self.assertEqual([record.story_key for record in records], ["story-repaired"])

    def test_malformed_coverage_migration_rolls_back_and_retries_after_repair(self) -> None:
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            legacy = root / "coverage_log.jsonl"
            legacy.write_text("{broken\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                CoverageMemoryStore.from_state_dir(root).read_records()

            legacy.write_text(
                json.dumps(
                    {
                        "date": "2026-09-10",
                        "brief_name": "general",
                        "story_key": "story-repaired",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = CoverageMemoryStore.from_state_dir(root).read_records()
            self.assertEqual([record.story_key for record in records], ["story-repaired"])


if __name__ == "__main__":
    unittest.main()
