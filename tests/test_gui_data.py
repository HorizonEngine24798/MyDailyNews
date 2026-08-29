from __future__ import annotations

import json
from pathlib import Path
import time
import unittest
import uuid

from gui import build_parser as build_gui_parser
from mydailynews.gui.data import GuiDataService


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "gui_data"


class GuiDataServiceTests(unittest.TestCase):
    def _temp_root(self) -> Path:
        path = TEMP_ROOT / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _write_config(self, root: Path) -> Path:
        payload = json.loads((REPO_ROOT / "config.example.json").read_text(encoding="utf-8-sig"))
        payload["output_dir"] = "output"
        payload["memory"]["state_dir"] = "state/memory"
        path = root / "config.local.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_report(self, root: Path) -> None:
        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-28_general_brief.md").write_text(
            "# Test Brief\n\nA concise report.\n",
            encoding="utf-8",
        )
        (output_dir / "2026-06-28_general_brief.json").write_text(
            json.dumps(
                {
                    "title": "Test Brief",
                    "selected_articles": [
                        {
                            "id": "article-1",
                            "headline": "A durable story",
                            "source": "Example News",
                            "topic": "AI policy",
                            "story_key": "durable-story",
                            "story_family_key": "durable",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_reports_and_feedback_use_saved_brief_json(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        self._write_report(root)
        service = GuiDataService(root, "config.local.json")

        reports = service.list_reports()["reports"]
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["brief_name"], "general")

        detail = service.report_detail("2026-06-28_general_brief.md")
        self.assertEqual(detail["feedback_items"][0]["story_key"], "durable-story")
        self.assertFalse(detail["has_audio"])
        self.assertEqual(detail["audio_url"], "")

        result = service.record_feedback(
            {
                "action": "more_like_this",
                "report_id": detail["id"],
                "item": detail["feedback_items"][0],
            }
        )

        self.assertEqual(result["event"]["brief_name"], "general")
        self.assertEqual(result["event"]["article_id"], "article-1")
        self.assertEqual(result["event"]["story_key"], "durable-story")
        self.assertTrue(result["learned_preferences_changed"])
        self.assertEqual(result["learned_preference_delta"]["topic_weights"], {"AI policy": 0.35})
        self.assertEqual(result["learned_preference_delta"]["source_weights"], {"Example News": 0.2})

        detail_after_feedback = service.report_detail("2026-06-28_general_brief.md")
        self.assertEqual(detail_after_feedback["feedback_items"][0]["feedback_count"], 1)
        self.assertEqual(detail_after_feedback["feedback_items"][0]["latest_feedback_action"], "more_like_this")

    def test_report_detail_exposes_matching_wav_audio(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        self._write_report(root)
        wav_path = root / "output" / "2026-06-28_general_brief.wav"
        wav_path.write_bytes(b"RIFF....WAVEfmt ")
        service = GuiDataService(root, "config.local.json")

        detail = service.report_detail("2026-06-28_general_brief.md")

        self.assertTrue(detail["has_audio"])
        self.assertEqual(detail["audio_url"], "/api/reports/2026-06-28_general_brief.md/audio")
        self.assertEqual(service.report_audio_path("2026-06-28_general_brief.md"), wav_path)

    def test_map_snapshot_resolves_event_loci_and_audits_searched_countries(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-28_perspectives_report.json").write_text(
            json.dumps(
                {
                    "stories": [
                        {
                            "story_id": "story-1",
                            "story_title": "Shipping disruption",
                            "summary": "Traffic was disrupted near a strategic waterway.",
                            "story_loci": [
                                {
                                    "label": "Strait of Hormuz",
                                    "country": "",
                                    "kind": "event_site",
                                    "confidence": "high",
                                    "reason": "The disruption occurred there.",
                                }
                            ],
                            "selected_sources": [
                                {"source_id": "us_test", "name": "US Test", "country": "US", "language": "en"},
                                {"source_id": "gb_test", "name": "GB Test", "country": "GB", "language": "en"},
                            ],
                            "source_yields": [
                                {"source_id": "us_test", "source_name": "US Test", "raw": 2, "final": 1},
                                {"source_id": "gb_test", "source_name": "GB Test", "raw": 0, "final": 0},
                            ],
                            "coverage_articles": [
                                {
                                    "title": "A US report",
                                    "url": "https://example.com/story",
                                    "source_name": "US Test",
                                    "source_country": "US",
                                    "source_language": "en",
                                    "published_at": "2026-06-28T10:00:00Z",
                                    "body": "large body must not reach the map API",
                                    "context_text": "large context must not reach the map API",
                                }
                            ],
                            "coverage_status": "ok",
                            "coverage_quality": {"status": "thin"},
                            "provider_statuses": {},
                            "framing_report": {"synthesis": "The source emphasized shipping risk."},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        service = GuiDataService(root, "config.local.json")

        snapshot = service.map_snapshot()

        self.assertEqual(snapshot["date"], "2026-06-28")
        story = snapshot["stories"][0]
        self.assertEqual(story["loci"][0]["resolution"], "named_place")
        self.assertEqual(
            {point["country"]: point["status"] for point in story["coverage_points"]},
            {"GB": "searched_empty", "US": "found"},
        )
        self.assertNotIn("body", story["coverage_articles"][0])
        self.assertNotIn("context_text", story["coverage_articles"][0])
        self.assertEqual(story["search_summary"]["searched_empty_country_count"], 1)

    def test_record_feedback_updates_learned_preferences_not_user_memory(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        self._write_report(root)
        service = GuiDataService(root, "config.local.json")
        profile_before = service.read_config()["config"]["user_memory"]
        detail = service.report_detail("2026-06-28_general_brief.md")

        result = service.record_feedback(
            {
                "action": "not_interested_in_topic",
                "report_id": detail["id"],
                "item": detail["feedback_items"][0],
            }
        )

        learned_path = root / "state" / "memory" / "learned_preferences.json"
        learned_payload = json.loads(learned_path.read_text(encoding="utf-8"))
        self.assertTrue(result["learned_preferences_changed"])
        self.assertEqual(learned_payload["topic_weights"]["AI policy"], -0.7)
        self.assertNotIn("Example News", learned_payload.get("source_weights", {}))
        self.assertEqual(service.read_config()["config"]["user_memory"], profile_before)

        repetitive = service.record_feedback(
            {
                "action": "too_repetitive",
                "report_id": detail["id"],
                "item": detail["feedback_items"][0],
            }
        )

        self.assertFalse(repetitive["learned_preferences_changed"])
        self.assertEqual(repetitive["learned_preferences"]["topic_weights"]["AI policy"], -0.7)

    def test_config_section_save_validates_full_config(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        service = GuiDataService(root, "config.local.json")

        saved = service.save_config_section(
            "user_memory",
            {
                "role": "AI engineer",
                "beats": {"AI policy": 2.0},
                "wants": ["infrastructure"],
                "avoid": ["celebrity coverage"],
            },
        )

        self.assertEqual(saved["config"]["user_memory"]["role"], "AI engineer")
        with self.assertRaisesRegex(ValueError, "Config section user_memory has unrecognized key"):
            service.save_config_section("user_memory", {"unknown": True})

    def test_preview_helpers_do_not_save_drafts(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        service = GuiDataService(root, "config.local.json")
        original_role = service.read_config()["config"]["user_memory"].get("role", "")

        profile = service.preview_user_memory(
            {
                "role": "AI engineer",
                "beats": {"AI policy": 4.5, "bad": "not-a-number"},
                "wants": ["infrastructure", "infrastructure"],
                "preferred_depth": "deep",
            }
        )
        self.assertIn("Role: AI engineer", profile["prompt"])
        self.assertIn("AI policy(3.00)", profile["prompt"])
        self.assertEqual(profile["profile"]["wants"], ["infrastructure"])

        learned = service.preview_learned_preferences(
            {
                "preferred_topics": ["AI", "AI"],
                "topic_weights": {"AI policy": 7, "noise": "bad"},
                "source_weights": {"Example News": -8},
            }
        )
        self.assertEqual(learned["preferences"]["preferred_topics"], ["AI"])
        self.assertEqual(learned["preferences"]["topic_weights"]["AI policy"], 3.0)
        self.assertEqual(learned["preferences"]["source_weights"]["Example News"], -3.0)
        self.assertEqual(service.read_config()["config"]["user_memory"].get("role", ""), original_role)

    def test_learned_preferences_and_story_store_edits_are_normalized(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        service = GuiDataService(root, "config.local.json")

        learned = service.save_learned_preferences(
            {
                "preferred_topics": ["AI", "AI", ""],
                "topic_weights": {"AI": 9, "bad": "not-a-number"},
                "source_weights": {"Example News": -9},
                "notes": "Visible evolving profile.",
            }
        )
        prefs = learned["preferences"]
        self.assertEqual(prefs["preferred_topics"], ["AI"])
        self.assertEqual(prefs["topic_weights"]["AI"], 3.0)
        self.assertEqual(prefs["source_weights"]["Example News"], -3.0)

        memory = service.save_story_store(
            {
                "stories": [
                    {
                        "story_key": "story-b",
                        "title": "Story B",
                        "tokens": ["Story", "Story"],
                        "status": "archived",
                    },
                    {
                        "story_key": "story-a",
                        "title": "Story A",
                        "tokens": ["A"],
                        "status": "stale",
                        "source_document_ids": ["source-a"],
                        "facts": [
                            {
                                "fact_id": "fact:a",
                                "text": "Story A was confirmed by its source.",
                                "source_id": "source-a",
                                "tokens": ["story", "confirmed"],
                                "user_visible": True,
                            }
                        ],
                    },
                ]
            }
        )

        keys = [item["story_key"] for item in memory["story_store"]]
        self.assertEqual(keys, ["story-a", "story-b"])
        status_by_key = {item["story_key"]: item["status"] for item in memory["story_store"]}
        self.assertEqual(status_by_key["story-a"], "stale")
        self.assertEqual(status_by_key["story-b"], "active")
        story_a_file = next(
            item for item in memory["story_store_file"]["stories"] if item["story_key"] == "story-a"
        )
        self.assertEqual(story_a_file["source_document_ids"], ["source-a"])
        self.assertEqual(story_a_file["facts"][0]["fact_id"], "fact:a")

    def test_memory_snapshot_exposes_display_rows_recall_packets_and_health(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        state_dir = root / "state" / "memory"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "story_store.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stories": [
                        {
                            "story_key": "story-a",
                            "story_family_key": "family-a",
                            "title": "Story A",
                            "topic": "AI policy",
                            "tokens": ["story", "a"],
                            "first_seen": "2026-06-27",
                            "last_seen": "2026-06-28",
                            "status": "active",
                        },
                        {
                            "story_key": "story-b",
                            "story_family_key": "family-b",
                            "title": "Story B",
                            "topic": "Markets",
                            "tokens": ["story", "b"],
                            "first_seen": "2026-06-26",
                            "last_seen": "",
                            "status": "active",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        (state_dir / "coverage_log.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "schema_version": 1,
                            "date": "2026-06-28",
                            "brief_name": "general",
                            "story_key": "story-a",
                            "story_family_key": "family-a",
                            "title": "Story A",
                            "prominence": "lead",
                            "article_ids": ["a"],
                        }
                    ),
                    json.dumps(
                        {
                            "schema_version": 1,
                            "date": "2026-06-28",
                            "brief_name": "general",
                            "story_key": "missing-story",
                            "story_family_key": "missing",
                            "title": "Missing Story",
                            "prominence": "body",
                            "article_ids": ["missing"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (state_dir / "feedback_events.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "schema_version": 1,
                            "created_at": "2026-06-28T12:00:00+00:00",
                            "action": "more_like_this",
                            "title": "Loose feedback",
                        }
                    ),
                    "{not-json",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (state_dir / "learned_preferences.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-06-28T13:00:00+00:00",
                    "preferred_topics": ["AI"],
                    "topic_weights": {"AI policy": 1.0},
                    "source_weights": {"Example News": 0.5},
                    "notes": "Keep watching.",
                }
            ),
            encoding="utf-8",
        )
        recall_dir = state_dir / "recall_packets"
        recall_dir.mkdir(parents=True, exist_ok=True)
        (recall_dir / "2026-06-28_general.json").write_text("{}", encoding="utf-8")
        service = GuiDataService(root, "config.local.json")

        snapshot = service.memory_snapshot()

        story_a = next(item for item in snapshot["story_store"] if item["story_key"] == "story-a")
        self.assertEqual(story_a["family"], "family-a")
        self.assertEqual(story_a["coverage_count"], 1)
        self.assertEqual(snapshot["coverage_records"][0]["story_key"], "story-a")
        self.assertEqual(snapshot["feedback_events"][0]["created_date"], "2026-06-28")
        self.assertEqual(snapshot["learned_preferences_summary"]["topic_weights"], 1)
        self.assertTrue(snapshot["recall_packets"]["exists"])
        self.assertEqual(snapshot["recall_packets"]["latest"]["brief_name"], "general")
        codes = {warning["code"] for warning in snapshot["health"]["warnings"]}
        self.assertIn("invalid_feedback_jsonl_rows", codes)
        self.assertIn("story_records_missing_last_seen", codes)
        self.assertIn("coverage_story_key_missing_from_store", codes)
        self.assertIn("feedback_rows_missing_identity", codes)
        self.assertEqual(snapshot["summary"]["health_warnings"], len(snapshot["health"]["warnings"]))

    def test_memory_repair_dispatch_uses_snapshot_row_ids_and_creates_backup(self) -> None:
        root = self._temp_root()
        self._write_config(root)
        state_dir = root / "state" / "memory"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "story_store.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stories": [
                        {
                            "story_key": "story-a",
                            "story_family_key": "family-a",
                            "title": "Story A",
                            "topic": "AI policy",
                            "tokens": ["story", "a"],
                            "first_seen": "2026-06-27",
                            "last_seen": "2026-06-28",
                            "status": "active",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (state_dir / "coverage_log.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "date": "2026-06-28",
                    "brief_name": "general",
                    "story_key": "story-a",
                    "story_family_key": "family-a",
                    "title": "Story A",
                    "prominence": "lead",
                    "article_ids": ["a"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        service = GuiDataService(root, "config.local.json")
        row_id = service.memory_snapshot()["coverage_records"][0]["row_id"]

        result = service.repair_memory({"action": "coverage_archive", "row_ids": [row_id], "confirm": True})

        self.assertEqual(result["repair"]["coverage_rows_archived"], 1)
        self.assertEqual(result["memory"]["summary"]["coverage_records"], 0)
        self.assertTrue(Path(result["repair"]["backup"]["path"]).exists())
        self.assertIn("story-a", (state_dir / "coverage_log.archive.jsonl").read_text(encoding="utf-8"))

    def test_run_manager_rejects_unknown_kind_and_runs_memory_inspect(self) -> None:
        root = self._temp_root()
        config_path = self._write_config(root)
        service = GuiDataService(REPO_ROOT, config_path)

        with self.assertRaisesRegex(ValueError, "Unsupported run kind"):
            service.start_run({"kind": "shell", "command": "echo bad"})

        started = service.start_run({"kind": "memory", "memory_action": "inspect"})["run"]
        self.assertEqual(started["status"], "running")
        self.assertIn("--memory inspect", started["command_display"])

        finished = started
        deadline = time.time() + 10
        while time.time() < deadline:
            finished = service.run_detail(started["id"])["run"]
            if finished["status"] != "running":
                break
            time.sleep(0.05)

        self.assertEqual(finished["status"], "completed")
        self.assertEqual(finished["returncode"], 0)
        self.assertIn("Memory state:", finished["stdout_tail"])
        self.assertEqual(service.list_runs()["runs"][0]["id"], started["id"])

    def test_gui_cli_flags_parse_without_pipeline_options(self) -> None:
        args = build_gui_parser().parse_args(["--host", "127.0.0.1", "--port", "9001"])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 9001)


if __name__ == "__main__":
    unittest.main()
