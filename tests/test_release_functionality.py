from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from mydailynews.ai import create_ai_client
from mydailynews.app.config import load_config
from mydailynews.app.models import AIConfig
from mydailynews.pipeline.stages import PipelineRunOptions
from mydailynews.pipeline.stage_artifacts import (
    STAGE_ARTIFACT_SCHEMA_VERSION,
    build_stage_artifact,
    build_stage_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "release_tests"


class ReleaseSmokeTests(unittest.TestCase):
    def _config_payload(self) -> dict:
        return json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8-sig"))

    def _write_config_payload(self, directory: Path, payload: dict, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_config_contract_keeps_llama_cpp_only_runtime(self) -> None:
        config = load_config(REPO_ROOT / "config.json")

        self.assertEqual(config.ai_summary.backend, "llama_cpp_server")
        self.assertEqual(config.ai_final.backend, "llama_cpp_server")
        self.assertEqual(config.ai_summary.effective_model_label, "Qwen3.5-35B-A3B-Q3_K_M")
        self.assertEqual(config.ai_final.effective_model_label, "Qwen3.5-35B-A3B-Q3_K_M")
        self.assertFalse(hasattr(config.ai_summary, "preset"))
        removed_ai_module = ".".join(("mydailynews", "ai", "client"))
        self.assertIsNone(importlib.util.find_spec(removed_ai_module))
        self.assertTrue(config.general_topics)
        self.assertTrue(config.topics_to_examine)
        self.assertTrue(config.rss_sources)
        self.assertEqual(config.cache.discovery_mode, "network_first")
        self.assertEqual(config.cache.article_text_retention_days, 3)
        self.assertEqual(config.cache.enrichment_retention_days, 30)
        self.assertEqual(config.cache.wikipedia_retention_days, 30)

        with self.subTest("summary and final share managed llama runtime"):
            shared_runtime_fields = (
                "backend",
                "base_url",
                "server_model",
                "manage_server",
                "server_executable",
                "server_model_path",
                "server_arguments",
                "server_log_dir",
                "server_auto_stop",
            )
            for field in shared_runtime_fields:
                self.assertEqual(getattr(config.ai_summary, field), getattr(config.ai_final, field), field)

        with self.subTest("factory trusts validated config"):
            with self.assertRaisesRegex(ValueError, "Unsupported ai backend: auto"):
                create_ai_client(AIConfig(backend="auto"))

        payload = self._config_payload()
        with self.subTest("legacy cache defaults"):
            legacy_cache_payload = deepcopy(payload)
            for key in (
                "discovery_mode",
                "article_text_retention_days",
                "enrichment_retention_days",
                "wikipedia_retention_days",
            ):
                legacy_cache_payload["cache"].pop(key, None)
            legacy_cache_payload["cache"]["http_retention_days"] = 7
            legacy_cache_path = self._write_config_payload(TEMP_ROOT, legacy_cache_payload, "legacy_cache")
            legacy_config = load_config(legacy_cache_path)
            self.assertEqual(legacy_config.cache.discovery_mode, "network_first")
            self.assertEqual(legacy_config.cache.article_text_retention_days, 3)
            self.assertEqual(legacy_config.cache.enrichment_retention_days, 30)
            self.assertEqual(legacy_config.cache.wikipedia_retention_days, 30)

        with self.subTest("removed ai preset"):
            preset_payload = deepcopy(payload)
            preset_payload["ai_summary"]["preset"] = "qwen3-8b"
            preset_path = self._write_config_payload(TEMP_ROOT, preset_payload, "preset")
            with self.assertRaisesRegex(ValueError, "ai_summary.preset is no longer supported"):
                load_config(preset_path)

        with self.subTest("removed backend aliases"):
            alias_payload = deepcopy(payload)
            alias_payload["ai_summary"]["backend"] = "auto"
            alias_path = self._write_config_payload(TEMP_ROOT, alias_payload, "backend_alias")
            with self.assertRaisesRegex(ValueError, "Unsupported ai_summary.backend 'auto'"):
                load_config(alias_path)

        with self.subTest("canonical backend spelling only"):
            hyphen_payload = deepcopy(payload)
            hyphen_payload["ai_summary"]["backend"] = "llama-cpp-server"
            hyphen_path = self._write_config_payload(TEMP_ROOT, hyphen_payload, "backend_hyphen")
            with self.assertRaisesRegex(ValueError, "Unsupported ai_summary.backend 'llama-cpp-server'"):
                load_config(hyphen_path)

        with self.subTest("general filtering error label"):
            general_filtering_payload = deepcopy(payload)
            general_filtering_payload["general_filtering"]["max_candidates_for_ai"] = ""
            general_filtering_path = self._write_config_payload(
                TEMP_ROOT,
                general_filtering_payload,
                "general_filtering_label",
            )
            with self.assertRaisesRegex(ValueError, "general_filtering.max_candidates_for_ai"):
                load_config(general_filtering_path)

        with self.subTest("detailed filtering error label"):
            filtering_payload = deepcopy(payload)
            filtering_payload["filtering"]["fill_selected_articles"] = "maybe"
            filtering_path = self._write_config_payload(TEMP_ROOT, filtering_payload, "filtering_label")
            with self.assertRaisesRegex(ValueError, "filtering.fill_selected_articles"):
                load_config(filtering_path)

    def test_removed_evaluation_release_surface_stays_removed(self) -> None:
        retired_modules = ("mydailynews.evaluation", "mydailynews.prompt_regression")
        for module_name in retired_modules:
            with self.subTest(module=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))

        retired_paths = (
            REPO_ROOT / "docs" / "evaluation",
            REPO_ROOT / "tools" / "baseline_eval.py",
            REPO_ROOT / "tools" / "prompt_regression_pack.py",
            REPO_ROOT / "tools" / "release_gate.py",
        )
        for path in retired_paths:
            with self.subTest(path=str(path.relative_to(REPO_ROOT))):
                self.assertFalse(path.exists())

    def test_cli_list_stages_smoke(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "main.py", "--list-stages"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Available pipeline stages:", result.stdout)
        self.assertIn("headline_select", result.stdout)
        self.assertNotIn("Config not found", result.stdout)

    def test_stage_options_and_artifacts_are_replay_ready(self) -> None:
        options = PipelineRunOptions.from_cli(
            brief="general",
            stop_after_stage="article-fetch",
            save_intermediate=False,
            no_save_intermediate=False,
            dump_stage_artifacts=True,
            stage_artifact_dir="output/stages",
        )
        payload = build_stage_payload(
            stage="headline_select",
            brief="general",
            summary={"selected": 1},
            next_stage_input={"selected": [{"id": "candidate-1"}]},
        )
        artifact = build_stage_artifact(
            run_label="20260611_000000",
            brief="general",
            stage="headline_select",
            generated_at="2026-06-11T00:00:00+00:00",
            summary=payload["summary"],
            next_stage_input=payload["next_stage_input"],
        )

        self.assertEqual(options.briefs, ("general",))
        self.assertEqual(options.stop_after_stage, "article_fetch")
        self.assertTrue(options.save_intermediate)
        self.assertTrue(options.dump_stage_artifacts)
        self.assertEqual(artifact["schema_version"], STAGE_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(artifact["next_stage"], "article_fetch")
        self.assertEqual(artifact["next_stage_input"]["selected"][0]["id"], "candidate-1")
        self.assertNotIn("payload", artifact)
        self.assertNotIn("intermediate", artifact)


if __name__ == "__main__":
    unittest.main()
