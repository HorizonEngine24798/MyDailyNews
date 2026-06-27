from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from mydailynews.app.config import load_config
from tools import autoconfig


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "autoconfig"


class FakeDownloadResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self) -> "FakeDownloadResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield from self._chunks


class FakeInteractiveStdin:
    def isatty(self) -> bool:
        return True


class AutoconfigTests(unittest.TestCase):
    def _temp_dir(self) -> Path:
        path = TEMP_ROOT / self.id().rsplit(".", 1)[-1] / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _catalog(self) -> dict:
        return autoconfig.load_json(REPO_ROOT / "profiles" / "model_catalog.json")

    def _example_config(self) -> dict:
        return autoconfig.load_json(REPO_ROOT / "config.example.json")

    def test_hardware_tier_selects_recommended_model(self) -> None:
        catalog = self._catalog()
        hardware = autoconfig.HardwareInfo(
            os_name="test",
            system_ram_gb=64,
            gpu_name="RTX test",
            gpu_vendor="nvidia",
            vram_gb=12,
        )

        tier = autoconfig.choose_tier(catalog, hardware)
        model = autoconfig.model_for_tier(catalog, tier)

        self.assertEqual(tier["id"], "nvidia_12gb")
        self.assertEqual(model["id"], "qwen3-14b-q4")

    def test_recommended_config_is_coupled_and_does_not_mutate_source(self) -> None:
        catalog = self._catalog()
        source = self._example_config()
        source_before = deepcopy(source)
        tier = next(item for item in catalog["tiers"] if item["id"] == "nvidia_8gb")
        model = autoconfig.model_for_tier(catalog, tier)

        recommended = autoconfig.build_recommended_config(source, tier, model)

        self.assertEqual(source, source_before)
        self.assertEqual(recommended["ai_summary"]["server_model"], "Qwen3-8B-Q4_K_M")
        self.assertEqual(recommended["ai_summary"]["context_window_tokens"], 8192)
        self.assertEqual(recommended["ai_summary"]["max_input_tokens"], 6000)
        self.assertEqual(recommended["ai_final"]["max_new_tokens"], 1024)
        self.assertEqual(recommended["general_filtering"]["max_headlines_per_ai_batch"], 6)
        self.assertEqual(recommended["filtering"]["max_selected_articles"], 5)
        self.assertTrue(recommended["enrichment"]["enabled"])
        self.assertEqual(recommended["enrichment"]["max_story_threads"], 8)
        self.assertEqual(recommended["enrichment"]["planner_max_questions_per_story"], 3)
        self.assertEqual(recommended["enrichment"]["max_fetched_research_pages_per_story"], 4)
        self.assertEqual(recommended["enrichment"]["max_selected_article_excerpt_chars"], 2800)
        self.assertEqual(recommended["enrichment"]["max_context_chars_per_article"], 2400)
        self.assertTrue(recommended["narrative_briefing"]["enabled"])
        self.assertEqual(recommended["narrative_briefing"]["target_words"], 1800)
        self.assertEqual(recommended["pipeline"]["default_series"], ["briefs", "enrichment", "narrative_brief"])
        self.assertEqual(recommended["analysis"]["evidence_distillation"]["max_input_tokens"], 5000)

    def test_recommended_config_rewrites_stale_enrichment_section(self) -> None:
        catalog = self._catalog()
        source = self._example_config()
        source["enrichment"]["mode"] = "simple"
        source["enrichment"]["max_entities"] = 5
        source.setdefault("runtime", {})["max_enrichment_workers"] = 4
        source.setdefault("cache", {})["wikipedia_retention_days"] = 30
        source["filtering"]["max_selected_per_event_cluster"] = 1
        source["general_filtering"]["prefer_multi_source_clusters"] = True
        tier = next(item for item in catalog["tiers"] if item["id"] == "nvidia_12gb")
        model = autoconfig.model_for_tier(catalog, tier)

        recommended = autoconfig.build_recommended_config(source, tier, model)

        self.assertTrue(recommended["enrichment"]["enabled"])
        self.assertEqual(recommended["enrichment"]["mode"], "story_llm")
        self.assertNotIn("max_entities", recommended["enrichment"])
        self.assertNotIn("max_enrichment_workers", recommended["runtime"])
        self.assertNotIn("wikipedia_retention_days", recommended["cache"])
        self.assertNotIn("max_selected_per_event_cluster", recommended["filtering"])
        self.assertNotIn("prefer_multi_source_clusters", recommended["general_filtering"])
        self.assertEqual(recommended["enrichment"]["max_story_threads"], 10)
        self.assertEqual(recommended["enrichment"]["cache_ttl_seconds"], 604800)

        path = self._temp_dir() / "recommended.json"
        path.write_text(json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8")
        load_config(path)

    def test_recommended_config_preserves_explicit_enrichment_opt_in(self) -> None:
        catalog = self._catalog()
        source = self._example_config()
        source["enrichment"]["enabled"] = True
        tier = next(item for item in catalog["tiers"] if item["id"] == "nvidia_8gb")
        model = autoconfig.model_for_tier(catalog, tier)

        recommended = autoconfig.build_recommended_config(source, tier, model)

        self.assertTrue(recommended["enrichment"]["enabled"])
        self.assertEqual(recommended["enrichment"]["mode"], "story_llm")
        self.assertEqual(recommended["enrichment"]["max_story_threads"], 8)

    def test_recommended_config_preserves_explicit_enrichment_opt_out(self) -> None:
        catalog = self._catalog()
        source = self._example_config()
        source["enrichment"]["enabled"] = False
        source["enrichment"]["mode"] = "story_llm"
        tier = next(item for item in catalog["tiers"] if item["id"] == "nvidia_8gb")
        model = autoconfig.model_for_tier(catalog, tier)

        recommended = autoconfig.build_recommended_config(source, tier, model)

        self.assertFalse(recommended["enrichment"]["enabled"])
        self.assertEqual(recommended["enrichment"]["mode"], "story_llm")
        self.assertEqual(recommended["enrichment"]["max_story_threads"], 8)

    def test_default_pipeline_preferences_preserve_generated_defaults(self) -> None:
        catalog = self._catalog()
        source = self._example_config()
        tier = next(item for item in catalog["tiers"] if item["id"] == "nvidia_8gb")
        model = autoconfig.model_for_tier(catalog, tier)
        recommended = autoconfig.build_recommended_config(source, tier, model)
        before = deepcopy(recommended)

        autoconfig.apply_pipeline_preferences(recommended, autoconfig.PipelinePreferences())

        self.assertEqual(recommended, before)

    def test_apply_pipeline_preferences_rewrites_user_workflow_shape(self) -> None:
        catalog = self._catalog()
        source = self._example_config()
        tier = next(item for item in catalog["tiers"] if item["id"] == "nvidia_8gb")
        model = autoconfig.model_for_tier(catalog, tier)
        recommended = autoconfig.build_recommended_config(source, tier, model)
        original_selected = recommended["filtering"]["max_selected_articles"]
        preferences = autoconfig.PipelinePreferences(
            workflow="narrative",
            brief_volume="compact",
            analysis_depth="fast",
            narrative_length="concise",
            server_mode="external",
            cache_mode="cache",
        )

        autoconfig.apply_pipeline_preferences(recommended, preferences)

        self.assertEqual(recommended["pipeline"]["default_series"], ["briefs", "narrative_brief"])
        self.assertFalse(recommended["enrichment"]["enabled"])
        self.assertTrue(recommended["narrative_briefing"]["enabled"])
        self.assertEqual(recommended["narrative_briefing"]["target_words"], 1000)
        self.assertLess(recommended["filtering"]["max_selected_articles"], original_selected)
        self.assertEqual(recommended["analysis"]["rollout"]["enabled"], False)
        self.assertEqual(recommended["analysis"]["evidence_distillation"]["enabled"], False)
        self.assertEqual(recommended["analysis"]["delta_extraction"]["enabled"], False)
        self.assertFalse(recommended["ai_summary"]["manage_server"])
        self.assertFalse(recommended["ai_summary"]["server_auto_stop"])
        self.assertFalse(recommended["ai_final"]["manage_server"])
        self.assertFalse(recommended["ai_final"]["server_auto_stop"])
        self.assertEqual(recommended["cache"]["discovery_mode"], "cache_first")

    def test_prompt_pipeline_preferences_accepts_names_and_numbers(self) -> None:
        stdout = io.StringIO()
        answers = ["research", "3", "deep", "long", "external", "cache"]

        with patch("tools.autoconfig.sys.stdin", FakeInteractiveStdin()), patch(
            "builtins.input",
            side_effect=answers,
        ), redirect_stdout(stdout):
            preferences = autoconfig.maybe_prompt_pipeline_preferences()

        self.assertEqual(
            preferences,
            autoconfig.PipelinePreferences(
                workflow="research",
                brief_volume="wide",
                analysis_depth="deep",
                narrative_length="long",
                server_mode="external",
                cache_mode="cache",
            ),
        )

    def test_existing_model_path_is_preserved_and_wired_to_both_ai_sections(self) -> None:
        temp_dir = self._temp_dir()
        model_path = temp_dir / "model.gguf"
        model_path.write_bytes(b"gguf")
        config = self._example_config()
        config["ai_summary"]["server_model_path"] = str(model_path)
        config["ai_summary"]["server_model"] = "local-model"
        config["ai_final"]["server_model_path"] = "PATH/TO/model.gguf"

        found = autoconfig.existing_model_path(config)
        autoconfig.set_model_path(config, found, "local-model")

        self.assertEqual(found, str(model_path))
        self.assertEqual(config["ai_summary"]["server_model_path"], str(model_path))
        self.assertEqual(config["ai_final"]["server_model_path"], str(model_path))
        self.assertEqual(config["ai_final"]["server_model"], "local-model")

    def test_download_model_streams_to_ignored_models_dir(self) -> None:
        temp_dir = self._temp_dir()
        model = {
            "filename": "tiny.gguf",
            "url": "https://example.test/tiny.gguf",
        }

        with patch("tools.autoconfig.requests.get", return_value=FakeDownloadResponse([b"gg", b"uf"])) as get:
            path = autoconfig.download_model(model, temp_dir)

        self.assertEqual(path.read_bytes(), b"gguf")
        get.assert_called_once()

    def test_main_writes_recommended_config_without_mutating_input(self) -> None:
        temp_dir = self._temp_dir()
        source_path = temp_dir / "config.local.json"
        target_path = temp_dir / "config.recommended.json"
        source = self._example_config()
        source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")

        stdout = io.StringIO()
        with patch(
            "tools.autoconfig.detect_hardware",
            return_value=autoconfig.HardwareInfo("test-os", 64, "RTX test", "nvidia", 24),
        ), patch(
            "tools.autoconfig.probe_config",
            return_value=autoconfig.ProbeReport(version_ok=True, server_ready=True, json_probe_ok=True),
        ), redirect_stdout(stdout):
            rc = autoconfig.main(
                [
                    "--config",
                    str(source_path),
                    "--write",
                    str(target_path),
                    "--model-catalog",
                    str(REPO_ROOT / "profiles" / "model_catalog.json"),
                    "--no-download-prompt",
                    "--no-preference-prompt",
                ]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(source_path.read_text(encoding="utf-8")), source)
        written = json.loads(target_path.read_text(encoding="utf-8"))
        self.assertEqual(written["ai_summary"]["server_model"], "Qwen3-30B-A3B-Q4_K_M")
        self.assertEqual(written["ai_summary"]["context_window_tokens"], 32768)
        self.assertTrue(written["enrichment"]["enabled"])
        self.assertEqual(written["enrichment"]["max_story_threads"], 16)
        self.assertEqual(written["enrichment"]["max_fetched_research_pages_per_story"], 10)
        self.assertEqual(written["enrichment"]["max_research_excerpt_chars"], 4000)
        self.assertEqual(written["pipeline"]["default_series"], ["briefs", "enrichment", "narrative_brief"])


if __name__ == "__main__":
    unittest.main()
