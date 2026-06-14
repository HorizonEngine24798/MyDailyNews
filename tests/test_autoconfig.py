from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

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
        self.assertEqual(recommended["analysis"]["evidence_distillation"]["max_input_tokens"], 5000)

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
                ]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(source_path.read_text(encoding="utf-8")), source)
        written = json.loads(target_path.read_text(encoding="utf-8"))
        self.assertEqual(written["ai_summary"]["server_model"], "Qwen3-30B-A3B-Q4_K_M")
        self.assertEqual(written["ai_summary"]["context_window_tokens"], 32768)


if __name__ == "__main__":
    unittest.main()
