from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mydailynews.app.config import load_config
from mydailynews.app.models import AnalysisConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


class AnalysisConfigTests(unittest.TestCase):
    def test_defaults_are_disabled_for_both_briefs(self) -> None:
        analysis = AnalysisConfig()

        for brief_name in ("general", "detailed"):
            brief = getattr(analysis, brief_name)
            self.assertFalse(brief.evidence_distillation.enabled)
            self.assertFalse(brief.delta_extraction.enabled)

    def test_public_configs_preserve_materialized_effective_settings(self) -> None:
        expected = {
            "config.example.json": ((True, False, 8000, 8000), (True, True, 10000, 8000)),
            "config.cpu-small.example.json": ((False, False, 3000, 2600), (True, False, 3000, 2600)),
            "config.nvidia-8gb.example.json": ((False, False, 5000, 4200), (True, True, 5000, 4200)),
            "config.nvidia-12gb.example.json": ((True, False, 8000, 8000), (True, True, 10000, 8000)),
            "config.nvidia-24gb.example.json": ((True, False, 16000, 14000), (True, True, 18000, 14000)),
            "config.remote-server.example.json": ((True, False, 10000, 8000), (True, True, 10000, 8000)),
        }

        for filename, brief_values in expected.items():
            path = REPO_ROOT / filename if filename == "config.example.json" else REPO_ROOT / "profiles" / filename
            with self.subTest(filename=filename):
                analysis = load_config(path).analysis
                for brief_name, values in zip(("general", "detailed"), brief_values):
                    brief = getattr(analysis, brief_name)
                    actual = (
                        brief.evidence_distillation.enabled,
                        brief.delta_extraction.enabled,
                        brief.evidence_distillation.max_input_tokens,
                        brief.delta_extraction.max_input_tokens,
                    )
                    self.assertEqual(actual, values)

    def test_removed_rollout_shape_is_rejected(self) -> None:
        payload = json.loads((REPO_ROOT / "config.example.json").read_text(encoding="utf-8"))
        payload["analysis"]["rollout"] = {"enabled": True, "profile": "balanced_local"}
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "analysis.*rollout"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
