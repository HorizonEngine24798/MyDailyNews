from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from mydailynews.evaluation import (
    build_evaluation_report,
    evaluate_guardrails,
    load_guardrails,
    percentile,
    render_report_markdown,
)


class BaselineEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.dataset_path = cls.repo_root / "docs" / "evaluation" / "baseline_dataset_v1.json"
        cls.artifact_root = cls.repo_root / "docs" / "evaluation" / "fixtures" / "run_artifacts"
        cls.guardrails_path = cls.repo_root / "docs" / "evaluation" / "guardrails_v1.json"

    def test_percentile_linear_interpolation(self) -> None:
        self.assertAlmostEqual(percentile([180.0, 200.0, 220.0], 95), 218.0, places=6)

    def test_build_report_aggregate_metrics(self) -> None:
        report = build_evaluation_report(self.dataset_path, self.artifact_root)
        aggregate = report["aggregate"]

        self.assertEqual(report["schema_version"], "baseline_eval_report.v1")
        self.assertAlmostEqual(float(aggregate["top5_story_value_precision"]), 0.7333, places=4)
        self.assertAlmostEqual(float(aggregate["novelty_ratio"]), 0.5625, places=4)
        self.assertAlmostEqual(float(aggregate["duplication_leak_rate"]), 0.1250, places=4)
        self.assertAlmostEqual(float(aggregate["decision_context_alignment"]), 0.6875, places=4)
        self.assertAlmostEqual(float(aggregate["brief_utility_score"]), 4.3333, places=4)
        self.assertAlmostEqual(float(aggregate["latency_p95_sec"]), 218.0, places=4)
        self.assertEqual(int(aggregate["oom_or_truncation_incidents"]), 2)
        self.assertEqual(int(aggregate["max_prompt_tokens_by_stage"]["headline_scoring"]), 2100)
        self.assertEqual(int(aggregate["max_prompt_tokens_by_stage"]["final_brief_generation"]), 3500)

    def test_guardrails_and_markdown_shape(self) -> None:
        report = build_evaluation_report(self.dataset_path, self.artifact_root)
        guardrails = load_guardrails(self.guardrails_path)
        result = evaluate_guardrails(report, guardrails)
        markdown = render_report_markdown(report, result)

        self.assertTrue(result["passed"])
        self.assertTrue(any(check["name"] == "latency_p95_sec" for check in result["checks"]))
        self.assertIn("## Aggregate Metrics", markdown)
        self.assertIn("## Guardrails", markdown)
        self.assertIn("case_2026_05_31_general_reader_general", markdown)

    def test_cli_writes_report_for_fixed_artifacts(self) -> None:
        out_dir = self.repo_root / ".codex_tmp_test" / "baseline_eval_cli"
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(self.repo_root / "tools" / "baseline_eval.py"),
            "--output-dir",
            str(out_dir),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            self.fail(
                "baseline_eval.py failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )

        report_json = out_dir / "baseline_eval_report.json"
        report_md = out_dir / "baseline_eval_report.md"
        self.assertTrue(report_json.exists())
        self.assertTrue(report_md.exists())

        payload = json.loads(report_json.read_text(encoding="utf-8"))
        self.assertIn("guardrails", payload)
        self.assertTrue(payload["guardrails"]["passed"])


if __name__ == "__main__":
    unittest.main()
