from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class ReleaseGateToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_release_gate_dry_run_lists_expected_steps(self) -> None:
        command = [
            sys.executable,
            str(self.repo_root / "tools" / "release_gate.py"),
            "--dry-run",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            self.fail(
                "release_gate.py --dry-run failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        stdout = completed.stdout
        self.assertIn("step=prompt_regression_tests", stdout)
        self.assertIn("step=baseline_eval_guardrails", stdout)
        self.assertIn("release decision: PASS", stdout)


if __name__ == "__main__":
    unittest.main()
