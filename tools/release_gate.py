from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateStepResult:
    name: str
    command: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prompt regression checks and baseline evaluation release gates.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "evaluation"),
        help="Output directory for baseline evaluation artifacts.",
    )
    parser.add_argument(
        "--no-enforce-guardrails",
        action="store_true",
        help="Pass-through to baseline_eval.py; do not fail on guardrail checks.",
    )
    parser.add_argument(
        "--skip-prompt-tests",
        action="store_true",
        help="Skip prompt regression unittest step.",
    )
    parser.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        help="Skip baseline evaluation step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser


def _run_step(name: str, command: List[str], *, dry_run: bool) -> GateStepResult:
    print(f"[release-gate] step={name}")
    print(f"[release-gate] command={' '.join(command)}")
    if dry_run:
        return GateStepResult(name=name, command=command, returncode=0, stdout="", stderr="")
    completed = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if completed.stdout.strip():
        print(completed.stdout.rstrip())
    if completed.returncode != 0 and completed.stderr.strip():
        print(completed.stderr.rstrip())
    return GateStepResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def main() -> int:
    args = build_parser().parse_args()
    python_exe = sys.executable
    steps: List[tuple[str, List[str]]] = []

    if not args.skip_prompt_tests:
        steps.append(
            (
                "prompt_regression_tests",
                [
                    python_exe,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_prompt_regression_pack.py",
                ],
            )
        )

    if not args.skip_baseline_eval:
        baseline_command = [
            python_exe,
            str(REPO_ROOT / "tools" / "baseline_eval.py"),
            "--output-dir",
            str(Path(args.output_dir)),
        ]
        if args.no_enforce_guardrails:
            baseline_command.append("--no-enforce-guardrails")
        steps.append(("baseline_eval_guardrails", baseline_command))

    if not steps:
        print("[release-gate] no steps selected")
        return 2

    results: List[GateStepResult] = []
    for name, command in steps:
        result = _run_step(name, command, dry_run=bool(args.dry_run))
        results.append(result)
        if not result.passed:
            break

    failed = [item for item in results if not item.passed]
    print("[release-gate] summary")
    for item in results:
        print(f"- {item.name}: {'PASS' if item.passed else 'FAIL'}")

    if failed:
        print("[release-gate] release decision: FAIL")
        return 1
    print("[release-gate] release decision: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
