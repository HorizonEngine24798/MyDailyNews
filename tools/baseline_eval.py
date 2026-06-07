from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.evaluation import (  # noqa: E402
    build_evaluation_report,
    evaluate_guardrails,
    load_guardrails,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline baseline quality/performance evaluation and guardrails.")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "docs" / "evaluation" / "baseline_dataset_v1.json"),
        help="Path to baseline evaluation dataset JSON.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(REPO_ROOT / "docs" / "evaluation" / "fixtures" / "run_artifacts"),
        help="Root directory containing run artifacts referenced by the dataset.",
    )
    parser.add_argument(
        "--guardrails",
        default=str(REPO_ROOT / "docs" / "evaluation" / "guardrails_v1.json"),
        help="Path to guardrail thresholds JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "evaluation"),
        help="Directory where report JSON/Markdown are written.",
    )
    parser.add_argument(
        "--no-enforce-guardrails",
        action="store_true",
        help="Do not return non-zero exit code when guardrails fail.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    artifact_root = Path(args.artifact_root)
    guardrails_path = Path(args.guardrails)
    output_dir = Path(args.output_dir)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 2
    if not artifact_root.exists():
        print(f"Artifact root not found: {artifact_root}")
        return 2
    if not guardrails_path.exists():
        print(f"Guardrails config not found: {guardrails_path}")
        return 2

    report = build_evaluation_report(dataset_path, artifact_root)
    guardrails = load_guardrails(guardrails_path)
    guardrail_result = evaluate_guardrails(report, guardrails)
    json_path, markdown_path = write_report(output_dir, report, guardrail_result)

    aggregate = report.get("aggregate", {})
    print("Baseline evaluation complete.")
    print(f"- report_json: {json_path}")
    print(f"- report_markdown: {markdown_path}")
    print(f"- cases: {report.get('dataset', {}).get('cases_count', 0)}")
    print(f"- top5_story_value_precision: {aggregate.get('top5_story_value_precision')}")
    print(f"- novelty_ratio: {aggregate.get('novelty_ratio')}")
    print(f"- decision_context_alignment: {aggregate.get('decision_context_alignment')}")
    print(f"- brief_utility_score: {aggregate.get('brief_utility_score')}")
    print(f"- latency_p95_sec: {aggregate.get('latency_p95_sec')}")
    print(f"- guardrails_passed: {guardrail_result.get('passed')}")

    if guardrail_result.get("passed") is False and not args.no_enforce_guardrails:
        print("Guardrails failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
