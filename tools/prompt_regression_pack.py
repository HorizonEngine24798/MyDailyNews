from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.prompt_regression import (  # noqa: E402
    PROMPT_REGRESSION_SCHEMA_VERSION,
    build_prompt_regression_pack,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate or verify prompt regression fixture pack.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "docs" / "evaluation" / "fixtures" / "prompt_regression_pack_v1.json"),
        help="Prompt regression fixture JSON path.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing fixture against current prompt renderer (no file writes).",
    )
    return parser


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def verify_fixture(expected: Dict[str, Any], generated: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(expected.get("schema_version", "")).strip() != PROMPT_REGRESSION_SCHEMA_VERSION:
        errors.append(
            f"Unsupported fixture schema_version={expected.get('schema_version')!r}; "
            f"expected {PROMPT_REGRESSION_SCHEMA_VERSION!r}."
        )
    expected_stages = expected.get("stages")
    generated_stages = generated.get("stages")
    if not isinstance(expected_stages, dict) or not isinstance(generated_stages, dict):
        return errors + ["Fixture/generation missing 'stages' map."]

    for stage_name, expected_stage in expected_stages.items():
        generated_stage = generated_stages.get(stage_name)
        if not isinstance(expected_stage, dict) or not isinstance(generated_stage, dict):
            errors.append(f"Stage {stage_name!r} missing in fixture or generated pack.")
            continue
        prompt = str(generated_stage.get("rendered_prompt", ""))
        fixture_prompt = str(expected_stage.get("rendered_prompt", ""))
        if not prompt:
            errors.append(f"Stage {stage_name!r} generated prompt is empty.")
            continue
        if not fixture_prompt:
            errors.append(f"Stage {stage_name!r} fixture prompt is empty.")
        max_chars = _safe_int(expected_stage.get("max_chars"), default=0)
        if max_chars > 0 and len(prompt) > max_chars:
            errors.append(f"Stage {stage_name!r} prompt chars {len(prompt)} exceeded max_chars {max_chars}.")
        baseline_chars = _safe_int(expected_stage.get("prompt_chars"), default=len(fixture_prompt))
        max_char_delta = _safe_int(expected_stage.get("max_char_delta"), default=0)
        if max_char_delta > 0 and abs(len(prompt) - baseline_chars) > max_char_delta:
            errors.append(
                f"Stage {stage_name!r} prompt char delta {abs(len(prompt) - baseline_chars)} exceeded "
                f"max_char_delta {max_char_delta}."
            )
        for clause in [str(item) for item in _safe_list(expected_stage.get("required_clauses")) if str(item).strip()]:
            if clause not in prompt:
                errors.append(f"Stage {stage_name!r} missing required clause in generated prompt: {clause!r}")
            if clause not in fixture_prompt:
                errors.append(f"Stage {stage_name!r} missing required clause in fixture snapshot: {clause!r}")
    return errors


def main() -> int:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = build_prompt_regression_pack()

    if args.verify:
        if not output_path.exists():
            print(f"Prompt regression fixture not found: {output_path}")
            return 2
        expected = json.loads(output_path.read_text(encoding="utf-8"))
        errors = verify_fixture(expected, generated)
        if errors:
            print("Prompt regression verification FAILED.")
            for item in errors:
                print(f"- {item}")
            return 1
        print("Prompt regression verification passed.")
        print(f"- fixture: {output_path}")
        for stage_name, stage_payload in sorted(generated.get("stages", {}).items()):
            if not isinstance(stage_payload, dict):
                continue
            print(f"- {stage_name}: prompt_chars={len(str(stage_payload.get('rendered_prompt', '')))}")
        return 0

    output_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prompt regression fixture written.")
    print(f"- path: {output_path}")
    for stage_name, stage_payload in sorted(generated.get("stages", {}).items()):
        if not isinstance(stage_payload, dict):
            continue
        print(f"- {stage_name}: prompt_chars={len(str(stage_payload.get('rendered_prompt', '')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

