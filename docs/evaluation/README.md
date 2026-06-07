# Baseline Evaluation and Guardrails (PR01)

This directory defines the fixed offline baseline for quality and runtime checks.

## One-Command Run

From repo root:

```powershell
python tools/baseline_eval.py
```

This command is reproducible and offline. It uses:

- `docs/evaluation/baseline_dataset_v1.json`
- `docs/evaluation/fixtures/run_artifacts/`
- `docs/evaluation/guardrails_v1.json`

Outputs are written to:

- `output/evaluation/baseline_eval_report.json`
- `output/evaluation/baseline_eval_report.md`

The command exits non-zero when guardrails fail. To generate a report without failing the command:

```powershell
python tools/baseline_eval.py --no-enforce-guardrails
```

## PR08 Release Gate

Prompt + guardrail release readiness (single command):

```powershell
python tools/release_gate.py
```

This runs:

1. Prompt regression pack tests (`tests/test_prompt_regression_pack.py`)
2. Baseline evaluation guardrails (`tools/baseline_eval.py`)

Prompt fixture-only verification:

```powershell
python tools/prompt_regression_pack.py --verify
```

## Dataset Contract

`baseline_dataset_v1.json` includes:

- Fixed profile set (general reader, operator/founder, policy-focused).
- Fixed historical artifact references.
- Per-story human judgments:
  - `would_regret_missing` (boolean).
  - `novelty` (1-5).
  - `personal_relevance` (1-5).
  - `utility` (1-5).
- Per-case brief utility rating.

This keeps metric definitions stable and versioned (`metric_definitions_version`).

## Metrics Produced

Quality metrics:

- `top5_story_value_precision`: top-5 stories judged as "should not miss".
- `novelty_ratio`: selected stories with high novelty score.
- `duplication_leak_rate`: selected stories that duplicate a prior selected event cluster.
- `decision_context_alignment`: selected stories aligned with profile relevance threshold.
- `brief_utility_score`: average utility score from case ratings.

Performance and pressure metrics:

- `latency_p95_sec`: p95 of `pipeline.total` from debug analytics artifacts.
- `max_prompt_tokens_by_stage`: max prompt tokens per stage from probe artifacts.
- `max_ai_input_tokens_by_bucket`: fallback/secondary token-pressure metric from debug analytics.
- `oom_or_truncation_incidents`: warnings indicating OOM/truncation/budget overflow.

## Guardrail Policy

`guardrails_v1.json` enforces:

- Absolute limits for latency and incident counts.
- Stage-level prompt token caps.
- Quality regression tolerances against baseline values.

This provides release-safe pass/fail gates for later PRs before prompt/schema changes land.
