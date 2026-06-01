# Prompt Change Checklist and Release Policy (PR08)

This checklist is required for any prompt, schema, or prompt-payload update across:

- Headline scoring
- Evidence distillation
- Delta extraction
- Final brief generation

## One-Command Release Gate

From repo root:

```powershell
python tools/release_gate.py
```

This command combines:

1. Deterministic prompt regression tests (`tests/test_prompt_regression_pack.py`)
2. Baseline evaluation + guardrail checks (`tools/baseline_eval.py`)

It returns non-zero when release gates fail.

## Prompt Authoring Policy

1. Keep prompts concise and operational.
2. Preserve strict JSON-only response instructions.
3. Avoid schema-changing edits unless parser updates and tests ship in the same PR.
4. Keep evidence-grounding language explicit:
   - only use supplied evidence
   - do not invent facts
5. Preserve compactness constraints to reduce local-model truncation risk.

## Required Pre-Merge Checks

1. Run prompt regression fixture verification:

```powershell
python tools/prompt_regression_pack.py --verify
```

2. Run PR08 release gate:

```powershell
python tools/release_gate.py
```

3. Confirm guardrails remain green in:
   - `output/evaluation/baseline_eval_report.json`
   - `output/evaluation/baseline_eval_report.md`

## When Prompts Intentionally Change

1. Rebuild fixture snapshot:

```powershell
python tools/prompt_regression_pack.py
```

2. Review prompt diffs for all four stages.
3. Adjust tests only when change is intentional and justified.
4. Re-run full release gate command.

## Rollback Guidance

If a prompt change causes unstable gates:

1. Revert prompt text and fixture together.
2. Keep parser/schema compatibility intact.
3. If guardrails are overly strict, downgrade to warning mode temporarily using:

```powershell
python tools/baseline_eval.py --no-enforce-guardrails
```

Then tune thresholds before re-enabling hard fail.

