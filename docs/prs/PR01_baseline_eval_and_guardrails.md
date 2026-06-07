# PR01: Baseline Evaluation and Guardrails

## Objective
Create a reliable quality and performance baseline before changing scoring or prompts.

Without this PR, later quality claims will be subjective and hard to validate.

## Why This PR First
All following PRs change judgment behavior. We need hard before/after numbers and stable regression checks.

## Scope
In scope:

1. Add an offline evaluation script for briefing quality samples.
2. Add standard metrics output for quality, latency, and token pressure.
3. Add release guardrails (pass/fail thresholds) for critical regressions.

Out of scope:

1. Any scoring prompt or schema changes.
2. Any output-format changes.

## Planned File Changes
Primary targets:

1. `tools/`:
   - New evaluator scripts and sample runner.
2. `tests/`:
   - Tests for metric calculation and artifact parsing.
3. `docs/`:
   - Evaluation runbook and metric definitions.
4. Optional:
   - `mydailynews/debug.py` metrics additions if needed for telemetry extraction.

## Implementation Plan
1. Add fixed evaluation dataset:
   - Small curated batch of historical inputs and expected editorial judgments.
   - Include at least three profile types (general reader, operator/founder, policy-focused).

2. Add scoring rubric for human eval:
   - "Would regret missing this" score.
   - Novelty score.
   - Personal relevance score.
   - Utility score.

3. Add automated report generator:
   - Reads run outputs and calculates core metrics.
   - Produces machine-readable JSON plus readable markdown summary.

4. Add regression gate config:
   - Fail if `latency_p95_sec` or prompt token pressure exceeds threshold.
   - Fail if quality metrics regress beyond tolerance.

## Runtime and Memory Impact
Expected runtime impact on normal pipeline: none.

Evaluation scripts run offline and are not part of main pipeline execution.

## Things To Be Careful About
1. Dataset leakage:
   - Do not use synthetic examples too similar to prompt text.
2. Metric drift:
   - Keep metric definitions versioned and immutable once adopted.
3. Overfitting to benchmark:
   - Keep small hidden holdout set for sanity checks.

## Tests
1. Unit tests for metric computation.
2. Snapshot tests for evaluator output shape.
3. Integration test on one fixed run artifact directory.

## Acceptance Criteria
1. One-command evaluation run is documented and reproducible.
2. Baseline report includes quality, latency, prompt-token, and warning metrics.
3. Regression gate thresholds are clearly defined for later PRs.

## Rollback Plan
If scripts are unstable, keep PR limited to passive reporting only and defer hard gating to PR08.
