# PR08: Prompt Regression Pack and Release Gates

## Objective
Lock in quality gains from PR02-PR07 and reduce future drift using prompt regression checks and release gates.

## Why This PR
Prompt-centric systems regress easily when templates or payload shapes change. We need repeatable safety checks.

## Scope
In scope:

1. Add prompt regression fixtures for headline, evidence, delta, and final brief stages.
2. Add quality and runtime release gates using PR01 metrics.
3. Add operational checklist for safe prompt updates.

Out of scope:

1. New ranking logic.
2. New LLM stage design.

## Planned File Changes
1. `tests/`
   - Prompt snapshot and schema-compat regression tests.
2. `tools/`
   - Automation wrappers for benchmark + gate checks.
3. `docs/`
   - Prompt change checklist and release policy.

## Implementation Plan
1. Add golden prompt fixtures:
   - Store rendered prompts for fixed inputs.
   - Assert critical instruction clauses remain present.

2. Add response-shape regression fixtures:
   - Ensure parser compatibility for expected model outputs.

3. Add release gate command:
   - Run benchmark, compare against baseline thresholds, emit pass/fail summary.

4. Add prompt-authoring policy:
   - Keep prompts concise.
   - Avoid schema-changing edits without matching parser/test updates.

## Runtime and Memory Impact
Expected runtime impact in production: none.

CI/test runtime increases modestly due benchmark checks.

## Things To Be Careful About
1. Snapshot brittleness:
   - Assert critical clauses, not every character.
2. Gate noise:
   - Use tolerance bands to avoid false failures.
3. Local model variance:
   - Separate deterministic prompt checks from stochastic quality checks.

## Tests
1. Prompt fixture rendering tests.
2. Schema parse compatibility tests.
3. End-to-end benchmark gate dry run.

## Acceptance Criteria
1. One command produces release readiness decision.
2. Prompt regressions are caught before merge.
3. Quality and runtime guardrails are enforceable and documented.

## Rollback Plan
If gates are too strict, downgrade hard-fail thresholds to warnings until tuned.
