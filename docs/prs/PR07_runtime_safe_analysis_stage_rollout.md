# PR07: Runtime-Safe Rollout of Optional Evidence/Delta Stages

## Objective
Use existing optional analysis stages for better synthesis quality, but only with conservative runtime and memory profiles.

## Why This PR
Evidence distillation and delta extraction already exist, but default-disabled operation leaves synthesis value on the table.

This PR is a rollout and tuning PR, not a new architecture PR.

## Scope
In scope:

1. Introduce rollout profiles for optional stages.
2. Enable stages selectively by brief mode and hardware profile.
3. Add guardrails to prevent token/memory blowups.

Out of scope:

1. New analysis stages.
2. Major schema redesign for evidence/delta packets.

## Planned File Changes
1. `config.example.json`
   - Add rollout profiles and recommended defaults.
2. `mydailynews/brief_execution.py`
   - Mode-aware stage enablement decisions (general vs detailed).
3. `mydailynews/analysis_pipeline.py`
   - Tighten compacting and fallback behavior where needed.
4. `tests/test_pipeline_basics.py`
   - Coverage for mode-aware enablement and fallback behavior.
5. `docs/`
   - Profile tuning notes for low/mid/high VRAM setups.

## Rollout Strategy
Recommended staged rollout:

1. Phase A:
   - Enable `evidence_distillation` for `detailed` only.
   - Keep `general` unchanged.
2. Phase B:
   - Enable `delta_extraction` for `detailed`.
3. Phase C:
   - Consider enabling one stage for `general` only if latency target holds.

## Implementation Plan
1. Add config presets:
   - `safe_local`
   - `balanced_local`
   - `quality_focused`
2. Keep conservative caps for safe profile:
   - Low `max_articles`, `max_input_tokens`, and compact payload mode preference.
3. Preserve deterministic delta scaffold as automatic fallback.
4. Add explicit warning counters for:
   - Prompt truncation pressure.
   - Stage skips and fallback reasons.
5. Validate with PR01 benchmark set.

## Runtime and Memory Impact
Expected:

1. Additional latency in `detailed` when enabled.
2. Increased prompt/token pressure, controlled by existing compaction loops.
3. No increase in simultaneous model residency beyond current safeguards.

## Things To Be Careful About
1. Stage explosion:
   - Do not enable both stages for `general` by default.
2. Context pressure:
   - Keep compaction and article-drop loops intact.
3. Failure policy:
   - Continue non-fatal fallback behavior to preserve pipeline completion.

## Tests
1. Mode-specific enablement tests.
2. Budget-compaction behavior tests.
3. Deterministic scaffold fallback tests.

## Acceptance Criteria
1. `detailed` quality improves with acceptable latency delta.
2. OOM/truncation incidents do not regress.
3. `general` mode remains stable and fast.

## Rollback Plan
Disable optional stages by config only; no structural rollback needed.
