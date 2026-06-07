# PR02: Headline Prompt Rubric Refresh

## Objective
Improve headline scoring quality using prompt engineering only, while keeping the existing JSON schema unchanged.

Current schema (`id`, `score`) stays intact in this PR for low-risk rollout.

## Why This PR
This is the cheapest quality lift:

1. No parser/model changes.
2. No new LLM calls.
3. Directly addresses mediocre relevance by clarifying scoring rubric and penalties.

## Scope
In scope:

1. Rewrite `HEADLINE_ANALYSIS_SYSTEM` and `HEADLINE_ANALYSIS_USER` instructions for stronger editorial judgment.
2. Add explicit examples of low-value noise vs high-value "must know" items.
3. Clarify handling of "topic match but low impact" cases.

Out of scope:

1. Schema expansion.
2. Selection logic changes.
3. Final brief format changes.

## Planned File Changes
1. `mydailynews/ai/prompts.py`
   - Update headline analysis prompt pair.
2. `tests/test_pipeline_basics.py`
   - Update/add assertions for prompt payload and expected parse behavior.
3. Optional docs note:
   - `readme` prompt behavior summary update.

## Implementation Plan
1. Add clear scoring dimensions in prompt text:
   - Personal relevance.
   - Impact.
   - Novelty.
   - Actionability.
   - Urgency.

2. Add explicit penalties:
   - Routine market noise without user-specific stake.
   - Minor incremental updates.
   - Repetitive rewrites of same event.

3. Add "regret test" framing:
   - "Would this reader regret missing this today?"

4. Keep response contract unchanged:
   - Continue returning only `decisions: [{id, score}]`.

5. Add prompt compactness check:
   - Keep instruction size modest to avoid avoidable token growth.

## Runtime and Memory Impact
Expected:

1. No additional calls.
2. Slight prompt token increase only.
3. Negligible runtime impact.

## Things To Be Careful About
1. Prompt over-constraint:
   - Too many hard rules can produce flat scoring.
2. JSON reliability:
   - Preserve strict JSON-only instruction clarity.
3. Batch behavior:
   - Ensure model still outputs all IDs in batched scoring.

## Tests
1. Prompt-build unit tests (contains new rubric anchors).
2. Existing schema parse tests still pass unchanged.
3. Compare PR01 baseline vs PR02 on fixed eval set.

## Acceptance Criteria
1. `top5_story_value_precision` improves meaningfully vs baseline.
2. No increase in invalid JSON or missing-decision rates.
3. No meaningful latency regression.

## Rollback Plan
Revert prompt text only. No schema or data migration required.
