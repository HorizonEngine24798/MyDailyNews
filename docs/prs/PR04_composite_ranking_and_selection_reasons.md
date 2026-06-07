# PR04: Composite Ranking and Selection Reasons

## Objective
Use multi-factor decision signals to rank/select stories more editorially, and expose explicit reasons for selected and skipped candidates.

## Why This PR
After PR03, we have richer signals. This PR turns those signals into consistent deterministic selection behavior and improves debuggability.

## Scope
In scope:

1. Add composite ranking function using multi-factor fields.
2. Preserve existing diversity constraints and hard caps.
3. Record concise reason codes for selection/skip outcomes.

Out of scope:

1. User profile schema expansion (PR05).
2. Final brief writing style changes (PR06).

## Planned File Changes
1. `mydailynews/headline_selection.py`
   - Composite score function.
   - Selection reason generation.
2. `mydailynews/models.py`
   - Optional metadata fields for reason codes.
3. `mydailynews/brief_execution.py`
   - Include selection rationale in stage artifacts/diagnostics.
4. `tests/test_pipeline_basics.py`
   - Ranking and reason-code tests.

## Composite Ranking Design
Recommended initial weighted formula:

1. `0.30 * personal_relevance`
2. `0.20 * impact`
3. `0.18 * novelty`
4. `0.15 * actionability`
5. `0.10 * urgency`
6. `0.07 * confidence`

Then apply deterministic adjustments:

1. Multi-source cluster bonus.
2. Source avoid/prefer penalties/bonuses.
3. Existing dedupe/diversity caps as hard constraints.

## Selection/Skip Reason Codes
Add compact reason strings, for example:

1. `selected_high_composite`
2. `selected_cluster_diversity`
3. `skipped_below_cutoff`
4. `skipped_source_cap`
5. `skipped_cluster_cap`
6. `skipped_duplicate`
7. `skipped_low_novelty`

These reason codes should appear in debug artifacts and optionally in JSON output metadata.

## Implementation Plan
1. Introduce composite scorer in `headline_selection.py`.
2. Keep old path behind a config switch for safe rollout.
3. Apply selection reason annotation in `select_articles`.
4. Emit aggregate counters by reason code for diagnostics.
5. Compare old vs new ranking on fixed eval corpus.

## Runtime and Memory Impact
Expected:

1. No additional LLM calls.
2. Negligible compute overhead (simple arithmetic).
3. Slightly larger diagnostic payloads only.

## Things To Be Careful About
1. Weight instability:
   - Overweighting novelty can suppress important ongoing stories.
2. Double-penalties:
   - Avoid penalizing same trait in both model and deterministic logic excessively.
3. Backward compatibility:
   - If multi-factor fields are missing, fallback to scalar score behavior.

## Tests
1. Deterministic ranking order tests with fixed fixtures.
2. Reason code assignment tests for each major skip path.
3. Regression test for source/cluster cap behavior.

## Acceptance Criteria
1. Selection rationale available in diagnostics.
2. Quality metrics improve on benchmark with no major latency change.
3. No regressions in diversity constraints.

## Rollback Plan
Disable composite mode by config and revert to scalar-score ranking path.
