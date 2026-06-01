# PR03: Multi-Factor Headline Decision Schema

## Objective
Expand headline decision output from one scalar score to structured editorial dimensions.

This PR introduces richer decision payloads so downstream ranking can separate topic match from importance.

## Why This PR
Single-score scoring collapses too many judgments and produces mushy rankings. This PR gives deterministic ranking logic richer inputs.

## Scope
In scope:

1. Expand headline decision schema with additional fields.
2. Parse/store those fields safely with defaults.
3. Preserve backward compatibility for cached and partial outputs.

Out of scope:

1. New LLM stages.
2. Major final brief structure changes.

## Planned File Changes
1. `mydailynews/ai/schemas.py`
   - Extend `HEADLINE_ANALYSIS_JSON_SCHEMA`.
2. `mydailynews/models.py`
   - Extend `HeadlineDecision` dataclass fields.
3. `mydailynews/ai/headline_analyzer.py`
   - Parse new fields.
   - Clamp ranges.
   - Handle absent fields robustly.
   - Bump cache fingerprint version.
4. `tests/test_pipeline_basics.py`
   - Add parser and fallback tests.

## Proposed Schema Additions
Per decision item:

1. `personal_relevance` (0-10)
2. `impact` (0-10)
3. `novelty` (0-10)
4. `urgency` (0-10)
5. `actionability` (0-10)
6. `confidence` (0-10)
7. `reason` (short string)
8. `skip_reason` (nullable string)
9. `angle_type` (enum-ish string; e.g. policy_change, market_shift, risk_signal)

## Implementation Plan
1. Extend schema and keep `id` and `score` required.
2. Parse optional fields defensively:
   - If absent, default to neutral values.
3. Clamp numeric ranges to 0-10.
4. Keep compatibility path:
   - Old `id` + `score` payloads still accepted.
5. Add analyzer metrics for new dimensions.
6. Version cache key to avoid stale parse assumptions.

## Runtime and Memory Impact
Expected:

1. No new LLM calls.
2. Slightly larger response JSON per batch.
3. Small decode/parse overhead.

## Things To Be Careful About
1. Field sparsity:
   - Some local models may omit optional fields.
2. JSON truncation risk:
   - Slightly larger output requires checking batch `max_new_tokens`.
3. Cache invalidation:
   - Fingerprint version bump is required.

## Tests
1. Schema conformance tests for full and minimal payloads.
2. Parsing fallback tests when only `id` and `score` are returned.
3. Cache-key version tests.

## Acceptance Criteria
1. Analyzer safely handles both old and new response shapes.
2. Invalid JSON/missing-decision rates do not regress materially.
3. New fields available to downstream ranking without breaking existing flow.

## Rollback Plan
Disable use of new fields in ranking and preserve parsing fallback to `score` only.
