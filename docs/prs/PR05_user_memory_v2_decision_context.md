# PR05: User Memory V2 (Decision Context Profile)

## Objective
Upgrade user personalization from shallow topic/source preferences to decision-context modeling.

The goal is better "for this user, why now" relevance without adding new LLM stages.

## Why This PR
Current `UserMemory` is useful but limited. Better personalization requires explicit profile dimensions used by both prompts and deterministic ranking.

## Scope
In scope:

1. Extend user profile schema with richer fields.
2. Ensure prompt serialization includes new fields concisely.
3. Add deterministic helper signals derived from profile fields.

Out of scope:

1. Fully learned user-feedback loops.
2. UI profile editor (if not already present).

## Planned File Changes
1. `mydailynews/models.py`
   - Extend `UserMemory` with new fields.
   - Update `to_prompt()` serialization.
2. `config.example.json`
   - Add documented example fields.
3. `mydailynews/headline_selection.py`
   - Optional deterministic boosts/penalties from profile dimensions.
4. `tests/test_pipeline_basics.py`
   - Profile serialization and ranking interaction tests.

## Proposed Profile Additions
Recommended fields:

1. `role`
2. `geography_focus`
3. `time_horizon` (breaking, tactical, strategic)
4. `beats` (weighted topical subdomains)
5. `wants` (explicit signal types)
6. `avoid` (explicit low-value classes)
7. `portfolio_or_stake_notes` (optional concise text)
8. `preferred_depth` (brief, analytical, deep)

## Implementation Plan
1. Extend dataclass with defaults so old configs still work.
2. Keep prompt serialization concise and structured.
3. Add deterministic hooks for selected fields:
   - Geography match boost.
   - Explicit avoid-list penalties.
4. Update cache fingerprints where prompt text changes materially.
5. Add config migration notes in docs.

## Runtime and Memory Impact
Expected:

1. No additional LLM calls.
2. Slightly larger prompt prefix (`memory` block).
3. Minor deterministic compute overhead.

## Things To Be Careful About
1. Prompt bloat:
   - Limit profile verbosity and cap list lengths.
2. Over-personalization:
   - Avoid narrowing so much that user misses truly major events.
3. Backward compatibility:
   - Missing new fields must not break config loading.

## Tests
1. Config loading tests for old and new profile shapes.
2. Prompt serialization snapshot tests.
3. Ranking behavior tests for geography/avoid/wants hooks.

## Acceptance Criteria
1. Profile v2 fields are optional and backward-compatible.
2. Decision-context alignment metric improves on benchmark.
3. No material latency regression.

## Rollback Plan
Keep new fields accepted but ignore them in ranking/prompting behind feature flags.
