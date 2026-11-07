# Story change analysis

Updated: 2026-11-07

## Production design

Delta analysis operates on same-day story groups:

1. **Group & Partition**: Group today's articles via `StoryGroupingPlanner`, then partition over-broad topic buckets in [`build_story_memory_context()`](../mydailynews/memory/context.py) (`_partition_story_group`):
   - Vetoes merging articles with strong (`score >= 0.42`) disjoint prior baselines in `StoryStore`.
   - Requires positive same-story cohesion (shared `story_key`, shared prior baseline, or pairwise hybrid retrieval score above threshold).
2. **Hybrid Historical Retrieval**: Retrieve up to 3 prior story candidates using lexical overlap + `intfloat/multilingual-e5-small` semantic embeddings (`Recall@1 = 0.9333`, `Recall@3 = 0.9867` on the 75-pair blind benchmark).
3. **Fact Operations**: Ask the analysis model for one operation per current fact against bounded prior facts: `add`, `repeat`, `replace`, `resolve`, or `uncertain`. Stories with no prior candidates get deterministic `add` operations without an LLM call.
4. **Structural Validation**: [`validate_fact_operations()`](../mydailynews/analysis/claim_delta.py) rejects responses with invented IDs, ambiguous ownership, multi-story citations, conflicts, or uncovered current facts.
5. **Story-Key Resolution**:
   - Reuses a prior `story_key` when validated operations cite facts from exactly one prior story.
   - When all validated operations are `add` with empty `prior_fact_ids`, [`_high_confidence_baseline_key()`](../mydailynews/analysis/story_delta.py) links to an unambiguous high-confidence prior baseline (`candidate_score >= 0.42`, no `numeric_conflict`) and sets `change_type = "incremental"`.
6. **Card-Budgeted Editor Selection**:
   - [`_bounded_editor_prompt()`](../mydailynews/analysis/story_delta.py) budgets `1` per story card against `card_limit = max_articles` (preventing multi-article clusters from starving later cards).
   - Verified `change_type == "unchanged"` stories (`safe_to_apply == True`) are deterministically clamped to `disposition = "omit"` (`materiality = 0`) in [`_validated_editor_decisions()`](../mydailynews/analysis/story_delta.py).
7. **Deferred Append-Only Writeback**: Persist selected stories to `StoryStore` after all briefs finish. `replace` and `resolve` record relationships without deactivating prior facts.

## Failure policy

Uncertain, malformed, missing, over-budget, or failed model output fails open so affected stories remain eligible for coverage. Consent-wall interstitials (`"Before you continue to Google..."`) are hard-stopped at fetch, extraction, story grouping, `StoryStore`, and E5 embedding layers.
