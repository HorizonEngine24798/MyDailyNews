# PR06: Final Brief Writer Upgrade (Structured Briefing Voice)

## Objective
Make final outputs read like a briefing engine rather than a generic summary by enforcing stronger structure and angle discipline.

## Why This PR
Selection quality alone is not enough. Writing format and synthesis framing determine perceived value.

## Scope
In scope:

1. Upgrade final brief prompt instructions for sharper briefing voice.
2. Add explicit per-topic or per-section framing elements:
   - Why it matters.
   - What changed.
   - Who is affected.
   - What to watch.
3. Keep JSON contract manageable and compact.

Out of scope:

1. Additional LLM stages.
2. Long-form narrative generation.

## Planned File Changes
1. `mydailynews/ai/prompts.py`
   - Update `BRIEF_SYSTEM` and `BRIEF_USER`.
2. `mydailynews/ai/schemas.py`
   - Add minimal required fields only if needed for structure.
3. `mydailynews/brief.py`
   - Normalize missing new slots gracefully.
4. `mydailynews/output.py`
   - Render new sections cleanly in markdown.
5. `tests/test_pipeline_basics.py`
   - Prompt/output shape tests.

## Proposed Output Shape Evolution
Keep top-level concise while adding structure:

1. `lead`
2. `knowns`
3. `unknowns`
4. `watch_signals`
5. `topic_reports` entries with:
   - `topic`
   - `why_it_matters`
   - `what_changed`
   - `who_is_affected`
   - `narrative_changes`
   - `what_to_watch`

## Implementation Plan
1. Tighten instruction language:
   - Reject generic phrasing.
   - Demand significance framing tied to supplied evidence.
2. Keep brevity constraints explicit:
   - Small character/line guidance for each field.
3. Preserve existing fallback slot logic:
   - Keep robust `_ensure_signal_slots` behavior.
4. Update markdown renderer with clean headings.
5. Validate with fixed sample outputs.

## Runtime and Memory Impact
Expected:

1. No extra calls.
2. Slightly larger output JSON.
3. Minimal runtime increase.

## Things To Be Careful About
1. Overly long outputs:
   - Tight constraints in prompt and schema are required.
2. Hallucination risk:
   - Maintain strict "only supplied evidence" wording.
3. Render compatibility:
   - Ensure markdown still renders cleanly when new fields are absent.

## Tests
1. Output shape tests for new/legacy paths.
2. Markdown rendering tests for new fields.
3. Regression tests for knowns/unknowns/watch fallback behavior.

## Acceptance Criteria
1. Human raters judge output as more actionable and less generic.
2. Output size remains bounded.
3. Existing consumers of JSON/markdown do not break.

## Rollback Plan
Keep renderer tolerant; revert prompt/schema to previous shape if quality regresses.
