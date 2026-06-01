# MyDailyNews Quality Upgrade Plan (2026-05-31)

## Purpose
This plan upgrades output quality from "good summarizer" to "useful personal briefing/radar" while keeping runtime and memory pressure bounded for local-first execution.

This is a delivery plan, not a brainstorming note. It is designed to be executed as a sequence of small, reviewable PRs.

## Product Direction
Target product behavior:

1. Briefing first, not newspaper replay.
2. Select fewer stories with clearer "why this matters now".
3. Prefer pattern-level synthesis over independent item summaries.
4. Keep uncertainty explicit (`knowns`, `unknowns`, `watch_signals`).
5. Stay stable on constrained local hardware.

## Current State Snapshot
What is already strong in the current codebase:

1. Deterministic pre-LLM filtering, dedupe, clustering, and diversity caps are already implemented.
2. Shared headline scoring pass avoids duplicate LLM work across general/detailed briefs.
3. Prompt budget trimming already exists across headline analysis, evidence, delta, and final brief generation.
4. Output contract already includes uncertainty/watch slots and reference lists.
5. Optional evidence/distillation and delta extraction stages already exist with deterministic fallback logic.

Primary limitation right now:

1. Headline scoring contract is still mostly single-dimensional (`id` + `score`), so editorial nuance is compressed too early.
2. User profile model is still shallow for decision-context personalization.
3. Selection rationale is mostly implicit (not explicit "selected because"/"skipped because").
4. Optional analysis stages are disabled in active config by default, reducing synthesis depth.

## Constraints We Must Keep
Hard constraints for this plan:

1. No uncontrolled increase in LLM call count.
2. No broad increase in model context or article payload sizes without measured gain.
3. Preserve current budget-trimming behavior and cache behavior.
4. Keep current worker defaults conservative (`1`) unless profiling shows safe headroom.

## Quality Targets
Track quality improvements with concrete metrics:

1. `top5_story_value_precision`: fraction of top-5 stories judged "should not miss".
2. `novelty_ratio`: selected stories that are meaningfully new vs repeated.
3. `duplication_leak_rate`: selected items that represent the same story cluster.
4. `decision_context_alignment`: selected stories aligned with user wants/avoid lists.
5. `brief_utility_score`: rater score for "actionable and strategic value".
6. `latency_p95_sec`: end-to-end latency p95.
7. `max_prompt_tokens_by_stage`: to guard regressions in token pressure.
8. `oom_or_truncation_incidents`: failures tied to context pressure.

Recommended acceptance thresholds for the full PR train:

1. +20% improvement in `top5_story_value_precision` vs baseline.
2. +25% improvement in `decision_context_alignment`.
3. `latency_p95_sec` increase <= 15%.
4. No increase in OOM incidents.

## PR Train Overview
Execution order and intent:

1. PR01: Baseline evaluation harness and release guardrails.
2. PR02: Headline scoring prompt rubric refresh (no schema change yet).
3. PR03: Multi-factor headline decision schema and parsing.
4. PR04: Composite ranking and explicit selection/skip reasons.
5. PR05: User memory v2 (decision-context profile).
6. PR06: Final brief prompt/output upgrade for briefing voice and angle discipline.
7. PR07: Runtime-safe rollout of optional evidence/delta stages.
8. PR08: Prompt regression pack, anti-regression checks, and rollout checklist.

PR docs:

1. `docs/prs/PR01_baseline_eval_and_guardrails.md`
2. `docs/prs/PR02_headline_prompt_rubric_refresh.md`
3. `docs/prs/PR03_multifactor_headline_decision_schema.md`
4. `docs/prs/PR04_composite_ranking_and_selection_reasons.md`
5. `docs/prs/PR05_user_memory_v2_decision_context.md`
6. `docs/prs/PR06_brief_writer_structured_briefing_voice.md`
7. `docs/prs/PR07_runtime_safe_analysis_stage_rollout.md`
8. `docs/prs/PR08_prompt_regression_and_release_gates.md`

## Cross-Cutting Implementation Rules
These rules apply in every PR:

1. Backward compatibility:
   - Keep default behavior stable when new config fields are absent.
   - Version cache fingerprints when prompt/schema contracts change.

2. Runtime safety:
   - Preserve existing prompt budget loops.
   - Add no new mandatory LLM stages.
   - Prefer improving payload quality over increasing payload size.

3. Deterministic before generative:
   - Perform cheap deterministic filtering/ranking before LLM calls.
   - Use LLM for judgment where deterministic signals are weak.

4. Observability:
   - Emit metrics for each new scoring dimension and selection reason.
   - Log any dropped-context behavior and reason code.

5. Testing discipline:
   - Add parsing tests for each schema addition.
   - Add regression tests for prompt building and token-budget behavior.
   - Add "no output-shape regression" tests for markdown/json render paths.

## Key Code Areas To Keep In Mind
When implementing PRs, these modules are tightly coupled:

1. `mydailynews/ai/prompts.py`
   - Prompt contract source of truth.
2. `mydailynews/ai/schemas.py`
   - JSON schema contract for model outputs.
3. `mydailynews/ai/headline_analyzer.py`
   - Scoring payload, parsing, batching, caching keys.
4. `mydailynews/models.py`
   - Data model surface for decisions and user memory.
5. `mydailynews/headline_selection.py`
   - Deterministic prefilter, scoring combination, selection caps.
6. `mydailynews/analysis_pipeline.py`
   - Optional evidence and delta stages, compacting, budget controls.
7. `mydailynews/brief.py`
   - Final brief prompt assembly and fallback slot normalization.
8. `mydailynews/output.py`
   - Markdown rendering contract and user-visible structure.
9. `mydailynews/brief_execution.py`
   - Stage orchestration, deterministic delta scaffold, failure policy.
10. `config.example.json` and `config.json`
    - Rollout defaults and new profile fields.

## Focus Areas and Common Failure Modes
What to focus on:

1. Improve decision quality before final writing style.
2. Preserve deterministic diversity controls while adding editorial nuance.
3. Keep output concise and strategic, not verbose.

What to be careful about:

1. Prompt bloat:
   - Richer schema can increase output size and retry risk.
2. Inconsistent scoring:
   - If multi-factor fields are optional or weakly constrained, ranking quality may become unstable.
3. Selection regressions:
   - Composite scoring can unintentionally over-penalize niche but important stories.
4. Profile overfit:
   - Excess personalization can reduce breadth too aggressively.
5. Hidden latency:
   - More detailed headline scoring outputs can increase decode time even without extra calls.

## Runtime and Memory Strategy
Preferred strategy:

1. First increase signal quality per call, not number of calls.
2. Keep shared headline pass and cache usage central.
3. Use compact or minimal analysis payload modes by default in final brief.
4. Enable optional analysis by profile:
   - `general`: optional disabled by default.
   - `detailed`: optional enabled with conservative caps.

## External Patterns To Borrow (Prompt-Focused)
Use these public patterns as design references:

1. Horizon:
   - Uses explicit scoring bands, reasoned scoring output, and separate enrichment prompts.
   - Link: `https://raw.githubusercontent.com/Thysrael/Horizon/main/src/ai/prompts.py`
2. ClawFeed:
   - Separates curation policy from digest writing format via editable templates.
   - Links:
     - `https://raw.githubusercontent.com/kevinho/clawfeed/main/templates/curation-rules.md`
     - `https://raw.githubusercontent.com/kevinho/clawfeed/main/templates/digest-prompt.md`
3. Meridian:
   - Emphasizes clustering and multi-stage analysis before final synthesis.
   - Link: `https://raw.githubusercontent.com/iliane5/meridian/main/README.md`

## Delivery and Review Workflow
For each PR in this train:

1. Ship schema/prompt changes with tests in the same PR.
2. Include a before/after quality sample set (small fixed corpus).
3. Report latency, max prompt tokens, and warning counts.
4. Keep rollout behind config flags when behavior is substantial.

## Definition of Done For This Plan
Plan execution is complete when:

1. All PR docs in `docs/prs/` are implemented and merged.
2. Quality metrics meet agreed targets.
3. Runtime and memory constraints remain within guardrails.
4. Output demonstrates clearer "why this matters" and better novelty filtering.
