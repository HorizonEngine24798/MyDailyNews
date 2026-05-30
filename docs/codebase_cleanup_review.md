# Codebase Cleanup Review

This document captures the main problems currently visible in the MyDailyNews codebase after the shallow-pipeline refactor and debug analytics work.

The goal is not just to list bugs. The goal is to identify the places where the code is likely to keep slowing us down: reliability mismatches, refactor residue, repo hygiene issues, and structural complexity that makes future tuning harder than it should be.


## Summary

The biggest remaining issues are:

1. Backend behavior is inconsistent, especially around prompt budgeting and structured output.
2. Invalid-JSON diagnostics and analytics do not share one coherent output path.
3. The `transformers` path still depends on prompt obedience for JSON.
4. Local model lifecycle management is still expensive.
5. The repo is being dirtied by runtime cache files and temporary artifacts.
6. There is still stale code and stale documentation from older pipeline designs.
7. The orchestrator is carrying too many responsibilities in one file.


## Status Update (May 30, 2026)

Since this review was first written, parts of it have been addressed:

- PR1 repo hygiene cleanup removed tracked runtime artifacts and tightened ignores.
- PR2 unified AI invalid-JSON artifacts under configurable output root.
- PR3 added backend budget enforcement on `llama_cpp_server` input prompts (input limit is now actively fitted instead of ignored), with regression tests.
- PR4 removed stale compatibility/dead surface:
  - removed legacy orchestrator `self.ai_client` alias
  - removed dead debug event categories
  - removed unused `scrapers/base.py` protocol
  - removed unused orchestrator helper `_newest_first(...)`
  - removed legacy single-`ai` config compatibility in favor of explicit `ai_summary` + `ai_final`
- PR5 started orchestrator decomposition:
  - extracted brief execution flow to `mydailynews/brief_execution.py`
  - extracted headline selection heuristics to `mydailynews/headline_selection.py`
  - extracted snapshot construction/windowing to `mydailynews/snapshot_helpers.py`
  - extracted shared snapshot scoring coordination to `mydailynews/shared_headline_scoring.py`
  - extracted article fetch/enrichment metric helpers to `mydailynews/article_pipeline.py`
- PR6 updated `docs/llama_cpp_port.md` to the current shallow pipeline and current backend seam.

Remaining priority from this review: broader model lifecycle strategy decisions.


## 1. Backend Budget Enforcement Is Inconsistent

### Problem

The pipeline tries to reason carefully about prompt/input budgets, but those guarantees are not actually enforced consistently across backends.

On the `transformers` path, `input_token_limit` is honored and passed all the way into tokenization and generation:

- `mydailynews/ai/transformers_client.py`
  - `complete_json(...)` computes `target_input_limit`
  - `_generate_with_oom_backoff(...)` carries that limit forward
  - `_generate(...)` uses `max_length=input_token_limit`

On the `llama_cpp_server` path, the same parameter is accepted and then ignored:

- `mydailynews/ai/llama_cpp_server_client.py`
  - `complete_json(...)`
  - `_ = input_token_limit`

### Why It Matters

This means the exact same pipeline configuration can behave differently depending on backend:

- prompt fits on one backend and silently overruns on another
- local prompt-budget tuning becomes misleading
- OOM and truncation debugging becomes harder because the caller believes a budget exists when it may not

### Cleanup Direction

We should make prompt-budget behavior explicit and consistent:

1. Either enforce input budgeting on all backends.
2. Or remove the abstraction claim and rename the parameter so callers know which backends truly support it.

The first option is much better.


## 2. Diagnostics Are Split Across Two Output Systems

### Problem

The new analytics artifact respects `config.output_dir`, but invalid-JSON artifacts do not.

Analytics:

- `mydailynews/debug.py`
  - `_DebugAnalytics.write_artifact(...)`
  - writes to `<output_dir>/diagnostics/analytics/...`

Invalid-JSON artifacts:

- `mydailynews/ai/base.py`
  - `_artifact_directory(...)`
  - always writes to `Path("output") / "diagnostics" / kind`

### Why It Matters

This creates an annoying debugging split:

- one class of diagnostics goes to configured output
- another class always goes to hardcoded `output/`

That is exactly the kind of inconsistency that makes incident debugging slower than it needs to be.

### Cleanup Direction

Unify all diagnostics under one configurable root:

1. Pass a diagnostics/output root into the AI artifact helpers.
2. Remove hardcoded `Path("output")` from `mydailynews/ai/base.py`.
3. Keep all debug artifacts under one tree so a failed run can be inspected from one place.


## 3. Structured Output Reliability Still Depends On Prompt Obedience

### Problem

The `transformers` backend still uses the weakest structured-output strategy:

- prompt for JSON
- hope the model follows instructions
- retry once or twice
- parse the result

This is stated directly in:

- `mydailynews/ai/transformers_client.py`
  - `# Transformers path currently relies on prompt-instructed JSON + retries.`

Even though the app now has smaller prompts and better diagnostics, this is still fundamentally fragile.

### Why It Matters

There are at least three separate failure modes:

1. Invalid JSON
   - malformed braces, stray text, markdown fences, truncation
2. Valid JSON but incomplete decisions
   - model emits JSON but omits one or more headline IDs
3. Valid structure but semantically wrong content
   - duplicate IDs, missing required business logic, empty sections

Prompting alone does not solve all three reliably.

### Cleanup Direction

Preferred options, in order:

1. Use a backend that supports schema-constrained or grammar-constrained generation.
2. Keep JSON schemas small and output contracts minimal.
3. Validate semantically after parsing, not just syntactically.
4. Use targeted retry/fallback strategies when output is incomplete.


## 4. Model Lifecycle Churn Is Still Expensive

### Problem

The local `transformers` path still does a lot of expensive model memory management:

- `mydailynews/ai/transformers_client.py`
  - `_cleanup_cuda(reason="pre_generate")`
  - `_cleanup_cuda(reason="post_generate")`
  - `_cleanup_cuda(reason="unload")`

The orchestrator also unloads clients during the brief flow:

- `mydailynews/orchestrator.py`
  - unload summary client before final synthesis
  - unload final client after final synthesis

### Why It Matters

This may reduce OOM risk, but it is also a real latency cost:

- repeated GC / CUDA cleanup overhead
- reduced locality
- extra model warmup pressure across stages
- harder to reason about throughput because part of the runtime is orchestration churn, not generation

### Cleanup Direction

We should decide explicitly what we want:

1. A persistent local inference service model.
2. Or an in-process model with aggressive unloading for survival.

Right now we are somewhere in between, which means we pay costs from both worlds.


## 5. Runtime Data Is Dirtying The Repo

### Problem

Normal runs are modifying files under the repository cache path:

- `config.json`
  - `cache.dir = ".cache/mydailynews"`

The caches then write under that path:

- `mydailynews/cache.py`
  - `HTTPCache(root_dir=...)`
  - `JSONCache(root_dir=...)`

At the moment, `git status` shows many modified files under `.cache/mydailynews/http/shared/...`.

There are also temporary local debug files in the repo root such as `.codex_tmp_*`.

### Why It Matters

This creates unnecessary development friction:

- noisy git status
- harder reviews
- risk of committing runtime artifacts by accident
- more confusion about what is source code versus generated state

### Cleanup Direction

We should tighten repo hygiene:

1. Ensure `.cache/` is ignored if it is not meant to be versioned.
2. Ensure `output/` is ignored if it is runtime output only.
3. Ignore `.codex_tmp_*` and other temp artifacts.
4. Consider moving runtime caches outside the repo entirely for local developer machines.


## 6. Refactor Residue Is Still Present

### Problem

There are several signs that the codebase still carries compatibility and conceptual residue from the older multi-stage design.

Examples:

- `mydailynews/orchestrator.py`
  - `self.ai_client = self.summary_ai_client`
  - marked as legacy compatibility alias

- `mydailynews/debug.py`
  - still includes `brief.article.ai` event filtering even though the per-article brief stage was removed from the hot path

- `docs/llama_cpp_port.md`
  - still describes the older, deeper pipeline:
    - headline narrative triage
    - article-level narrative revision

### Why It Matters

Residue like this makes the codebase harder to trust:

- some names describe the old system
- some docs describe the old system
- some compatibility fields imply dependencies that may no longer exist

That slows down future changes because every edit requires rediscovering which abstractions are real and which are leftovers.

### Cleanup Direction

We should do a cleanup pass with a bias toward deletion:

1. Remove compatibility aliases that are no longer needed.
2. Remove dead debug event categories.
3. Update or replace outdated docs that describe the old pipeline.
4. Rename remaining objects if their current role is narrower than their historical name.


## 7. The Orchestrator Is Too Large And Carries Too Many Concerns

### Problem

`mydailynews/orchestrator.py` is currently doing all of the following:

- pipeline setup
- shared snapshot construction
- per-brief orchestration
- headline limiting
- score reuse
- article fetch coordination
- enrichment coordination
- output writing
- warning aggregation
- metrics and analytics instrumentation

### Why It Matters

This makes the file a bottleneck for every change:

- performance tuning touches it
- reliability tuning touches it
- debug instrumentation touches it
- backend coordination touches it
- output coordination touches it

Large orchestrators are common in early refactors, but if they stay large they become the place where every shortcut accumulates.

### Cleanup Direction

Split by responsibility:

1. pipeline runner / top-level orchestration
2. snapshot builder
3. selection pipeline
4. article retrieval coordinator
5. analytics recorder / run statistics collector

The current code is workable, but it is past the point where further tuning will stay pleasant without decomposition.


## 8. Default Runtime Parallelism Is Extremely Conservative

### Problem

The default runtime config still uses:

- `max_http_workers = 1`
- `max_article_workers = 1`
- `max_enrichment_workers = 1`

in `config.json`.

### Why It Matters

This is safe, but it means:

- RSS fetch is serialized
- article retrieval is serialized
- enrichment is serialized

That is fine for cautious debugging, but it undercuts the performance benefits of the shallower LLM pipeline.

### Cleanup Direction

We should choose sane defaults separately for:

1. LLM generation concurrency
2. network and fetch concurrency

These do not need to be equally conservative.


## 9. Documentation Drift Still Exists

### Problem

The main README is now closer to reality, but not all docs are aligned.

In particular:

- `docs/llama_cpp_port.md` still describes the older deeper pipeline
- some wording across the repo still reflects “summary” and “narrative” assumptions from earlier designs

### Why It Matters

Outdated docs are not harmless in a refactor-heavy project:

- they cause incorrect assumptions
- they slow onboarding
- they encourage preserving old abstractions because they still look documented

### Cleanup Direction

We should treat docs as part of the refactor, not as an afterthought:

1. update `docs/llama_cpp_port.md`
2. explicitly describe the current shallow pipeline
3. document which backend is preferred for scoring and why
4. document what the debug analytics artifact contains


## Recommended Cleanup Order

If we want to reduce risk while improving velocity, this is the order I would use:

1. Unify diagnostics and output paths.
2. Fix backend budget consistency.
3. Decide the long-term scorer backend strategy for structured output.
4. Clean repo hygiene: `.cache`, `output`, temp files.
5. Remove refactor residue and stale docs.
6. Split orchestration responsibilities once behavior is stable.


## Recommended Principle

The main theme across these issues is consistency.

Right now the codebase is not failing because it lacks ideas. It is failing because similar concepts are implemented differently in different places:

- one backend enforces prompt budgets and another does not
- one diagnostics system respects config and another does not
- one doc reflects the new architecture and another reflects the old one

The next cleanup pass should therefore optimize for consistency first, then elegance.
