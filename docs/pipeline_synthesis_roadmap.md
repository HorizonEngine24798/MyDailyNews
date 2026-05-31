# Pipeline Synthesis Roadmap

## Purpose

This document proposes a concrete roadmap to improve synthesis quality in `MyDailyNews` while keeping runtime and memory bounded for local-first execution.

It covers:

- Two new optional LLM stages:
  - Evidence distillation pass (with integrated reader Q&A scaffold)
  - Delta extraction pass
- Non-LLM improvements discussed previously (items 1-6, excluding post-pass critic)
- OOM and speed constraints, including how to keep the current safety profile
- Sequential PR plan with independently verifiable steps

---

## Current Constraints and Why They Matter

The current pipeline is intentionally shallow:

```text
fetch -> prefilter -> headline scoring LLM -> article fetch -> optional enrichment -> final synthesis LLM
```

The existing code already contains meaningful OOM/runtime protections:

- `mydailynews/brief.py`
  - Prompt budget loop trims excerpt size, drops prior reports, and finally drops lower-ranked articles if token estimates exceed budget.
- `mydailynews/ai/transformers_client.py`
  - Adaptive OOM backoff lowers `max_new_tokens` and `input_token_limit` across retries.
  - CUDA cleanup before/after generation.
- `mydailynews/ai/llama_cpp_server_client.py`
  - Truncates system/user prompts to fit configured input token limits.
- `mydailynews/ai/headline_analyzer.py`
  - Dynamic batch building by token budget.
  - Cache support for JSON stage outputs.
- `config` defaults
  - Worker counts are conservative (`1`) to avoid concurrent LLM pressure on VRAM.

### OOM Reasoning for New Stages

Adding stages increases risk primarily by:

- Increasing the number of model loads/inference runs
- Increasing total prompt assembly and intermediate payload size
- Increasing worst-case generation length if schemas are unconstrained

The safest approach is:

1. Use exactly one small-model call per new stage (no per-article calls).
2. Keep stage outputs strictly structured and compact.
3. Reuse caching and token budgeting logic already present in headline scoring/final brief.
4. Fail open: if the new stage fails, continue with existing pipeline behavior.

This keeps the current guardrails intact while allowing targeted extra reasoning.

---

## Proposed Functional Additions

## 1) Evidence Distillation Pass (Optional Additional LLM Step, User-Specified)

### Goal

Convert selected articles + context into a compact, cross-source evidence packet before final prose generation.

### Includes Reader-Q&A Scaffold

Integrate Q&A in the same distillation pass to avoid an extra call:

- `reader_questions`: high-value why/how/what-next questions
- `answers`: concise evidence-grounded responses linked to article IDs

### Suggested Output Contract (JSON)

- `story_clusters`: grouped narrative threads
- `key_claims`: claims with support article IDs
- `consensus_points`
- `contested_points`
- `known_unknowns`
- `watch_signals`
- `reader_qa`: array of `{question, answer, article_ids}`
- `source_coverage`: per cluster source diversity metadata

## 2) Delta Extraction Pass (Optional Additional LLM Step, User-Specified)

### Goal

Explicitly compute changes versus prior reports so the brief emphasizes movement, not recap.

### Suggested Output Contract (JSON)

- `new`
- `escalated`
- `weakened`
- `reframed`
- `unchanged_but_important`
- `evidence_gaps`

### Dependency Behavior

- If prior reports are absent and `require_prior_reports=true`, stage is skipped with warning.
- If prior reports are absent and `require_prior_reports=false`, stage may run in "intra-run delta" mode using only current evidence.

---

## Pipeline Integration (One/Both/Neither)

## Stage Placement

Insert after enrichment and before final brief generation:

```text
... -> enrichment
    -> [optional evidence distillation]
    -> [optional delta extraction]
    -> final brief synthesis
```

## Behavior Matrix

- Neither enabled:
  - Existing behavior unchanged.
- Only evidence enabled:
  - Final brief consumes evidence packet + selected article payload.
- Only delta enabled:
  - Final brief consumes delta packet + selected article payload.
- Both enabled:
  - Final brief consumes evidence packet + delta packet + selected article payload.
  - Delta stage should preferentially use distilled packet as compact input when available.

## Failure Policy (Critical for Stability)

- Any new-stage JSON or backend error should append warnings and continue.
- Existing final brief path remains the fallback.
- Pipeline should not abort unless final brief generation itself fails (current behavior).

---

## Config Design Proposal

Add a new optional top-level section, defaulting to disabled:

```json
{
  "analysis": {
    "evidence_distillation": {
      "enabled": false,
      "model_role": "summary",
      "include_reader_qa": true,
      "max_input_tokens": 2300,
      "max_new_tokens": 700,
      "max_articles": 8,
      "max_article_chars": 700,
      "max_context_sources_per_article": 2,
      "max_story_clusters": 6,
      "max_claims_per_cluster": 4,
      "max_questions": 6,
      "cache_ttl_seconds": 604800
    },
    "delta_extraction": {
      "enabled": false,
      "model_role": "summary",
      "input_source": "evidence_or_articles",
      "require_prior_reports": false,
      "max_input_tokens": 1700,
      "max_new_tokens": 380,
      "max_prior_reports": 3,
      "cache_ttl_seconds": 604800
    }
  }
}
```

### Notes

- `model_role` should map to existing clients (`summary` or `final`) so no new backend abstraction is required.
- Defaults should keep both features off for backward compatibility.
- Config parsing must tolerate missing `analysis`.

---

## Technical Design Changes

## New/Updated Data Models

Add dataclasses in `mydailynews/models.py`:

- `EvidenceDistillationConfig`
- `DeltaExtractionConfig`
- `AnalysisConfig`
- `EvidencePacket` (or dict contract with schema validation)
- `DeltaPacket`

Add `analysis: AnalysisConfig` to `AppConfig`.

## Config Loading

Update `mydailynews/config.py`:

- Parse optional `analysis` section.
- Keep old configs valid (no new required fields).

## Prompts and Schemas

Update:

- `mydailynews/ai/prompts.py`
  - add distillation prompt templates
  - add delta extraction prompt templates
- `mydailynews/ai/schemas.py`
  - add strict schemas for both stages

## Stage Execution Module

Add `mydailynews/analysis_pipeline.py` (or similarly named module):

- Build compact stage inputs from selected articles/context/prior reports
- Token-budget fitting (same defensive style as `BriefGenerator`)
- Run `complete_json(...)`
- Parse + validate + return warnings + cache

## Brief Generation Integration

Update `mydailynews/brief.py`:

- Extend prompt payload with optional:
  - `evidence_packet`
  - `delta_packet`
- Keep legacy behavior when absent.

Update `mydailynews/brief_execution.py`:

- Run optional stages before `BriefGenerator.generate(...)`.
- Attach warnings to metadata.

## Caching

Reuse `JSONCache`:

- Cache key should include:
  - model label/backend
  - relevant config subset
  - selected article IDs + normalized short text hashes
  - prior report IDs/dates for delta
  - schema version field

---

## OOM and Speed Strategy

## Recommended Defaults (Safe Profile)

- Run new stages on `ai_summary` model.
- One call per stage, per brief.
- Conservative token ceilings:
  - distillation: `max_input_tokens <= 2300`, `max_new_tokens <= 700`
  - delta: `max_input_tokens <= 1700`, `max_new_tokens <= 380`
- Keep `runtime.max_*_workers` unchanged for LLM sections (effectively serial).

## Relaxation (Balanced Profile)

Only after instrumentation confirms headroom:

- Increase distillation `max_input_tokens` by +10% to +20%.
- Increase final brief article count or excerpt chars modestly, but only if evidence packet is used to reduce final prompt pressure.
- Do not relax all limits simultaneously.

## Additional Guardrails to Add

- Per-stage hard cap on input articles (`max_articles`).
- Per-stage hard cap on chars per article (`max_article_chars`).
- Early compacting:
  - use `article_text[:N]`
  - limit context sources/items
  - trim prior reports to configured max.
- Emit stage-level debug metrics:
  - estimated tokens
  - input chars
  - cache hit/miss
  - stage latency

---

## Non-LLM Improvements (Previously Discussed, Keep)

These should be implemented alongside/around the new stages to improve quality without extra model depth.

## 1) Deterministic Event Clustering Before Synthesis

- Cluster by normalized title tokens, URL canonicalization, and temporal proximity.
- Compute cluster-level metadata:
  - unique sources
  - latest timestamp
  - representative headline
  - supporting article IDs

Benefit:
- Reduces duplication and improves cross-source synthesis quality before any LLM step.

## 2) Source-Diversity Constraints in Selection

- Add caps such as:
  - max articles per source
  - min source count per selected set (when available)
- Prefer candidates supported by multiple outlets when scores are close.

Benefit:
- Lowers single-outlet framing bias and improves corroboration quality.

## 3) Story-State Features (Non-LLM Derivation)

Build deterministic hints per cluster:

- `is_new`
- `is_continuing`
- `freshness_delta`
- `source_agreement_score`
- `source_disagreement_signal`

Benefit:
- Gives LLM higher-signal structure with little runtime cost.

## 4) Stronger Prior-Report Delta Scaffolding

- Compute deterministic overlap between prior major headlines and current selected items.
- Pass overlap/non-overlap scaffolding to delta stage/final prompt.

Benefit:
- Makes "what changed" explicit even when delta stage is disabled.

## 5) Fixed Output Slots for Known/Unknown/Watch

- Enforce output slots in final JSON contract:
  - knowns
  - unknowns
  - watch_signals

Benefit:
- Improves analytical usefulness and uncertainty reporting without requiring extra calls.

## 6) Curated Background Packs Beyond Wikipedia

- Topic-level static background notes (local files) with trusted reference summaries.
- Keep small and source-attributed.

Benefit:
- Better context than ad hoc wiki-only lookup; zero runtime network cost for pack content.

---

## Risks and Pitfalls

## 1) Hallucinated Cross-Document Synthesis

Risk increases in multi-document settings. Mitigation:

- Require article IDs for claims in distillation output.
- Prefer short claim units over long freeform narratives.
- Keep final prompt instruction strict: do not assert unsupported facts.

## 2) Latency Creep

Two extra stages can double LLM latency if unconstrained. Mitigation:

- Small model for both stages by default.
- Cache stage outputs aggressively.
- Keep to one call per stage.

## 3) Prompt Bloat from Rich Intermediate JSON

Distilled output can become too large and hurt final stage. Mitigation:

- Cap list lengths and string sizes in schema.
- Provide optional compact mode for final prompt embedding.

## 4) Silent Degradation When Prior Reports Are Sparse

Delta stage can become generic. Mitigation:

- Explicit "insufficient prior evidence" status in delta output.
- Skip stage when configured to require prior reports.

## 5) Configuration Complexity

Too many toggles can confuse tuning. Mitigation:

- Preset profiles in docs:
  - `safe`
  - `balanced`
- Keep defaults off and conservative.

---

## Validation Plan

## Unit Tests

- Config parsing with `analysis` absent/present/partial.
- Distillation input builder trimming behavior.
- Delta stage skip behavior with no prior reports.
- Cache key stability and cache hit behavior.
- Fallback behavior when stage returns invalid JSON.

## Integration Tests

- Four mode matrix:
  - none
  - evidence only
  - delta only
  - both
- Assert final output generation succeeds in all modes.
- Assert warnings are surfaced, not fatal, for stage failures.

## Runtime/Resource Checks

- Compare `--debug` analytics before/after:
  - total runtime
  - AI request counts
  - stage token estimates
  - OOM retries occurrence

---

## Sequential PR Plan

Each PR should be independently reviewable and deployable.

## PR1: Config and Model Plumbing

Scope:

- Add `analysis` config dataclasses + loader support.
- Keep defaults disabled.

Verification:

- Existing configs run unchanged.
- New config fields parse correctly.

## PR2: Schemas + Prompts + Stage Scaffolding

Scope:

- Add distillation/delta schemas and prompt templates.
- Add analysis pipeline module with no orchestrator wiring yet.

Verification:

- Unit tests for prompt build, schema parse, budget trimming.

## PR3: Evidence Distillation Integration (Optional)

Scope:

- Wire distillation stage into `run_brief`.
- Add warnings/fallback behavior.
- Add caching.

Verification:

- Mode: evidence-only works.
- Disabled mode unchanged.

## PR4: Reader-Q&A in Distillation Output

Scope:

- Extend distillation schema/prompt/output with Q&A scaffold.
- Add output rendering support if needed.

Verification:

- Distillation output includes bounded `reader_qa`.

## PR5: Delta Extraction Integration (Optional)

Scope:

- Wire delta stage with `require_prior_reports` behavior.
- Add fallback behavior and caching.

Verification:

- Mode: delta-only works with/without prior reports per config.

## PR6: Final Prompt Consumption Improvements

Scope:

- Update `BriefGenerator` prompt payload to consume evidence/delta packets cleanly.
- Add compact embedding mode to avoid prompt bloat.

Verification:

- Both-enabled mode works.
- Prompt token estimates remain within configured budgets.

## PR7: Non-LLM Event Clustering and Source Diversity

Scope:

- Deterministic clustering and source-diversity aware selection.
- Add cluster metadata into candidate/context payloads.

Verification:

- Candidate diversity metrics improve in debug outputs.
- No extra LLM calls introduced.

## PR8: Prior-Report Scaffolding + Known/Unknown/Watch Contract

Scope:

- Deterministic delta hints when stage disabled.
- Enforce final output slots for uncertainty and watch signals.

Verification:

- Final JSON always includes uncertainty/watch fields.
- Regression tests pass for legacy and new modes.

## PR9: Background Packs and Documentation Finalization

Scope:

- Add local background pack mechanism and sample packs.
- Document tuning playbook (`safe` vs `balanced` profiles).

Verification:

- Background pack retrieval works without network dependency.

---

## External Rationale References

- SummN (multi-stage summarization for long input):
  - https://aclanthology.org/2022.acl-long.112/
- Multi-document hallucination findings (NAACL Findings 2025):
  - https://aclanthology.org/2025.findings-naacl.293/
- AP guidance on AI-assisted summaries with human editing (May 8, 2024):
  - https://www.ap.org/the-definitive-source/behind-the-news/updates-to-generative-ai-standards/
- AP verification emphasis in AI/data workflows (March 31, 2026):
  - https://www.ap.org/insights/ai-and-data-journalism-why-verification-matters-more-than-ever/
- Google on diversity/eligibility in news surfaces:
  - https://support.google.com/news/publisher-center/answer/9607025?hl=en
