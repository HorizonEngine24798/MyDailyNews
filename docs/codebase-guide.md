<!-- generated-by: gsd-doc-writer -->
# Codebase guide

This is the starting point for reading and changing MyDailyNews. It connects the
product-level architecture to the Python modules that implement it. For installation
and normal use, start with [Setup](setup.md). For diagrams and runtime boundaries,
see [Architecture](architecture.md).

## System at a glance

MyDailyNews is a local-first, staged news pipeline. It retrieves current news,
scores and selects headlines using the reader profile, fetches article text, groups
articles into stories, compares each story's facts with bounded history, lets an
editor model choose coverage, writes reports, updates local memory, and optionally
runs enrichment, perspectives, and text-to-speech modules.

```text
config + reader profile
    -> news discovery and URL deduplication
    -> headline scoring and deterministic selection
    -> article retrieval and same-day story grouping
    -> evidence distillation
    -> historical candidate retrieval and fact operations
    -> validated story cards and editorial selection
    -> structured general and detailed briefs
    -> optional enrichment / perspectives
    -> narrative brief
    -> optional TTS
    -> output files, memory, and caches
```

The default configured module series is `briefs -> narrative_brief`. Enrichment,
perspectives, and TTS are available but disabled by default. Configuration controls
both which modules run and which analysis stages are enabled.

## Recommended reading path

For a first pass, read these files in order. Stop when you reach the subsystem you
want to investigate.

1. [`README.md`](../README.md) — product purpose, setup entrypoint, and outputs.
2. [`docs/architecture.md`](architecture.md) — diagrams, module boundaries, model
   calls, and storage roles.
3. [`main.py`](../main.py) — CLI parsing, configuration loading, runtime validation,
   and construction of the orchestrator.
4. [`mydailynews/pipeline/orchestrator.py`](../mydailynews/pipeline/orchestrator.py) —
   top-level module dispatch, shared source snapshots, multiple brief execution, and
   deferred memory writes.
5. [`mydailynews/pipeline/brief_execution.py`](../mydailynews/pipeline/brief_execution.py)
   — the complete execution path for one general or detailed brief.
6. [`mydailynews/pipeline/brief_stages.py`](../mydailynews/pipeline/brief_stages.py) —
   the retrieval, scoring, selection, article-fetch, and story-grouping stage
   adapters.
7. [`mydailynews/pipeline/brief_analysis_stages.py`](../mydailynews/pipeline/brief_analysis_stages.py)
   — construction and failure handling for evidence and story-change analysis.
8. [`mydailynews/analysis/story_delta.py`](../mydailynews/analysis/story_delta.py) and
   [`mydailynews/analysis/claim_delta.py`](../mydailynews/analysis/claim_delta.py) —
   story cards, fact-operation validation, historical key reuse, and editor
   decisions.
9. [`mydailynews/briefing/generator.py`](../mydailynews/briefing/generator.py) — the
   final structured-brief prompt, token budgeting, response normalization, and
   fallbacks.
10. [`mydailynews/memory/story_store.py`](../mydailynews/memory/story_store.py) —
    durable story records, source facts, lifecycle, and writeback.
11. [`mydailynews/pipeline/narrative_brief.py`](../mydailynews/pipeline/narrative_brief.py)
    and [`mydailynews/pipeline/tts_module.py`](../mydailynews/pipeline/tts_module.py) —
    downstream narrative and audio generation.

The shortest useful trace is `main()` -> `NewsOrchestrator.run()` ->
`NewsOrchestrator.run_briefs()` -> `run_brief()`.

## Runtime call path

### 1. Startup and configuration

[`main.py`](../main.py) parses the CLI, loads configuration with `load_config()`,
checks it with `find_runtime_config_issues()`, constructs `PipelineRunOptions`, and
calls `NewsOrchestrator.run()`. Memory-management commands are dispatched directly
to [`mydailynews/memory/cli.py`](../mydailynews/memory/cli.py).

The configuration contract is split between:

- [`mydailynews/app/config.py`](../mydailynews/app/config.py): strict JSON loading,
  defaults, validation, and environment overrides.
- [`mydailynews/app/models.py`](../mydailynews/app/models.py): configuration and
  runtime dataclasses.
- [`mydailynews/app/runtime_config.py`](../mydailynews/app/runtime_config.py): checks
  that the chosen model-server mode and local runtime are ready.
- [`config.example.json`](../config.example.json): the public example containing all
  supported top-level configuration sections.

### 2. Module orchestration

`NewsOrchestrator.run()` chooses a standalone module or the configured series.
`NewsOrchestrator.run_series()` runs enabled modules in configuration order.
`NewsOrchestrator.run_briefs()` prepares prior reports, builds one shared source
snapshot, scores reusable headline candidates, and invokes the general and detailed
brief specifications.

The canonical module and stage names live in
[`mydailynews/pipeline/stages.py`](../mydailynews/pipeline/stages.py). Brief names and
their goals, topics, filters, and output suffixes are assembled in
[`mydailynews/pipeline/brief_specs.py`](../mydailynews/pipeline/brief_specs.py).

### 3. One brief

`run_brief()` in
[`mydailynews/pipeline/brief_execution.py`](../mydailynews/pipeline/brief_execution.py)
runs the following stages:

1. `_prepare_candidates_stage()` gathers candidates for the brief and deduplicates
   URLs.
2. `_limit_headlines_stage()` applies deterministic input limits before model
   scoring.
3. `_score_headlines_stage()` obtains `HeadlineDecision` values, reusing shared
   decisions when available.
4. `_select_articles_stage()` applies thresholds, profile signals, source and story
   caps, learned preferences, memory signals, and the final input budget.
5. `_fetch_articles_stage()` populates full article text where possible.
6. `_story_grouping_stage()` uses `StoryGroupingPlanner.plan()` and normalizes the
   result so every selected article has a disposition.
7. `_run_evidence_stage()` optionally creates a bounded evidence packet.
8. `_run_delta_stage()` retrieves story history, builds fact operations, validates
   them, and obtains editorial decisions.
9. `BriefGenerator.generate()` writes the reader-facing structured brief payload.
10. `write_markdown()` and `write_json()` persist the report; memory updates are
    captured for deferred writeback.
11. `write_brief_handoff()` saves the selected article and story-boundary artifact
    used by optional downstream modules.

The exact checkpoint order is `BRIEF_STAGE_ORDER` in
[`mydailynews/pipeline/stages.py`](../mydailynews/pipeline/stages.py). This is the best
source when adding, removing, or debugging a stage.

### 4. Headline scoring and selection

The model-facing headline scorer is `HeadlineAnalyzer` in
[`mydailynews/ai/headline_analyzer.py`](../mydailynews/ai/headline_analyzer.py).
Selection policy is intentionally separate in
[`mydailynews/domain/headline_selection.py`](../mydailynews/domain/headline_selection.py).
That module derives profile-match annotations, combines model and memory signals,
enforces selection caps, and returns `SelectedArticle` objects with inspectable
reason codes.

Shared snapshot and scoring behavior is implemented by:

- [`mydailynews/pipeline/snapshot_helpers.py`](../mydailynews/pipeline/snapshot_helpers.py)
- [`mydailynews/pipeline/shared_headline_scoring.py`](../mydailynews/pipeline/shared_headline_scoring.py)

### 5. Story grouping, comparison, and editing

Same-day grouping is owned by
[`mydailynews/story_grouping/planner.py`](../mydailynews/story_grouping/planner.py).
[`mydailynews/story_grouping/normalization.py`](../mydailynews/story_grouping/normalization.py)
validates and repairs the group shape without making semantic news judgments.

Historical comparison then follows this boundary:

1. `build_story_memory_context()` in
   [`mydailynews/memory/context.py`](../mydailynews/memory/context.py) retrieves up to
   three plausible prior story baselines.
2. `StoryDeltaAnalyzer` in
   [`mydailynews/analysis/story_delta.py`](../mydailynews/analysis/story_delta.py)
   creates bounded current and prior evidence for each story.
3. The analysis model assigns one of `add`, `repeat`, `replace`, `resolve`, or
   `uncertain` to every current fact. There is no separate same-story/new-story
   model verdict.
4. `validate_fact_operations()` in
   [`mydailynews/analysis/claim_delta.py`](../mydailynews/analysis/claim_delta.py)
   checks IDs, ownership, conflicts, and complete current-fact coverage. It does not
   reinterpret the fact semantics.
5. A validated citation to facts owned by exactly one prior story allows
   deterministic reuse of that story key.
6. The editor model receives validated story cards and chooses `full_report`,
   `continuing_bullet`, or `omit`. It cannot change fact operations.

Malformed, ambiguous, missing, or over-budget analysis fails open: affected stories
remain eligible for coverage. See [Story quality](story-quality.md) for the complete
contract and failure policy.

### 6. Reports, memory, and optional modules

`BriefGenerator` creates the structured general and detailed reports.
[`mydailynews/briefing/output.py`](../mydailynews/briefing/output.py) serializes them
as Markdown and JSON. Memory writes occur only after all configured briefs have
analyzed the same pre-run baseline; the queue is managed by `PendingMemoryWrite` in
[`mydailynews/pipeline/brief_execution.py`](../mydailynews/pipeline/brief_execution.py).

After `briefs`, the orchestrator can run:

- [`mydailynews/pipeline/enrichment_module.py`](../mydailynews/pipeline/enrichment_module.py):
  optional additional retrieval and story context.
- [`mydailynews/pipeline/perspectives_report.py`](../mydailynews/pipeline/perspectives_report.py):
  optional claim-led perspectives report.
- [`mydailynews/pipeline/narrative_brief.py`](../mydailynews/pipeline/narrative_brief.py):
  narrative synthesis over current structured outputs and available optional
  context.
- [`mydailynews/pipeline/tts_module.py`](../mydailynews/pipeline/tts_module.py):
  Markdown-to-WAV dispatch through the TTS package.

## Package map

| Package | Responsibility | Start with |
|---|---|---|
| `mydailynews/ai` | AI client contracts, backends, prompts, schemas, and token budgets | `mydailynews/ai/base.py`, `mydailynews/ai/factory.py`, `mydailynews/ai/prompts.py`, `mydailynews/ai/schemas.py` |
| `mydailynews/analysis` | Evidence distillation, fact operations, story cards, and coverage policy | `mydailynews/analysis/evidence.py`, `mydailynews/analysis/story_delta.py`, `mydailynews/analysis/claim_delta.py` |
| `mydailynews/app` | Configuration, runtime validation, and shared dataclasses | `mydailynews/app/config.py`, `mydailynews/app/runtime_config.py`, `mydailynews/app/models.py` |
| `mydailynews/briefing` | Structured and narrative report generation and serialization | `mydailynews/briefing/generator.py`, `mydailynews/briefing/narrative.py`, `mydailynews/briefing/output.py` |
| `mydailynews/common` | SQLite access, caches, warnings, time, and serialization helpers | `mydailynews/common/storage.py`, `mydailynews/common/cache.py` |
| `mydailynews/diagnostics` | Debug events, metrics, artifacts, and CLI reporting | `mydailynews/diagnostics/debug.py`, `mydailynews/diagnostics/reporting.py` |
| `mydailynews/domain` | Deterministic selection and domain-level scoring policy | `mydailynews/domain/headline_selection.py` |
| `mydailynews/enrichment` | Optional story research, retrieval, and context synthesis | `mydailynews/enrichment/runner.py`, `mydailynews/enrichment/research.py`, `mydailynews/enrichment/synthesis.py` |
| `mydailynews/evaluation` | Evaluation corpus contracts and retrieval diagnostics | `mydailynews/evaluation/schema.py`, `mydailynews/evaluation/retrieval_diagnostics.py` |
| `mydailynews/gui` | Local web server, data access, run management, and static UI | `mydailynews/gui/server.py`, `mydailynews/gui/data.py`, `mydailynews/gui/runs.py` |
| `mydailynews/memory` | Story, coverage, recall, feedback, preferences, health, and repair | `mydailynews/memory/story_store.py`, `mydailynews/memory/coverage.py`, `mydailynews/memory/recall.py` |
| `mydailynews/perspectives` | Source adapters used by optional perspectives analysis | `mydailynews/perspectives/sources.py` |
| `mydailynews/pipeline` | Cross-module orchestration and stage adapters | `mydailynews/pipeline/orchestrator.py`, `mydailynews/pipeline/brief_execution.py`, `mydailynews/pipeline/brief_stages.py` |
| `mydailynews/retrieval` | News search, article retrieval, prior reports, and HTTP-backed sources | `mydailynews/retrieval/google_news.py`, `mydailynews/retrieval/article.py`, `mydailynews/retrieval/reports.py` |
| `mydailynews/scrapers` | RSS discovery | `mydailynews/scrapers/rss.py` |
| `mydailynews/story_grouping` | Same-day story planning, normalization, and artifacts | `mydailynews/story_grouping/planner.py`, `mydailynews/story_grouping/normalization.py`, `mydailynews/story_grouping/models.py` |
| `mydailynews/tts` | Kokoro loading, text preparation, chunking, and WAV output | `mydailynews/tts/kokoro_backend.py` |

## Core data contracts

Most cross-stage runtime objects are dataclasses in
[`mydailynews/app/models.py`](../mydailynews/app/models.py).

| Type | Meaning |
|---|---|
| `AppConfig` | Fully loaded configuration passed into the orchestrator. |
| `NewsCandidate` | A discovered headline with source, URL, snippet, time, metadata, and annotations. |
| `HeadlineDecision` | Model scoring plus selection reason and rank fields. |
| `SelectedArticle` | A candidate selected for deeper work, with article text and optional context. |
| `PriorReport` | A prior structured report available to prompts and fallback history. |
| `BriefOutput` | Paths and counts for one structured brief. |
| `PipelineResult` | Aggregated outputs and warnings for the requested module or series. |

Two subsystem-specific contracts are also important:

- `StoryGroup` in
  [`mydailynews/story_grouping/models.py`](../mydailynews/story_grouping/models.py)
  defines a same-day story and its member article IDs.
- `ClaimEvidence`, `FactOperationRequest`, `FactOperation`, and
  `FactOperationValidation` in
  [`mydailynews/analysis/claim_delta.py`](../mydailynews/analysis/claim_delta.py)
  define the auditable boundary between model-proposed semantics and deterministic
  validation.

Evidence and delta packets are bounded JSON-shaped dictionaries because they are
both prompt payloads and persisted report artifacts. When changing one, inspect its
producer, every filtering/compaction helper, the final generator, memory writeback,
and tests before treating the change as complete.

## Files written at runtime

| Location | Contents | Primary implementation |
|---|---|---|
| `output/` | Structured and narrative Markdown/JSON, handoffs, optional WAV files, and diagnostics | `mydailynews/briefing/output.py`, `mydailynews/pipeline/handoff.py` |
| `state/memory/memory.sqlite3` | Story, coverage, feedback, and metadata tables | `mydailynews/common/storage.py`, `mydailynews/memory/` |
| **state/memory/learned_preferences.json** | Human-editable learned preference state | `mydailynews/memory/learned_preferences.py` |
| `.cache/mydailynews/cache.sqlite3` | Discovery, article, enrichment, and AI synthesis cache entries | `mydailynews/common/cache.py` |

These directories contain generated local state. They are inputs to later runs but
are not source code.

## Where to look for common questions

| Question | Primary files | Useful tests |
|---|---|---|
| Why was a headline included or excluded? | `mydailynews/domain/headline_selection.py`, `mydailynews/ai/headline_analyzer.py`, `mydailynews/memory/ranking.py` | `tests/test_headline_analyzer.py`, `tests/test_memory_module.py` |
| Where did an article come from? | `mydailynews/scrapers/rss.py`, `mydailynews/retrieval/google_news.py`, `mydailynews/pipeline/snapshot_helpers.py` | `tests/test_registry_rss_retriever.py`, `tests/test_gnews_retriever.py` |
| Why were articles grouped together? | `mydailynews/story_grouping/planner.py`, `mydailynews/story_grouping/normalization.py`, `mydailynews/pipeline/brief_stages.py` | `tests/test_story_grouping.py` |
| How is a fact classified and validated? | `mydailynews/analysis/story_delta.py`, `mydailynews/analysis/claim_delta.py`, `mydailynews/ai/prompts.py`, `mydailynews/ai/schemas.py` | `tests/test_claim_delta.py`, `tests/test_story_delta_editor.py`, `tests/test_story_identity_architecture.py` |
| Why did the editor omit or shorten a story? | `mydailynews/analysis/story_delta.py`, `mydailynews/memory/recall.py`, `mydailynews/analysis/policy_filter.py` | `tests/test_story_delta_editor.py`, `tests/test_brief_delta_integration.py` |
| How is the final prompt constructed? | `mydailynews/briefing/generator.py`, `mydailynews/briefing/final_budget.py` | `tests/test_brief_delta_integration.py`, `tests/test_narrative_briefing.py` |
| How is story history persisted? | `mydailynews/memory/story_store.py`, `mydailynews/memory/coverage.py`, `mydailynews/memory/context.py` | `tests/test_memory_writeback.py`, `tests/test_story_memory_context.py`, `tests/test_sqlite_storage.py` |
| How do optional modules connect? | `mydailynews/pipeline/orchestrator.py`, `mydailynews/pipeline/handoff.py`, `mydailynews/pipeline/enrichment_module.py`, `mydailynews/pipeline/narrative_brief.py`, `mydailynews/pipeline/tts_module.py` | `tests/test_pipeline_module_contract.py`, `tests/test_handoff_and_enrichment_module.py`, `tests/test_tts_module.py` |
| How does the GUI read and modify state? | `mydailynews/gui/server.py`, `mydailynews/gui/data.py`, `mydailynews/gui/runs.py` | `tests/test_gui_server.py`, `tests/test_gui_data.py` |

## Practical investigation tools

List the pipeline's canonical checkpoints:

```bash
python main.py --config <config-path> --list-stages
```

Stop after a stage and inspect the output and diagnostics produced so far:

```bash
python main.py --config <config-path> --debug --stop-after-stage story_grouping
```

Run the complete test suite:

```bash
python -B -m unittest discover -s tests
```

Run one focused test module through discovery:

```bash
python -B -m unittest discover -s tests -p "test_claim_delta.py" -v
```

When changing a cross-stage packet or dataclass, search for both its constructor and
serialized field names. Several boundaries deliberately support older stored packet
versions, so changing only the producer can leave memory, recall, GUI, or report
consumers inconsistent.

## Related documentation

- [Architecture](architecture.md) — diagrams, module order, call counts, and storage
  boundaries.
- [Story quality](story-quality.md) — current fact-operation and editor design.
- [Configuration](configuration.md) — supported settings and defaults.
- [CLI](cli.md) — run modes, standalone modules, and flags.
- [Evaluation](evaluation.md) — retrieval and story-operation evaluation tools.
- [Troubleshooting](troubleshooting.md) — startup, runtime, and output problems.
