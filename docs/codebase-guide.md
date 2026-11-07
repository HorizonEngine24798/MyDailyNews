# Codebase guide

Map from product architecture to Python modules. For install/usage see [Setup](setup.md). For runtime flow see [Architecture](architecture.md).

## Pipeline at a glance

```text
config + reader profile
  -> news discovery + URL deduplication (with 2-step RSS self-healing)
  -> headline scoring + deterministic selection
  -> article fetch (consent-wall guarded) + same-day story grouping & event-cohesion partitioning
  -> evidence distillation
  -> hybrid lexical + multilingual E5 historical retrieval & fact operations
  -> validated story cards + editorial selection (unchanged omitted deterministically)
  -> structured general & detailed briefs
  -> optional enrichment / perspectives_report
  -> narrative brief
  -> optional TTS
  -> output/, state/memory/, .cache/mydailynews/
```

Default series: `briefs -> narrative_brief`. `enrichment`, `perspectives_report`, and `tts` are opt-in.

## Reading path

1. [`README.md`](../README.md) — overview and outputs.
2. [`docs/architecture.md`](architecture.md) — flow diagrams, LLM calls, and state boundaries.
3. [`main.py`](../main.py) — CLI entrypoint and config checks.
4. [`mydailynews/pipeline/orchestrator.py`](../mydailynews/pipeline/orchestrator.py) — module dispatch, shared snapshots, deferred memory writes.
5. [`mydailynews/pipeline/brief_execution.py`](../mydailynews/pipeline/brief_execution.py) — single-brief execution (`BRIEF_STAGE_ORDER` in [`stages.py`](../mydailynews/pipeline/stages.py)).
6. [`mydailynews/pipeline/brief_stages.py`](../mydailynews/pipeline/brief_stages.py) & [`brief_analysis_stages.py`](../mydailynews/pipeline/brief_analysis_stages.py) — stage adapters.
7. [`mydailynews/analysis/story_delta.py`](../mydailynews/analysis/story_delta.py) & [`claim_delta.py`](../mydailynews/analysis/claim_delta.py) — story cards, fact-operation validation, high-confidence baseline resolution, editor decisions.
8. [`mydailynews/briefing/generator.py`](../mydailynews/briefing/generator.py) — structured brief prompt and normalization.
9. [`mydailynews/memory/story_store.py`](../mydailynews/memory/story_store.py) & [`story_records.py`](../mydailynews/memory/story_records.py) — durable story records and facts.
10. [`mydailynews/pipeline/perspectives_report.py`](../mydailynews/pipeline/perspectives_report.py) & [`mydailynews/perspectives/`](../mydailynews/perspectives/planning.py) — claim-led perspectives pipeline.

## Package map

| Package | Responsibility | Key modules |
|---|---|---|
| `mydailynews/ai` | AI client contracts, backends, prompts, schemas, token budgets | `base.py`, `factory.py`, `prompts.py`, `schemas.py` |
| `mydailynews/analysis` | Evidence distillation, fact operations, story cards, coverage policy | `evidence.py`, `story_delta.py`, `claim_delta.py` |
| `mydailynews/app` | Config loading, runtime checks, dataclasses | `config.py`, `runtime_config.py`, `models.py` |
| `mydailynews/briefing` | Structured and narrative brief generation and output | `generator.py`, `narrative.py`, `output.py` |
| `mydailynews/common` | SQLite storage, HTTP/AI cache, consent-wall guards, utilities | `storage.py`, `cache.py`, `utils.py` |
| `mydailynews/diagnostics` | Debug events, metrics, CLI reporting | `debug.py`, `reporting.py` |
| `mydailynews/domain` | Deterministic headline selection and text similarity | `headline_selection.py`, `text_similarity.py` |
| `mydailynews/enrichment` | Optional story thread research and synthesis | `runner.py`, `research.py`, `synthesis.py` |
| `mydailynews/evaluation` | Offline evaluation corpus schemas and retrieval diagnostics | `schema.py`, `retrieval_diagnostics.py` |
| `mydailynews/gui` | Local web UI server, state inspection, run control | `server.py`, `data.py`, `runs.py` |
| `mydailynews/memory` | Story store/records, E5 embeddings, hybrid ranking, context partitioning, repair | `story_store.py`, `story_records.py`, `multilingual_embeddings.py`, `ranking.py`, `context.py` |
| `mydailynews/perspectives` | Perspectives schemas, planning, coverage retrieval, claim synthesis, rendering | `schemas.py`, `planning.py`, `coverage.py`, `synthesis.py`, `rendering.py`, `sources.py` |
| `mydailynews/pipeline` | Cross-module orchestration and stage execution | `orchestrator.py`, `brief_execution.py`, `perspectives_report.py`, `narrative_brief.py` |
| `mydailynews/retrieval` | Article fetch, Google News, GDELT, GNews, registry RSS | `article.py`, `google_news.py`, `registry_rss.py`, `gdelt.py` |
| `mydailynews/scrapers` | RSS parsing and 2-step feed self-healing (HTML link discovery + Google News `site:` fallback) | `rss.py` |
| `mydailynews/story_grouping` | Same-day story grouping planner and normalization | `planner.py`, `normalization.py`, `models.py` |
| `mydailynews/tts` | Kokoro TTS synthesis and WAV output | `kokoro_backend.py` |

## Core data contracts

Defined in [`mydailynews/app/models.py`](../mydailynews/app/models.py), [`story_grouping/models.py`](../mydailynews/story_grouping/models.py), [`analysis/claim_delta.py`](../mydailynews/analysis/claim_delta.py), and [`memory/story_records.py`](../mydailynews/memory/story_records.py):

| Type | Meaning |
|---|---|
| `AppConfig` | Validated runtime configuration. |
| `NewsCandidate` / `HeadlineDecision` / `SelectedArticle` | Discovery, scoring, and selection records. |
| `StoryGroup` | Same-day story cluster and member article IDs. |
| `SourceFact` / `StoryRecord` | Durable historical story and extracted atomic facts. |
| `FactOperationRequest` / `FactOperationValidation` | Model-proposed fact operations (`add`, `repeat`, `replace`, `resolve`, `uncertain`) and deterministic validation result. |
| `BriefOutput` / `PerspectivesReportOutput` / `PipelineResult` | Module output paths, counts, and warnings. |

## Runtime state files

| Path | Contents | Owner |
|---|---|---|
| `output/` | Markdown/JSON briefs, perspectives, handoffs, WAVs, diagnostics | `mydailynews/briefing/output.py`, `mydailynews/perspectives/rendering.py` |
| `state/memory/memory.sqlite3` | Stories, facts, coverage, feedback, E5 embeddings | `mydailynews/common/storage.py`, `mydailynews/memory/` |
| `state/memory/learned_preferences.json` | Human-editable learned preference weights | `mydailynews/memory/learned_preferences.py` |
| `.cache/mydailynews/cache.sqlite3` | HTTP, article text, enrichment, and AI synthesis caches | `mydailynews/common/cache.py` |

## Common lookup table

| Question | Primary files | Tests |
|---|---|---|
| Why was a headline selected/skipped? | `domain/headline_selection.py`, `ai/headline_analyzer.py`, `memory/ranking.py` | `tests/test_headline_analyzer.py`, `tests/test_memory_module.py` |
| How does RSS discovery & self-healing work? | `scrapers/rss.py`, `retrieval/registry_rss.py`, `data/sources.rss.json` | `tests/test_registry_rss_retriever.py` |
| Why were articles grouped or split? | `story_grouping/planner.py`, `memory/context.py` (`_partition_story_group`) | `tests/test_story_grouping.py`, `tests/test_story_memory_context.py` |
| How are facts compared and stories omitted? | `analysis/story_delta.py`, `analysis/claim_delta.py` | `tests/test_claim_delta.py`, `tests/test_story_delta_editor.py`, `tests/test_story_identity_architecture.py` |
| How does perspectives planning/synthesis work? | `pipeline/perspectives_report.py`, `perspectives/planning.py`, `perspectives/coverage.py`, `perspectives/synthesis.py` | `tests/test_perspectives_report_module.py`, `tests/test_perspectives_relevance_diagnostics.py` |

## Debug & test commands

```bash
python main.py --config config.recommended.json --list-stages
python main.py --config config.recommended.json --debug --stop-after-stage story_grouping
python -B -m unittest discover -s tests -p "test_*.py"
```
