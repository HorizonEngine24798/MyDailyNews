# Configuration

This is the config reference. For first-run commands, use [setup](setup.md).

Do not edit tracked defaults directly. Copy `config.example.json` to a local config and let autoconfig write `config.recommended.json`.

## Files

- `config.example.json`: committed portable sample for LM Studio's local server.
- `config.local.json`: ignored local working config.
- `config.local*.json`: ignored machine-specific variants.
- `config.recommended.json`: ignored autoconfig output.
- `profiles/model_catalog.json`: committed model and hardware-tier recommendations.
- `profiles/config.*.example.json`: committed loadable example profiles.

## AI Sections

`ai_summary` and `ai_final` intentionally duplicate fields so headline scoring and final synthesis can use different prompt/output budgets while sharing the same server.

Important fields:

- `backend`: must be `llama_cpp_server`.
- `base_url`: OpenAI-compatible endpoint, usually `http://127.0.0.1:1234/v1`.
- `server_model`: model label sent to the endpoint.
- `context_window_tokens`: app-side record of the server's effective per-slot context window.
- `max_input_tokens` and `max_new_tokens`: model-role ceilings. One shared resolver reserves context overhead and clamps every request so input plus output cannot exceed the context window.
- `manage_server`: keep `false` for LM Studio or Docker.
- `server_executable`, `server_model_path`, and `server_arguments`: only used when `manage_server=true`.

Stage token fields are optional preferences. Leave them `null` to inherit the model-role ceilings; use positive values only when a stage has a measured reason to be smaller. Disabling work uses the stage's `enabled` flag, not a zero token budget.

## Coupled Limits

Tune these together for smaller hardware:

- model size, context size, and GPU offload settings
- `ai_summary` and `ai_final` token limits
- headline batch sizes and headline token limits
- selected article caps
- evidence, delta, and enrichment budgets

Autoconfig writes these as a coupled profile. Changing one knob by hand can make the model overload, time out, or return malformed JSON.

## Pipeline Modules

`pipeline.default_series` controls the default top-level module order:

```json
{
  "pipeline": {
    "default_series": ["briefs", "enrichment", "narrative_brief"]
  }
}
```

Allowed modules are `briefs`, `enrichment`, `narrative_brief`, `tts`, and `perspectives_report`. Unknown or duplicate module names fail config parsing. Disabled optional modules listed in the series are skipped with a warning.

Standalone module runs can reuse artifacts from disk. `--date` is accepted only for standalone `enrichment`, `narrative_brief`, `tts`, and `perspectives_report` runs. `--markdown-path` is accepted only for standalone `tts` runs.

## Narrative Briefing

`narrative_briefing` controls the module that turns saved brief JSON into polished narrative Markdown.

Important fields:

- `enabled`: allow the narrative module to run.
- `max_input_tokens` and `max_new_tokens`: optional preferences; leave `null` to use the shared resolver and `ai_final` ceilings.
- `target_words`: soft length target.
- `editorial_style`: natural-language guidance for the narrative pass.

Inside the default series, narrative briefing consumes artifacts produced earlier in the same run. As a standalone module, it loads same-day brief and enrichment JSON from disk. If it fails, existing structured briefs remain written and the pipeline records a warning.

## TTS Audio

`tts` controls the optional Kokoro audio module. It is disabled by default. In a series, it speaks only the Markdown artifacts for modules listed in `tts.modules`.

Important fields:

- `enabled`: allow TTS in the default series.
- `modules`: report modules to speak: `briefs`, `enrichment`, `narrative_brief`, and/or `perspectives_report`.
- `backend`: currently `kokoro`.
- `model_id`, `voice`, `lang_code`, and `speed`: Kokoro voice settings.
- `max_chunk_chars`: paragraph chunking limit for synthesis.

Add `tts` at the end of `pipeline.default_series` when audio should be part of normal runs; the GUI keeps this in sync with the selected `tts.modules`. See [TTS audio](tts.md).

## RSS Sources

`mydailynews/data/sources.rss.json` is the single source catalog. It owns feed URLs, source metadata, categories, and required tags. Settings only select the catalog entries used by normal RSS runs:

```json
{
  "sources": {
    "rss": ["gb_bbc_world", "us_npr", "us_ars_technica"]
  }
}
```

The perspectives report reads the same catalog and selects active sources from its planner tags. Do not put feed URLs or source metadata in a config file.

## Perspectives Report

`perspectives_report` writes one claim-led source-country coverage and framing report from same-day enrichment and brief evidence artifacts. One planner call creates broad story queries and selects a bounded verification set. Broad coverage is retrieved once; targeted primary/independent evidence is then retrieved only for selected claims, followed by one verifier call per claim with usable evidence and one claim-led synthesis call per eligible story.

Important fields:

- `enabled`: allow the module to run.
- `gnews_api_key`: optional GNews API key. Leave empty to skip the GNews provider. You can also set `MYDAILYNEWS_GNEWS_API_KEY` or `GNEWS_API_KEY`.
- `coverage_scope`: optional source countries to include. Leave empty for global/unfiltered discovery.
- `coverage_regions`: optional source-registry regions to target, such as `middle_east`, `europe`, or `asia_pacific`.
- `coverage_timespan_days`: coverage search lookback window.
- `coverage_max_records_per_story`: maximum coverage rows to keep per story.
- `minimum_source_countries`: evidence-quality threshold used to mark thin coverage.
- `verification_enabled`: enable bounded targeted retrieval and focused verifier calls.
- `verification_claims_per_story`: selected verification claims per story, capped at `2`.
- `verification_claims_per_run`: selected verification claims per run, capped at `12`.
- `verification_queries_per_claim`: targeted queries per selected claim, capped at `2`.
- `verification_documents_per_claim`: retained evidence documents per selected claim, capped at `4`.

Coverage retrieval is not filtered by language. The report still records each article's detected or registry language so language diversity remains visible in diagnostics.

Report generation does not cache final coverage sets. It rebuilds accepted rows per run from live provider responses, while still using the normal HTTP/article/AI caches.

Run it standalone with `python main.py --module perspectives_report --date YYYY-MM-DD`, or use `briefs -> enrichment -> perspectives_report -> narrative_brief -> tts` in `pipeline.default_series` so narrative markers can reference validated claim cards.

## Memory

`memory` controls the coverage-memory layer used during article selection and final/narrative prompting.

Important fields:

- `enabled`: turn story keys, recent-coverage rank adjustments, story caps, coverage writeback, and recall packets on or off.
- `state_dir`: durable memory directory. The default is `state/memory`.
- `coverage_window_days` and `coverage_retention_days`: recent-history lookback and retention.
- `story_stale_after_days` and `story_retention_days`: story lifecycle limits.
- `recent_story_penalty`, `recent_lead_penalty`, and `material_update_boost`: deterministic rank adjustments.
- `max_selected_per_story` and `max_selected_per_story_family`: same-run diversity caps.
- `recall_prompt_enabled` and `save_recall_packets`: compact coverage guidance and debug packets.
- `feedback_enabled`: file-backed feedback events used by the GUI and learned preferences.

Memory files are inspectable under `state/memory/`, including `coverage_log.jsonl`, `story_store.json`, `feedback_events.jsonl`, `learned_preferences.json`, `backups/`, and `recall_packets/`. `story_store.json` is the single durable record for story identity, lifecycle, semantic state, retrieval signals, and bounded source evidence. On an existing installation, the store reads legacy `story_index.json` and `story_ledger.json` together until the next normal write creates `story_store.json`; the legacy files are then retained as migration backups but no longer read.

The memory layer does not add LLM calls and does not mutate `user_memory`. Learned preferences are stored separately and applied as bounded deterministic rank adjustments.

## Enrichment

`enrichment` controls the post-brief story-context module.

Modes:

- `story_llm`: group selected articles into story threads, retrieve related context, synthesize compact context, and write enrichment artifacts.
- `disabled`: skip enrichment, equivalent to `enabled=false`.

Main knobs:

- thread planning: `max_story_threads`, `planner_max_questions_per_story`
- retrieval: `search_results_per_query`, `max_queries_per_story`, `max_fetched_research_pages_per_story`
- excerpting: `excerpt_strategy`, selected/research excerpt limits
- optional LLM preferences: `planner_max_input_tokens`, `planner_max_new_tokens`, `synthesis_max_input_tokens`, `synthesis_max_new_tokens`; `null` inherits the model-role limits
- caching: `cache_ttl_seconds`

Autoconfig rewrites the enrichment block from `profiles/model_catalog.json` while preserving explicit local opt-outs such as `enabled=false` or `mode="disabled"`.

## Runtime

`runtime` controls pipeline-level concurrency and snapshot reuse:

- `max_http_workers`: headline/source fetch concurrency.
- `max_article_workers`: selected-article and enrichment research-page fetch concurrency.
- `use_shared_snapshot`: fetch candidate sources once and reuse them across enabled brief modes.

Story enrichment is deterministic and sequential.

## Cache

`cache` controls local discovery, article text, enrichment retrieval, and AI synthesis caches. The default directory is `.cache/mydailynews`.

Useful fields:

- `enabled`: turn local caches on or off.
- `discovery_mode`: network/cache behavior for feed and news discovery.
- `article_text_retention_days`, `enrichment_retention_days`, and `http_retention_days`: cache pruning windows.
- `ai_enabled` and `synth_fresh_seconds`: AI synthesis cache behavior.

## Migration Notes

Config loading uses strict unknown-key validation. Update older local configs instead of relying on compatibility shims.

Removed keys and behaviors include:

- `enrichment.mode="simple"` and the old Wikipedia/related-news enrichment path.
- `enrichment.past_news_days`, `enrichment.max_past_news_results`, `enrichment.max_wikipedia_results`, and `enrichment.max_entities`.
- `cache.wikipedia_retention_days`.
- `runtime.max_enrichment_workers`.
- `memory.recall_packet_enabled`; use `memory.recall_prompt_enabled` and `memory.save_recall_packets`.
- old event-cluster selection/filtering configuration.

## Runtime Checks

`load_config` handles syntax and schema parsing. `main.py` runs readiness checks before the pipeline starts, including token/context mismatches and managed-server file checks.
