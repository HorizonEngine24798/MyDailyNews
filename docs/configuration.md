# Configuration

Public users should not edit tracked project defaults directly.

Recommended flow:

```powershell
copy config.example.json config.local.json
python tools/autoconfig.py --config config.local.json --write config.recommended.json
python main.py --config config.recommended.json
```

## Files

- `config.example.json`: committed portable sample with placeholder paths.
- `config.local.json`: ignored local working config.
- `config.local*.json`: ignored machine-specific variants.
- `config.recommended.json`: ignored autoconfig output.
- `profiles/model_catalog.json`: committed model and hardware-tier recommendations.
- `profiles/config.*.example.json`: committed loadable example profiles.

## AI Sections

`ai_summary` and `ai_final` intentionally duplicate managed-server fields so each role can tune prompt and output limits separately while sharing the same server.

Important fields:

- `backend`: must be `llama_cpp_server`.
- `base_url`: OpenAI-compatible endpoint exposed by `llama-server`.
- `manage_server`: start and stop `llama-server` from the app.
- `server_executable`: path to `llama-server`.
- `server_model_path`: local GGUF path.
- `server_model`: model label sent to the endpoint.
- `server_arguments`: llama.cpp launch arguments.
- `context_window_tokens`: app-side record of the effective context window.
- `max_input_tokens` and `max_new_tokens`: prompt and output budgets.

Keep `max_input_tokens + max_new_tokens` lower than the context window passed to llama.cpp with `-c`.

## Coupled Limits

Do not lower only one field when tuning for smaller hardware. Tune these together:

- llama.cpp context size and GPU offload arguments
- `ai_summary` and `ai_final` token limits
- headline batch sizes and headline token limits
- selected article caps
- evidence and delta article caps
- evidence and delta prompt/output limits

Autoconfig writes these as a coupled profile.

## Narrative Briefing

`narrative_briefing` controls the optional final pass that turns saved brief JSON into polished narrative Markdown.

Important fields:

- `enabled`: run the narrative pass after the regular brief outputs. This is `true` in generated configs by default.
- `max_input_tokens` and `max_new_tokens`: optional overrides for this pass. Leave `null` to reuse `ai_final` limits.
- `target_words`: soft length target for the Markdown brief; the prompt still asks the model not to compress away material developments.
- `editorial_style`: natural-language guidance for the human-readable narrative pass.

When enabled, the pass loads same-day general and detailed JSON briefs when they exist, removes URL/link fields before prompting to reduce context load, and writes:

```text
output/YYYY-MM-DD_narrative_brief.md
output/YYYY-MM-DD_narrative_brief.json
```

This stage deliberately avoids SSML, pause markers, pronunciation tags, and provider-specific TTS markup. A future TTS-prep stage should consume the narrative Markdown and adapt it to the selected TTS backend.

Narrative generation is a post-output pass. If it fails, the structured general/detailed briefs remain written and the pipeline records a warning instead of failing the whole run.

## Enrichment

`enrichment.enabled` defaults to `false`. `enrichment.mode` controls the optional post-fetch context stage when enrichment is explicitly enabled:

- `story_llm`: selected articles are grouped into LLM-planned story threads, searched with cached DDG HTML retrieval, synthesized into compact internal context articles, and attached as `story_llm_research_context` sources.
- `disabled`: skip enrichment, equivalent to `enabled=false`.

The main story-thread budget knobs are `max_story_threads`, `planner_max_questions_per_story`, `search_results_per_query`, `max_fetched_research_pages_per_story`, `max_selected_article_excerpt_chars`, `max_research_excerpt_chars`, and `cache_ttl_seconds`. Autoconfig rewrites the `enrichment` block from `profiles/model_catalog.json` `story_enrichment_budget` recommendations and keeps enrichment disabled unless the source config explicitly opts in. Local configs can still override the generated values manually. Runtime enrichment uses these values directly and skips over-budget planner/synthesis work instead of applying hidden excerpt or fetch-count fallback tiers.

The previous Wikipedia/related-news enrichment mode has been removed. `load_config` now rejects unrecognized keys consistently across config sections, so stale enrichment keys such as `past_news_days`, `max_past_news_results`, `max_wikipedia_results`, and `max_entities` fail as ordinary unknown keys. `enrichment.mode` must be `story_llm` or `disabled`.

When both enrichment and evidence are enabled, the pipeline runs a shared `story_grouping` stage after article fetch so both consumers receive the same story boundaries. See [shared story grouping](shared_story_grouping_plan.md).

## Runtime

`runtime` controls only pipeline-level concurrency and snapshot reuse:

- `max_http_workers`: headline/source fetch concurrency.
- `max_article_workers`: selected-article text fetch concurrency.
- `use_shared_snapshot`: fetch candidate sources once and reuse them across enabled brief modes.

Story enrichment is deterministic and sequential. The old `runtime.max_enrichment_workers` key has been removed.

## Migration Notes

This revamp ships strict unknown-key validation. Update older local configs instead of relying on compatibility shims.

Removed keys and behaviors:

- `enrichment.mode="simple"` and the old Wikipedia/related-news enrichment path.
- `enrichment.past_news_days`, `enrichment.max_past_news_results`, `enrichment.max_wikipedia_results`, and `enrichment.max_entities`.
- `cache.wikipedia_retention_days`.
- `runtime.max_enrichment_workers`.
- Old event-cluster selection/filtering configuration. Event-cluster diversity heuristics were intentionally retired in favor of source caps, topic caps, ranking, novelty, duplicate-link checks, and optional shared story grouping after article fetch.

## Runtime Checks

`main.py` validates runtime readiness before starting the pipeline. It reports placeholder paths, missing managed-server model files, unresolved `llama-server`, and token/context mismatches.

`load_config` remains a syntax and schema parser; runtime readiness checks live separately in `mydailynews.app.runtime_config`.
