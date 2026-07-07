# Configuration

Public users should not edit tracked project defaults directly.

Recommended flow:

```bash
cp config.example.json config.local.json
python tools/autoconfig.py --config config.local.json --write config.recommended.json
python main.py --config config.recommended.json
```

In an interactive terminal, `autoconfig` asks usage-preference questions after hardware detection. Those answers adjust the default module series, brief volume, evidence/delta depth, narrative length, and discovery cache mode. Add `--no-preference-prompt` for scripted runs that should keep the standard defaults.

## Files

- `config.example.json`: committed portable sample for LM Studio's local server.
- `config.local.json`: ignored local working config.
- `config.local*.json`: ignored machine-specific variants.
- `config.recommended.json`: ignored autoconfig output.
- `profiles/model_catalog.json`: committed model and hardware-tier recommendations.
- `profiles/config.*.example.json`: committed loadable example profiles.

## AI Sections

`ai_summary` and `ai_final` intentionally duplicate AI fields so each role can tune prompt and output limits separately while sharing the same OpenAI-compatible server.

Important fields:

- `backend`: must be `llama_cpp_server`.
- `base_url`: OpenAI-compatible endpoint, usually `http://127.0.0.1:1234/v1` for LM Studio.
- `manage_server`: keep this `false` for LM Studio or Docker.
- `server_executable`: only used when `manage_server=true`.
- `server_model_path`: only used when `manage_server=true`.
- `server_model`: model label sent to the endpoint.
- `server_arguments`: only used when `manage_server=true`.
- `context_window_tokens`: app-side record of the effective context window.
- `max_input_tokens` and `max_new_tokens`: prompt and output budgets.

Keep `max_input_tokens + max_new_tokens` lower than the context window configured in LM Studio.

## Coupled Limits

Do not lower only one field when tuning for smaller hardware. Tune these together:

- LM Studio model, context size, and GPU offload settings
- `ai_summary` and `ai_final` token limits
- headline batch sizes and headline token limits
- selected article caps
- evidence and delta article caps
- evidence and delta prompt/output limits

Autoconfig writes these as a coupled profile.

## Narrative Briefing

`narrative_briefing` controls the optional module that turns saved brief JSON into polished narrative Markdown.

Important fields:

- `enabled`: allow the narrative module to run. This is `true` in generated configs by default.
- `max_input_tokens` and `max_new_tokens`: optional overrides for this pass. Leave `null` to reuse `ai_final` limits.
- `target_words`: soft length target for the Markdown brief; the prompt still asks the model not to compress away material developments.
- `editorial_style`: natural-language guidance for the human-readable narrative pass.

When run as a standalone module, narrative briefing loads same-day general and detailed JSON briefs when they exist and uses same-day enrichment JSON when enrichment is enabled. Inside the default module series, it consumes only the structured briefs and enrichment JSON produced earlier in the same run. It removes URL/link fields before prompting to reduce context load, and writes:

```text
output/YYYY-MM-DD_narrative_brief.md
output/YYYY-MM-DD_narrative_brief.json
```

This stage deliberately avoids SSML, pause markers, pronunciation tags, and provider-specific TTS markup. The optional TTS module consumes the narrative Markdown and adapts plain text to the selected TTS backend.

Narrative generation is a post-brief module. If it fails, the structured general/detailed briefs remain written and the pipeline records a warning instead of failing the whole run.

## TTS Audio

`tts` controls the optional Kokoro audio module. It is disabled by default and consumes the narrative Markdown output:

```json
{
  "tts": {
    "enabled": false,
    "backend": "kokoro",
    "model_id": "hexgrad/Kokoro-82M",
    "voice": "af_heart",
    "lang_code": "a",
    "speed": 1.0,
    "max_chunk_chars": 1200
  }
}
```

Kokoro currently supports Python 3.10 through 3.12. If your base environment is Python 3.13 or newer, create a Python 3.12 environment first:

```bash
conda create -n mydailynews-tts python=3.12
conda activate mydailynews-tts
python -m pip install -r requirements.txt
```

Then run:

```bash
python main.py --module tts --markdown-path output/YYYY-MM-DD_general_brief.md
```

If `--markdown-path` is omitted, TTS falls back to the same-day narrative Markdown selected by `--date`.

It writes:

```text
output/<input-markdown-stem>.wav
output/<input-markdown-stem>_audio.json
```

Add `tts` after `narrative_brief` in `pipeline.default_series` only when audio should be part of normal runs.

## Memory

`memory` controls the lightweight coverage-memory layer used during article selection and final/narrative prompting.

Important fields:

- `enabled`: turn story keys, recent-coverage rank adjustments, story caps, coverage writeback, and recall packets on or off. Set this to `false` to preserve the old selection behavior.
- `state_dir`: durable memory directory. The default is `state/memory`, separate from `output/` so ordinary output cleanup does not reset ranking memory.
- `coverage_window_days`: recent-history lookback for repeat penalties.
- `coverage_retention_days`: coverage-log retention; the default is 30 days.
- `story_stale_after_days`: mark story-index records `stale` when they have not been updated for more than this many days; the default is 7.
- `story_retention_days`: prune story-index records older than this many days; the default is 30.
- `recent_story_penalty` and `recent_lead_penalty`: bounded deterministic rank penalties for repeated stories and recent leads.
- `material_update_boost`: offset for recent-coverage penalties when headline/delta signals indicate a material new phase.
- `max_selected_per_story` and `max_selected_per_story_family`: same-run diversity caps.
- `recall_prompt_enabled`: pass compact coverage guidance into final and narrative prompts.
- `save_recall_packets`: save compact per-brief recall packets for debugging and inspection.
- `feedback_enabled`: enable the file-backed feedback event surface for `too_repetitive`, `not_relevant`, `not_interested_in_topic`, and `more_like_this`.

Memory writes inspectable JSON files:

```text
state/memory/coverage_log.jsonl
state/memory/coverage_log.archive.jsonl
state/memory/story_index.json
state/memory/feedback_events.jsonl
state/memory/learned_preferences.json
state/memory/backups/
state/memory/recall_packets/YYYY-MM-DD_general.json
state/memory/recall_packets/YYYY-MM-DD_detailed.json
```

The `memory` section is required in current configs. The memory layer does not add LLM calls and does not mutate `user_memory` preferences. Learned preferences live in `state/memory/learned_preferences.json`, are visible/editable in the GUI, are updated by supported feedback events, and are applied as bounded deterministic rank adjustments in later article selection. GUI repair tools create timestamped backups in `state/memory/backups/` before rewriting Story Index, Coverage Memory, or Feedback Events. The GUI Runs tab can also launch safe CLI-owned memory inspect, prune, and export commands.

Useful memory maintenance commands:

```bash
python main.py --config config.local.json --memory inspect
python main.py --config config.local.json --memory prune
python main.py --config config.local.json --memory export --memory-export memory-export.json
python main.py --config config.local.json --memory reset --confirm-memory-reset
```

## Enrichment

`enrichment.enabled` defaults to `true`. `enrichment.mode` controls the post-brief enrichment module:

- `story_llm`: selected articles are loaded from same-day handoff/brief files, grouped into LLM-planned story threads, searched with cached DDG HTML retrieval, synthesized into compact internal context articles, and written to `output/YYYY-MM-DD_enrichment.json`.
- `disabled`: skip enrichment, equivalent to `enabled=false`.

The main story-thread budget knobs are `max_story_threads`, `planner_max_questions_per_story`, `planner_max_input_tokens`, `planner_max_new_tokens`, `search_results_per_query`, `max_queries_per_story`, `max_fetched_research_pages_per_story`, `max_selected_article_excerpt_chars`, `max_research_excerpt_chars`, `synthesis_max_input_tokens`, `synthesis_max_new_tokens`, and `cache_ttl_seconds`. `omitted_article_policy` defaults to `skip`, so planner omissions are recorded instead of becoming automatic singleton enrichment work; set it to `fallback_singleton` to restore the older behavior. `planner_allow_misc_group=true` lets the planner classify low-value leftovers as `misc`, and `enrich_misc_story=false` keeps those groups in artifacts without searching, fetching, synthesizing, or attaching context. `excerpt_strategy` defaults to `relevant_windows`, which keeps a lead plus story-relevant body windows; set it to `prefix` for old prefix truncation. Planner and synthesis token settings are interpreted as stage budgets, but they do not shrink requests below the active AI client's `max_input_tokens` or `max_new_tokens`; by default enrichment has at least the same prompt and output room as the main pipeline. Autoconfig rewrites the `enrichment` block from `profiles/model_catalog.json` `story_enrichment_budget` recommendations while preserving explicit local opt-outs such as `enabled=false` or `mode="disabled"`. Local configs can still override the generated values manually. Runtime enrichment uses these values directly and skips over-budget planner/synthesis work instead of applying hidden excerpt or fetch-count fallback tiers.

The previous Wikipedia/related-news enrichment mode has been removed. `load_config` now rejects unrecognized keys consistently across config sections, so stale enrichment keys such as `past_news_days`, `max_past_news_results`, `max_wikipedia_results`, and `max_entities` fail as ordinary unknown keys. `enrichment.mode` must be `story_llm` or `disabled`.

When evidence is enabled, the structured brief pipeline can run `story_grouping` after article fetch to provide shared story boundaries for evidence. Standalone enrichment plans its own story threads from the saved handoff/brief inputs and prints progress for input collection, story planning, per-story retrieval, and per-story synthesis completion.

## Module Series

`pipeline.default_series` controls the default top-level module order when `--module` is omitted:

```json
{
  "pipeline": {
    "default_series": ["briefs", "enrichment", "narrative_brief"]
  }
}
```

Allowed module names are `briefs`, `enrichment`, `narrative_brief`, and `tts`. Unknown or duplicate module names fail config parsing. Disabled optional modules listed in the series are skipped with a warning. In series mode, downstream modules consume only artifacts created earlier in the same run; standalone module commands are the disk-rerun path. `--date` is accepted only for standalone `enrichment`, `narrative_brief`, and `tts` runs. `--markdown-path` is accepted only for standalone `tts` runs.

CLI examples:

```bash
python main.py --module briefs
python main.py --module enrichment --date 2026-06-25
python main.py --module narrative_brief --date 2026-06-25
python main.py --module tts --markdown-path output/2026-06-25_general_brief.md
python main.py --module series --skip-module enrichment
```

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
- `memory.recall_packet_enabled`; use `memory.recall_prompt_enabled` and `memory.save_recall_packets` instead.
- Old event-cluster selection/filtering configuration. Event-cluster diversity heuristics were intentionally retired in favor of source caps, topic caps, ranking, novelty, duplicate-link checks, and optional shared story grouping after article fetch.

## Runtime Checks

`main.py` validates runtime readiness before starting the pipeline. In the default LM Studio mode it checks token/context mismatches; if `manage_server=true`, it also reports missing managed-server model files and unresolved `llama-server`.

`load_config` remains a syntax and schema parser; runtime readiness checks live separately in `mydailynews.app.runtime_config`.
