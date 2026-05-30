# Implementing MyDailyNews With llama.cpp

This document describes the current, code-accurate path for running MyDailyNews on llama.cpp.

## Status

- Backend abstraction is already implemented (`AIClient` protocol + `create_ai_client` factory).
- Both backends exist today:
  - `transformers`
  - `llama_cpp_server`
- Structured output schemas are already used in pipeline calls (`headline_analyzer` and `brief`).
- The runtime now uses dual model roles:
  - `ai_summary` (headline scoring)
  - `ai_final` (final brief synthesis)

## Current Pipeline

The runtime flow is intentionally shallow:

```text
fetch + dedupe
-> heuristic prefilter
-> headline scoring LLM (score contract)
-> article fulltext retrieval
-> optional non-LLM enrichment
-> one-shot final brief synthesis
-> markdown/json write
```

Removed from hot path:

- per-article LLM enrichment planning
- per-article LLM brief generation
- narrative-map contracts

## Integration Seam (Actual)

You do not replace the whole pipeline. You swap AI backend configuration.

Primary seam:

- `mydailynews/ai/factory.py`
- `create_ai_client(config, debug)`

Shared call contract used by pipeline stages:

```python
complete_json(
    system: str,
    user: str,
    label: str = "ai.complete_json",
    *,
    max_new_tokens: int | None = None,
    input_token_limit: int | None = None,
    json_schema: JSONSchemaSpec | None = None,
) -> dict
```

Main callers:

- `mydailynews/ai/headline_analyzer.py`
- `mydailynews/brief.py`

## What Stays The Same

These modules remain the same conceptually when switching to llama.cpp:

- `main.py`
- `mydailynews/orchestrator.py`
- `mydailynews/scrapers/rss.py`
- `mydailynews/retrieval/*`
- `mydailynews/enrichment.py`
- `mydailynews/brief.py`
- `mydailynews/output.py`
- `mydailynews/debug.py`

All retrieval, enrichment, selection, and output orchestration remains Python-side.

## What Changes

Only backend selection and runtime config change.

Use backend `llama_cpp_server` in `ai_summary` and/or `ai_final`.

Example:

```json
{
  "ai_summary": {
    "backend": "llama_cpp_server",
    "server_model": "qwen3-1.7b-q4km",
    "base_url": "http://127.0.0.1:8080/v1",
    "max_input_tokens": 3072,
    "max_new_tokens": 512,
    "json_retries": 1,
    "temperature": 0.0,
    "top_p": 0.9,
    "response_format": "json_schema",
    "request_timeout_seconds": 300
  },
  "ai_final": {
    "backend": "llama_cpp_server",
    "server_model": "qwen3-8b-q4km",
    "base_url": "http://127.0.0.1:8080/v1",
    "max_input_tokens": 4096,
    "max_new_tokens": 2048,
    "json_retries": 1,
    "temperature": 0.0,
    "top_p": 0.9,
    "response_format": "json_schema",
    "request_timeout_seconds": 300
  }
}
```

Notes:

- `server_model` is the model identifier sent to llama-server.
- `response_format=json_schema` enables schema payloads when provided by callers.
- The same server can serve both roles, or you can split ports/models.

## Desktop Setup

1. Start llama-server with a GGUF model.

```powershell
llama-server.exe `
  -m D:\Models\qwen3-8b-instruct-q4_k_m.gguf `
  -c 16384 `
  --host 127.0.0.1 `
  --port 8080 `
  --n-gpu-layers 99
```

2. Set backend config in `config.json` (`ai_summary` and/or `ai_final`).
3. Run:

```powershell
python main.py --config config.json --debug
```

## Structured Output Guidance

Current behavior by backend:

- `llama_cpp_server`: can use `response_format=json_schema` for constrained output.
- `transformers`: currently prompt+retry JSON strategy (schema argument is accepted but not decoder-enforced).

Practical recommendation:

1. Start with simple schemas (`headline_analysis`, `final_brief`).
2. Keep contracts minimal and stable.
3. Validate semantic completeness after parse (not only JSON syntax).

## Known Gap

As of current implementation, `input_token_limit` is not strictly enforced on the `llama_cpp_server` path (it is accepted by the call interface but not hard-trimmed client-side). Keep prompt budgets conservative until backend parity is tightened.

## Android Porting Note

Python orchestration does not move to Android directly. The portable pieces are:

- JSON contracts
- pipeline sequencing rules
- candidate limits and tuning profiles

An Android implementation should mirror these contracts in Kotlin/NDK while using llama.cpp native inference.

## Migration Checklist

1. Choose summary/final GGUF models.
2. Bring up llama-server and verify `/v1/chat/completions`.
3. Switch `ai_summary` and `ai_final` backends.
4. Run with `--debug` and inspect diagnostics artifacts.
5. Tune:
   - `max_candidates_for_ai`
   - `max_headlines_per_ai_batch`
   - `max_input_tokens`
   - `max_new_tokens`
