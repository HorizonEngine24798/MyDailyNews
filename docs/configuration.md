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

## Runtime Checks

`main.py` validates runtime readiness before starting the pipeline. It reports placeholder paths, missing managed-server model files, unresolved `llama-server`, and token/context mismatches.

`load_config` remains a syntax and schema parser; runtime readiness checks live separately in `mydailynews.app.runtime_config`.
