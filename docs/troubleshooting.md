# Troubleshooting

## Config Not Found

`python main.py` looks for `config.local.json`.

Create one:

```powershell
copy config.example.json config.local.json
python tools/autoconfig.py --config config.local.json --write config.recommended.json
```

Then run:

```powershell
python main.py --config config.recommended.json
```

## Placeholder Or Missing Paths

If the CLI reports `server_executable` or `server_model_path`, edit your local config or rerun autoconfig.

Verify llama.cpp:

```powershell
llama-server --version
```

Verify the GGUF path points to an existing file.

## Startup Timeout

Inspect:

```text
output/diagnostics/llama_server/
```

Common causes:

- model too large for VRAM or RAM
- context window too large
- wrong llama.cpp build for your GPU backend
- port already in use
- stale executable path

## Invalid JSON

Invalid JSON usually means the prompt/output budget is too aggressive for the selected model or context window.

Lower these together:

- `max_headlines_per_ai_batch`
- headline input and output token limits
- selected article caps
- evidence and delta article caps
- `ai_summary.max_input_tokens`
- `ai_final.max_input_tokens`
- `max_new_tokens`

Then rerun autoconfig or choose a smaller hardware profile.
