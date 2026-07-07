# Troubleshooting

## Config Not Found

`python main.py` looks for `config.local.json`.

Create one:

```bash
cp config.example.json config.local.json
python tools/autoconfig.py --config config.local.json --write config.recommended.json
```

Then run:

```bash
python main.py --config config.recommended.json
```

## LM Studio Connection

Make sure LM Studio's local server is started and reachable at:

```text
http://127.0.0.1:1234
```

In Docker, Compose overrides the app to use `http://host.docker.internal:1234/v1`.

## Startup Timeout

Inspect:

```text
output/diagnostics/llama_server/
```

Common causes:

- model too large for VRAM or RAM
- context window too large
- wrong LM Studio GPU offload/runtime setting
- port already in use
- LM Studio local server not started

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
