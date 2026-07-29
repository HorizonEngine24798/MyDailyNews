# Managed llama-server Mode

Most users should skip this page. MyDailyNews defaults to an already-running OpenAI-compatible server such as LM Studio at `http://127.0.0.1:1234/v1`.

Use managed mode only when you want MyDailyNews to start and stop a local `llama-server` process.

## Install llama.cpp

Install `llama-server` from the upstream llama.cpp project:

- llama.cpp README: <https://github.com/ggml-org/llama.cpp>
- llama.cpp releases: <https://github.com/ggml-org/llama.cpp/releases>

After installing, verify the binary:

```bash
llama-server --version
```

If it is not on `PATH`, use the full path in `server_executable`.

## Config

Set both `ai_summary` and `ai_final` to managed mode. Use the same `base_url`, executable, model path, and server arguments when both roles should share one process.

```json
{
  "manage_server": true,
  "base_url": "http://127.0.0.1:8080/v1",
  "server_executable": "PATH/TO/llama-server",
  "server_model_path": "PATH/TO/model.gguf",
  "server_arguments": ["--no-webui", "--reasoning", "off", "-ngl", "999", "-c", "16384", "-np", "1"],
  "server_spec_default": true,
  "server_auto_stop": true
}
```

Runtime checks require `server_executable` and `server_model_path` when `manage_server=true`. `server_spec_default` defaults to `true`, enabling llama.cpp's draftless `ngram-mod` speculative decoding; set it to `false` if an older server does not support the flag or benchmarking shows a regression.

## Launch Command

Once managed mode is set in `config.local.json`, autoconfig can print the command MyDailyNews would use:

```bash
python tools/autoconfig.py --config config.local.json --write config.recommended.json --print-launch-command --no-server-probe
```

If the output says `External model server`, the written config is still in external-server mode.

## External Mode

External mode is the normal path:

```json
{
  "manage_server": false,
  "base_url": "http://127.0.0.1:1234/v1",
  "server_model": "your-loaded-model-label"
}
```

Use external mode for LM Studio, Docker, an already-running `llama-server`, or any compatible local endpoint.
