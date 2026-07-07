# llama.cpp Setup

MyDailyNews defaults to LM Studio's OpenAI-compatible local server at
`http://127.0.0.1:1234/v1`. Use this page only if you intentionally want the
older managed `llama-server` path.

## Install Options

Windows:

```bash
winget install llama.cpp
```

macOS:

```bash
brew install llama.cpp
```

Build from source:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build -j --target llama-server llama-cli
```

Prebuilt binaries are also available from the llama.cpp releases page.

## Verify The Binary

```bash
llama-server --version
```

If `llama-server` is not on `PATH`, set `server_executable` in `config.local.json` to the full path.

## Managed Server Mode

With `manage_server=true`, MyDailyNews starts `llama-server`, waits for the OpenAI-compatible endpoint, reuses the same process for summary and final model roles, and stops it when done if `server_auto_stop=true`. Keep `manage_server=false` when using LM Studio or Docker.

The effective command is:

```text
llama-server -m PATH/TO/model.gguf --host 127.0.0.1 --port 8080 --no-webui --reasoning off -ngl 999 -c 16384 -np 1
```

Run autoconfig to print the exact command for your config:

```bash
python tools/autoconfig.py --config config.local.json --write config.recommended.json --print-launch-command
```

## External Server Mode

The default path is an already-running compatible server such as LM Studio:

```json
{
  "manage_server": false,
  "base_url": "http://127.0.0.1:1234/v1",
  "server_model": "your-loaded-model-label"
}
```

This is useful for LM Studio, an already-running llama.cpp server, or another local OpenAI-compatible server.
