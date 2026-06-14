# llama.cpp Setup

MyDailyNews expects a local `llama-server` binary and a local GGUF model file.

## Install Options

Windows:

```powershell
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

```powershell
llama-server --version
```

If `llama-server` is not on `PATH`, set `server_executable` in `config.local.json` to the full path.

## Managed Server Mode

With `manage_server=true`, MyDailyNews starts `llama-server`, waits for the OpenAI-compatible endpoint, reuses the same process for summary and final model roles, and stops it when done if `server_auto_stop=true`.

The effective command is:

```text
llama-server -m PATH/TO/model.gguf --host 127.0.0.1 --port 8080 --no-webui --reasoning off -ngl 999 -c 16384 -np 1
```

Run autoconfig to print the exact command for your config:

```powershell
python tools/autoconfig.py --config config.local.json --write config.recommended.json --print-launch-command
```

## External Server Mode

If you already run a compatible server, use `profiles/config.remote-server.example.json` as a starting point and set:

```json
{
  "manage_server": false,
  "base_url": "http://127.0.0.1:8080/v1",
  "server_model": "your-loaded-model-label"
}
```

This is useful for LM Studio, an already-running llama.cpp server, or another local OpenAI-compatible server. It remains a secondary path for v1.
