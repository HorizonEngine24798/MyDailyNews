# MyDailyNews

MyDailyNews is a local-first news briefing pipeline. It collects headlines from RSS and Google News RSS, deduplicates and clusters them, asks a local llama.cpp model to score candidates, retrieves and enriches selected articles, then writes Markdown and JSON daily briefs.

The supported public runtime is llama.cpp through its OpenAI-compatible `llama-server`. MyDailyNews can spawn, probe, reuse, and stop that server when `manage_server=true`.

## Quick Start

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install or build llama.cpp, then create a local config:

```powershell
copy config.example.json config.local.json
python tools/autoconfig.py --config config.local.json --write config.recommended.json
python main.py --config config.recommended.json
```

Useful run options:

```powershell
python main.py --brief general
python main.py --brief detailed
python main.py --no-enrichment
python main.py --debug
python main.py --list-stages
```

`python main.py` looks for `config.local.json` by default. Local configs, recommended configs, downloaded models, output, logs, and caches are intentionally ignored by git.

## llama.cpp Runtime

The supported backend value is:

```json
{
  "backend": "llama_cpp_server"
}
```

Managed mode is the default public path:

```json
{
  "backend": "llama_cpp_server",
  "base_url": "http://127.0.0.1:8080/v1",
  "manage_server": true,
  "server_executable": "PATH/TO/llama-server",
  "server_model_path": "PATH/TO/model.gguf",
  "server_arguments": ["--no-webui", "--reasoning", "off", "-ngl", "999", "-c", "16384", "-np", "1"],
  "server_auto_stop": true
}
```

`server_executable` is your local `llama-server` binary. `server_model_path` is the local GGUF model file. Keep `max_input_tokens + max_new_tokens` below the context window passed with `-c`.

See [llama.cpp setup](docs/llama_cpp_setup.md) and [configuration](docs/configuration.md) for platform notes.

## Hardware Profiles

There is no single universal config. Start with `config.example.json`, then let autoconfig choose a conservative tier from `profiles/model_catalog.json`.

Committed examples:

```text
profiles/config.cpu-small.example.json
profiles/config.nvidia-8gb.example.json
profiles/config.nvidia-12gb.example.json
profiles/config.nvidia-24gb.example.json
profiles/config.remote-server.example.json
```

The catalog recommends Qwen-family quantized GGUF models and can prompt interactive users to download one into ignored `models/`.

See [hardware profiles](docs/hardware_profiles.md) for the tuning model.

## Pipeline

```text
config
-> fetch prior reports
-> fetch RSS headlines
-> fetch Google News RSS topic headlines
-> merge duplicate URLs
-> dedupe similar titles
-> annotate event clusters
-> score candidates with local llama.cpp
-> select articles deterministically
-> fetch article text
-> enrich selected stories
-> optionally distill evidence
-> optionally extract narrative deltas
-> generate final brief JSON
-> write Markdown and JSON
```

The model is not the pipeline controller. Python owns fetching, dedupe, candidate limits, deterministic selection, fallback scaffolds, output normalization, diagnostics, and cache behavior.

## Output And Diagnostics

Briefs are written under `output/`:

```text
output/YYYY-MM-DD_general_brief.md
output/YYYY-MM-DD_general_brief.json
output/YYYY-MM-DD_detailed_brief.md
output/YYYY-MM-DD_detailed_brief.json
```

Managed server logs are written under:

```text
output/diagnostics/llama_server/
```

Stage artifacts are written when `--stop-after-stage`, `--dump-stage-artifacts`, or `--save-intermediate` is used.

## Configuration

Main sections:

- `ai_summary`: local llama.cpp client for headline scoring and summary-role analysis.
- `ai_final`: local llama.cpp client for final brief synthesis.
- `general_topics`: broad-topic query definitions.
- `topics_to_examine`: focused-topic query definitions.
- `general_filtering` and `filtering`: candidate limits, selection caps, and prompt budgets.
- `user_memory`: reader preferences and briefing style.
- `sources`: RSS feeds, Google News RSS settings, and prior-report lookup.
- `analysis`: evidence and delta stage settings.
- `cache`: HTTP, article text, enrichment, Wikipedia, and AI cache settings.

See [configuration](docs/configuration.md) for details.

## Tests

Run the maintained no-GPU/no-network suite:

```powershell
python -B -m unittest discover -s tests
python -B main.py --list-stages
```

Optional llama.cpp probing belongs in local autoconfig runs, not public CI.

## Troubleshooting

Most runtime failures are config or hardware-fit issues: missing `llama-server`, missing GGUF model path, context/token mismatch, server startup timeout, or invalid JSON from an overloaded model.

Start with [troubleshooting](docs/troubleshooting.md), then inspect `output/diagnostics/llama_server/`.

## Attribution

- Horizon inspiration: https://github.com/Thysrael/Horizon. MyDailyNews uses an original local-first implementation, with RSS normalization and the staged fetch/dedupe/score/enrich/summarize pipeline shape adapted from or substantially inspired by Horizon. See `LICENSE` for the retained Horizon MIT notice.
- Qwen public model organization: https://huggingface.co/Qwen
- License: MIT
