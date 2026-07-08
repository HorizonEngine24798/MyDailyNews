# Setup

These commands assume Git Bash on Windows. On macOS or Linux, use `source .venv/bin/activate` instead of `source .venv/Scripts/activate`.

## Requirements

- Python 3.10 through 3.12. Python 3.12 is the safest default because the optional Kokoro TTS dependency supports 3.10 through 3.12.
- LM Studio, llama.cpp, or another OpenAI-compatible local server.
- A model loaded and served at `http://127.0.0.1:1234` when using the default LM Studio path.

## Install

```bash
cd MyDailyNews
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Configure

Create a local config, then let autoconfig write a tuned copy:

```bash
test -f config.local.json || cp config.example.json config.local.json
python tools/autoconfig.py --config config.local.json --write config.recommended.json
```

Autoconfig detects hardware best-effort, probes the configured endpoint when possible, and asks about the default run shape. It keeps MyDailyNews in external-server mode by default; LM Studio still owns model loading and serving.

Edit `config.local.json` first if you need different feeds, topics, or reader preferences. Run autoconfig again after changing hardware-sensitive settings.

## Run The Pipeline

Start LM Studio's local server, then run:

```bash
python main.py --config config.recommended.json
```

Outputs usually include:

```text
output/YYYY-MM-DD_general_brief.md
output/YYYY-MM-DD_general_brief.json
output/YYYY-MM-DD_detailed_brief.md
output/YYYY-MM-DD_detailed_brief.json
output/YYYY-MM-DD_enrichment.md
output/YYYY-MM-DD_enrichment.json
output/YYYY-MM-DD_narrative_brief.md
output/YYYY-MM-DD_narrative_brief.json
```

## Run The GUI

```bash
python gui.py --config config.recommended.json --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Useful Commands

```bash
python main.py --config config.recommended.json --module briefs
python main.py --config config.recommended.json --module enrichment --date "$(date +%F)"
python main.py --config config.recommended.json --module narrative_brief --date "$(date +%F)"
python main.py --config config.recommended.json --module tts --markdown-path output/YYYY-MM-DD_general_brief.md
python main.py --config config.recommended.json --debug
python main.py --config config.recommended.json --list-stages
python main.py --config config.recommended.json --memory inspect
python main.py --config config.recommended.json --memory prune
```

## Docker

Docker still expects the model server to run on the host:

```bash
docker compose build
docker compose run --rm app
docker compose up gui
```

Compose sets `MYDAILYNEWS_AI_BASE_URL=http://host.docker.internal:1234/v1` inside the container so the app can reach LM Studio on the host. See [Docker](docker.md).

## TTS

TTS is disabled by default. To include audio in normal runs, set `tts.enabled=true` and add `tts` after `narrative_brief` in `pipeline.default_series`.

Standalone TTS:

```bash
python main.py --config config.recommended.json --module tts --markdown-path output/YYYY-MM-DD_general_brief.md
```

See [TTS audio](tts.md) for backend details.

## Test

```bash
python -B -m unittest discover -s tests
```
