# MyDailyNews

Local news briefing pipeline. It fetches headlines, scores and summarizes them through a local OpenAI-compatible server such as LM Studio, then writes Markdown/JSON briefs. Optional TTS turns any saved Markdown brief into a WAV file.

Use Git Bash for the commands below.

## First Setup

```bash
cd /d/Project/MyDailyNews
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
test -f config.local.json || cp config.example.json config.local.json
notepad config.local.json
```

Start LM Studio's local server, reachable at:

```text
http://127.0.0.1:1234
```

Then write the tuned config. Autoconfig still detects your hardware and writes
matching token/batch budgets, but MyDailyNews does not start or stop the model
server:

```bash
python tools/autoconfig.py --config config.local.json --write config.recommended.json
```

Autoconfig asks what default run shape you want, including whether to include TTS audio.

## Run It

```bash
cd /d/Project/MyDailyNews
source .venv/Scripts/activate
python main.py --config config.recommended.json
```

Outputs land in `output/`, usually like:

```text
output/YYYY-MM-DD_general_brief.md
output/YYYY-MM-DD_general_brief.json
output/YYYY-MM-DD_detailed_brief.md
output/YYYY-MM-DD_detailed_brief.json
output/YYYY-MM-DD_enrichment.md
output/YYYY-MM-DD_enrichment.json
output/YYYY-MM-DD_narrative_brief.md
output/YYYY-MM-DD_narrative_brief.json
output/<input-markdown-stem>.wav
output/<input-markdown-stem>_audio.json
```

## Run The GUI

```bash
cd /d/Project/MyDailyNews
source .venv/Scripts/activate
python gui.py --config config.recommended.json --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Docker

```bash
docker compose build
docker compose run --rm app
docker compose up gui
```

Docker sets `MYDAILYNEWS_AI_BASE_URL=http://host.docker.internal:1234/v1`
inside the container so it can reach LM Studio on the host. See `docs/docker.md`.

## Useful Commands

```bash
python main.py --config config.recommended.json --module briefs
python main.py --config config.recommended.json --module enrichment --date "$(date +%F)"
python main.py --config config.recommended.json --module narrative_brief --date "$(date +%F)"
python main.py --config config.recommended.json --debug
python main.py --config config.recommended.json --list-stages
python main.py --config config.recommended.json --memory inspect
python main.py --config config.recommended.json --memory prune
```

## TTS Audio
Kokoro needs Python 3.10 through 3.12. If your main `.venv` uses one of those versions, TTS is already installed by the first setup command:

```bash
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

If your main Python is newer, use a Python 3.12 environment for runs that need TTS:

```bash
conda create -n mydailynews-tts python=3.12 -y
conda activate mydailynews-tts
python -m pip install -r requirements.txt
```

Run TTS by itself against any saved Markdown brief:

```bash
cd /d/Project/MyDailyNews
source .venv/Scripts/activate
python main.py --config config.local.json --module tts --markdown-path output/YYYY-MM-DD_general_brief.md
```

If you use the separate Conda TTS environment:

```bash
cd /d/Project/MyDailyNews
conda activate mydailynews-tts
python main.py --config config.local.json --module tts --markdown-path output/YYYY-MM-DD_general_brief.md
```

## Config Defaults

`config.example.json` keeps TTS off. To include audio in normal runs, set `tts.enabled=true` and add `tts` after `narrative_brief` in `pipeline.default_series`.

To turn TTS off:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("config.local.json")
c = json.loads(p.read_text())
c["tts"]["enabled"] = False
c["pipeline"]["default_series"] = [m for m in c["pipeline"]["default_series"] if m != "tts"]
p.write_text(json.dumps(c, indent=2) + "\n")
PY
```

## Test

```bash
cd /d/Project/MyDailyNews
source .venv/Scripts/activate
python -B -m unittest discover -s tests
```

## Troubleshooting

Validate JSON:

```bash
python -m json.tool config.local.json >/tmp/mydailynews-config.json
```

Confirm the configured model server URL:

```bash
python tools/autoconfig.py --config config.local.json --write config.recommended.json --print-launch-command --no-server-probe
```

It should print `External model server: http://127.0.0.1:1234/v1`.

## License

MIT. See `LICENSE`.
