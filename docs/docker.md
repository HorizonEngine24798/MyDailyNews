# Docker

This Docker setup runs the Python app in a container. It does not run a model
server in the container; keep LM Studio's local server on the host machine and
point the app at it.

## Build

From the repo root:

```bash
docker compose build
```

The image uses Python 3.12 because Kokoro TTS supports Python 3.10 through
3.12. The image also installs `libsndfile1` for the optional audio path.

## Config

Create your local config if you do not already have one, then write the
recommended config from the host machine:

```bash
cp config.example.json config.local.json
python tools/autoconfig.py --config config.local.json --write config.recommended.json
```

Run autoconfig on the host, not from the GUI running inside Docker. Hardware
detection inside a container may not see the same GPU resources as Windows.

For normal Windows runs, `config.recommended.json` should point both
`ai_summary` and `ai_final` at LM Studio:

```json
{
  "base_url": "http://127.0.0.1:1234/v1",
  "manage_server": false,
  "server_executable": "",
  "server_model_path": "",
  "server_auto_stop": false
}
```

Why Docker still works: inside Docker, `127.0.0.1` means the container itself.
The Compose file sets `MYDAILYNEWS_AI_BASE_URL=http://host.docker.internal:1234/v1`,
which overrides the app's AI base URL only inside the container.

## Run The Pipeline

```bash
docker compose run --rm app
```

Pass normal CLI arguments after the service name:

```bash
docker compose run --rm app python main.py --config config.recommended.json --module briefs
```

## Run The GUI

```bash
docker compose up gui
```

Open:

```text
http://127.0.0.1:8765
```

The Compose file bind-mounts the repo into `/app`, so `config.local.json`,
`config.recommended.json`, `output/`, `state/`, and `.cache/` stay on your
machine.
