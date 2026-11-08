# Docker & Turnkey Setup

MyDailyNews can run completely inside Docker containers. This eliminates the need to install Python, PyTorch, C++ compilers, or system audio libraries on your host machine.

---

## 1. Quick Start Options

Pick the option that fits your setup:

### Option A: Zero-GPU Cloud Mode (Easiest & Fastest)
You don't need a local GPU or local model weights. You can connect MyDailyNews to OpenAI, OpenRouter, Groq, Mistral, or any OpenAI-compatible provider:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your provider details:
   ```env
   MYDAILYNEWS_AI_BASE_URL=https://openrouter.ai/api/v1
   MYDAILYNEWS_AI_BACKEND=openai_compatible
   MYDAILYNEWS_AI_API_KEY=sk-or-v1-...
   MYDAILYNEWS_AI_MODEL=qwen/qwen-2.5-72b-instruct
   ```
3. Start the GUI:
   ```bash
   docker compose up gui
   ```
4. Open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

---

### Option B: Turnkey Local LLM (Bundled Ollama)
If you want to run everything locally without installing anything on your host OS:

1. Start both MyDailyNews and Ollama in Docker:
   ```bash
   docker compose --profile local-llm up
   ```
2. Pull your desired model into the Ollama container (in a separate terminal):
   ```bash
   docker exec -it mydailynews-ollama ollama pull qwen2.5:7b
   ```
3. Open [http://127.0.0.1:8765](http://127.0.0.1:8765) and configure your model name.

---

### Option C: Connect to Host `llama-server` or LM Studio
If you already run `llama-server` (`llama.cpp`) or LM Studio on your host machine:

1. Start your host model server listening on port `8080` (e.g. `http://127.0.0.1:8080`).
2. Run the GUI container:
   ```bash
   docker compose up gui
   ```
   *Docker Compose automatically routes requests to `http://host.docker.internal:8080/v1` to reach the host.*

---

## 2. Running CLI Commands in Docker

To execute CLI pipelines rather than the GUI:

```bash
# Run full default pipeline
docker compose run --rm app

# Run a specific module (e.g. briefs only)
docker compose run --rm app python main.py --module briefs

# Run diagnostic tests inside the container
docker compose run --rm app python tools/run_story_retrieval_diagnostics.py
```

---

## 3. How the Docker Setup Works

If you are new to Docker, here is how the pieces fit together:

- **Image (`mydailynews:local`)**: Built from `Dockerfile`. Contains Debian Linux, Python 3.12, system audio libraries (`libsndfile1`), and all Python dependencies in `requirements.txt`.
- **Containers**:
  - `mydailynews-gui`: The web application and HTTP server listening on port `8765`.
  - `mydailynews-cli`: One-off tasks that exit when finished.
  - `mydailynews-ollama`: (Optional) Local model server container.
- **Bind Mount (`.:/app`)**: Your local project directory is mounted inside the container. Generated briefs in `output/`, memory in `state/`, and cache in `.cache/` are written directly to your host machine's drive, so nothing is lost when the container stops.
- **Port Mapping (`8765:8765`)**: Forwards port 8765 from inside the container to `http://127.0.0.1:8765` on your computer.
- **Auto-Seeding Entrypoint (`docker-entrypoint.sh`)**: If you don't have a `config.local.json` or `config.recommended.json` yet, the container automatically creates `config.local.json` from `config.example.json` so you never get a startup error on a fresh clone.
- **Model Downloads (`MYDAILYNEWS_ALLOW_MODEL_DOWNLOAD=1`)**: The container enables downloading the multilingual embedding model (`intfloat/multilingual-e5-small`) on first use if it is not already cached locally.
