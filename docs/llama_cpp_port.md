# Implementing MyDailyNews With llama.cpp

Status update:
- Backend abstraction (`AIClient` protocol + factory) is now implemented.
- `transformers` and `llama_cpp_server` adapters now exist in `mydailynews/ai/`.
- Structured per-stage JSON schemas now exist in `mydailynews/ai/schemas.py`.

This document describes how to implement the current MyDailyNews pipeline with llama.cpp while preserving the same application behavior:

```text
general brief + detailed brief
Google News RSS and RSS discovery
prior report ingestion
headline narrative triage
full article retrieval
enrichment
article-level narrative revision
final JSON and Markdown outputs
```

The short version: keep almost all Python retrieval/orchestration code, replace `mydailynews/ai/client.py` with a llama.cpp-backed client that still exposes this interface:

```python
complete_json(system: str, user: str, label: str) -> dict
```

That seam is already in the codebase.


## References Checked

Primary references used for this design:

- llama.cpp repository: https://github.com/ggml-org/llama.cpp
- llama.cpp server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp build docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- llama.cpp Android docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md
- llama-cpp-python package docs: https://pypi.org/project/llama-cpp-python/

Relevant current capabilities:

- llama.cpp uses GGUF model files.
- llama-server provides an OpenAI-compatible HTTP API.
- llama-server supports `/v1/chat/completions`.
- llama-server supports JSON output and schema-constrained JSON through `response_format`.
- llama.cpp supports grammar-based constrained generation.
- llama.cpp can be built with CUDA, Vulkan, Metal, OpenCL, and Android targets depending on device/backend.
- llama.cpp has official Android build notes using Termux or cross-compilation with the Android NDK.


## What Stays The Same

These modules should not need conceptual changes:

```text
main.py
mydailynews/config.py
mydailynews/models.py
mydailynews/orchestrator.py
mydailynews/scrapers/rss.py
mydailynews/retrieval/google_news.py
mydailynews/retrieval/reports.py
mydailynews/retrieval/article.py
mydailynews/retrieval/wikipedia.py
mydailynews/retrieval/past_news.py
mydailynews/enrichment.py
mydailynews/brief.py
mydailynews/output.py
mydailynews/debug.py
```

The retrieval pipeline is independent of the model runtime. Python still does all internet retrieval and local file ingestion.

The LLM still only receives retrieved text.

The output JSON schema stays the same.

The prompts can mostly stay the same, although llama.cpp should ideally use constrained JSON decoding instead of relying only on prompt instructions and retry.


## What Changes

The main change is this file:

```text
mydailynews/ai/client.py
```

Current client:

```text
LocalAIClient
  -> Hugging Face tokenizer/model
  -> transformers generate(...)
  -> safe_json_load(...)
```

llama.cpp client:

```text
LlamaCppClient
  -> llama-server HTTP request
  -> response_format JSON or JSON schema
  -> parse response message content
  -> safe_json_load(...)
```

or:

```text
LlamaCppPythonClient
  -> llama_cpp.Llama(...)
  -> create_chat_completion(...)
  -> response_format JSON or JSON schema
  -> safe_json_load(...)
```

The rest of the app should continue calling:

```python
client.complete_json(system, user, label="...")
```


## Recommended Desktop Path

Use `llama-server` first.

Why:

- It is closest to how a future mobile service boundary might work.
- It avoids binding Python directly to llama.cpp internals.
- It can be swapped for on-device Android inference later.
- It supports OpenAI-style chat completions.
- It supports structured JSON response formats.

### 1. Get A GGUF Model

Use a GGUF instruct/chat model with enough context for this pipeline.

Good starting points conceptually:

```text
Qwen2.5 7B Instruct GGUF Q4_K_M or Q5_K_M
Llama 3.1/3.2 8B Instruct GGUF Q4_K_M
Gemma 3 instruct GGUF if compatible and small enough
Phi-class small models for mobile experiments
```

For this pipeline, context length matters because headline triage can include:

```text
brief mode goal
reader memory
topic list
prior report summaries
up to 36-60 candidate headlines/snippets
```

Desktop target:

```text
ctx-size: 8192 to 16384
```

Phone target:

```text
ctx-size: 4096 to 8192, depending on RAM
```

### 2. Run llama-server

Example desktop command:

```powershell
llama-server.exe `
  -m D:\Models\qwen2.5-7b-instruct-q4_k_m.gguf `
  -c 16384 `
  --host 127.0.0.1 `
  --port 8080 `
  --n-gpu-layers 99
```

Notes:

- `-m` points to the GGUF model.
- `-c` controls context size.
- `--n-gpu-layers 99` tries to offload layers to GPU on supported builds.
- On CPU-only or mobile, use fewer/no GPU layers.
- On CUDA desktop, build or download a CUDA-capable llama.cpp binary.

### 3. Add AI Backend Config

Current config is Transformers-specific:

```json
"ai": {
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "device": "auto",
  "torch_dtype": "auto"
}
```

For llama.cpp, use a backend-discriminated config:

```json
"ai": {
  "backend": "llama_cpp_server",
  "base_url": "http://127.0.0.1:8080/v1",
  "model": "local-gguf",
  "max_input_tokens": 12288,
  "max_new_tokens": 1400,
  "json_retries": 1,
  "temperature": 0.0,
  "top_p": 0.9,
  "response_format": "json_schema"
}
```

Keep the existing fields if you want backward compatibility:

```json
"backend": "transformers"
```

Then load either:

```python
LocalAIClient(config.ai, debug)
```

or:

```python
LlamaCppServerClient(config.ai, debug)
```

### 4. Implement LlamaCppServerClient

New file option:

```text
mydailynews/ai/llama_cpp_client.py
```

Sketch:

```python
from __future__ import annotations

from typing import Any, Dict

import requests

from ..debug import DebugLogger
from ..models import AIConfig
from ..utils import safe_json_load
from .client import AIJsonError


class LlamaCppServerClient:
    def __init__(self, config: AIConfig, debug: DebugLogger | None = None) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.base_url = config.base_url.rstrip("/")

    def complete_json(self, system: str, user: str, label: str = "ai.complete_json") -> Dict[str, Any]:
        attempts = max(1, self.config.json_retries + 1)
        last_text = ""
        for attempt in range(1, attempts + 1):
            attempt_user = user if attempt == 1 else self._retry_user_prompt(user)
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": attempt_user},
                ],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_new_tokens,
                "response_format": {"type": "json_object"},
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            raw = response.json()
            text = raw["choices"][0]["message"]["content"]
            last_text = text or ""
            parsed = safe_json_load(last_text)
            if parsed is not None:
                return parsed
        raise AIJsonError(f"{label}: llama.cpp did not return valid JSON after {attempts} attempt(s)")
```

The real implementation should include:

- debug logs matching the current client
- HTTP error conversion into useful exceptions
- optional `response_format` JSON schema per call
- maybe health check against `/health`
- no API key needed for local llama-server unless you configure one

### 5. Use JSON Schema Instead Of JSON Mode Where Possible

Current code validates required keys after parsing.

With llama.cpp, we can push more structure into decoding by sending a JSON schema.

For example, headline triage could use:

```json
{
  "type": "object",
  "required": ["topic_narratives", "decisions"],
  "properties": {
    "topic_narratives": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["topic", "narrative", "status", "why_it_matters", "candidate_ids", "prior_report_ids"],
        "properties": {
          "topic": {"type": "string"},
          "narrative": {"type": "string"},
          "status": {"type": "string"},
          "why_it_matters": {"type": "string"},
          "candidate_ids": {"type": "array", "items": {"type": "string"}},
          "prior_report_ids": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "decisions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "topic", "score", "reason", "summary"],
        "properties": {
          "id": {"type": "string"},
          "topic": {"type": "string"},
          "score": {"type": "number"},
          "topic_relevance": {"type": "number"},
          "narrative_importance": {"type": "number"},
          "novelty": {"type": "number"},
          "source_rank": {"type": "integer"},
          "source_value": {"type": "string"},
          "narratives": {"type": "array", "items": {"type": "string"}},
          "narrative_role": {"type": "string"},
          "reason": {"type": "string"},
          "summary": {"type": "string"},
          "tags": {"type": "array", "items": {"type": "string"}},
          "duplicate_of": {"type": ["string", "null"]}
        }
      }
    }
  }
}
```

Then send:

```json
"response_format": {
  "type": "json_schema",
  "schema": { ... }
}
```

This should reduce invalid JSON retries substantially.

Practical warning: schema-constrained decoding can make small models slower or more brittle if the schema is too complex. Start with `{"type": "json_object"}` first, then add per-call schemas once basic generation is stable.


## Alternative Desktop Path: llama-cpp-python

Instead of running `llama-server`, use `llama-cpp-python` in-process.

Potential dependency:

```text
llama-cpp-python
```

Client sketch:

```python
from llama_cpp import Llama

class LlamaCppPythonClient:
    def __init__(self, config, debug=None):
        self.llm = Llama(
            model_path=config.model_path,
            n_ctx=config.max_input_tokens,
            n_gpu_layers=config.n_gpu_layers,
            chat_format=config.chat_format,
        )

    def complete_json(self, system, user, label="ai.complete_json"):
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        text = result["choices"][0]["message"]["content"]
        parsed = safe_json_load(text)
        if parsed is None:
            raise AIJsonError(...)
        return parsed
```

Pros:

- no separate server process
- simpler packaging for a pure Python desktop app
- direct control over model object lifetime

Cons:

- Python wheel/build issues can be fiddly, especially with CUDA/Vulkan
- less similar to Android app architecture
- server route is easier to test with curl/OpenAI-compatible clients

Recommendation: use `llama-server` for desktop experimentation, then direct C++/JNI or an Android wrapper later.


## Android Path

For Android, the Python codebase will not move over as-is. The pipeline should be split into:

```text
portable orchestration concepts
Android-native retrieval/storage/UI code
llama.cpp native inference backend
shared JSON contracts
```

### Android Architecture

A likely Android version:

```text
Kotlin app
  NewsRepository
    fetch RSS and Google News RSS with OkHttp
    parse feeds
    fetch selected article HTML
    extract readable text or fallback to snippets
    read previous JSON reports from app storage

  PipelineOrchestrator
    run general pass
    run detailed pass
    apply same candidate limiting and selection rules

  LlamaCppEngine
    JNI bridge to llama.cpp
    load GGUF model once
    run chat completion with JSON grammar/schema

  OutputStore
    save general/detailed JSON
    render UI
    later feed TTS
```

Keep the JSON shapes the same as the Python version. That lets us compare desktop and Android output during development.

### Android Build Options

Official llama.cpp docs describe two Android paths:

1. Termux build/run, useful for experiments.
2. Android NDK cross-compile, useful for packaging.

A production phone app probably wants the NDK path.

Basic cross-compile shape:

```bash
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -B build-android

cmake --build build-android --config Release -j8
```

For Adreno GPU experiments, llama.cpp also has OpenCL backend notes. For a first Android implementation, CPU-only with a small quantized model is simpler.

### Android Model Choice

Phone constraints are very different from the RTX 3090 desktop.

Desktop current target:

```text
7B instruct model
Q4/Q5 or FP16 depending backend
8k-16k context
```

Phone likely target:

```text
1B-4B instruct model
Q4_K_M or smaller
4k-8k context
shorter batches
fewer selected articles
more aggressive snippets
```

To keep quality acceptable on mobile:

- reduce `max_candidates_for_ai`
- reduce `max_headlines_per_ai_batch`
- reduce `article_text_max_chars`
- maybe disable enrichment by default
- run only one pass unless the phone is charging/on Wi-Fi
- use prior reports sparingly


## Migration Plan

### Phase 1: Backend Abstraction

Add a factory:

```text
mydailynews/ai/factory.py
```

```python
def create_ai_client(config, debug):
    if config.backend == "transformers":
        return LocalAIClient(config, debug)
    if config.backend == "llama_cpp_server":
        return LlamaCppServerClient(config, debug)
    if config.backend == "llama_cpp_python":
        return LlamaCppPythonClient(config, debug)
    raise ValueError(...)
```

Update `orchestrator.py`:

```python
self.ai_client = create_ai_client(config.ai, self.debug)
```

### Phase 2: Config Expansion

Extend `AIConfig`:

```python
backend: str = "transformers"
base_url: str = "http://127.0.0.1:8080/v1"
model: str = "local-gguf"
model_path: str = ""
n_gpu_layers: int = -1
chat_format: str = "chatml"
timeout_seconds: int = 300
response_format: str = "json_object"
```

Keep old fields for Transformers.

### Phase 3: Server Client

Implement:

```text
mydailynews/ai/llama_cpp_client.py
```

Start with:

```json
"response_format": {"type": "json_object"}
```

Preserve current retry behavior.

### Phase 4: JSON Schemas

Add:

```text
mydailynews/ai/schemas.py
```

Define schemas for:

```text
headline triage
article brief
enrichment planning
final brief
```

Update `complete_json(...)` to accept optional schema:

```python
complete_json(system, user, label, schema=None)
```

Callers pass schema. llama.cpp backend uses schema-constrained generation; Transformers backend ignores schema or uses prompt-only JSON until we add a local constrained decoder there.

### Phase 5: Prompt Size Optimization

llama.cpp on mobile will be context-constrained.

Add explicit compression helpers:

```text
summarize prior reports more aggressively
limit snippets per candidate
reduce topic descriptions in general pass
cap candidate metadata fields
```

Potential config:

```json
"mobile_profile": {
  "max_candidates_for_ai": 20,
  "max_selected_articles": 5,
  "article_text_max_chars": 1200,
  "prior_report_days": 3
}
```

### Phase 6: Android Prototype

First Android prototype can be ugly but useful:

```text
Termux or debug APK
manual model file path
single run button
write JSON to app storage
show rendered Markdown-like output
```

Then add:

```text
scheduled daily run
charging/Wi-Fi constraints
model manager
TTS
thumbs up/down feedback
```


## Prompt Changes For llama.cpp

Minimal changes:

```text
Keep current prompts.
Use llama.cpp JSON mode.
Keep retry-on-invalid-json.
```

Better changes:

```text
Keep current instructions.
Add JSON schema constrained decoding.
Remove some redundant "return valid JSON only" phrasing once schema mode is stable.
Shorten prompts for small mobile models.
```

Important: do not rely on model intelligence alone for JSON validity. llama.cpp's grammar/schema support is the main reason this migration is attractive.


## Performance Expectations

The pipeline has several LLM stages:

```text
2 headline triage calls, one per pass if batch size equals max candidates
up to 16 article brief calls by default: 10 general + 6 detailed
up to 16 enrichment planning calls if enrichment enabled
2 final synthesis calls
```

That is a lot for a phone.

Desktop llama.cpp can handle this with a 7B model if the GPU/backend is configured well.

Phone llama.cpp probably needs reduced defaults:

```text
general max_selected_articles: 5-7
detailed max_selected_articles: 3-4
enrichment disabled or limited
smaller model
smaller context
```

A good mobile strategy is to preserve the same architecture but run a lighter config profile.


## Concrete Desktop Config Example

```json
{
  "ai": {
    "backend": "llama_cpp_server",
    "base_url": "http://127.0.0.1:8080/v1",
    "model": "local-gguf",
    "max_input_tokens": 12288,
    "max_new_tokens": 1400,
    "json_retries": 1,
    "temperature": 0.0,
    "top_p": 0.9,
    "timeout_seconds": 300,
    "response_format": "json_object"
  }
}
```

Run server:

```powershell
llama-server.exe -m D:\Models\qwen2.5-7b-instruct-q4_k_m.gguf -c 16384 --host 127.0.0.1 --port 8080 --n-gpu-layers 99
```

Run app:

```powershell
python main.py --config config.llama.json --debug
```


## Why This Is A Good Fit

MyDailyNews already has the right boundary:

```text
retrieval/orchestration/output code
  depends on complete_json(...)
```

So llama.cpp does not require rewriting the news pipeline. It requires replacing one local inference adapter and optionally adding JSON schemas.

That is exactly the right direction for later Android deployment.
