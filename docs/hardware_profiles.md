# Hardware Profiles

MyDailyNews is model-agnostic at the pipeline boundary, but it is not automatically hardware-agnostic. Model size, quantization, context size, GPU offload, KV cache, prompt limits, and batch sizes must fit together.

Use:

```bash
python tools/autoconfig.py --config config.local.json --write config.recommended.json
```

Autoconfig detects hardware best-effort, recommends a Qwen-family GGUF model from `profiles/model_catalog.json`, keeps the app in external-server mode for LM Studio, probes the configured endpoint when possible, and writes a recommended config.

## Tiers

| Profile | Model class | Context | Notes |
| --- | --- | --- | --- |
| CPU small | 4B Q4 | 4k | Slow, useful for smoke runs |
| NVIDIA 8 GB | 8B Q4 | 8k | Small batches and article caps |
| NVIDIA 12-16 GB | 14B Q4 | 16k | Moderate default for consumer GPUs |
| NVIDIA 20-24 GB | 30B-A3B Q4 | 32k | Higher quality, still probe first |
| External server | user managed | server dependent | Default; LM Studio owns model loading |

## Symptoms Of Oversizing

- LM Studio fails during model load
- startup hangs or times out
- CUDA, Vulkan, Metal, or ROCm memory errors
- very slow partial CPU offload
- request timeouts on large prompts or high per-call output caps
- malformed JSON from overloaded prompts
- final brief pruning many selected articles

Reduce the model class, context window, AI prompt/output budgets, selected article caps, enrichment fetch/excerpt limits, and batch sizes together.
