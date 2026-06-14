# Hardware Profiles

MyDailyNews is model-agnostic at the pipeline boundary, but it is not automatically hardware-agnostic. Model size, quantization, context size, GPU offload, KV cache, prompt limits, and batch sizes must fit together.

Use:

```powershell
python tools/autoconfig.py --config config.local.json --write config.recommended.json
```

Autoconfig detects hardware best-effort, recommends a Qwen-family GGUF model from `profiles/model_catalog.json`, optionally prompts to download it, probes llama.cpp when possible, and writes a recommended config.

## Tiers

| Profile | Model class | Context | Notes |
| --- | --- | --- | --- |
| CPU small | 4B Q4 | 4k | Slow, useful for smoke runs |
| NVIDIA 8 GB | 8B Q4 | 8k | Small batches and article caps |
| NVIDIA 12-16 GB | 14B Q4 | 16k | Moderate default for consumer GPUs |
| NVIDIA 20-24 GB | 30B-A3B Q4 | 32k | Higher quality, still probe first |
| Remote server | user managed | server dependent | Use `manage_server=false` |

## Symptoms Of Oversizing

- llama.cpp fails during model load
- startup hangs or times out
- CUDA, Vulkan, Metal, or ROCm memory errors
- very slow partial CPU offload
- request timeouts on large prompts
- malformed JSON from overloaded prompts
- final brief pruning many selected articles

Reduce the model class, context window, selected article caps, and batch sizes together.
