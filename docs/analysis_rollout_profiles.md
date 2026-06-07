# Analysis Rollout Profiles (PR07)

This note documents the runtime-safe rollout presets for optional analysis stages.

## Profiles

### `safe_local` (recommended for low VRAM)
- Target: 4 GB to 8 GB VRAM.
- `general`: evidence OFF, delta OFF.
- `detailed`: evidence ON, delta OFF.
- Conservative caps: lower `max_input_tokens`, `max_new_tokens`, `max_articles`, and `max_article_chars`.

### `balanced_local` (recommended for mid VRAM)
- Target: 8 GB to 12 GB VRAM.
- `general`: evidence OFF, delta OFF.
- `detailed`: evidence ON, delta ON.
- Moderate caps and guarded compaction fallback.

### `quality_focused` (recommended for high VRAM)
- Target: 16 GB+ VRAM.
- `general`: evidence ON, delta OFF.
- `detailed`: evidence ON, delta ON.
- Higher caps while still preserving non-fatal fallback behavior.

## Rollout Guidance

1. Phase A: run `safe_local` first and validate latency/guardrails.
2. Phase B: move to `balanced_local` when detailed-mode runtime is stable.
3. Phase C: use `quality_focused` only if hardware headroom is consistent.

## Safety Rules

1. Keep `general` fast by default.
2. Preserve deterministic delta scaffold fallback whenever delta output is empty or stage is skipped.
3. Treat analysis stage failures as non-fatal and continue brief generation.
