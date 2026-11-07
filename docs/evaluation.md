# Story-quality & retrieval evaluation

Offline evaluation uses synthetic and 75-pair blind real-news corpora under `evals/` (`change_monitoring.real_blind.v1`). Gold labels are scorer-only and never enter prompts.

## Commands

```powershell
# Hybrid retrieval benchmark (lexical + multilingual-e5-small, no LLM needed)
python tools/run_story_retrieval_diagnostics.py
```

## Reference Metrics (2026-11-05)

- **Stage 0 Retrieval (`change_monitoring.real_blind.v1`)**: `Recall@1 = 0.9333` (`70/75`), `Recall@3 = 0.9867` (`74/75`).
- **Unit Test Suite**: 277 tests across `tests/test_*.py`.
