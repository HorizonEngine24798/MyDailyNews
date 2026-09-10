# Focused story-quality checks

The superseded broad-delta evaluator has been removed with `DeltaExtractor`.
Current checks use the shared offline corpus only for narrow, independently
scored questions.

## Retrieval diagnostic

Measure whether production story retrieval supplies the correct historical
candidate before asking an LLM to decide identity:

```powershell
python tools/run_story_retrieval_diagnostics.py
```

The report includes recall at bounded candidate counts. It does not measure
delta correctness or final-brief quality.

## Experimental operation checks

The structured-delta and daily-story experiments are documented in
[story-experiments.md](story-experiments.md). They test narrow fact-operation
contracts and are not alternate production implementations.

The corpora under `evals/cases/` contain synthetic and blind real-news cases.
Gold story IDs, labels, and fact expectations are scorer data and must never be
included in model input.

## Test suite

Production behavior is covered directly by the story-delta editor, claim-delta,
story-memory, retrieval, and brief-integration tests. This avoids retaining a
second end-to-end implementation solely for evaluation.
