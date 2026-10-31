# Focused story-quality checks

The superseded broad-delta evaluator has been removed with `DeltaExtractor`.
Current checks use the shared offline corpus only for narrow, independently
scored questions.

## Retrieval diagnostic

Measure whether production story retrieval supplies the correct bounded set of
historical candidates and their facts for fact-operation comparison:

```powershell
python tools/run_story_retrieval_diagnostics.py
```

The report includes recall at bounded candidate counts. It measures candidate
recall only; it does not measure fact-operation correctness, editorial
selection, or final-brief quality.

The corpora under `evals/cases/` contain synthetic and blind real-news cases.
Gold story IDs, labels, and fact expectations are scorer data and must never be
included in model input.

## Test suite

Production behavior is covered directly by the story-delta editor, claim-delta,
story-memory, retrieval, and brief-integration tests. This avoids retaining a
second end-to-end implementation solely for evaluation.
