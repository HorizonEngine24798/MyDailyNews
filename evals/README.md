# Evaluation data

The versioned corpus lives under `cases/`. See
[`docs/evaluation.md`](../docs/evaluation.md) for commands, labels, metrics, and
the anti-leak design.

When adding an arc:

1. Use at least two chronological days unless the trap is inherently same-day.
2. Keep source documents free of canonical IDs and expected labels.
3. Add required and forbidden facts to the arc's fact catalog.
4. Include both a plausible positive and a confusing negative when possible.
5. Tag the failure mechanism, not the expected answer.
6. Run the oracle and fault-injection tests before accepting the case.
7. Include unrelated-only days and occasional `documents: []` days in long
   arcs; real feeds contain both noise and silence.
8. A source-empty day must also have `expectations: []`. Use the
   `hallucinate_quiet_days` fault to prove that invented output is detected.

Do not treat the committed `holdout` label as secrecy. It is only a stable
regression slice.
