# Story identity and change analysis

Updated: 2026-09-11

## Production design

Delta analysis operates on a same-day story group, not on each article in
isolation:

1. Group today's articles that describe the same concrete story.
2. Retrieve at most three plausible historical story candidates.
3. Ask the identity model to select exactly one supplied prior story, select no
   prior story, or return `uncertain`.
4. If a prior story was selected, ask the analysis model for one operation per
   current fact: `add`, `repeat`, `replace`, `resolve`, or `uncertain`.
5. Reject the entire operation response when IDs are invented, facts cross
   story boundaries, operations conflict, or current facts are left uncovered.
6. Send the bounded story cards to one editor model. It chooses
   `full_report`, `continuing_bullet`, or `omit`; no more than five stories can
   receive full-report treatment in one brief.
7. Persist every selected story after all briefs have analyzed the same frozen
   historical baseline. A story omitted from today's prose still advances
   durable evidence and active-fact state for tomorrow.

Identity, fact comparison, and editorial selection are deliberately separate
decisions. The identity model cannot invent a historical key, the operation
model cannot choose whether a story is published, and the editor cannot retire
facts.

## Failure policy

Uncertain, malformed, missing, over-budget, or failed model output fails open:
the affected story remains visible. Only a validated `story-cards.v1` editor
decision may defer a story.

Validated `replace` and `resolve` operations update `StoryStore.active_fact_ids`.
The cited prior facts remain in the bounded evidence history for audit and
retrieval diagnostics, but they are excluded from the next active baseline.
This also applies to an editor-omitted story, so the next run compares against
the newest known state rather than the last state shown to the reader.

## Known semantic weakness

The deterministic validator is strong on structure but weak on meaning. Its
final veto for `replace` and `resolve` uses English marker regular expressions.
A marker in an unrelated or negated clause can wrongly authorize retirement,
while valid synonyms and non-English text can be rejected. False positives are
the dangerous case because they can remove the wrong prior fact from active
memory; false negatives merely keep a story visible and retain stale facts.

Do not expand the keyword lists. The intended follow-up is an independent,
batched semantic verifier over only the proposed destructive prior/current
fact tuples. It should return `confirm`, `reject`, or `uncertain`, and any error,
disagreement, or uncertainty should fail open. The current regex behavior is
retained until that verifier is designed and evaluated.

## Deliberate non-features

The production path has no broad delta extractor, parallel evaluator, replay
harness, semantic reranker, knowledge graph, persisted 5W1H grid, OpenIE layer,
or mandatory atomic-fact extraction pass. Those alternatives added duplicate
contracts or more failure surfaces without being necessary for the selected
workflow.
