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
   historical baseline. A story omitted from today's prose still adds durable
   evidence for tomorrow without deactivating earlier facts.

Identity, fact comparison, and editorial selection are deliberately separate
decisions. The identity model cannot invent a historical key, the operation
model cannot choose whether a story is published, and the editor cannot retire
facts.

## Failure policy

Uncertain, malformed, missing, over-budget, or failed model output fails open:
the affected story remains visible. Only a validated `story-cards.v1` editor
decision may defer a story.

Validated `replace` and `resolve` operations record the model-proposed
relationship, but do not deactivate the cited prior facts. Current non-repeat
facts are added to `StoryStore.active_fact_ids`, leaving the evidence history
append-only and avoiding destructive semantic mutation.

## Semantic boundary

The operation model owns the semantic classification. The deterministic
validator checks only response shape, supplied evidence IDs, story ownership,
conflicts, and complete coverage. It does not use English keyword lists to
approve or veto semantic relationships. Model output remains stochastic, so
malformed or `uncertain` results fail open and no operation automatically
deactivates historical evidence.

## Deliberate non-features

The production path has no broad delta extractor, parallel evaluator, replay
harness, semantic reranker, knowledge graph, persisted 5W1H grid, OpenIE layer,
or mandatory atomic-fact extraction pass. Those alternatives added duplicate
contracts or more failure surfaces without being necessary for the selected
workflow.
