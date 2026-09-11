# Story change analysis

Updated: 2026-09-11

## Production design

Delta analysis operates on a same-day story group, not on each article in
isolation:

1. Group today's articles that describe the same concrete story.
2. Retrieve at most three plausible historical story candidates.
3. Ask the analysis model for one operation per current fact against all
   bounded retrieved facts: `add`, `repeat`, `replace`, `resolve`, or
   `uncertain`. There is no separate same-story/new-story verdict.
4. Reject the entire operation response when IDs are invented, a cited fact has
   ambiguous ownership, cited facts span multiple historical stories,
   operations conflict, or current facts are left uncovered.
5. Reuse a historical story key only when a valid operation response cites
   facts owned by exactly one prior story. An `add` may cite one such fact as a
   continuity anchor even though the current proposition is new. Otherwise
   retain the current grouped story key and fail open when the response is
   unsafe.
6. Send the bounded story cards to one editor model. It chooses
   `full_report`, `continuing_bullet`, or `omit`; no more than five stories can
   receive full-report treatment in one brief.
7. Persist every selected story after all briefs have analyzed the same frozen
   historical baseline. A story omitted from today's prose still adds durable
   evidence for tomorrow without deactivating earlier facts.

Fact comparison and editorial selection are deliberately separate decisions.
Historical retrieval and story-key reuse are deterministic, the operation model
cannot choose whether a story is published, and the editor cannot alter or
retire facts.

## Failure policy

Uncertain, malformed, missing, over-budget, or failed model output fails open:
the affected story remains visible. Only a validated `story-cards.v2` editor
decision may defer a story.

Validated `replace` and `resolve` operations record the model-proposed
relationship, but do not deactivate the cited prior facts. Current non-repeat
facts are added to `StoryStore.active_fact_ids`, leaving the evidence history
append-only and avoiding destructive semantic mutation.

## Semantic boundary

The operation model owns the semantic classification. The deterministic
validator checks only response shape, supplied evidence IDs, unambiguous story
ownership, single-story historical linkage, conflicts, and complete coverage.
It does not use English keyword lists to approve or veto semantic
relationships. Model output remains stochastic, so malformed or `uncertain`
results fail open and no operation automatically deactivates historical
evidence.

## Deliberate non-features

The production path has no broad delta extractor, parallel evaluator, replay
harness, semantic reranker, knowledge graph, persisted 5W1H grid, OpenIE layer,
or mandatory atomic-fact extraction pass. Those alternatives added duplicate
contracts or more failure surfaces without being necessary for the selected
workflow.
