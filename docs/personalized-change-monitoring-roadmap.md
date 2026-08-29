# Personalized Change-Monitoring Roadmap

## Product promise

MyDailyNews is a personalized change-monitoring system, not a general news
aggregator. A user defines a profile, receives briefings, and can signal what
was useful or what should remain under watch. The system should report an
ongoing story again only when there is a specific, material change.

```text
User profile + lightweight feedback
            ↓
Story memory: what the user has already seen
            ↓
New evidence: what materially changed
            ↓
Personalized, source-bounded brief
            ↓
Useful / keep-watch signals
            ↺
```

## Existing foundations

- Editable user profile and deterministic profile-aware candidate ranking.
- Local coverage memory, story keys/families, story index, and recall packets.
- Recent-story penalties, material-update boosts, and story-family caps.
- Prior-report context and a delta-extraction stage.
- File-backed feedback, a GUI/API path, and learned topic/source preferences.
- Isolated CPU-model profiles and tests for small local models.

## Current gaps

- A material update is still inferred too broadly by model output rather than
  demonstrated as a comparison between old and new source-backed facts.
- Story identity and recall are not yet a reliable fact-level state ledger.
- The active local configuration has feedback disabled and no feedback events
  have been collected.
- The GUI exposes four feedback actions, which is too much friction for daily
  use.
- Learned preferences mostly affect broad topic/source ranking, not a durable
  watched story or entity.
- The small model is asked to do too much synthesis before deterministic
  evidence and delta logic have narrowed the task.

## Implementation status (2026-08-29)

- Done: versioned offline evaluation schema, 15 multi-day adversarial arcs, 74
  documents, 50 simulated days (including four source-empty days), trap/split
  metrics, oracle controls, fault injection, and fake RSS retrieval through the
  real parser.
- Done: local-model delta adapter with latency, token, retry, and throughput
  diagnostics.
- Done: removed news-specific regex verb lists and model-angle whitelists from
  deterministic materiality decisions; lexical fallback now suppresses only a
  confirmed duplicate and otherwise fails open.
- Done: enforced safe `omit` decisions before final generation and stopped
  omitted or prompt-dropped stories from being recorded as user-visible
  coverage or sent to enrichment; suppressed source, evidence, recall, and
  prior-report prose are also removed from writer-facing context.
- Done: generalized lexical identity and retrieval to Unicode and numeric
  signals, removed topic-specific weak-token lists from control decisions, and
  added an arbitrary elf/toenail-magic regression arc.
- Done: added a decision-only delta contract for constrained local models. It
  removed full-schema truncation in the Qwen3 1.7B CPU run (56/56 valid
  requests), but the 74-document benchmark still scored only 0.6234 identity
  F1, 0.2297 delta accuracy, and 0.3649 display accuracy in 32m 27.5s.
- Revealed: unrelated daily noise is not being removed at the evaluated stage.
  The local-delta adapter bypasses production headline selection, while the
  deterministic relevance fallback marked 28 of 29 irrelevant items eligible.
  The noise-heavy arcs had only 0.0968 display accuracy.
- Next: add adjudicated real-source snapshots and claim-to-fact annotation so
  final-prose faithfulness can be measured rather than left unavailable.
- Next: replace the story index with a source-backed fact/state ledger, then
  implement the two-button Useful / Keep watching interaction against it.
- Next: add a full-pipeline evaluation adapter (or a separate ranking contract)
  so profile selection is measured before delta classification; keep claim
  faithfulness unavailable until mapped source facts are emitted.

## Product rule

For a previously shown story, the pipeline must identify a new fact, status
change, contradiction, resolution, or relevance change. Otherwise it suppresses
the story or renders at most a compact continuing-watch item.

## Target story ledger

Each durable story should keep a compact, inspectable state record:

```text
Story family: AI export controls / Taiwan / China
Last shown: 2026-08-26
Last user-visible facts:
- Taiwan charged nine people over alleged server exports.
- Status moved from investigation to charges.

New source facts:
- A court hearing date was announced.
- No change to the allegations or export-control policy.

Decision:
- Incremental procedural update → suppress or one-line watch item.
```

The decision should be structured before generation:

```text
new | escalated | policy/status change | contradiction | resolution |
incremental | no material change | insufficient evidence
```

## Feedback design

Feedback belongs inline with each story and should expose only two actions:

- **Useful**: this item was worth showing.
- **Keep watching**: track this story and surface it again only on material
  change.

No interaction is not negative feedback. It is too ambiguous for a daily
product. Later, reading/click behavior can be a weak supplemental signal.

| Signal | Immediate effect | Longer-term effect |
| --- | --- | --- |
| Useful | Small temporary rank boost for the story family | Modest updates to matching topic/entity/source preferences |
| Keep watching | Creates or refreshes watched-story state | Preserves a state baseline and lowers the threshold for verified material updates |
| No action | No direct change | None; never infer dislike from silence |

## Delivery order

1. Build a labelled evaluation set and report its metrics before changing the
   pipeline.
2. Implement source-bounded story state and material-delta classification.
3. Enforce the product rule: no material delta, no full repeated story.
4. Replace the visible feedback controls with Useful and Keep watching; enable
   feedback in the active configuration.
5. Add watched-story/entity effects to ranking and delta thresholds.
6. Use the local model only to explain a structured, evidence-backed delta in a
   short story card.
7. Reserve frontier models for optional deep synthesis or periodic quality
   review, not basic repeat suppression.

## CPU-model role

A small model should receive a short, bounded task:

> Using only these cited old and new facts, explain what changed, why it matters
> to this profile, and what remains uncertain.

It should not independently retrieve the story, decide story identity, infer
novelty from a large corpus, or decide whether an unsupported claim is true.

## Definition of success

- Most repeated stories are suppressed unless a human label agrees that a
  material change occurred.
- Every displayed repeat names the change explicitly.
- The user increasingly marks displayed items Useful or Keep watching.
- The daily CPU brief remains concise, source-bounded, and runs in a predictable
  time budget.
