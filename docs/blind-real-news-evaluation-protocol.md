# Blind real-news evaluation protocol

Date established: 2026-08-30

## Purpose

This protocol prevents the synthetic-corpus leakage that invalidated the
phrase-based transition classifier. The real-news corpus and the replacement
semantic system are developed by isolated work tracks. Production semantic
code must be frozen before the corpus is unsealed or scored.

## Isolation and freezing

1. The corpus track owns source selection, paraphrased evidence, provenance,
   story-family assignment, labels, and corpus validation.
2. The semantic-system track may not inspect corpus documents, labels, source
   URLs, label distributions, or intermediate scores before code freeze.
3. The corpus is split by complete real-world event family, never by article
   pair. Wire copies and near duplicates may not cross splits.
4. Before the first run, record hashes for the corpus, semantic source files,
   prompts, model files, configuration, and evaluator.
5. Thresholds and abstention policy are fixed before scoring.
6. The first blind report is immutable. It is never overwritten or presented
   as blind after anyone inspects its failures.
7. Inspected failures may enter a development corpus only. A new, untouched
   story-family holdout is required for the next blind claim.

## Prohibited implementation shortcuts

- phrase or regular-expression lists that map wording to transitions;
- entity, publisher, location, topic, or story-specific branches;
- label lookup through IDs, URLs, dates, source names, or metadata;
- thresholds selected against blind labels or blind score distributions;
- treating newer claims as truth without attribution and temporal scope;
- silently forcing an uncertain semantic relation into a story merge.

Regular expressions used only for syntax-neutral operations such as sentence
boundary detection or explicit numeric/date parsing are allowed. They may not
assign semantic change labels.

## Pre-registered system contract

The replacement system operates in bounded stages:

1. high-recall candidate retrieval;
2. candidate-bounded event-coreference judgment;
3. source-span-backed claim extraction or sentence fallback;
4. semantic claim alignment and bidirectional relation inference;
5. attributed, time-scoped claim/event graph update;
6. novelty and materiality policy separated from semantic truth;
7. abstention or visible unmerged output when evidence is insufficient.

Every normalized claim must cite an exact source span. Failed or unsupported
normalization falls back to the original source sentence.

## Pre-registered reporting

Report both oracle-identity and chronological full-stream results. At minimum:

- retrieval recall@1 and recall@3;
- story identity precision, recall, F1, and overmerge rate;
- claim support precision/recall;
- claim-relation macro-F1;
- delta-type and materiality macro-F1;
- unchanged/redundant suppression precision and recall;
- right-thread plus right-novel-claim accuracy;
- abstention coverage and accuracy at retained coverage;
- per-story bootstrap confidence intervals;
- CPU latency, peak memory, context size, and stored-state growth;
- results by source, event family, time gap, language, and adversarial slice.

The headline metric is the chronological full-stream result. Oracle modes are
diagnostics and must be labelled as such.

## Interpretation

A single blind set can falsify a design but cannot establish generality. Strong
results require replication on another untouched collection period, source
mix, and set of event families. Synthetic and inspected cases remain regression
tests only.

## Run history and protocol status

The original preregistration above is preserved. The entries below document
what happened after it was established; they do not retroactively weaken its
requirements.

### Invalid aborted preflight

An initial launch was terminated before report creation or result inspection.
It produced no valid blind result. No scores, predictions, story text, labels,
or failure cases from that launch were inspected.

Its audit records are:

- `output/evaluations/semantic_thread_real_blind_first_20260830.freeze.json`
- `output/evaluations/semantic_thread_real_blind_first_20260830.abort.json`

The preflight was invalid because a harness audit found three information-flow
and replay defects:

1. evaluation categories and tags could reach retrieval;
2. separate per-family stores exposed the corpus family partition; and
3. sequential same-day writeback allowed a later document to see an earlier
   document from the same date, unlike the intended production replay.

The aborted launch must never be reported as a blind score or used to tune the
system.

### Corrected harness

Before a valid diagnostic launch, the evaluator was changed so that:

- production candidates strip evaluation category, tag, and metadata fields
  before retrieval and model inference;
- all event families are chronologically interleaved in one global StoryStore;
- every same-day batch is annotated against frozen prior-day state and written
  back only after all documents in that batch have been decided;
- oracle mode routes documents to canonical story keys rather than testing
  end-to-end retrieval; normal StoryStore lifecycle retention remained active,
  so it isolates semantics only while a prior baseline is retained;
- canonical and predicted cluster IDs are recorded only after inference and
  are never model or retrieval inputs;
- pairwise clustering is calculated globally, including false merges and false
  splits; and
- the running model basename is checked against the intended local GGUF, while
  the corpus, model, configuration, evaluator, and semantic dependencies are
  recorded by path, size, and SHA-256 hash.

### Implemented diagnostic cascade

The frozen candidate system implements the preregistered design as follows:

1. heuristic retrieval returns at most three candidate story baselines;
2. deterministic handling is limited to first observations and
   normalized-exact repetitions;
3. Qwen performs a bounded candidate-identity decision over opaque claim IDs
   and text;
4. for linked stories, AlignScore ranks the top two prior body claims for each
   current body claim using the maximum bidirectional entailment or
   contradiction probability, without a fitted threshold;
5. Qwen labels fixed claim-pair slots in bounded requests;
6. formal code aggregates atomic relations into a delta and validates all
   evidence references before the candidate identity gate; and
7. StoryStore records the bounded evidence and semantic event ledger.

AlignScore never decides story identity. Its post-hoc edge audit and NLI-gated
variant are diagnostics. The gated rows share the underlying Qwen retrieval
and writeback history and therefore do not represent an independent pipeline.
Materiality is a fixed change-type policy baseline rather than a separately
evidence-scored prediction.

### Broad blind diagnostic — completed

The authorized run completed as a broad diagnostic over the complete sealed
corpus, not as a claim that every preregistered reporting item was implemented.
Its executed scope was:

- all 18 event families and 120 documents across the sealed genre-diverse
  collection;
- both within-retention canonical-route oracle and chronological full-stream
  modes;
- structural-only and Qwen 2,048-token variants; and
- AlignScore pair ranking, edge audit, and the diagnostic NLI-gated rows.

A 4,096-token sweep was outside this diagnostic's scope. Separate inspected
development prompts fit inside the 2,048-token budget without truncation, so
the broad run prioritized event-family coverage rather than repeating a larger
context that is not expected to address semantic-label errors.

The current evaluator reports identity, relationship, delta, combined
thread-plus-delta accuracy, the semantic-required subset, accepted coverage,
abstention, confusion matrices, global pairwise clustering, category/tag
slices, model wall-clock latency, final state-file size and story count, and
AlignScore edge agreement. It does not yet calculate all preregistered
measures: retrieval recall@1/recall@3,
claim-support precision/recall, claim-relation macro-F1, suppression
precision/recall, per-story bootstrap intervals, peak memory, or complete
source/family/time-gap/language slices. Materiality is reported as secondary
accuracy over fixed policy values rather than macro-F1. These omissions must
remain explicit when interpreting the run.

### Immutable completion records

The evaluator completed at `2026-08-30T19:03:41.2990165+04:00`. The completion
record marks the output `completed_valid_broad_diagnostic`, accepts it as the
first valid blind score, and explicitly marks it as not confirmatory-protocol
complete.

- freeze:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.freeze.json`,
  2,182 bytes, SHA-256
  `ed8b5adefd92176dd0fc140b1cf1ea63ab6d1f2f97bca423c21c18eb3cde4e60`;
- completion:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.complete.json`,
  1,331 bytes, SHA-256
  `7ce6850ca03da4c32d1e1ab42f7aec66657d30acfb398bb1bbeff30d4c79bac7`;
- errata:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.errata.json`,
  1,189 bytes, SHA-256
  `3ca446d1bae9f2b24b27ab5d87c41caae3e484486c9bf434b5228ed37cc74b97`;
- JSON report:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1/report.json`,
  2,853,412 bytes, SHA-256
  `efdd1cb3b089f3ac55505c274516a33141a16546d1f6e56f68fc4af3e0bc96a4`;
  and
- Markdown report:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1/report.md`,
  1,849 bytes, SHA-256
  `67b3815eb5414a00e5a6ddc7dbf47614ea7cc386e0e474c2fc90f0a46b1b903e`.

The frozen corpus SHA-256 was
`b220a3bbe7f50c62d9e0f33cb17e238253b855a4c8918974e3fac851fdab95fa`,
and all freeze-manifest checks recorded in the completion record passed.

| Mode / variant | Documents | Identity | Relationship | Delta | Thread + delta | Semantic-required thread + delta | Pairwise cluster F1 | Accepted coverage | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle / structural only | 120 | 0.1500 | 0.1500 | 0.3500 | 0.1500 | 0.0000 | 1.0000 | 0.0000 | 0.3250 |
| Oracle / Qwen 2,048 | 120 | 0.4750 | 0.4750 | 0.3583 | 0.1583 | 0.0256 | 1.0000 | 1.0000 | 0.0083 |
| Oracle / Qwen 2,048 + diagnostic NLI gate | 120 | 0.2083 | 0.2083 | 0.3500 | 0.1500 | 0.0000 | 0.6051 | 1.0000 | 0.2750 |
| Full stream / structural only | 120 | 0.1500 | 0.1500 | 0.3417 | 0.1500 | 0.0000 | 0.0000 | 0.0000 | 0.1583 |
| Full stream / Qwen 2,048 | 120 | 0.2083 | 0.3000 | 0.3500 | 0.1500 | 0.0000 | 0.1639 | 1.0000 | 0.0083 |
| Full stream / Qwen 2,048 + diagnostic NLI gate | 120 | 0.1500 | 0.1750 | 0.3417 | 0.1500 | 0.0000 | 0.0179 | 1.0000 | 0.1417 |

Accepted coverage is evidence-contract acceptance among attempted Qwen
decisions. In an NLI-gated row it is not retained coverage after the post-hoc
gate.

### Retrieval and exact-thread result

Chronological full stream is the headline result. Retrieval classified 18 of
78 same-story continuations as `same_story` and treated 60 as new. It kept 23
of 24 related-theme hard negatives separate and linked one. Relationship
accuracy of 0.3000 therefore hides a harder exact-thread problem: only seven
of the 19 semantic attempts used the correct canonical thread. Global
pairwise clustering recorded 20 true-positive pairs, five false merges, and
199 false splits, for precision 0.8000, recall 0.0913, and F1 0.1639.

The Qwen variant did not improve the end-to-end headline metric over the
structural baseline. Both scored 0.1500 thread-plus-delta; the Qwen semantic
subset scored 0/19. Continuation delta accuracy was 1/78, but that delta was
not attached to the correct end-to-end thread. Overall delta accuracy is
inflated by non-continuation cases whose expected delta is `new`.

### Within-retention oracle result

The oracle used canonical story-key routing, but it was not a full
teacher-forced semantic comparison across all continuations. The replay used
normal `StoryStore.update_selected()` lifecycle behavior, including the
default 30-day retention window. Only 39 of 78 continuations still had a prior
canonical baseline and triggered Qwen; the other 39 appeared as first
observations after the baseline had been pruned.

For the 39 retained-prior comparisons, the expected distribution was 17
material updates, eight status changes, eight resolutions, four unchanged
reports, and two incremental updates. Qwen correctly classified one unchanged
case and none of the 35 actual changes. It predicted `correction` 22 times even
though this retained-prior subset contained no correction-labeled example.
The corpus's correction and reframing continuation cases did not receive an
oracle semantic comparison, so the run cannot isolate model quality on those
classes.

Oracle pairwise clustering F1 of 1.0000 is a consequence of forced canonical
story keys and is not model clustering evidence. Likewise, oracle identity and
relationship scores include 18 trivial new-story observations. The relevant
semantic result is 1/39 thread-plus-delta, not the constructed cluster score.

### AlignScore audit and diagnostic gate

The full-stream audit found agreement on 8 of 36 checkable claim edges
(0.2222); only 2 of 18 decisions with checkable edges had complete agreement.
The oracle audit found agreement on 18 of 83 edges (0.2169), with complete
agreement for 4 of 36 checkable decisions.

The gate was conservative but not discriminative. Full-stream clustering F1
fell from 0.1639 to 0.0179, continuation delta accuracy fell to zero, and
abstention rose from 0.0083 to 0.1417. In oracle mode, semantic-required
thread-plus-delta fell from 1/39 to zero and abstention rose to 0.2750; the gate
rejected the one correct Qwen delta. AlignScore uses an unfitted three-way
argmax, and these rows share Qwen retrieval and writeback history, so the audit
cannot be treated as ground truth or as an independent safe pipeline.

### Retention and storage interpretation

The final state snapshot contained one retained story in every replay: 6,601
bytes for oracle and 6,472 bytes for full stream, identical between structural
and Qwen variants. The corpus dates are widely separated, and every writeback
applied 30-day lifecycle pruning. The one-story count is therefore the live
state at the final corpus date, not peak storage, corpus cardinality, or
evidence that unrelated stories were merged.

StoryStore schema version 4 persists bounded semantic events and canonical
typed claim relations across reloads. It retains at most 32 deduplicated source
facts and 24 thread events per story, protects user-visible and semantically
cited evidence, and does not store full article text. This fixes duplicate
evidence growth and loss of semantic writeback, but whole-story expiry still
creates a demonstrated long-horizon retrieval gap. A production design needs
an explicit choice among longer retention, archival rehydration, or a separate
durable retrieval tier.

### Blindness after completion

The valid report was produced once from the frozen model, code, configuration,
and corpus. No thresholds, prompts, retrieval rules, retention settings,
models, or labels were tuned from its results, and the run was not repeated.
The inspected results may falsify this model/configuration but cannot establish
generality. This collection is now development data; every future blind claim
requires a fresh, untouched story-family holdout.
