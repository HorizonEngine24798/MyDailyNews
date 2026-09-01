# Claim-thread architecture and evaluation

Date: 2026-08-30

Status: implemented experimental architecture; first valid broad blind
diagnostic completed and frozen.

## Current architecture

Story understanding runs as bounded, evidence-constrained stages:

1. heuristic retrieval supplies at most three source-backed prior-story
   candidates;
2. deterministic code handles only first observations and normalized-exact
   repetitions;
3. for every other comparison, Qwen chooses one bounded candidate and labels
   its relationship as the same event, a direct successor, related context, a
   different event, or uncertain;
4. when the reports link, headline claims are excluded if body/source claims
   exist, and AlignScore ranks the two closest prior body claims for each
   current body claim;
5. Qwen labels those fixed claim pairs independently;
6. deterministic ontology aggregation converts the atomic labels into a story
   delta; and
7. an evidence-contract validator and the candidate identity gate reject
   unsupported references before StoryStore writeback.

The model receives opaque, request-bounded claim and candidate IDs plus claim
text. Evaluation categories, tags, source fields, URLs, and other retrieval
metadata are not model inputs. There are no phrase, regular-expression,
entity, publisher, location, topic, or story-specific semantic branches.

AlignScore ranks pairs by the maximum forward or reverse entailment or
contradiction probability. It uses no fitted threshold and never decides story
identity. Its independent post-hoc edge audit can also produce a diagnostic
NLI-gated variant. That variant shares Qwen's retrieval and writeback history,
so it is not an independently safe pipeline.

## Semantic output contract

The fixed-pair relation vocabulary covers:

- equivalence, added specificity, and weaker restatement;
- ordinary, completed, intensified, and weakened successors;
- explicit replacement, conflict, and a new unaligned fact; and
- non-substantive or uncertain evidence.

Formal aggregation maps those relations to `correction`, `resolved`,
`escalated`, `weakened`, `status_change`, `incremental`, `unchanged`, or
`uncertain`. The validator checks that every cited claim belongs to the bounded
request and that entailment directions and unchanged-story coverage are
internally consistent. Invalid or insufficient decisions remain visible and
uncertain rather than being silently merged or omitted.

Materiality and display disposition are currently fixed policy baselines
derived from the aggregated change type. They are not separately inferred
from evidence and are secondary diagnostics, not headline semantic metrics.

## Storage contract

StoryStore schema version 4 retains at most:

- 32 distinct source facts per story;
- 24 semantic thread events per story;
- 40 source document IDs per story; and
- four recent thread events in model/delta context.

Facts are deduplicated by normalized exact proposition. The most recently
user-visible facts are protected, and remaining capacity retains recent
distinct evidence. A thread event stores its date, bounded source article IDs,
relationship, change type, materiality, disposition, short summary, bounded
added/repeated/superseded claim lists, and typed claim relations. StoryStore
does not duplicate full article text.

## Evaluation status

### Superseded synthetic regressions

Earlier perfect or near-perfect tables in this document measured an inspected,
small synthetic corpus with the rejected rule-led transition classifier. They
are useful only as historical regression records. They do not evaluate the
current Qwen/AlignScore cascade and provide no evidence of generality.

The historical reports remain at:

- `output/evaluations/claim_delta_20260830/final/report.md`
- `output/evaluations/claim_thread_20260830/final/report.md`

### Broad blind diagnostic — complete

The first valid run covered all 18 event families and 120 documents in the
sealed, genre-diverse real-news corpus. It compared structural-only and Qwen
variants at a 2,048-token request context in chronological full-stream and
canonical-route oracle modes, and recorded an AlignScore diagnostic gate. The
run completed once; there was no post-score tuning or rerun against this
holdout.

Immutable records:

- freeze:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.freeze.json`,
  SHA-256
  `ed8b5adefd92176dd0fc140b1cf1ea63ab6d1f2f97bca423c21c18eb3cde4e60`;
- completion:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.complete.json`,
  SHA-256
  `7ce6850ca03da4c32d1e1ab42f7aec66657d30acfb398bb1bbeff30d4c79bac7`;
- errata:
  `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.errata.json`,
  SHA-256
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

The completion record accepts this as the first valid blind broad diagnostic,
not as a protocol-complete confirmatory evaluation.

| Mode / variant | Documents | Identity | Relationship | Delta | Thread + delta | Semantic-required thread + delta | Pairwise cluster F1 | Accepted coverage | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle / structural only | 120 | 0.1500 | 0.1500 | 0.3500 | 0.1500 | 0.0000 | 1.0000 | 0.0000 | 0.3250 |
| Oracle / Qwen 2,048 | 120 | 0.4750 | 0.4750 | 0.3583 | 0.1583 | 0.0256 | 1.0000 | 1.0000 | 0.0083 |
| Oracle / Qwen 2,048 + diagnostic NLI gate | 120 | 0.2083 | 0.2083 | 0.3500 | 0.1500 | 0.0000 | 0.6051 | 1.0000 | 0.2750 |
| Full stream / structural only | 120 | 0.1500 | 0.1500 | 0.3417 | 0.1500 | 0.0000 | 0.0000 | 0.0000 | 0.1583 |
| Full stream / Qwen 2,048 | 120 | 0.2083 | 0.3000 | 0.3500 | 0.1500 | 0.0000 | 0.1639 | 1.0000 | 0.0083 |
| Full stream / Qwen 2,048 + diagnostic NLI gate | 120 | 0.1500 | 0.1750 | 0.3417 | 0.1500 | 0.0000 | 0.0179 | 1.0000 | 0.1417 |

Accepted coverage measures evidence-contract validation among attempted Qwen
decisions. For an NLI-gated row it does not mean that the post-hoc gate retained
the decision. Materiality and display remain secondary because the evaluation
bypasses upstream article selection and uses fixed policy baselines.

### What the result says about retrieval

Full-stream retrieval was conservative but had very low continuation recall.
Of 78 same-story continuations, 18 were classified as `same_story` and 60 were
treated as new observations. Of 24 related-theme hard negatives, 23 remained
separate and one was linked. Among the 19 semantic attempts, all were accepted
by the evidence-contract validator, but only seven used the correct canonical
thread. Global clustering recorded 20 true-positive pairs, five false merges,
and 199 false splits: precision 0.8000, recall 0.0913, and F1 0.1639.

Qwen therefore produced a small retrieval-recall lift over the structural
baseline, but no end-to-end delta benefit. Full-stream thread-plus-delta stayed
at 0.1500, exactly the structural baseline, and semantic-required
thread-plus-delta was 0/19. Overall delta accuracy of 0.3500 is dominated by
non-continuation `new` cases; continuation delta accuracy was only 1/78, and
that delta did not belong to the correct end-to-end thread.

### What the result says about semantic deltas

The oracle was canonical routing within retained StoryStore history, not a
full teacher-forced comparison for every continuation. Normal StoryStore
writeback pruned records after the configured 30-day retention window, so only
39 of 78 continuations had a canonical prior baseline and reached Qwen. The
other 39 were handled as first observations after their baselines had expired.

On those 39 bounded comparisons, expected deltas were 17 material updates,
eight status changes, eight resolutions, four unchanged reports, and two
incremental updates. Qwen got one unchanged case right: semantic-required
thread-plus-delta was 1/39, and none of the 35 actual changes was classified
correctly. It predicted `correction` 22 times despite no correction-labeled
case occurring in this retained-prior subset. True correction and reframing
continuations fell outside the retained oracle comparisons, so this run does
not isolate Qwen performance on those classes.

Oracle pairwise cluster F1 of 1.0000 is constructed by canonical story-key
routing. It is not evidence that Qwen solved clustering. The oracle identity
and relationship numbers also include 18 trivially correct new-story rows and
should be read together with the 1/39 semantic result.

### AlignScore audit and gate

In full stream, only 8 of 36 checkable claim edges agreed with the independent
AlignScore audit (0.2222), and all checkable edges agreed for only 2 of 18
decisions. In oracle mode, 18 of 83 edges agreed (0.2169), with complete
agreement for 4 of 36 checkable decisions.

The gate converted disagreement into visible uncertainty, but it was not a
useful selector. Full-stream cluster F1 fell from 0.1639 to 0.0179 and
abstention rose from 0.0083 to 0.1417. Oracle semantic-required accuracy fell
from 1/39 to zero and abstention rose to 0.2750; the gate rejected the sole
correct Qwen delta. Because it shares Qwen's retrieval and writeback history
and uses unfitted three-way NLI argmax, it remains an audit diagnostic rather
than a repair mechanism.

### Retention and final storage snapshot

Every replay ended with one retained story: 6,601 bytes in oracle mode and
6,472 bytes in full-stream mode, with the same final sizes for structural and
Qwen variants. This is a final live-state snapshot after chronological
30-day pruning across widely separated corpus dates. It is not peak storage,
does not mean that the corpus contained one story, and does not show that
StoryStore collapsed all threads together.

The gap analysis shows both retention loss and an independent retrieval
failure inside the retained window:

| Gap from prior canonical document | Continuations | Prior available | Treated as first observation | Correct cluster |
| --- | ---: | ---: | ---: | ---: |
| Same day | 5 | 1 | 4 | 1 |
| 1–7 days | 18 | 18 | 9 | 5 |
| 8–30 days | 15 | 15 | 9 | 1 |
| 31–90 days | 14 | 5 | 13 | 0 |
| 91–365 days | 17 | 0 | 16 | 0 |
| More than 365 days | 9 | 0 | 9 | 0 |

Thus 35 of 40 continuations beyond 30 days had no surviving canonical anchor.
But all 33 continuations from one through 30 days had a retained prior and
retrieval still treated 18 as new; longer retention alone cannot repair the
threading system.

Schema version 4 does persist bounded semantic thread events and typed claim
relations across reloads, while exact repeated facts are compacted and
user-visible and cited evidence is protected. Those persistence fixes address
unbounded duplicate evidence and lost semantic writeback. They do not address
the separate lifecycle limit: a story that is not re-observed within 30 days
is removed, so long-horizon continuation currently requires longer retention,
archival rehydration, or another durable retrieval tier.

## Research basis

The staged design follows published evidence that this is not one monolithic
classification task:

- cross-document event coreference must distinguish a concrete event from
  merely shared participants or topic
  ([Caciularu, Cohan, Beltagy, Peters, and Dagan, 2022](https://aclanthology.org/2022.emnlp-main.58/));
- news storylines require explicit temporal and subevent structure
  ([Caselli and Vossen, 2017](https://aclanthology.org/W17-2711/));
- source sentences should be decomposed into smaller independently checkable
  facts before comparison
  ([Molecular Facts, 2024](https://aclanthology.org/2024.findings-emnlp.215/));
- AlignScore is an alignment-based factual-consistency scorer, not an event
  identity system
  ([Zha et al., 2023](https://aclanthology.org/2023.acl-long.634/)); and
- temporal natural-language inference remains materially harder than ordinary
  entailment
  ([Vashishtha et al., 2021](https://aclanthology.org/2021.blackboxnlp-1.31/)).

The blind failure does not refute that architecture. It shows that the current
heuristic retriever, 30-day identity lifetime, sentence fallback, Qwen 1.7B
semantic judge, and uncalibrated AlignScore gate are not adequate
implementations of it.

## Current limits and next work

The system still uses sentence-level source claims rather than a molecular
fact extractor, and StoryStore is a bounded event ledger rather than a rich
temporal knowledge graph. The broad diagnostic shows both an upstream
retrieval/long-horizon problem and a downstream small-model semantic-label
problem. AlignScore can rank and audit claim pairs but is not an
event-coreference model or a calibrated correctness oracle.

The completed holdout was not tuned against or rerun. Its now-inspected
failures may be used only as development data. Retrieval, retention, claim
decomposition, model, ontology, or calibration changes must be developed on a
separate collection, and any later blind performance claim requires a fresh,
untouched story-family holdout.
