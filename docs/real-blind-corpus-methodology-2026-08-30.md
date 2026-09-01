# Real-news blind corpus methodology

Date: 2026-08-30  
Corpus: `evals/cases/change_monitoring.real_blind.v1.json`  
Protocol: `docs/blind-real-news-evaluation-protocol.md`

## Purpose and isolation

This corpus tests whether a change-monitoring system can associate a new
article with the correct event thread and identify what, if anything, changed.
It is a falsification set for general semantic behavior, not a source of
features, prompts, thresholds, or examples for implementation.

Corpus selection and gold annotation were completed on an isolated work track.
The semantic-system track must not inspect documents, URLs, source names,
labels, label frequencies, or intermediate results before its implementation,
model configuration, thresholds, and abstention policy are frozen. The first
score is immutable. Once failures are inspected, this version becomes an
inspected regression set and cannot support another blind claim.

Every real-world event family is wholly contained in this holdout. No article,
rewrite, near-duplicate, or related subthread from a family is allocated to a
different split.

## Collection and source policy

The collection cutoff was 2026-08-30. Cases were selected from public primary
sources: government agencies, regulators, intergovernmental organizations,
standards bodies, official sports organizations, teams, broadcasters, and
unions. Publisher and wire-service prose was not copied. Guardian Open
Platform content was excluded because its terms prohibit this use; BBC feeds
or pages were not used as evaluation text. Syndicated AP, Reuters, AFP, and
other third-party material was excluded even if it appeared on an otherwise
eligible site.

Each document stores:

- the official publisher and source-page title;
- an absolute source URL and timezone-aware publication timestamp;
- a short original snippet and body paraphrase;
- tags identifying provenance or an adversarial document property.

The paraphrases preserve time, attribution, provisional status, and explicit
uncertainty. They intentionally avoid reproducing long passages or house style.
The provenance manifest binds every document ID to exactly one event family.

Some official pages are updated in place. For these, the corpus records the
historical state represented by the document timestamp and notes the snapshot
limitation in the manifest. Corrections and amendments are modeled as later,
time-scoped evidence; they do not overwrite the earlier published state.

## Family construction

Families were chosen before evaluation to cover different causal and temporal
shapes, not merely different nouns. A family normally contains a main event
thread, at least one continuation, a resolution or later state where available,
and a difficult negative that shares topic, actors, source, venue, or wording
without being the same story.

The corpus includes these structural challenges:

- incident, diagnosis, mitigation, recovery, and post-incident findings;
- proposal, draft, approval, entry into force, implementation, and repeal;
- preliminary, corrected, revised, final, and retrospectively amended facts;
- tentative agreement, operational status, formal approval, and ratification;
- forecast or plan versus observed outcome;
- partial recovery, incremental checkpoints, and material state transitions;
- an apparent resolution that closes only one subthread;
- stale retellings, translated restatements, and same-event duplicates;
- same-source, same-actor, same-location, and same-day hard negatives;
- cross-source continuations, multilingual evidence, long noisy releases, and
  multi-year gaps.

## Annotation contract

Gold labels were assigned from the time-scoped content in each document and
its preceding family history, without running or inspecting production
semantic rules. Names, URLs, dates, publishers, and document IDs were never
used as label shortcuts.

Relationship labels mean:

- `new_story`: first occurrence of a distinct canonical event thread;
- `same_story`: continuation of an already observed canonical event thread;
- `related_theme`: topically or institutionally related but a distinct thread;
- `uncertain`: evidence is insufficient to force a merge or separation.

Delta labels mean:

- `new`: introduction of a distinct thread;
- `material_update`: important new evidence without a formal state transition;
- `status_change`: a plan, legal, operational, medical, competition, or policy
  state changes;
- `correction`: a previously stated fact is explicitly fixed or superseded;
- `resolved`: an open question or tracked subthread reaches a documented end;
- `reframed`: the event proceeds under materially changed scope or terms;
- `incremental`: new but non-material progress worth a compact update;
- `unchanged`: no reportable novelty relative to stored history;
- `uncertain`: the evidence does not justify a more specific relation.

Materiality is annotated separately from semantic relation. A real new fact can
be non-material; a resolution can leave another subthread open. Required and
forbidden fact IDs make claim-level scoring possible and prevent a system from
receiving credit for repeating stale or contradicted claims.

## Coverage matrix

Requested genre coverage is balanced at the event-family level:

| Genre | Whole families |
| --- | ---: |
| Sports | 2 |
| Celebrity / culture | 2 |
| Economy / business | 3 |
| War / conflict | 2 |
| Climate / extreme weather | 2 |
| Science / health science | 2 |
| AI / technology policy and standards | 2 |
| Space | 2 |
| General technology incidents / regulation | 2 |

Tags overlap where a family genuinely spans domains. The sealed corpus has 18
families, 120 documents, 107 chronological evaluation days, 38 named official
sources, and 32 source domains. Source dates span 2022 through 2026; seven
documents are from 2026.

Broad structural coverage is also redundant rather than one-example-only:

| Structural category | Minimum families represented |
| --- | ---: |
| Formal policy, law, agreement, or regulatory lifecycle | 8 |
| Incident, disruption, medical event, or recovery | 5 |
| Correction, revision, provisional-to-final, or retrospective amendment | 4 |
| Cross-source or cross-institution continuation | 4 |
| Stale rewrite, translation, or near-duplicate suppression | 4 |
| Same-event or closely related hard negative | 18 |
| Long/noisy source or long temporal gap | 10 |
| Post-cutoff 2026 evidence | 4 |

## Validation and sealing

`tests/test_real_blind_corpus.py` validates the corpus through the production
evaluation schema and independently checks:

- unique event, document, and expectation IDs;
- exact document-to-expectation and document-to-provenance coverage;
- whole-family canonical-story containment;
- absolute public HTTP(S) provenance URLs;
- all-holdout and all-real categorization;
- minimum requested genre and adversarial-slice coverage;
- presence of all required relationship, delta, and materiality behaviors;
- hash agreement with the seal.

The seal records SHA-256 digests for the corpus, provenance manifest, and this
methodology before the first run. Any change to those artifacts invalidates the
seal and requires an explicit reseal before scoring. The seal status begins as
`sealed_unscored`; the immutable first-run report must be written to a new file
and referenced from a new seal version, never inserted by silently editing the
original results.

## Pre-registered evaluation

The system must be frozen before unsealing. Report both oracle-identity
diagnostics and chronological full-stream behavior; the latter is the headline
result. Metrics, thresholds, confidence intervals, resource measurements, and
abstention reporting are those pre-registered in the isolation protocol. Do
not choose thresholds against blind predictions or score distributions.

## Limitations and replication

This is a manually curated adversarial corpus, so its label distribution does
not estimate ordinary news traffic. Reliance on official sources biases the set
toward institutional writing and underrepresents local, investigative, and
eyewitness reporting. It is English-heavy despite multilingual traps.

Historical families may occur in a model's pretraining data. Later 2026 source
material reduces that risk for part of the set but does not make the collection
pretraining-blind. The relevant guarantee is implementation blindness: the
system was not designed or tuned on these documents or labels.

A strong score can reject some failure hypotheses but cannot establish general
semantic generality. Replication requires a separately sealed corpus collected
from a later window, with untouched event families and a changed source mix.
Rolling public-domain feeds, including eligible original VOA material after
third-party-credit screening, are a suitable future source for that replication
set.
