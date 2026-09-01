# Handover: story-thread retrieval, claim deltas, and blind real-news evaluation

- Date: 2026-08-30
- Repository: `C:\Users\daroi\Desktop\Project\MyDailyNews`
- Branch / starting commit: `main` at `e77dee7` (`docs: record StoryStore checkpoint`)
- Current verification: 290 tests pass; Qwen server stopped
- Status: substantial experimental implementation in a dirty, uncommitted
  worktree; not production-ready

## Executive summary

The current work answers two questions decisively:

1. Story continuation is partly a retrieval/event-coreference problem, but
   retrieval alone is not enough. The original heuristic was strong on the
   inspected 74-document synthetic corpus, yet the diverse real-news replay
   fragmented most continuations.
2. Qwen3 1.7B Q4_K_M is not reliable enough to own semantic story deltas. In
   the valid blind diagnostic, it got only 1 of 39 evidence-bounded deltas
   correct when the correct retained story was supplied, and none of 35 actual
   changes. The failure was not caused by JSON parsing, transport errors, or a
   full context window.

The broader architecture remains appropriate: high-recall retrieval, bounded
event identity, molecular source claims, bidirectional claim comparison,
temporal story state, and a separate materiality/display policy. What failed
was this implementation of those stages: a lexical retriever with a 30-day
identity lifetime, sentence-level claim fallback, Qwen 1.7B as the semantic
judge, and an uncalibrated AlignScore gate.

The most important valid results are:

| Diagnostic | Result |
| --- | ---: |
| Full-stream pairwise clustering precision / recall / F1 | 0.8000 / 0.0913 / 0.1639 |
| Full-stream semantic-required thread + delta | 0/19 |
| Correct exact thread among full-stream semantic attempts | 7/19 |
| Within-retention oracle semantic thread + delta | 1/39 |
| Correct actual changes in that oracle subset | 0/35 |
| AlignScore edge agreement, full stream / oracle | 8/36 / 18/83 |
| Full-stream pairwise F1 after diagnostic NLI gate | 0.0179 |

Do not deploy the specialized Qwen semantic adapter, use AlignScore as a story
identity model, or turn on reranker hard rejection based on these results.

## Read this before continuing

The authoritative reading order is:

1. this handover;
2. [`claim-thread-architecture-2026-08-30.md`](claim-thread-architecture-2026-08-30.md);
3. [`blind-real-news-evaluation-protocol.md`](blind-real-news-evaluation-protocol.md);
4. [`real-blind-corpus-methodology-2026-08-30.md`](real-blind-corpus-methodology-2026-08-30.md); and
5. the earlier [`HANDOVER-qwen-cpu-evaluation-2026-08-29.md`](HANDOVER-qwen-cpu-evaluation-2026-08-29.md)
   only for the work that predates this document.

The earlier handover is now historical. Its 255-test count, schema-v1 storage
description, and proposed deterministic transition work are superseded here.

The worktree contains user work and all of the work described below. Do not
reset, clean, checkout, or overwrite it. Generated reports, downloaded models,
the CPU environment, llama.cpp, and live state are ignored by Git.

## What is actually wired today

This distinction is easy to lose because the evaluator and production code
share contracts:

| Component | In normal pipeline? | Default state | Current meaning |
| --- | --- | --- | --- |
| Heuristic StoryStore candidate retrieval | Yes | Enabled with memory | Supplies at most three source-backed candidates. |
| Qwen3 0.6B story reranker | Optionally | Disabled | Scores the heuristic candidates, but the current context integration re-sorts them by heuristic score; only hard rejection reliably changes the final set today. |
| Claim evidence scaffold and validator | Yes | Always built in delta stage | First observation/exact repetition remain structural; other cases require accepted semantic evidence. |
| Generic `DeltaExtractor` model call | Yes when configured | Existing delta configuration | Its decisions are merged only if they satisfy the claim evidence contract. |
| Specialized `QwenSemanticDeltaInferencer` fixed-pair cascade | No | Evaluation-only | Used by `tools/run_semantic_thread_evaluation.py`; it is not selected by the production factory. |
| AlignScore NLI scorer | No | Evaluation-only | Ranks claim pairs and audits edges in the semantic evaluator. |
| StoryStore schema v4 writeback | Yes in code | Normal memory write | Bounded facts/events and typed claim relations are persisted. |
| Live schema-v4 canonical store | Not yet created | Migration pending | Current live state is still a legacy `story_index.json`; the next normal write creates `story_store.json`. |

The normal production path is:

1. memory-enabled brief execution creates StoryStore and coverage-store
   instances;
2. selection assigns each article a provisional new key and retrieves at most
   three possible prior-story keys;
3. brief analysis builds story-specific memory, optionally calls the existing
   generic `DeltaExtractor`, always builds the structural claim scaffold,
   validates any model result, and then applies the candidate identity gate;
4. only an accepted, candidate-bounded same-story decision reuses a prior key;
   rejected, new, and uncertain decisions keep a provisional key and remain
   visible; and
5. after the brief files are written, StoryStore receives all selected
   articles, including selected articles omitted from the rendered brief. Only
   rendered IDs mark facts user-visible, and only rendered articles enter the
   coverage log.

This means StoryStore and the general evidence safety contract are real
pipeline behavior. It does not mean the evaluated fixed-pair Qwen/AlignScore
cascade powers normal briefs.

The generic `DeltaExtractor` and the specialized semantic evaluator should
not be confused. In the checked local CPU configurations the base delta switch
is off, while general/detailed rollout entries explicitly enable the generic
extractor. No specialized fixed-pair output mode is configured, so its normal
`full` model contract is used when enabled.

## Starting point inherited from the 2026-08-29 handover

Before this work, the project had:

- a 74-document, 15-arc inspected synthetic/adversarial corpus, including four
  absurd magic/elf documents intended to expose narrow topic assumptions;
- a unified StoryStore architecture and candidate-bounded identity gate;
- heuristic top-three retrieval that scored 24/25 prior-day continuations on
  that inspected corpus;
- a Qwen3 1.7B CPU decision-only benchmark with 56/56 valid JSON responses but
  only 0.2297 delta accuracy and 0.1923 continuation-delta accuracy; and
- oracle ablations showing that simply supplying the correct candidate or a
  compact fact packet did not make the model identify deltas reliably.

The final inherited ablation made the bottleneck especially clear: an oracle
candidate was supplied on every continuation, relationship accuracy reached
0.9615, and the correct key was linked 0.8077 of the time, yet delta accuracy
given a correct link was only 0.0952. Supplying both an oracle candidate and
an oracle fact packet produced 0.0000 delta accuracy.

That work established reliable CPU execution and bounded JSON, but not semantic
quality. The next work deliberately moved away from generic prompt/context
sweeps and toward retrieval, claim state, and blinded real-news evaluation.

## Detailed chronology of this work

### 1. StoryStore housekeeping and evidence retention

The first concern was whether StoryStore was growing by retaining repeated or
low-value source material. The store was changed from a loose compact index to
schema version 4 with bounded evidence and semantic events:

- facts per story were reduced from 80 to 32;
- exact normalized repeats now share a content-addressed fact ID based on fact
  kind and normalized proposition, rather than article ID;
- duplicate provenance prefers a user-visible occurrence, otherwise the most
  recent occurrence;
- the latest user-visible facts and facts cited by semantic decisions are
  protected during eviction;
- at most 24 semantic thread events and 40 source document IDs are retained per
  story;
- a baseline exposes at most four recent facts and four recent thread events;
- full article text is not stored in StoryStore; extraction keeps a headline
  plus at most six sentences from the first 2,400 source characters, each
  capped at 420 characters; and
- lifecycle defaults remain seven days to stale and 30 days to whole-story
  deletion.

The store persists event date, bounded source IDs, relationship, change type,
materiality, disposition, summary, added/repeated/superseded claims, and typed
claim relations. The post-evaluation audit found that persistence still had a
stale local relation whitelist. It was replaced with imports of the canonical
`CLAIM_RELATIONS` and `ENTAILMENT_VALUES`, and a round-trip test now covers
every canonical value.

The current live state is not large: `state/memory/story_index.json` is 35,960
bytes with 27 records last seen on 2026-08-25 or 2026-08-26. It contains no
facts or thread events, `story_store.json` is absent, and the next normal
memory-enabled write should perform the migration. Preserve the legacy file as
the migration backup.

### 2. A direct Qwen continuation spike

Before choosing more models, a contaminated oracle spike tested whether Qwen
1.7B could solve continuation and delta tasks when given deliberately easier
inputs. Ten true continuations and ten negative pairs were tried with compact
source, oracle fact-ledger, raw-pair, and staged-decision prompts.

The best relationship accuracy was 0.5000; every variant scored 0.0000 delta
accuracy on true pairs and 0.0000 same-story recall. This ruled out the idea
that a small prompt adjustment or an already-correct candidate would be enough.

Historical report:
`output/evaluations/qwen_continuation_spike_20260830/report.md`.

### 3. Research and model selection

Research indicated that the product problem should be decomposed rather than
treated as fact-checking or one large classification prompt:

- cross-document event coreference distinguishes a concrete event from shared
  entities or topic
  ([Caciularu et al., 2022](https://aclanthology.org/2022.emnlp-main.58/));
- event storylines require temporal and subevent structure
  ([Event StoryLine, 2017](https://aclanthology.org/W17-2711/));
- claims should be decomposed into independently checkable molecular facts
  ([Molecular Facts, 2024](https://aclanthology.org/2024.findings-emnlp.215/));
- temporal entailment remains substantially harder than ordinary NLI
  ([Vashishtha et al., 2021](https://aclanthology.org/2021.blackboxnlp-1.31/));
- alignment-based factual consistency can audit claim support, but it is not
  event identity
  ([AlignScore, 2023](https://aclanthology.org/2023.acl-long.634/)); and
- random article splits can leak news events, so complete event families and
  time-aware holdouts are required
  ([Laban et al., 2023](https://aclanthology.org/2023.findings-eacl.55/)).

Two pair models were downloaded and evaluated on 52 scorer-constructed pairs
from the inspected standard and absurd corpora:

| Model | Accuracy at 0.5 | Positive recall | Negative recall | AUC |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-Reranker-0.6B | 0.8462 | 0.8077 | 0.8846 | 0.9504 |
| ModernBERT-large-zeroshot-v2.0 NLI | 0.5000 | 0.0000 | 1.0000 | 0.7167 |

Qwen reranker separated pairs reasonably well. ModernBERT had some ranking
signal but unusable default calibration. ModernBERT was subsequently deleted
at the user's request. Qwen reranker remains available locally.

The four absurd elf/magic pair examples were all classified correctly by the
Qwen reranker, but that tiny inspected slice is only a topic-diversity smoke,
not proof of generality.

Pair reports:

- `output/evaluations/pair_models_20260830/qwen-reranker/report.md`
- `output/evaluations/pair_models_20260830/modernbert-nli/report.md`

AlignScore was added later as a claim-pair scorer, not as the second story
reranker. The upstream training checkpoint was converted into a compact
498,616,700-byte safetensors runtime containing the RoBERTa encoder and
three-way NLI head. The approximately 1.83 GiB training checkpoint and partial
download were removed after conversion.

### 4. Optional Qwen reranker integration

The Qwen reranker was wrapped behind a small protocol so the heuristic remains
the candidate generator. It can only reorder the retrieved set or, when an
explicit experimental option is enabled, reject candidates below a threshold.
Torch and Transformers are imported lazily so a normal installation does not
acquire or load the model.

On the inspected 74-document retrieval replay, threshold rejection at 0.5
reduced historical continuation recall from 0.9600 to 0.8000. Standard-news
recall fell from 0.9565 to 0.7826. The model was therefore not good enough to
replace or hard-filter the heuristic. Production configuration keeps
`story_reranker_enabled = false` and `story_reranker_hard_rejection = false`.
If used later, it should initially reorder a high-recall union, not remove it.
New/unrelated rejection was 0.9592, but one of the four absurd documents also
received a false candidate, reinforcing that precision and recall must be
measured outside authored examples.

Report:
`output/evaluations/reranked_story_retrieval_20260830/report.md`.

### 5. Initial claim ledger and the rejected rule-led success

StoryStore was extended to record source claims and thread events, and an
initial deterministic claim/state classifier was evaluated on the inspected
74-document corpus. It produced attractive results:

- oracle claim-delta accuracy: 1.0000;
- end-to-end identity and delta accuracy: 0.9865;
- continuation joint accuracy: 0.9615; and
- the one remaining continuation miss was the known indirect `rum-03` case.

These scores are not evidence of a solved problem. Audit showed that the
transition classifier used semantic phrase/verb patterns and other narrow
signals that matched the authored corpus. The absurd magic/elf examples were
useful but did not make the templated transition distribution blind. The user
was right to distrust the result.

Those reports are retained only as superseded regression history:

- `output/evaluations/claim_delta_20260830/final/report.md`
- `output/evaluations/claim_thread_20260830/final/report.md`

The phrase/regex transition logic was removed. Current deterministic code may
decide only a first observation or normalized-exact repetition. It may use
generic lexical/numeric signals to retrieve or flag identity uncertainty, but
it may not map wording, actor, publisher, topic, location, or corpus IDs to a
semantic transition.

### 6. General evidence contract

`mydailynews/analysis/claim_delta.py` now defines a backend-independent,
source-bounded contract:

- every claim has an opaque ID, text, kind, source provenance, publication
  time, and optional prior story key;
- a semantic decision must select only a supplied prior story and cite only
  supplied current/prior claim IDs;
- canonical relationship values are `same_story`, `related_theme`,
  `distinct_story`, and `uncertain`;
- canonical change values include `new`, `material_update`, `status_change`,
  `correction`, `resolved`, `incremental`, `escalated`, `weakened`,
  `reframed`, `unchanged`, and `uncertain`;
- typed claim edges include equivalence, support, added detail, weaker
  restatement, new fact, non-substantive text, contradiction, supersession,
  temporal successor, context-only, and uncertainty; and
- entailment direction is explicit: current entails prior, prior entails
  current, both, neither, or uncertain.

The validator enforces candidate ownership, claim-ID membership, legal
relationship/change combinations, directional requirements, and safe
unchanged coverage. A `same_story + unchanged` decision must cover every
substantive current body claim and may contain only unchanged-compatible
edges. Invalid or missing semantic decisions remain visible as uncertain; they
cannot silently merge, update confirmed state, or authorize omission.

`build_deterministic_delta_scaffold()` and
`merge_claim_delta_with_model()` are wired into the normal delta stage before
the existing candidate identity gate. This production wiring is important;
the specialized small-model adapter described next is not production-wired.

### 7. Fixed-pair Qwen and AlignScore cascade

Several non-blind development smokes were used to simplify the small model's
job:

1. A one-call identity-plus-delta schema collapsed into a few labels and was
   rejected by the evidence validator.
2. Splitting identity from a single anchor delta improved identity but biased
   all reasoning toward one selected claim.
3. Set-level atomic output still produced mostly uncertainty.
4. Fixed claim pairs finally gave reliable JSON and complete contract
   acceptance, but semantic labels remained wrong.

The final experimental cascade is:

1. structural first-observation/exact-repeat handling;
2. one Qwen identity call across at most three bounded candidate stories;
3. if linked, exclude headline claims when body claims exist;
4. AlignScore ranks the top two prior body claims for each current body claim
   using the maximum bidirectional entailment or contradiction probability;
5. Qwen labels fixed pair slots in batches of at most six;
6. deterministic, text-blind ontology aggregation derives the story delta;
7. the evidence validator and candidate identity gate accept or reject it; and
8. StoryStore writes bounded facts and typed thread events.

The Qwen pair labels are `equivalent`, `adds_specificity`,
`weaker_restatement`, ordinary/completed/intensified/weakened successor,
`explicit_replacement`, `conflict`, `new_unaligned`, `non_substantive`, and
`uncertain`. Formal aggregation, rather than text matching, maps these to the
canonical delta classes.

The final four-document development smoke proved that the plumbing worked but
not the semantics. Full-stream identity was 0.7500 and all three comparisons
were accepted, yet delta accuracy was 0.2500 and semantic-required
thread-plus-delta was zero. The model reversed the important outcomes: an
unchanged report became a correction and a status change became unchanged.
AlignScore agreed with only 1/5 checkable full-stream edges and 1/3 oracle
edges, so its gate abstained broadly.

Report:
`output/evaluations/semantic_thread_harness_smoke_20260830_fixed_pairs/report.md`.

### 8. Diverse blind real-news corpus

In parallel with the general implementation, an isolated corpus track built a
real-news falsification set without revealing documents, sources, labels, or
label distributions to the semantic-system track.

The sealed corpus contains:

- 18 complete event families, 120 documents, and 107 chronological days;
- 38 named official sources and 32 source domains;
- dates from 2022 through 2026;
- overlapping coverage of sports, celebrity/culture, economy/business,
  war/conflict, climate/extreme weather, science/health, AI/technology, space,
  environment, and general technology; and
- continuations, resolutions, corrections, policy lifecycles, multilingual
  evidence, noisy source pages, translations, multi-year gaps, and a
  related-event hard negative in every family.

Only official/primary-source material was used. The corpus stores original
short paraphrases and provenance rather than publisher prose. The Guardian
Open Platform was excluded because its terms prohibit this use; third-party
wire copy was also excluded. RSS discovery was treated as discovery, not as a
content license.

Sealed artifact hashes:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Corpus | 194,883 | `b220a3bbe7f50c62d9e0f33cb17e238253b855a4c8918974e3fac851fdab95fa` |
| Provenance | 9,282 | `5eb9c8ae5d0e84579ed82f217e44f1f1fc2b2857f74078c97141181a06fba6ef` |
| Methodology bound by seal | 8,863 | `fb4fb952c2343ba551db1e0329dbb4654beb8340cd1b106dbe5b7e8050db4c58` |
| Original seal | 1,792 | `1822da4936ef5509931a28710c4d3b06048a1c80e514b43970e0f1f302ae2f54` |

The corpus is now inspected development data. It may not support another blind
claim. A fresh future blind set must contain untouched complete event families.

### 9. Harness audit and invalid aborted preflight

An initial blind process was launched and terminated before it created a report
or exposed any score, prediction, story, or label. A concurrent audit found
three invalid information-flow/replay properties:

1. evaluation category and tags could enter lexical retrieval;
2. a separate store per event family revealed corpus partition boundaries; and
3. sequential same-day writeback let later fixtures see earlier same-day
   fixtures.

The launch is invalid and must never be cited as a result. Its records are:

- `output/evaluations/semantic_thread_real_blind_first_20260830.freeze.json`
- `output/evaluations/semantic_thread_real_blind_first_20260830.abort.json`

The aborted freeze is 1,244 bytes with SHA-256
`3744b56ecb1bbd94e10b03a9ed5dec336016f795a01d810947bb3197b4ad7725`;
the abort record is 804 bytes with SHA-256
`d7cb7c9db0071a9d8eaf3364100fd1eb16a316e67f91a2792da624253fa7a9c8`.

The corrected harness strips evaluation-only metadata, interleaves every
family in one global chronological store, annotates a whole date against frozen
prior-day state before any writeback, records cluster IDs only after inference,
uses global false-merge/false-split clustering metrics, and binds the running
server model basename plus file hashes.

The valid freeze also bound a 289-test passing tree. The later canonical
ontology persistence test raised the current total to 290 without changing or
rerunning the semantic evaluation.

### 10. First valid broad blind diagnostic

The frozen run evaluated all 18 families and 120 documents once, using Qwen3
1.7B Q4_K_M, a 2,048-token request context, both canonical-route oracle and
chronological full-stream modes, structural and Qwen variants, AlignScore pair
ranking/audit, and a diagnostic NLI gate.

It was frozen at approximately 18:37 and completed at
`2026-08-30T19:03:41.2990165+04:00`. The completion record accepts it as the
first valid blind score but explicitly records
`confirmatory_protocol_complete = false`: this is a broad falsification
diagnostic, not a completed confirmatory benchmark.

The server had a 4,096-token capacity, but every observed request completed
without truncation and the maximum observed prompt-plus-generation sequence
was 1,764 tokens. This run did not compare 2K with 4K, but it shows that context
truncation did not cause the failures reported here.
The 2K evaluator reserved 128 output tokens and a 256-token safety margin,
leaving 1,664 input tokens and admitting at most four current and six prior
claims per semantic comparison.

| Mode / variant | Identity | Relationship | Delta | Thread + delta | Semantic subset | Pair F1 | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full stream / structural | 0.1500 | 0.1500 | 0.3417 | 0.1500 | 0.0000 | 0.0000 | 0.1583 |
| Full stream / Qwen | 0.2083 | 0.3000 | 0.3500 | 0.1500 | 0.0000 | 0.1639 | 0.0083 |
| Full stream / Qwen + NLI gate | 0.1500 | 0.1750 | 0.3417 | 0.1500 | 0.0000 | 0.0179 | 0.1417 |
| Oracle-route / structural | 0.1500 | 0.1500 | 0.3500 | 0.1500 | 0.0000 | 1.0000* | 0.3250 |
| Oracle-route / Qwen | 0.4750 | 0.4750 | 0.3583 | 0.1583 | 0.0256 | 1.0000* | 0.0083 |
| Oracle-route / Qwen + NLI gate | 0.2083 | 0.2083 | 0.3500 | 0.1500 | 0.0000 | 0.6051* | 0.2750 |

`*` Oracle cluster keys are assigned by the harness; oracle pairwise F1 is not
model clustering ability.

The full-stream result separates into two failures:

- retrieval/threading: only 18/78 same-story continuations were called
  `same_story`; 60 were treated as new. The system kept 23/24 related-theme
  negatives separate, so it was conservative, but exact pairwise recall was
  only 0.0913. Of the 19 semantic attempts, just seven used the correct
  canonical thread. Global clustering counted 20 true-positive pairs, five
  false merges, 199 false splits, and 6,916 true negatives;
- semantics: the Qwen path got 0/19 attempted full-stream cases right on both
  thread and delta. Its 0.1500 overall score is entirely explained by 18
  straightforward new-story observations and is identical to the structural
  baseline.

The oracle result is even more important but needs its erratum. Oracle routing
used the correct canonical key only while the baseline survived StoryStore's
normal 30-day retention. Therefore only 39/78 continuations reached semantic
inference. On those 39 comparisons:

- 39/39 calls returned accepted, evidence-referencing decisions;
- only one unchanged case had the right delta;
- none of 35 actual changes was correct;
- the model predicted `correction` 22 times even though this retained subset
  contained no gold correction; and
- it never correctly produced a material update, status change, resolution,
  or incremental update.

This demonstrates that schema validity and evidence-ID validity are necessary
but cannot prove semantic truth.

AlignScore corroborated instability but did not fix it. Full-stream agreement
was 8/36 checkable edges (0.2222), oracle agreement was 18/83 (0.2169), and the
gate rejected the sole correct oracle delta. The gate shares Qwen retrieval and
writeback history and is only a post-hoc diagnostic.

Model-call time was 447.655 seconds for 19 full-stream comparisons and 932.516
seconds for 39 oracle comparisons, about 23.6-23.9 seconds per comparison. All
116 model requests completed with zero model errors or no-decisions.

### 11. Retention diagnosis

The broad corpus exposed that whole-story deletion and evidence compaction are
different concerns. Compact facts control size; deleting identity after 30
days prevents long-running stories from being retrieved.

| Gap from preceding canonical document | Continuations | Prior available | Treated as first observation | Correct cluster |
| --- | ---: | ---: | ---: | ---: |
| Same day | 5 | 1 | 4 | 1 |
| 1-7 days | 18 | 18 | 9 | 5 |
| 8-30 days | 15 | 15 | 9 | 1 |
| 31-90 days | 14 | 5 | 13 | 0 |
| 91-365 days | 17 | 0 | 16 | 0 |
| More than 365 days | 9 | 0 | 9 | 0 |

For gaps beyond 30 days, 35/40 continuations had no surviving canonical story
representation. But all 33 one-to-30-day continuations had a retained prior
and retrieval still treated 18 as new. Longer retention is necessary but not
sufficient.

Every replay ended with one retained story because the last two fixture dates
were separated by 32 days. The replay peaked at 16 live records. The final
6.5-KB state is a post-pruning snapshot, not evidence of a one-key collapse.

The appropriate design is a small hot evidence store plus durable compact story
anchors. Long-lived anchors should retain identity, aliases/entities/event
loci, first/last seen, compact key claims or last state, and enough provenance
to rehydrate archival evidence without retaining every article or every old
sentence.

### 12. Post-result changes

The blind report was not overwritten or rerun after inspection. Only general
mechanical corrections were made afterward:

- an immutable errata record documents the 30-day oracle limitation,
  constructed oracle clustering metric, and final-only state sizes;
- evaluator prose now says “canonical routing over retained history” rather
  than claiming full teacher-forced semantic isolation;
- evaluator console output is compact instead of printing all nested slices;
- StoryStore relation/entailment persistence now consumes the canonical
  ontology and round-trips all values; and
- documentation was rewritten to remove the rejected synthetic-success
  narrative.

No prompt, threshold, retrieval rule, model, ontology aggregation, or semantic
behavior was tuned from the blind results. The semantic run was not repeated.

## Immutable valid-run artifacts

The local `output/` tree is ignored by Git. Preserve these files:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Freeze record | 2,182 | `ed8b5adefd92176dd0fc140b1cf1ea63ab6d1f2f97bca423c21c18eb3cde4e60` |
| Completion record | 1,331 | `7ce6850ca03da4c32d1e1ab42f7aec66657d30acfb398bb1bbeff30d4c79bac7` |
| Errata record | 1,189 | `3ca446d1bae9f2b24b27ab5d87c41caae3e484486c9bf434b5228ed37cc74b97` |
| JSON report | 2,853,412 | `efdd1cb3b089f3ac55505c274516a33141a16546d1f6e56f68fc4af3e0bc96a4` |
| Markdown report | 1,849 | `67b3815eb5414a00e5a6ddc7dbf47614ea7cc386e0e474c2fc90f0a46b1b903e` |

Paths:

- `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.freeze.json`
- `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.complete.json`
- `output/evaluations/semantic_thread_real_blind_broad_20260830_v1.errata.json`
- `output/evaluations/semantic_thread_real_blind_broad_20260830_v1/report.json`
- `output/evaluations/semantic_thread_real_blind_broad_20260830_v1/report.md`

The report manifest contains 33 hashed dependencies and matches the frozen
corpus, Qwen model, AlignScore runtime, configuration, evaluator, and semantic
adapters. The current tree intentionally differs from the frozen tree in
StoryStore persistence and evaluator presentation as described above. Never
silently treat a run from current code as the original blind run.

For exact reproducibility, the frozen/current differences are:

| File | Frozen SHA-256 | Current SHA-256 |
| --- | --- | --- |
| `mydailynews/memory/story_store.py` | `4fea99d32041cfe5804f62f4690da941d4ad796b8f8fd33fe93cbfda7def812a` | `3abaf997578def35a2ff372e72d44797bf9be47c182ccd6493ce3c105a097a3f` |
| `tools/run_semantic_thread_evaluation.py` | `fe92d61b2ad167bf61e3e7789612d28b3bde7acff18ef78c0ad667650d036689` | `8d09aa27e5ac1a38f243f718bd220940646cf96f7b7658079b22bfa4b8a7d840` |

The corpus, configuration, models, claim contract, Qwen semantic adapter, and
AlignScore adapter still match the frozen manifest. Recover the frozen file
versions before claiming byte-for-byte reproduction; do not overwrite the
immutable report.

## Local model assets

All assets are ignored by Git and available only on this machine:

| Asset | Local size | SHA-256 / status |
| --- | ---: | --- |
| `models/Qwen3-1.7B-Q4_K_M.gguf` | 1,282,439,264 bytes | `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5` |
| `models/Qwen3-Reranker-0.6B/` | 1,207,492,002 bytes | model weights `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b` |
| `models/AlignScore-base/` | 501,331,338 bytes | runtime `38c104cb1c828a47811bf5ed6dbae579904e4d1cab68864d2ff3ada7c0fb222e` |
| ModernBERT reranker | removed | Redownload only for a new justified experiment. |
| Original AlignScore training checkpoint | removed after conversion | Recoverable from upstream if conversion must be repeated. |

llama.cpp remains at `tools/llama.cpp/b10631/llama-server.exe`. No
`llama-server` process is running at handover. The evaluated binary was build
10631, commit `5d5cb4c3a`, with SHA-256
`09a5d872e41eb5d028917cd396304040aaba7f4498979fa2566dd6fa703bf8df`.

## Current architecture and file map

### Identity, retrieval, and memory

- `mydailynews/memory/story_retrieval.py`: heuristic top-three candidate
  generation from aliases, entity/event/numeric tokens, fact overlap, and
  numeric conflict.
- `mydailynews/memory/story_reranker.py`: model-agnostic bounded reranker
  protocol; preserves heuristic candidates unless explicit rejection is used.
- `mydailynews/ai/qwen_story_reranker.py`: lazy Qwen3 0.6B implementation.
- `mydailynews/memory/context.py`: builds bounded per-article candidate
  baselines and optionally applies the reranker.
- `mydailynews/memory/ranking.py` and `mydailynews/memory/recall.py`: annotate
  selection candidates and copy only an accepted prior key into live identity.
- `mydailynews/analysis/identity_gate.py`: accepts only supplied candidate
  keys; unknown or ambiguous links fail open.
- `mydailynews/memory/story_store.py`: schema-v4 identity, lifecycle, compact
  evidence, semantic event ledger, migration, and writeback.

### Claims and semantic decisions

- `mydailynews/analysis/claim_delta.py`: canonical claims, relations,
  entailment directions, backend protocol, structural assessment, and
  validation.
- `mydailynews/analysis/deterministic_delta.py`: production scaffold and merge;
  deterministic semantics limited to first observation and exact repetition.
- `mydailynews/ai/qwen_semantic_delta.py`: evaluation-only Qwen identity and
  fixed-pair classifier.
- `mydailynews/ai/alignscore_nli.py`: evaluation-only local AlignScore NLI
  adapter.
- `mydailynews/analysis/delta.py`, `mydailynews/ai/prompts.py`, and
  `mydailynews/ai/schemas.py`: generic production delta prompt/schema support
  and bounded claim evidence.
- `mydailynews/pipeline/brief_analysis_stages.py`: production context,
  optional reranker, generic model call, deterministic scaffold merge, and
  candidate gate.
- `mydailynews/pipeline/brief_execution.py`: creates stores and writes selected
  articles back after output generation, with rendered IDs controlling
  visibility and coverage.

### Evaluation and corpus

- `tools/run_pair_model_evaluation.py`: inspected pair-ranker comparison.
- `tools/run_reranked_story_retrieval_evaluation.py`: inspected StoryStore
  reranking replay.
- `tools/run_qwen_continuation_spike.py`: contaminated small-model capability
  spike.
- `tools/run_claim_delta_evaluation.py` and
  `tools/run_claim_thread_evaluation.py`: superseded rule-led synthetic
  evaluations; never cite as general performance.
- `tools/run_semantic_thread_evaluation.py`: corrected chronological semantic
  evaluator and report manifest.
- `evals/cases/change_monitoring.real_blind.v1.json`: now-inspected real corpus.
- `tests/test_claim_delta.py`: evidence-contract validation.
- `tests/test_qwen_semantic_delta.py`: fixed-pair schema and aggregation.
- `tests/test_real_blind_corpus.py`: corpus/provenance/seal validation.
- `tests/test_semantic_thread_evaluator.py`: metadata isolation,
  chronological batching, and global clustering.
- `tests/test_story_identity_architecture.py`: StoryStore bounds, migration,
  reranking limits, semantic event persistence, and canonical ontology
  round-trip.

## Safety and anti-overfitting invariants

Preserve these unless a change is explicitly justified and tested:

- no semantic transition rules based on phrases, regex verb maps, entities,
  publishers, locations, topics, IDs, URLs, or source names;
- no examples copied from an evaluation corpus into prompts or code;
- no blind-set threshold or abstention calibration;
- deterministic omission only for a validated exact repetition of a confirmed
  story;
- uncertain semantic evidence remains visible and does not update confirmed
  transition state;
- a semantic model may cite only claims and story candidates in its current
  bounded request;
- source attribution, negation, quantity, modality, and time must survive
  comparison;
- AlignScore/NLI may rank or audit claim pairs but may not decide story
  identity by itself;
- a reranker cannot introduce a candidate that retrieval did not supply; and
- evaluation categories/tags/family IDs never enter retrieval or model input.

## Known limitations and unresolved risks

### Retrieval and memory

- The 30-day whole-story expiry destroys identity needed for long-horizon
  threads. A two-tier hot-evidence/durable-anchor design is not implemented.
- Retrieval recall is independently poor even inside 30 days.
- Score-only reranking is currently undone when `memory/context.py` deduplicates
  and sorts the candidates again by heuristic score. Hard rejection changes
  the set, but it is unsafe at the measured recall.
- There is no global StoryStore byte/story-count cap; churn can grow linearly
  inside the retention window, and the single JSON file is fully loaded and
  rewritten on every update.
- Missing/malformed `last_seen` dates are not automatically pruned, and
  backdated writes can move `last_seen` backward.
- Retrieval token arrays preserve early unique tokens once full, so stale
  vocabulary can crowd out newer state.
- Exact-text fact deduplication controls size but collapses independent-source
  provenance into one retained fact row. Paraphrased duplicates remain
  separate and can consume the 32 slots.
- Fact eviction protects evidence cited by the current update and recently
  visible facts, but not every fact referenced by older retained thread
  events. Old event edges can therefore become dangling.
- There is no concurrent-writer lock, cross-file transaction, or
  corrupt-canonical recovery fallback.
- Current live state has not exercised schema-v4 migration or semantic
  writeback in a normal production run.

### Claims and semantics

- “Claims” are source sentences, not molecular fact extraction. Composite
  sentences can contain multiple propositions or document commentary.
- The generic production schema, prompts, and `DeltaExtractor` normalizer
  still allow only the older eight relation labels. They do not yet accept
  `weaker_restatement`, `new_fact_in_story`, or `non_substantive`, even though
  the v3 validator and StoryStore do. A live generic model therefore cannot
  express the complete ontology.
- After an accepted semantic decision, the merge updates relation edges and
  superseded claims but does not recompute the scaffold's `added_claims` and
  `repeated_claims`. A paraphrased unchanged claim can be persisted as added.
- Qwen 1.7B is not a viable semantic judge under the current contract.
- The validator proves boundedness and internal consistency, not semantic
  truth.
- AlignScore is uncalibrated for this domain and its gate is not
  discriminative.
- Materiality/display values in the semantic evaluator are fixed policy
  baselines, not separately inferred evidence-backed predictions.
- The specialized Qwen/AlignScore cascade is not production-wired.
- The experimental fixed-pair path at 2K sees only a bounded prefix of claims;
  if AlignScore is absent, zero-score ties reduce pair selection to stable ID
  order, and aggregation precedence can let one spurious conflict dominate.

### Evaluation

- The valid run used 2K only; it is not a 2K-versus-4K or small-versus-large
  model comparison.
- The oracle retained the production 30-day lifetime and therefore compared
  only 39/78 continuations.
- Same-day batches intentionally do not see same-day writeback; four of five
  same-day continuations lacked a prior in this replay.
- The evaluator does not yet report every preregistered metric: retrieval
  recall@1/@3, claim-support precision/recall, relation macro-F1, suppression
  precision/recall, bootstrap confidence intervals, peak RAM, or complete
  source/family/time/language slices.
- The original 74-document and real 120-document corpora are both inspected.
  Neither can support another blind claim.
- The semantic evaluator processes fixture articles directly and bypasses
  fetch, multi-article selection/grouping, rendering, and a full brief. It is a
  component replay, not an end-to-end production release test.

## Next work, in order

### 1. Correct live contract and persistence mismatches

Align the generic prompt/schema/normalizer with the canonical relation
ontology; preserve score-only reranker ordering through context construction;
recompute `added_claims` and `repeated_claims` from accepted semantic edges;
and keep every retained event's cited facts reachable. Decide explicitly how
independent-source corroboration survives exact-text deduplication. Add focused
integration and round-trip tests before changing semantic behavior.

### 2. Build a two-tier story memory

Keep the 30-day hot store for bounded facts and recent events, but retain a
small durable anchor after hot evidence expires. Define explicit global byte
and record budgets. Retrieval should be able to rehydrate archived evidence
after an anchor match. Add tests for exact retention cutoffs, malformed dates,
backdated writes, global growth, migration, and concurrent writes.

### 3. Restore high-recall candidate generation

Measure recall@1 and recall@3 by time gap on a development collection. Use a
union of generic lexical/numeric retrieval and dense/event representations if
needed. The Qwen reranker may reorder that union; do not enable hard rejection
until it improves recall on genuinely new development families.

### 4. Add molecular claim extraction

Split source sentences into atomic, source-span-backed propositions while
preserving attribution, modality, numbers, negation, and time. Fall back to the
original source sentence when normalization is unsupported. Do not allow a
free-form summary to become durable state without cited source spans.

### 5. Evaluate a stronger semantic backend on development data

Use the same bounded identity and fixed-pair contracts so model comparison is
fair. First build a true all-continuation oracle that bypasses retention and
test relation/delta accuracy separately. Compare a materially larger model,
and calibrate any NLI/consistency scorer on development families only. Do not
reuse the real corpus as a blind claim.

### 6. Separate semantic truth from materiality and display

After claim relations are reliable, derive materiality/profile value and
display policy in a separate, testable layer. A model's prose or confidence
must never directly authorize suppression.

### 7. Wire only a proven semantic backend into production

The production scaffold/validator is ready for a backend, but the specialized
Qwen adapter is not wired. Add a backend interface/configuration only after an
evaluated model clears predeclared development thresholds. Run a normal
memory-enabled brief to test schema-v4 migration before considering rollout.

### 8. Create a new blind corpus only after freezing the next system

Use a later collection window, untouched event families, and a changed source
mix. Preserve complete-family isolation and immutable first scoring. The
current real corpus may now be used for development diagnostics, but any tuning
must be recorded as such.

## Verification and operational commands

From PowerShell:

```powershell
cd C:\Users\daroi\Desktop\Project\MyDailyNews

git -c safe.directory=C:/Users/daroi/Desktop/Project/MyDailyNews status --short

.\.venv-cpu-test\Scripts\python.exe -B -m unittest discover -s tests

git -c safe.directory=C:/Users/daroi/Desktop/Project/MyDailyNews diff --check

Get-Process llama-server -ErrorAction SilentlyContinue
```

Expected test result at this handover: `Ran 290 tests ... OK`. `diff --check`
is clean apart from Windows LF/CRLF notices, and no llama server should be
running.

Do not rerun or overwrite
`semantic_thread_real_blind_broad_20260830_v1`. If a development evaluation is
needed, use an explicitly new output directory and label it inspected. Start a
local model server only for a concrete bounded hypothesis, and stop it after
the run.

## Current worktree

The branch is `main` at `e77dee7`. All work described in this handover is
currently uncommitted. Tracked modifications include configuration/model
contracts, prompts/schemas, delta/scaffold logic, memory context/retrieval/
StoryStore, pipeline wiring, roadmap text, and story-identity tests. Untracked
work includes the three real-corpus artifacts, semantic/reranker/AlignScore
adapters, claim contract, evaluator tools/tests, and the new documentation.

Generated `models/`, `output/`, `state/`, `.venv-cpu-test/`, CPU configs, and
`tools/llama.cpp/` are ignored. An older ignored local Codex-agent bridge may
also still exist. Do not use `git clean`, `git reset --hard`, or checkout-based
reversion on this tree.

## Copy/paste prompt for the next agentic chat

```text
Work in C:\Users\daroi\Desktop\Project\MyDailyNews.

Read docs/HANDOVER-story-thread-semantic-evaluation-2026-08-30.md completely,
then read docs/claim-thread-architecture-2026-08-30.md,
docs/blind-real-news-evaluation-protocol.md, and
docs/real-blind-corpus-methodology-2026-08-30.md. Preserve the dirty worktree;
do not reset, clean, or overwrite unrelated changes.

Treat the 120-document real-news report as the immutable first valid broad
blind diagnostic. It is now inspected development data. Do not tune against it
and then describe another score on it as blind. The initial preflight was
aborted and invalid; the old perfect synthetic claim/thread reports were
overfit and superseded.

Qwen3 1.7B failed semantic deltas even with a retained correct story (1/39,
0/35 actual changes). AlignScore disagreement gating did not rescue it. The
Qwen3 0.6B reranker is optional and disabled; hard rejection reduced inspected
retrieval recall. The specialized Qwen semantic cascade is evaluation-only,
while the general claim scaffold/validator and StoryStore v4 are production-
wired.

Continue with a generally applicable two-tier story memory: bounded 30-day hot
facts/events plus durable compact story anchors and archival rehydration.
Measure retrieval recall by time gap on development data, then add molecular
source-span claims and evaluate a stronger semantic backend through the same
fixed evidence contract. Preserve fail-open visibility and all anti-overfitting
invariants. Run the full unittest suite before handoff.
```
