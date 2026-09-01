# Handover: Qwen CPU evaluation and next story-understanding work

> Historical handover. For the current implementation, real-news blind
> diagnostic, storage changes, and next work, read
> [`HANDOVER-story-thread-semantic-evaluation-2026-08-30.md`](HANDOVER-story-thread-semantic-evaluation-2026-08-30.md).

Date: 2026-08-29
Last updated: 2026-08-30
Repository: `C:\Users\daroi\Desktop\Project\MyDailyNews`

## Executive state

The expanded evaluation harness and first production StoryStore slice are
working and all 255 tests pass. The corpus still contains 15 story arcs, 74
dated documents, 50 simulated days, four completely
source-empty days, and 25 intentionally unrelated daily-noise documents around
sparse tracked-story updates.

The CPU-only Qwen3 1.7B Q4_K_M benchmark completed successfully after the delta
task was split into an explicit classification-only contract. The complete run
took 32m 27.5s for 56 model requests. All 56 responses were valid structured
JSON with no retries or transport errors. Resource limits were respected and
the llama server was stopped afterward.

Execution reliability is solved for this bounded experiment; semantic quality
is not. The main full-run scores were:

| Metric | Qwen result |
|---|---:|
| Story identity pairwise F1 | 0.6234 |
| Relationship accuracy | 0.5811 |
| Delta-type accuracy | 0.2297 |
| Continuation delta accuracy | 0.1923 |
| Novelty F1 | 0.9412 |
| Linked material-continuation recall | 0.8125 |
| Display-policy accuracy | 0.3649 |
| Unchanged full-report rate | 0.8205 |
| Quiet-day abstention | 1.0000 |

Do not interpret the high novelty score in isolation. The model over-predicted
`material_update`, over-merged unrelated stories, and repeated most unchanged
items. Claim-level faithfulness is still unmeasured.

Follow-up evaluator work completed the final bounded capability protocol using
the same corpus. Reports now separate candidate recall, relationship judgment,
actual candidate-key linkage, delta given a correctly linked identity, and
display given correct semantics. The two oracle modes are prominently marked
as contaminated capability ceilings.

| Condition | Candidate supplied | Relationship given candidate | Correct key linked | Delta given correct link | New-story overmerge |
|---|---:|---:|---:|---:|---:|
| Broad baseline | 0.8846 | 0.8696 | 0.6522 | 0.2000 | 0.5745 |
| Gold-blind top three | 0.6154 | 0.9375 | 0.8750 | 0.1429 | 0.7234 |
| Oracle candidate | 1.0000 | 0.9615 | 0.8077 | 0.0952 | 0.7872 |
| Oracle candidate + fact packet | 1.0000 | 0.8077 | 0.5000 | 0.0000 | 0.7447 |

The perfect candidate result is decisive: candidate availability is not the
remaining Qwen bottleneck. The model recognized most supplied continuations,
but frequently failed to return the supplied key and almost never identified
the specific delta. The fact packet did not rescue it. It mostly mapped every
continuation class to generic `material_update` and said `same_story` even when
no candidate existed.

The evaluator was hardened after adversarial review. Selectively missing
candidate metadata now counts as a miss; unknown/current/future document
references are rejected; production candidate keys must agree with the prior
prediction they cite; and conditional identity requires an actual link to the
correct supplied candidate. Saved predictions were rescored into new output
directories, leaving every original report untouched. See
[`docs/evaluation.md`](evaluation.md#completed-qwen3-17b-results-2026-08-29)
for full results, runtimes, resource diagnostics, and commands.

Original final-run artifacts are local under
`output/evaluations/qwen_final_investigation_20260829/`; the hardened linked-
identity rescores are under its `rescored-linked-identity-v2/` child. These paths
are ignored by Git. The llama server was stopped and verified absent after the
last run.

The four-mode ablation is complete. Do not continue generic Qwen prompt,
context, or quantization sweeps. Accept this model as a constraint and return
to architecture work. A new model run is justified only by a narrowly stated
regression question after an architectural change.

The first architecture slice requested after that stopping point is now also
complete. Production uses provisional current-story keys, retrieves at most
three source-backed prior candidates, enforces candidate-bounded model links,
and persists an inspectable unified story store. The gold-assisted retrieval diagnostic
recalled 24/25 prior-day continuations at rank one (0.9600), supplied no
candidate for 46/47 truly new stories (0.9787), and retrieved the one labelled
related-theme case. This diagnostic uses private canonical identity only for
historical writeback; it isolates retrieval and is not an end-to-end score.

## Product promise being evaluated

MyDailyNews should behave as a personalized change monitor:

1. A user defines a profile.
2. The system finds profile-relevant stories amid unrelated daily news.
3. It maintains stable story state across days.
4. It emits another full brief only for a source-backed material change.
5. The user can eventually provide only `Useful` and `Keep watching` feedback.

The long-term architecture should make the model explain a small,
source-grounded comparison. It should not ask a tiny model to independently
retrieve, resolve identity, remember state, infer novelty from raw history,
choose policy, and write a report in one call.

## Repository and worktree state

The evaluation, product, and CPU-tooling work is split into reviewable commits
on `main`:

| Commit | Purpose |
|---|---|
| `c7b0f4e` | Domain-general story matching, conservative repeat suppression, and writer-context filtering |
| `534737d` | Offline adversarial change-monitor evaluator and 74-document corpus |
| `775ae77` | Lightweight-install TTS fallback when `soundfile` is unavailable |
| `38a6bd7` | Guarded CPU diagnostics, replay tools, and local-asset ignore rules |
| `d248db7` | Stage-conditioned model investigations and adversarial evaluator hardening |
| `42e4a1d` | Source-backed story identity gate, heuristic retrieval, and provenance writeback |
| `486f7a7` | Qwen investigation results, retrieval results, roadmap, and handover |
| `8d2efa8` | Unified StoryStore, legacy migration, separated heuristic retrieval, and memory housekeeping |

The first seven checkpoints are already on `origin/main`; `8d2efa8` is the
tested StoryStore checkpoint prepared in this final follow-up and should be
pushed together with this handover update.

An older local-only Codex CLI comparison bridge remains deliberately outside
these commits. Its tracked hooks appear as local modifications in
`mydailynews/ai/factory.py`, `mydailynews/app/config.py`, and
`mydailynews/app/models.py`; its implementation file is ignored. Do not reset,
clean, checkout, or overwrite those files without first deciding whether to
keep that local bridge.

The downloaded model, llama.cpp binaries, CPU virtual environment,
machine-specific CPU configs, and generated benchmark reports are ignored and
were not pushed. They remain available on this machine at the paths below.

Useful first checks:

```powershell
cd C:\Users\daroi\Desktop\Project\MyDailyNews
git -c safe.directory=C:/Users/daroi/Desktop/Project/MyDailyNews status --short
.\.venv-cpu-test\Scripts\python.exe -m unittest discover -s tests
```

Expected test result at handover: `Ran 255 tests ... OK`.

## Production story architecture completed

The following is now in the normal memory-enabled brief path:

1. Candidate retrieval no longer assigns a prior story key. Every current item
   receives a collision-safe provisional key and keeps that key through
   selection.
2. The heuristic retriever uses title aliases, recurring entity tokens, event
   tokens, numeric identity, source-body fact overlap, and numeric-conflict
   penalties. It returns at most three candidates at the measured default
   threshold of `0.25`. It is explicitly named heuristic because it combines
   hand-weighted lexical signals, not sparse and dense retrieval.
3. An unvalidated retrieval candidate may softly affect ranking through recent
   coverage, but its penalty is capped at `-0.35` and it cannot trigger the old
   hard recent-story suppression. The item reaches delta classification when
   capacity permits.
4. `candidate-gated.v1` validates every model decision. No candidate forces
   `distinct_story/new/full_report`; an unknown or ambiguous prior key becomes
   uncertain and remains visible; only an explicitly supplied prior key may be
   linked. Missing model decisions are synthesized conservatively.
5. `state/memory/story_store.json` is the single durable source of truth. Each
   record combines identity, lifecycle, semantic delta state, exact bounded
   source facts, provenance, aliases/retrieval signals, source-document
   history, and last-user-visible fact IDs.
6. Delta context contains at most three retrieved baselines and at most four
   cited facts per baseline. The compact `decision_only` prompt preserves that
   provenance rather than reducing the baseline back to unsupported prose.
7. After output, one StoryStore write records all selected source observations
   while marking only rendered articles as user-visible, then applies story
   lifecycle retention. If `story_store.json` is absent, legacy
   `story_index.json` and `story_ledger.json` records are merged in memory; the
   first normal write creates the canonical store. The legacy files remain as
   untouched migration backups and are ignored afterward.

The new diagnostic command is:

```powershell
.\.venv-cpu-test\Scripts\python.exe tools\run_story_retrieval_diagnostics.py
```

Current result on all 74 documents:

| Retrieval measure | Result |
|---|---:|
| Historical prior-day continuations | 25 |
| Recall@1 / Recall@3 | 0.9600 / 0.9600 |
| Mean reciprocal rank | 0.9600 |
| Truly new stories with no candidate | 0.9787 (46/47) |
| Related-theme candidate recall | 1.0000 (1/1) |
| Mean candidates per document | 0.3919 |

One historical continuation remains missed: `rum-03`, where a supposed buyer's
unrelated logistics partnership indirectly resolves an acquisition rumor. Its
best score is `0.2103`. Lowering the global threshold enough to catch this is a
poor trade-off; recovering it requires an explicit actor/relation edge rather
than broader lexical matching. The safe failure is a visible new item, not a
silent merge or omission.

This slice does not yet prove material change. StoryStore stores source facts,
but deterministic old-fact/new-fact comparison, contradiction/status handling,
and final display derivation remain next. No new Qwen benchmark or production
news run was performed for this architecture turn, so `story_store.json` will
be created or populated by the next normal memory-enabled write.

## Evaluation corpus added

The corpus is in [`evals/cases/change_monitoring.v1.json`](../evals/cases/change_monitoring.v1.json).

Current size:

- 15 multi-day arcs;
- 74 documents;
- 50 simulated dates;
- 12 development arcs / 56 development documents;
- 3 holdout arcs / 18 holdout documents;
- four dates with `documents: []` and `expectations: []`;
- 25 unrelated documents in the two new long/noisy arcs.

The two added arcs are:

- `case-014`: Lumen seawall sensors across six dates, one source-empty day,
  several unrelated-only days, one verified operational update, and a stale
  next-day rewrite.
- `case-015`: Nacre Library roof restoration across seven dates, three
  source-empty days, unrelated-only days between updates, a material project
  pause, and a stale rewrite.

The empty-day schema is intentional. Corpus validation still requires every
arc to contain at least one document overall, but no longer requires every day
to have one.

The scorer now reports:

- `quiet_days`;
- `quiet_day_outputs`;
- `quiet_day_false_output_rate`;
- `quiet_day_abstention_rate`.

The `hallucinate_quiet_days` negative-control adapter invents one unsupported
prediction on each empty day. The oracle abstains on all four; the fault is
caught on all four and produces four unknown prediction keys.

## Model-contract work

The original full delta schema asks the model to emit several redundant
editorial category arrays plus a verbose story decision. With a 256-token
output ceiling, Qwen filled or began filling those arrays and hit the length
limit before closing the JSON object.

Two full-contract smoke attempts failed before producing a score. Their raw
artifacts are:

- `output/diagnostics/ai_invalid_json/20260829_182508_620_delta_extraction_batch_1_1_case-001.txt`
- `output/diagnostics/ai_invalid_json/20260829_182508_621_delta_extraction_batch_1_1_case-001.json`
- `output/diagnostics/ai_invalid_json/20260829_183211_983_delta_extraction_batch_1_1_case-001.txt`
- `output/diagnostics/ai_invalid_json/20260829_183211_998_delta_extraction_batch_1_1_case-001.json`

Changes made to support a constrained model:

- `DeltaExtractionConfig.output_mode` now accepts `full` (default) or
  `decision_only`.
- `DELTA_DECISION_JSON_SCHEMA` contains only fields consumed by identity,
  change, materiality, and display policy.
- `DELTA_DECISION_USER` sends compact current source evidence and deduplicated
  candidate baselines instead of a full editorial report template.
- The normal full prompt also no longer repeats a complete JSON example that
  the server already receives through `response_format`.
- The evaluation CLI can override output mode, model output ceiling, and delta
  batch width.
- Decision-only evaluation forces `max_articles_dropped_to_avoid_split=0`, so
  a small context cannot silently drop lower-ranked test documents.
- A failed or malformed model call no longer discards the entire run. The local
  model adapter records `model_error`, uses deterministic fallback for affected
  documents, and continues.
- Prediction metadata now retains `decision_summary`; future reports can expose
  model prose for a separate faithfulness review.
- Future reports count `model_fallback_used` and `model_error_predictions`
  directly in `prediction_counts`.

Decision-only mode is experimental. It normalizes the unused editorial delta
lists to empty arrays. Do not enable it blindly for the complete production
pipeline until downstream consumers are checked or those lists are derived
deterministically from the decisions.

## CPU setup and exact profile

Hardware observed:

- 8 logical processors;
- approximately 16,241 MB physical memory;
- CPU only;
- no llama-server remains running at handover.

Local assets:

- llama.cpp server: `tools/llama.cpp/b10631/llama-server.exe`
- model: `models/Qwen3-1.7B-Q4_K_M.gguf`
- GGUF file size: approximately 1.28 GB
- server-reported parameters: 2,031,739,904
- quantization: Q4_K Medium
- training context metadata: 40,960 tokens
- benchmark context: 2,048 tokens

Server profile used:

```powershell
tools\llama.cpp\b10631\llama-server.exe `
  -m models\Qwen3-1.7B-Q4_K_M.gguf `
  -c 2048 -t 4 -tb 4 -b 256 -ub 128 `
  --device none -ngl 0 `
  --host 127.0.0.1 --port 8081 --parallel 1 `
  --metrics --reasoning off --prio -1 --poll 0 --no-webui
```

This uses four of eight logical processors, one request slot, no GPU offload,
and below-normal process priority. Do not increase threads or parallel slots on
this machine without a new guarded measurement.

Full benchmark command:

```powershell
.\.venv-cpu-test\Scripts\python.exe tools\run_change_monitor_evals.py `
  --adapter delta-model `
  --config config.cpu-qwen1.7b-full.json `
  --model-role summary `
  --fixture-mode direct `
  --delta-output-mode decision_only `
  --model-max-new-tokens 384 `
  --delta-max-articles-per-batch 2 `
  --output output\evaluations\qwen_cpu_20260829\qwen-full-decision-only
```

The direct fixture seam was used because the deterministic direct and fake-RSS
runs produced identical scores. This isolates model latency from parser
overhead while the separate RSS test still checks production parser behavior.

## Benchmark sequence and artifacts

### Generic model diagnostic

Report:
`output/evaluations/qwen_cpu_20260829/model-diagnostic/20260829_182044/report.json`

- Three of three toy responses were parseable JSON.
- Server-reported generation was roughly 11-12 tokens/s on small prompts.
- The `brief_summary` toy answer omitted the requested two bullets despite
  being valid JSON. Syntax validity was therefore already stronger than
  instruction following.

### Decision-only smoke

Report:
[`output/evaluations/qwen_cpu_20260829/qwen-smoke-decision-only-case-001/evaluation_report.json`](../output/evaluations/qwen_cpu_20260829/qwen-smoke-decision-only-case-001/evaluation_report.json)

- Four requests, all valid, zero retries, zero fallback.
- Runtime: 121.8 seconds.
- Structured output throughput: 4.0321 tokens/s.
- Identity F1: 1.0000.
- Delta accuracy: 0.5000.
- Display accuracy: 0.2500.

### Full decision-only benchmark

Primary artifacts:

- [JSON report](../output/evaluations/qwen_cpu_20260829/qwen-full-decision-only/evaluation_report.json)
- [Markdown report](../output/evaluations/qwen_cpu_20260829/qwen-full-decision-only/evaluation_report.md)
- [Resource samples](../output/evaluations/qwen_cpu_20260829/qwen-full-decision-only/resource_samples.csv)
- `runner.stdout.log` and `runner.stderr.log` in the same directory
- server logs under `output/evaluations/qwen_cpu_20260829/server-balanced/`

Execution diagnostics:

| Measure | Result |
|---|---:|
| Documents | 74 |
| Model requests | 56 |
| Valid / invalid / transport errors | 56 / 0 / 0 |
| Retries | 0 |
| Input / output tokens | 27,526 / 9,815 |
| Model-call duration | 1,947.4 s |
| End-to-end duration | 1,947.5 s (32m 27.5s) |
| Output throughput | 5.0401 tokens/s |
| Mean / p95 per-prediction latency | 26.32 s / 42.82 s |
| Model decisions missing, then deterministically filled | 3 of 74 |

Resource watchdog:

- 647 samples at roughly three-second intervals;
- mean model CPU: 43.49% of the full machine;
- peak model CPU: 50.0%;
- minimum available physical memory: 3,204 MB;
- mean available physical memory: 5,283.7 MB;
- peak llama-server working set: 6,089.5 MB;
- no sustained low-memory or high-CPU guard event;
- server stopped after completion; available memory rose to about 9,655 MB.

The peak working set is much larger than the GGUF file. Treat the current
profile as feasible on this 16 GB PC, but not as evidence that it is ready for
a 6-8 GB phone. A phone also has less memory available to one process and may
have much lower sustained CPU performance due to thermal throttling.

## Quality analysis

### Qwen versus deterministic fallback

| Metric | Deterministic | Qwen decision-only |
|---|---:|---:|
| Story identity F1 | 0.2666 | 0.6234 |
| Relationship accuracy | 0.7432 | 0.5811 |
| Delta accuracy | 0.7162 | 0.2297 |
| Continuation delta accuracy | 0.1923 | 0.1923 |
| Novelty F1 | 0.9268 | 0.9412 |
| Material-continuation recall | 0.5625 | 0.9375 |
| Linked material-continuation recall | 0.0000 | 0.8125 |
| Display accuracy | 0.5135 | 0.3649 |
| Selection F1 | 0.6796 | 0.6542 |
| False suppression | 0.0000 | 0.0000 |
| Unchanged full-report rate | 0.8462 | 0.8205 |
| Quiet-day abstention | 1.0000 | 1.0000 |

The heuristic's apparently high relationship/delta scores are also
class-imbalance-sensitive: the added unrelated documents are usually first
observations. Its zero linked-continuation recall still shows that it does not
understand accumulated stories. Qwen materially improves linked identity, but
not enough to drive correct change or display policy.

### Main Qwen confusions

- 27 gold `new_story` items were labeled `same_story`.
- 34 gold `new` items were labeled `material_update`.
- Five gold `unchanged` items were labeled `material_update`; only two
  unchanged items were classified correctly.
- Seven gold `status_change` items became generic `material_update`.
- 32 gold omissions became full reports and six became continuing bullets.
- Only one gold omission was actually omitted.
- Three documents had no model decision and used deterministic fallback. There
  was no model exception: the valid response simply omitted those article IDs.

The model often reported confidence `1.0` even when wrong. Do not use its raw
confidence as a release gate without calibration.

### Holdout degradation

| Metric | Development | Holdout |
|---|---:|---:|
| Identity F1 | 0.6896 | 0.4210 |
| Delta accuracy | 0.2500 | 0.1667 |
| Continuation delta accuracy | 0.2273 | 0.0000 |
| Display accuracy | 0.4286 | 0.1667 |

This is a small holdout, but the drop is large enough to reject any claim that
the current prompt/model combination is robust.

### Unrelated noise and quiet days

The two new noisy arcs contain 31 documents total: six tracked-story documents
and 25 unrelated documents. Their aggregate results were:

- identity F1: 0.3810;
- delta accuracy: 0.2581;
- display accuracy: 0.0968;
- unchanged full-report rate: 0.9259;
- quiet-day abstention: 1.0000.

Quiet-day success is structural: when there are no source documents, the
adapter does not call the model and emits nothing. This proves that the pipeline
can remain silent, not that Qwen chose to abstain.

Noise failure is also partly architectural. The local-delta adapter deliberately
feeds all fixture documents into delta extraction and uses the deterministic
profile-relevance function. It bypasses the production headline-selection LLM.
Consequently Qwen and the heuristic have exactly the same profile-relevance
accuracy (0.5405) and irrelevant-selection rate (0.9655); 28 of 29 irrelevant
documents were considered eligible. The delta safety policy also refuses an
`omit` for a distinct new story, even if it is irrelevant, because omission is
only trusted for a high-confidence same-story unchanged item.

The new data therefore revealed a missing evaluation boundary: relevance and
delta cannot be meaningfully judged as one stage when upstream selection is
bypassed.

### Accuracy, faithfulness, instruction following, and novelty

- Accuracy: moderate pairwise identity but poor relationship and semantic
  delta accuracy. Over-merging is the dominant identity failure.
- Faithfulness: unavailable. `claim_evaluation_coverage` is 0 because model
  decisions are not mapped to corpus fact IDs. Valid JSON and source-bounded
  prompts are not substitutes for factual scoring.
- Instruction following: serialization was perfect under the compact schema,
  but semantic instruction following was weak (`instruction_following_score`
  0.4325). The model ignored first-observation and concrete-story constraints.
- Novelty: F1 0.9412 is misleadingly strong. Most unrelated noise documents are
  genuinely new, so overcalling new/material changes gets high recall. Repeat
  behavior is better represented by continuation delta accuracy (0.1923) and
  unchanged full-report rate (0.8205).
- Novel expression: not evaluated. This benchmark classifies decisions and
  does not generate a final news brief.

No frontier-model predictions were run against this 74-document corpus in this
turn. A future frontier comparison must emit the same prediction contract;
otherwise differences in retrieval, labels, or judge prompts will make the
comparison invalid.

## Architectural/adversarial review

What is sound:

- Gold labels, trap tags, fact catalogs, and splits remain private to the
  scorer; adapters receive only profile and source documents.
- Oracle scores are perfect, so the metric ceiling is reachable.
- Fault adapters demonstrate that over-merging, omission, unsupported claims,
  missing records, and quiet-day hallucinations are detected.
- Direct and fake-RSS deterministic results are identical.
- Every source document receives a prediction; missing predictions cannot
  disappear from denominators.
- Decision-only mode is explicit and defaults to off, avoiding silent behavior
  changes for larger models.
- Small-context batching no longer drops test articles to avoid another call.
- Resource constraints are measured rather than inferred from model file size.

Remaining threats:

- This is a stage adapter, not a complete production-pipeline adapter.
- Story identity measures deterministic candidate recall plus model choice, not
  unconstrained model entity resolution.
- Profile relevance is deterministic in this adapter, so the Qwen benchmark
  does not isolate personalization ability.
- Quiet-day abstention is a no-input control path, not model abstention.
- Claim/fact faithfulness is unmeasured and must remain reported as `null`.
- Gold labels are hand-authored and need independent adjudication.
- The public holdout is a regression split, not a secret benchmark.
- Tags on the two long/noisy arcs are highly correlated; their identical slice
  scores are not independent evidence.
- Catching `ValueError` in the local-model adapter preserves a long run but can
  hide malformed output behind fallback. The error and fallback counters must
  be treated as quality failures, not successes.
- The compact baseline list still exposes up to 12 prior stories. Qwen appears
  too willing to choose one, even for unrelated current evidence.
- The current evaluator does not store fact mappings for `decision_summary`;
  merely retaining the text does not make it faithful.

## Recommended next work, in order

Do not run another full Qwen benchmark until a story-state or stage-boundary
change creates a specific regression hypothesis. The next order should be:

1. Build deterministic source-fact differencing on top of StoryStore.
   Extract bounded fact records or source spans, compare current evidence with
   the last user-visible facts, and represent additions, contradictions,
   numeric changes, status transitions, resolutions, and insufficient evidence.
2. Derive display policy from validated relevance, identity, and fact delta.
   A full repeated story must require a demonstrated material fact change;
   otherwise emit a continuing watch line or omit it. Never let model prose
   directly authorize suppression.
3. Add a full-pipeline adapter or separate ranking adapter that runs real
   headline selection amid the 25 unrelated documents. Measure must-select
   recall, irrelevant selection rate, and top-k ranking before delta.
4. Add bounded relation edges for indirect continuations such as `rum-03`, and
   explicitly test current-day story grouping. Do not lower global retrieval
   thresholds merely to catch one indirect relationship.
5. Extend cases with explicit old/new source spans and require the pipeline to
   emit fact IDs or citations. Only then enable claim-level faithfulness scores.
6. Run fast smoke gates first (`case-014`, `case-015`, holdout), followed by the
   74-document retrieval diagnostic and full suite. Run Qwen only for a narrow
   post-architecture regression hypothesis.
7. Add adjudicated real-source snapshots and a private rotating recent-news set.
8. After relevance and deterministic delta policy are reliable, connect
   `Useful` and `Keep watching` to StoryStore state. Do not infer dislike from no
   interaction.

The key design conclusion is that a tiny model can reliably serialize a narrow
classification contract, but it should not own memory, candidate retrieval,
relevance, identity, delta proof, display policy, and prose simultaneously.

## Files most relevant to continue

- [`docs/evaluation.md`](evaluation.md): framework, commands, metrics, and full benchmark summary.
- [`docs/personalized-change-monitoring-roadmap.md`](personalized-change-monitoring-roadmap.md): product promise and delivery order.
- [`evals/cases/change_monitoring.v1.json`](../evals/cases/change_monitoring.v1.json): corpus and gold labels.
- [`mydailynews/evaluation/schema.py`](../mydailynews/evaluation/schema.py): corpus and prediction contracts.
- [`mydailynews/evaluation/scoring.py`](../mydailynews/evaluation/scoring.py): metrics, including quiet-day behavior.
- [`mydailynews/evaluation/adapters.py`](../mydailynews/evaluation/adapters.py): heuristic, local model, oracle, and fault adapters.
- [`mydailynews/evaluation/retrieval_diagnostics.py`](../mydailynews/evaluation/retrieval_diagnostics.py): gold-assisted isolation of production candidate retrieval.
- [`mydailynews/memory/story_store.py`](../mydailynews/memory/story_store.py): canonical identity, lifecycle, semantic state, bounded evidence, migration, and writeback.
- [`mydailynews/memory/story_retrieval.py`](../mydailynews/memory/story_retrieval.py): isolated heuristic candidate retrieval and source-signal extraction.
- [`mydailynews/analysis/identity_gate.py`](../mydailynews/analysis/identity_gate.py): candidate-bounded identity invariant.
- [`mydailynews/memory/context.py`](../mydailynews/memory/context.py): bounded source-backed candidate context.
- [`mydailynews/analysis/delta.py`](../mydailynews/analysis/delta.py): full and decision-only delta execution.
- [`mydailynews/ai/prompts.py`](../mydailynews/ai/prompts.py): compact decision prompt.
- [`mydailynews/ai/schemas.py`](../mydailynews/ai/schemas.py): compact decision schema.
- [`tools/run_change_monitor_evals.py`](../tools/run_change_monitor_evals.py): CLI and CPU-profile overrides.
- [`tools/run_story_retrieval_diagnostics.py`](../tools/run_story_retrieval_diagnostics.py): reproducible 74-document retrieval diagnostic.
- [`tests/test_evaluation_harness.py`](../tests/test_evaluation_harness.py): anti-leak, fault, compact-contract, and fallback tests.
- [`tests/test_story_identity_architecture.py`](../tests/test_story_identity_architecture.py): identity-gate, provenance, heuristic-retrieval, migration, and corpus regression tests.

## Copy/paste prompt for the next agentic chat

```text
Work in C:\Users\daroi\Desktop\Project\MyDailyNews.

Read docs/HANDOVER-qwen-cpu-evaluation-2026-08-29.md completely, then read
docs/evaluation.md and docs/personalized-change-monitoring-roadmap.md. Preserve
the dirty worktree and do not reset, clean, or overwrite unrelated changes.

Continue from the completed 74-document Qwen CPU benchmark, four-mode ablation,
and first production story architecture slice. Do not run another generic
model, prompt, context, or quantization experiment. Candidate-gated identity,
heuristic top-three retrieval, and unified `story_store.json`
source/provenance writeback are implemented; read their tests before changing
them. Build the next bounded
slice: deterministic old-fact/new-fact comparison and display policy based on
validated identity plus cited source evidence. Preserve fail-open behavior,
the elf/toenail-magic and numeric regressions, and the retrieval diagnostic
floors (recall@3 >= 0.95; new-story no-candidate rate >= 0.97). Keep
faithfulness unavailable unless emitted claims map to source facts/spans. Qwen
is an accepted weak component and may explain a validated decision, but it must
not own StoryStore identity, suppression, or state transition.
```
