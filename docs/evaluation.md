# Evaluation framework

## What this evaluates

The evaluator measures the product promise: retrieve profile-relevant source
material, keep story identity stable across days, identify a genuinely new
state, and choose full report, continuing bullet, or omission without inventing
facts.

It does not collapse this into one attractive but ambiguous score. The report
keeps identity, semantic change, materiality, display policy, selection,
faithfulness, and runtime separate.

## Run it

Run the production deterministic baseline:

```powershell
python tools/run_change_monitor_evals.py --adapter heuristic
```

Route the same cases through the real production RSS parser and an in-memory
fake HTTP transport (still no network):

```powershell
python tools/run_change_monitor_evals.py --adapter heuristic --fixture-mode rss
```

Verify the harness itself with the scripted oracle:

```powershell
python tools/run_change_monitor_evals.py --adapter oracle `
  --fail-under story_identity_pairwise_f1=1 `
  --fail-under delta_type_accuracy=1
```

Score output from any pipeline, model, or external evaluation framework:

```powershell
python tools/run_change_monitor_evals.py `
  --adapter predictions `
  --predictions path\to\predictions.json
```

Run the repository's delta task directly against a configured local model:

```powershell
.\tools\prepare_cpu_test_config.ps1 `
  -SourceConfig config.local.json `
  -DestinationConfig config.cpu-qwen1.7b-full.json `
  -FullPipeline

python tools/run_change_monitor_evals.py `
  --adapter delta-model `
  --config config.cpu-qwen1.7b-full.json `
  --model-role summary `
  --delta-output-mode decision_only `
  --model-max-new-tokens 384 `
  --delta-max-articles-per-batch 2 `
  --arc case-001
```

CPU configs are generated from the user's local profile and are intentionally
Git-ignored. The preparation script refuses to overwrite an existing file.

This path uses the normal structured `DeltaExtractor`, bounded prior-story
memory, and fake sources. The report embeds request/retry/token diagnostics and
calculated output tokens per second when the client reports token usage. The
`decision_only` contract removes redundant editorial arrays and limits the
model to identity/change/disposition classification. This matters for small
models: the full contract can spend its output budget serializing structure
before reaching the decision. Start with one `--arc`, then remove the filter
for a full run. `--split development` and `--split holdout` are also available.

Optional CI gates support both quality floors and error ceilings:

```powershell
python tools/run_change_monitor_evals.py --adapter predictions `
  --predictions path\to\predictions.json `
  --fail-under story_identity_pairwise_f1=0.85 `
  --fail-under material_continuation_recall=0.80 `
  --fail-over false_suppression_rate=0.05 `
  --fail-over unchanged_full_report_rate=0.20
```

Reports are written as JSON and Markdown under `output/evaluations/` by
default. The JSON includes every prediction so a failure can be traced to one
arc, date, and document.

## Architecture

```text
Versioned corpus
  |-- public profile + dated source documents --> fixture provider
  |                                              |-- direct candidate seam
  |                                              `-- real RSS parser seam
  |
  `-- private gold labels ----------------------> scorer

Adapter (oracle | production fallback | local delta model | prediction file)
  --> structured predictions
  --> overall metrics + trap slices + development/holdout slices
  --> JSON/Markdown report + optional CI gates
```

The fixture provider is offline and deterministic. It can supply candidates
directly for fast tests or render a fake RSS source and pass every evaluation
case through the real RSS parser with `--fixture-mode rss`. Neither path
receives gold labels.

The local-delta adapter exercises the existing model client and structured
delta prompt without live retrieval. The prediction-file adapter is the wider
plug-in boundary. A local Qwen run, a
frontier model, the full application pipeline, Promptfoo, DeepEval, or a human
annotation tool can all be scored if they emit the same records. A generic eval
framework can therefore orchestrate calls without owning the product-specific
story and display metrics.

The local-delta adapter is deliberately a stage evaluation: deterministic
lexical recall proposes bounded prior stories, then the model decides identity,
change, materiality, and disposition. Its identity score therefore measures the
combined candidate-recall/model decision path, not unconstrained entity
resolution. Use the prediction-file contract for a true full-pipeline run.

## Corpus v1

`evals/cases/change_monitoring.v1.json` contains 15 multi-day arcs, 74 dated
documents, 50 simulated days, and four source-empty days. It includes:

- an elf interested in invented toenail magic, to prevent news-domain rules
  from masquerading as general intelligence;
- numeric identity conflicts such as Model 3 versus Model 4;
- proposal, committee, enactment, correction, and resolution transitions;
- same actor/different event and different actor/same phrase traps;
- syndication, exact duplicates, stale republication, and source disagreement;
- headline/body contradiction and required/forbidden source facts;
- niche profile relevance versus globally prominent distractions;
- keep-watch behavior, Unicode names, aliases, acronym collision, and long
  distractor context.
- 25 unrelated daily-noise documents mixed around sparse material updates;
- several-day gaps with unrelated-only coverage, fully source-empty days, and
  stale rewrites after the tracked event resumes.

Every current document has gold labels for canonical story, relationship,
change type, materiality, display policy, profile relevance, selection, and
required/forbidden fact IDs. The taxonomy is product-oriented rather than tied
to the current model schema.

Case IDs are intentionally opaque (`case-001`, and so on). Failure-mode names
live only in private trap tags and documentation, so the adapter cannot infer a
required behavior from a routing identifier.

The committed `holdout` split is a regression split, not a secret benchmark.
A true release gate should keep a second, private set outside the repository and
rotate part of it periodically.

## Metrics and interpretation

| Metric | What failure means |
|---|---|
| Story identity pairwise precision/recall/F1 | Stories are over-merged or fragmented across days. |
| Continuation delta accuracy | The system cannot state what changed after first report. |
| Novelty detection precision/recall/F1 | New information is missed or repeated information is mislabeled as new. |
| Material continuation recall | A real update may be missed. |
| Linked material continuation recall | An update was noticed only by treating the continuing story as brand new. |
| False suppression rate | A material, selectable update was hidden; this is the highest-risk error. |
| Unchanged full-report rate | Repeated facts still consume the daily brief. |
| Quiet-day abstention rate | The system invented output on a day with no source documents. |
| Display and selection metrics | Structured instructions are not being enforced. |
| Required-fact recall | The output omitted the actual source-backed delta. |
| Forbidden/unsupported-claim rate | The output crossed its evidence boundary. |
| Claim evaluation coverage | How much output has fact-level annotations; zero means faithfulness is unmeasured, not perfect. |
| p50/p95 latency | CPU feasibility and tail behavior. |

The deterministic adapter intentionally reports no fact IDs, so its claim-level
coverage is zero and faithfulness scores are `null`. It measures policy code,
not final prose.

### Expanded deterministic baseline

The 74-document production-fallback run on 2026-08-29 produced:

| Metric | Result |
|---|---:|
| Story identity pairwise F1 | 0.2666 |
| Relationship accuracy | 0.7432 |
| Overall delta accuracy | 0.7162 |
| Continuation delta accuracy | 0.1923 |
| Novelty detection F1 | 0.9268 |
| Linked material continuation recall | 0.0000 |
| Display policy accuracy | 0.5135 |
| False suppression rate | 0.0000 |
| Unchanged full-report rate | 0.8462 |
| Profile relevance accuracy | 0.5405 |
| Quiet-day abstention rate | 1.0000 |

This is a safety baseline, not a quality target. It confirms that the fallback
fails open instead of hiding unfamiliar updates, while quantifying why lexical
identity and duplicate logic cannot deliver the product promise by themselves.
The apparently strong binary novelty score is not enough: linked-continuation
recall shows that the fallback often notices changed words only by fragmenting
an ongoing story into a new one.

### Qwen3 1.7B CPU benchmark

The bounded `decision_only` run used Qwen3 1.7B Q4_K_M through llama.cpp with
a 2,048-token context, 1,400 input tokens, 384 output tokens, at most two
articles per batch, four of eight logical CPU threads, one server slot, and
below-normal process priority.

| Measure | Result |
|---|---:|
| Documents / model requests | 74 / 56 |
| End-to-end delta runtime | 1,947.5 s (32m 27.5s) |
| Structured output throughput | 5.0401 tokens/s |
| Valid requests / retries | 56 / 0 |
| Predictions using deterministic fallback | 3 |
| Story identity pairwise F1 | 0.6234 |
| Relationship accuracy | 0.5811 |
| Overall / continuation delta accuracy | 0.2297 / 0.1923 |
| Novelty detection F1 | 0.9412 |
| Linked material continuation recall | 0.8125 |
| Display policy accuracy | 0.3649 |
| Unchanged full-report rate | 0.8205 |
| Quiet-day abstention rate | 1.0000 |

The compact contract fixed execution reliability but not semantic quality. The
model over-merged unrelated events (`new_story` became `same_story` 27 times)
and overused `material_update` (34 gold `new` items and five gold `unchanged`
items received that label). High novelty F1 is therefore not evidence of good
change monitoring; the noise-heavy corpus contains many genuinely new
distractors, and an always-new/material policy scores deceptively well.

The noisy/quiet arcs made the architectural boundary visible. Their display
accuracy was 0.0968 and unchanged full-report rate was 0.9259. This adapter
feeds every fixture document into delta classification and retains the current
deterministic profile-relevance function; it does not exercise the production
headline-selection model. Profile relevance was therefore identical for Qwen
and the heuristic baseline, and 28 of 29 irrelevant documents were considered
eligible. Treat this as a combined recall/policy-stage failure, not a clean
measurement of Qwen's personalization ability. A full-pipeline adapter is the
next required extension.

Claim-level faithfulness remains unmeasured because the decision adapter does
not emit mapped fact IDs. Structured validity was perfect, but that must not be
reported as factual accuracy.

The guarded run averaged 43.49% total-machine CPU and peaked at 50.0%. Minimum
available physical memory was 3,204 MB and peak llama-server working set was
6,089.5 MB. The watchdog never fired, and the server was stopped after the run.

## Prediction contract

Each record identifies one fixture document and supplies structured behavior:

```json
{
  "arc_id": "case-001",
  "date": "2026-01-03",
  "document_id": "elf-03",
  "predicted_story_id": "local-cluster-7",
  "relationship": "same_story",
  "delta_type": "status_change",
  "material": true,
  "display": "full_report",
  "profile_relevance": "must_select",
  "selected": true,
  "reported_fact_ids": ["elf_f4"],
  "unsupported_claims": [],
  "latency_ms": 842.3
}
```

`predicted_story_id` need not match the gold name; scoring compares pairwise
cluster membership. `reported_fact_ids` may be `null` when claim mapping has not
been performed. Do not replace `null` with an empty list: those mean
"unmeasured" and "measured, but no facts reported," respectively.

## Adversarial review of the harness

The following safeguards are implemented and covered by tests:

- Public adapter input excludes canonical IDs, expected labels, trap tags,
  splits, and fact catalogs.
- Public case IDs are opaque rather than labels such as `rumor-correction` or
  `headline-body-mismatch`.
- Corpus validation rejects duplicate IDs, missing expectations, invalid labels,
  malformed source URLs/timestamps, impossible same-story chronology,
  overlapping required/forbidden facts, and non-chronological days.
- JSON booleans and claim-ID arrays are type-checked rather than truthiness-
  coerced, so values such as `"false"` cannot silently become `true`.
- A perfect oracle proves the scoring ceiling is reachable.
- Fault injection proves the scorer detects over-merging, omission of material
  updates, missing predictions, unsupported claims, and invented output on a
  source-empty day.
- Missing predictions fail as explicit uncertain/non-selected records; they are
  never silently removed from denominators.
- Duplicate, extra, or invalid prediction records produce a non-zero CLI exit;
  malformed submissions cannot pass only because their aggregate score is high.
- Claim metrics are unavailable when claim annotations are absent, avoiding a
  false faithfulness score of 100%.
- Trap and split slices reduce the chance that an aggregate score hides one
  catastrophic failure class.
- Production enforcement tests remove omitted articles from source, evidence,
  recall, prior-report, handoff, and coverage paths. If every item is an
  unchanged repeat, final generation is skipped and a deterministic empty brief
  is emitted instead of asking a model to improvise from no current sources.

Remaining risks and planned extensions:

- Gold labels are hand-authored and need a second-person adjudication pass.
- Fact-ID mapping from free prose is not automated yet; initially use human
  annotation or a blinded frontier judge plus spot checks.
- The production adapter currently covers deterministic retrieval/identity/
  profile/delta behavior, not every orchestrator stage or final prose.
- Real-world source snapshots should be added after licensing and privacy
  review. Synthetic data is excellent for traps but cannot reproduce all feed
  noise, entity ambiguity, and editorial language.
- The local-delta adapter is a stage test and currently bypasses the production
  headline-selection stage; unrelated-noise selection needs a full-pipeline
  adapter or a separate ranking contract.
- A private rotating holdout and mutation tests should be added before using
  scores as a release gate.

## Dataset strategy

Do not make the long-term set all-synthetic. Use three layers:

1. Semi-synthetic adversarial arcs for precise coverage of rare failures and
   deterministic regression tests.
2. Frozen, redistributable real-source snapshots with human labels for
   ecological validity.
3. A private rotating set of recent stories and profiles for release decisions
   and anti-overfitting.

Start with v1 as the executable contract. Add a new case whenever a production
failure appears, but keep a portion of new cases private so the pipeline cannot
special-case the public suite.
