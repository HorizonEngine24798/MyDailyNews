<!-- generated-by: gsd-doc-writer -->

# Story quality: findings and next steps

Updated: 2026-09-05

## Where the project stands

MyDailyNews is intended to remember an ongoing news story and report it again
only when the new coverage contains a meaningful change. The surrounding
pipeline works, local inference now works on this machine, and a durable
`StoryStore` already exists. The unresolved problem is the core semantic one:
the tested models do not reliably distinguish a genuinely new fact from a
restatement, correction, status change, or unrelated event.

This is a quality problem rather than a CPU-enablement problem. Do not wire the
experimental semantic backend into normal briefs yet.

## What is working

- `llama.cpp` build `b10819` is installed locally under
  `tools/llama.cpp/b10819/`.
- `models/Qwen3-4B-Q4_K_M.gguf` is available as the small CPU-friendly model.
- The local Gemma 4 12B QAT Q4 model is also usable through `llama.cpp`.
- The RTX 3090 is detected and both models can be offloaded to it.
- The evaluation harness can replay multi-day stories, isolate semantic
  decisions with oracle candidate selection, and report identity and delta
  accuracy separately.
- `StoryStore` schema v4 already persists bounded source facts, thread events,
  source provenance, and the IDs of facts last shown to the user.

The installed models, llama.cpp binaries, and evaluation outputs are local
ignored assets rather than repository dependencies.

## Model findings

The most useful comparison used the 120-document, 18-family real-news corpus,
the same 2,048-token semantic contract, no AlignScore gate, and oracle mode.
The replay required a semantic decision for 39 cases. Its retained 30-day
lifecycle means this is not an all-continuation test.

| Model | Story identity on semantic cases | Correct identity + delta | Abstentions | Model time |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 4B Q4_K_M | 38/39 | 3/39 (7.69%) | 0/39 | 50.5 s |
| Gemma 4 12B QAT Q4_0 | 29/39 | 0/39 (0%) | 6/39 | 75.0 s |

On the older 74-document corpus, both models reached only 3/25 (12%) on the
same required identity-plus-delta measure. Both corpora have now been inspected
and should be treated as development sets, not as fresh blind evidence.

Approximate `llama-bench` generation rates were 21.5 tokens/s for Qwen on CPU,
177.7 tokens/s for Qwen with GPU offload, and 87.5 tokens/s for Gemma with GPU
offload. These establish that inference is practical; they do not predict
semantic accuracy.

### Interpretation

1. A larger model did not rescue the current contract. Gemma was slower and
   less accurate under a harness originally shaped around Qwen, so this is a
   failure of Gemma under this contract—not proof that Gemma is generally the
   weaker model.
2. Qwen usually found the correct story when the candidate was supplied, but
   almost always assigned the wrong delta. That isolates semantic differencing
   as the immediate bottleneck in the oracle slice.
3. The current task asks too much at once: sentence-level claims, many relation
   labels, aggregation precedence, materiality, and display policy. Composite
   news sentences also contain several propositions, attribution, modality,
   quantities, and time, which makes a single relation label brittle.
4. AlignScore/NLI did not behave like an oracle in earlier experiments. It may
   help rank or audit pairs, but it should not decide story identity or suppress
   a story by itself.

## Simplest useful design

Keep one persistent component: `StoryStore`. Do not add a separately persisted
`StoryPatch` type yet.

A stored fact should be small and source-backed:

- stable fact ID;
- normalized fact text;
- source article ID and URL;
- a verbatim supporting quote;
- active/inactive state; and
- optional `replaces_fact_id` for a correction or state transition.

The proposed flow is:

1. Retrieve the likely story.
2. Give the model the story's active facts and one new article.
3. Ask only which source-backed facts are new and whether any replace an
   existing fact.
4. Deterministically reject unknown fact IDs, missing/non-verbatim quotes, and
   malformed replacements.
5. Append validated facts and retire replaced facts in `StoryStore`.
6. Separately decide whether the validated change is important enough to show.

The model's interpretation cannot be made fully deterministic. What can be
deterministic is the accepted mutation: the same validated proposal applied to
the same store produces the same result, is traceable to source text, and can
be replayed or rejected.

A single mutable story summary is simpler on paper but loses provenance and
makes repetition, contradiction, and correction hard to audit. Fact IDs,
quotes, optional replacement, and `last_user_visible_fact_ids` are the minimum
complexity that protects the product promise.

## Next experiment

Build an evaluation-only function with no production writes:

```text
detect_changes(existing_story_facts, new_article)
  -> new_facts[{text, source_quote, replaces_fact_id?}]
```

For the first experiment:

- use the correct prior story/facts directly, so retrieval is out of scope;
- remove materiality, display choice, relation taxonomies, and prose summaries;
- require every proposed fact to quote the article verbatim;
- score new-fact precision/recall, false-new rate on repetitions, quote support,
  and replacement accuracy; and
- set pass criteria before comparing models or changing prompts.

If this narrow task fails, stop. More storage types or retrieval machinery will
not fix the core capability. If it succeeds, minimally extend the existing
`StoryStore` to accept validated facts and optional replacements, then test the
same flow with real retrieval.

## Guardrails to preserve

- Do not add phrase-, entity-, publisher-, location-, URL-, or corpus-specific
  semantic rules, and do not copy evaluation examples into prompts.
- Gold labels, family IDs, and diagnostic tags must never enter retrieval or
  model input.
- Preserve source attribution, negation, quantities, modality, and time.
- Uncertain evidence must not mutate confirmed fact state or authorize
  suppression.
- A model may cite only the bounded facts and source text in its request.
- Deterministic omission is safe only for a validated exact repetition of a
  confirmed story.
- A reranker can reorder supplied candidates but cannot recover a story that
  retrieval omitted.

## Retrieval decision for personal production

The present lexical top-three retrieval gate is not a sufficient production
boundary. It can omit the correct story before the semantic reranker sees it;
the reranker therefore cannot compensate for brittle wording, aliases, or a
long time gap.

The expected upper bound of 10,000 retained stories is small enough for exact
dense retrieval. Store one compact embedding per `StoryRecord`, re-embedding
it when its durable facts change. For each new article, embed it once and
compare it with every retained story embedding. This does not require a vector
database or approximate-nearest-neighbour index at that scale.

Do not run the Qwen cross-encoder over all 10,000 stories in normal operation:
that requires one model inference per article-story pair (and the current
implementation scores pairs sequentially). Instead:

1. take the dense top 100 stories;
2. union them with the lexical top 10, preserving strong rare-name, number,
   and exact-phrase matches;
3. rerank that compact union with the semantic reranker;
4. choose the best candidate only when its calibrated score clears the
   new-story threshold; otherwise create a new story; and
5. preserve the reranker order through memory-context construction.

Before adopting it, use exhaustive reranking only as an evaluation baseline:
compare lexical top 3, exhaustive reranking across every retained story, and
dense-plus-lexical retrieval followed by reranking. Measure same-story
recall@1/@3/@100, top-1 identity accuracy, false links for genuinely new
stories, time-gap slices, and latency. The already inspected corpora remain
development data, so freeze the choice before collecting a new holdout.

## Deferred work

### After fact differencing is proven

- Extract molecular propositions while preserving attribution, negation,
  modality, quantities, and time; fall back to the original source sentence
  when extraction is unsupported.
- Implement the dense-plus-lexical retrieval experiment above once fact
  differencing is proven.
- Split memory into a bounded hot evidence store and a tiny durable story
  anchor so a 30-day expiry does not erase long-running story identity.
- Derive materiality and display policy from validated fact changes in a
  separate testable layer. Model confidence or prose must not directly
  authorize suppression.
- Add a full-pipeline evaluation covering article selection, same-day grouping,
  rendering, and the final brief.
- Enable only two explicit feedback signals: `Useful` and `Keep watching`.
  Absence of interaction is not negative feedback.
- Freeze the next system before collecting a new blind corpus with later dates,
  unseen story families, and a changed source mix.

### Production hardening

- Reconcile the generic prompt/schema/normalizer with the canonical relation
  labels if the relation system remains in use.
- Preserve semantic reranker ordering through context construction and
  recompute added/repeated claims after accepted semantic edges.
- Ensure retained events cannot reference evicted facts and decide how
  corroboration from independent sources survives exact-text deduplication.
- Add global byte/story limits, robust date pruning, backdated-write handling,
  migration coverage, a concurrent-writer lock, atomic recovery, and tests for
  corrupt canonical state.
- Add independently adjudicated gold labels, source-span/fact-ID faithfulness
  scoring, confidence intervals, resource measurements, and private rotating
  holdouts.

### Ideas intentionally not scheduled

- a graph database or Graphiti-style memory layer;
- a universal event/relation ontology;
- a separately persisted StoryPatch abstraction;
- learned materiality before factual delta is reliable;
- multi-agent adjudication; and
- map zoom, clustering, timelines, animation, polygons, or external geocoding.

The geographic map is already implemented as a view over the perspectives
report (`story_loci`, `/api/map`, and a local place lookup). Expand it only if
measured missing locations or marker overlap justify the work. A country that
was searched but yielded no result must not be described as having no coverage.

## Evaluation references

- [Evaluation framework](evaluation.md)
- [Real-news evaluation protocol](blind-real-news-evaluation-protocol.md)
- [Frozen corpus methodology](real-blind-corpus-methodology-2026-08-30.md)
- `evals/cases/change_monitoring.real_blind.v1.json`
- Local comparison reports under `output/evaluations/model_compare_*_20260905/`

The protocol and methodology documents remain separate because the sealed
corpus and its integrity tests refer to them directly.
