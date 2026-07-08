# MyDailyNews Architecture

MyDailyNews is a local-first pipeline around an OpenAI-compatible chat endpoint. The app owns fetching, ranking, prompting, local state, and output files. The model server owns model loading, GPU offload, context size, and serving.

## Runtime Model

Default runtime:

```text
CLI or GUI -> config loader -> runtime checks -> NewsOrchestrator
    -> retrieval, ranking, analysis, AI calls
    -> output/, state/memory/, .cache/mydailynews/
```

By default, MyDailyNews expects an already-running local server such as LM Studio at `http://127.0.0.1:1234/v1`. It sends `/chat/completions` requests and leaves the server running.

The managed `llama-server` path is still available for advanced users. Set `manage_server=true` with `server_executable`, `server_model_path`, and `server_arguments` to let MyDailyNews start and stop `llama-server`. See [managed llama-server mode](llama_cpp_setup.md).

## Overall Flow

```mermaid
flowchart LR
    CONFIG["config + runtime checks"] --> BRIEFS["briefs module"]
    SOURCES["feeds, searches,<br/>prior reports"] --> BRIEFS
    MEMORY["state/memory<br/>coverage + preferences"] --> BRIEFS
    BRIEFS --> BRIEF_CALLS["LLM calls:<br/>score, group, evidence,<br/>delta, final brief"]
    BRIEF_CALLS --> STRUCTURED["structured briefs<br/>Markdown + JSON"]
    BRIEF_CALLS --> HANDOFF["handoff<br/>selected articles"]
    HANDOFF --> ENRICH["enrichment module"]
    ENRICH --> ENRICH_CALLS["LLM calls:<br/>plan threads + synthesize<br/>per enriched story"]
    ENRICH_CALLS --> ENRICHED["enrichment context<br/>Markdown + JSON"]
    STRUCTURED --> NARRATIVE["narrative brief<br/>1 LLM call"]
    ENRICHED --> NARRATIVE
    MEMORY --> NARRATIVE
    NARRATIVE --> TTS["optional TTS"]
    STRUCTURED --> OUTPUT["output/"]
    ENRICHED --> OUTPUT
    NARRATIVE --> OUTPUT
    TTS --> OUTPUT
    BRIEFS --> MEMORY
    BRIEFS --> CACHE[".cache/mydailynews"]
    ENRICH --> CACHE
```

## Briefs Module

```mermaid
flowchart TD
    SNAPSHOT["source snapshot<br/>feeds + searches + prior reports"] --> SCORE["headline scoring"]
    SUMMARY["summary_ai_client"] --> SCORE_CALL["LLM: headline scoring<br/>ceil(candidates / batch_size)<br/>plus single-item replays on bad JSON"]
    SCORE_CALL --> SCORE
    SCORE --> SELECT["deterministic selection<br/>caps, novelty, learned prefs"]
    MEMORY["state/memory"] --> SELECT
    SELECT --> FETCH["article text fetch"]
    FETCH --> GROUP["story grouping"]
    SUMMARY --> GROUP_CALL["LLM: story grouping<br/>0..N planner calls"]
    GROUP_CALL --> GROUP
    GROUP --> EVIDENCE["evidence distillation"]
    FETCH --> EVIDENCE
    ANALYSIS["analysis client<br/>summary or final"] --> EVIDENCE_CALL["LLM: evidence<br/>0..N article batches"]
    EVIDENCE_CALL --> EVIDENCE
    EVIDENCE --> DELTA["delta extraction"]
    SNAPSHOT --> DELTA
    ANALYSIS --> DELTA_CALL["LLM: delta<br/>0..N article/prior batches"]
    DELTA_CALL --> DELTA
    EVIDENCE --> FINAL_BRIEF["final brief generation"]
    DELTA --> FINAL_BRIEF
    MEMORY --> FINAL_BRIEF
    FINAL["final_ai_client"] --> FINAL_CALL["LLM: final brief<br/>1 call per brief<br/>general and detailed"]
    FINAL_CALL --> FINAL_BRIEF
    FINAL_BRIEF --> REPORTS["general + detailed<br/>Markdown/JSON"]
    FINAL_BRIEF --> HANDOFF["handoff JSON"]
    FINAL_BRIEF --> MEMORY
    SCORE_CALL --> AI_CACHE["AI synth cache"]
    GROUP_CALL --> AI_CACHE
    EVIDENCE_CALL --> AI_CACHE
    DELTA_CALL --> AI_CACHE
    FETCH --> HTTP_CACHE["HTTP/article cache"]
    REPORTS --> OUTPUT["output/"]
    HANDOFF --> OUTPUT
```

## Enrichment Module

```mermaid
flowchart TD
    HANDOFF["brief handoff<br/>or same-day brief JSON"] --> INPUTS["selected articles"]
    INPUTS --> TEXT["ensure article text"]
    TEXT --> PLAN["story-thread planning"]
    SUMMARY["summary_ai_client"] --> PLAN_CALL["LLM: plan story threads<br/>0..N planner calls<br/>skipped if shared grouping exists"]
    PLAN_CALL --> PLAN
    PLAN --> SEARCH["search for context<br/>DDG HTML retrieval"]
    SEARCH --> FETCH["fetch research pages"]
    FETCH --> SYNTH["synthesize compact<br/>story context"]
    SUMMARY --> SYNTH_CALL["LLM: synthesize context<br/>1 call per enriched story thread"]
    SYNTH_CALL --> SYNTH
    SYNTH --> ATTACH["attach context<br/>to story threads"]
    ATTACH --> ENRICHED["enrichment Markdown/JSON"]
    ENRICHED --> NARRATIVE["narrative brief input"]
    SEARCH --> HTTP_CACHE["enrichment HTTP cache"]
    FETCH --> HTTP_CACHE
    PLAN_CALL --> AI_CACHE["AI synth cache"]
    SYNTH_CALL --> AI_CACHE
    ENRICHED --> OUTPUT["output/"]
```

LLM call groups: headline scoring is batched, story grouping and enrichment planning can split into multiple planner calls, evidence and delta are optional batched analysis calls, final brief is normally one call per structured brief, narrative brief is normally one call, and enrichment synthesis is one call per enriched story thread. Cache hits skip eligible calls; JSON/transport retries can add attempts.

AI roles: `summary_ai_client` scores and plans; the configurable analysis client runs evidence/delta; `final_ai_client` writes final and narrative briefs.

Storage roles: `.cache/mydailynews/` stores network/article/enrichment fetches, `.cache/mydailynews/synth` stores reusable AI responses, `state/memory/` stores durable coverage/preferences, and `output/` stores reports, handoffs, diagnostics, and stage artifacts.

## Module Flow

The default module series is:

```text
briefs -> enrichment -> narrative_brief
```

`tts` is available but disabled by default. Add it after `narrative_brief` when audio should be part of normal runs.

Modules:

- `briefs`: fetches candidates, scores headlines, selects articles, fetches article text, runs analysis, writes general and detailed briefs.
- `enrichment`: groups selected articles into story threads, retrieves related context, and writes enrichment artifacts.
- `narrative_brief`: turns structured brief and enrichment JSON into a narrative Markdown brief.
- `tts`: turns a saved Markdown brief into WAV audio and audio metadata.

## State Boundaries

- `output/`: generated Markdown, JSON, WAV, diagnostics, and stage artifacts.
- `state/memory/`: durable coverage, story, feedback, learned-preference, recall, and backup files.
- `.cache/mydailynews/`: discovery, article text, enrichment retrieval, and AI synthesis caches.

Deleting `output/` removes generated reports. Deleting `state/memory/` resets local memory. Deleting `.cache/mydailynews/` only forces refetching or regeneration.

## Stable Entrypoints

- `main.py`: CLI.
- `gui.py`: local web GUI launcher.
- `tools/autoconfig.py`: hardware detection and recommended config writer.
- `mydailynews/app/config.py`: strict config loading.
- `mydailynews/app/runtime_config.py`: runtime readiness checks.
- `mydailynews/pipeline/orchestrator.py`: module orchestration.
