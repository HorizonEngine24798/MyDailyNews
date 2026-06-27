# TODO

## Pipeline Operability Matrix

Verify the pipeline runs cleanly across representative environments and LLM
choices before treating the current optional enrichment budgets as broadly
portable.

- Windows, Linux, and macOS shell/runtime differences.
- Managed and unmanaged `llama.cpp` server flows.
- Remote OpenAI-compatible llama.cpp server profiles.
- Small, medium, and larger context-window models.
- JSON response reliability for headline scoring, story-thread planning,
  story synthesis, evidence, delta, final brief, and narrative brief stages.
- Network-enabled, cache-first, and no-network/cached rerun behavior.
- Stage-artifact usefulness for diagnosing planner inputs, retrieval rows,
  fetched-page statuses, synthesis budgets, context attachments, and final
  brief inputs.
- Clear failure messages for missing model files, unresolved executables,
  context-budget mismatch, malformed JSON, and network fetch failures.

Acceptance target: a maintainer can run the documented smoke path on each
supported profile, inspect artifacts when a stage fails, and tell whether the
issue is environment setup, model behavior, retrieval/network state, or a
pipeline bug.

## Standalone Story Enrichment Module

Move the heavy story-thread research pass out of the default brief run and make
it a separate post-brief command/tool.

- Consume saved brief JSON and stage artifacts rather than rerunning discovery,
  scoring, selection, or article fetch.
- Prefer `analysis.evidence_packet.story_clusters` as the story list; fall back
  to final brief `topic_reports`, selected article IDs, or major headlines only
  when evidence clusters are absent.
- Reuse the existing research and synthesis pieces, but skip the current
  planner LLM step because the main pipeline already formed story clusters.
- Write an enrichment addendum JSON/Markdown file linked to the original brief,
  with optional hooks to regenerate a final brief from the addendum later.
- Keep separate cache, retry, and budget controls so routine briefs stay cheap
  and the enrichment tool can spend a full research-pass budget intentionally.
