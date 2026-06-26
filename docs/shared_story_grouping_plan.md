# Shared Story Grouping

Shared story grouping is the boundary-planning stage used by the enrichment revamp.

## Purpose

When both story enrichment and evidence distillation are enabled, the pipeline runs `story_grouping` after `article_fetch` and before `enrichment`. The stage asks the summary-role model for neutral story groups over the selected articles, then passes the same groups to enrichment and evidence.

This avoids two downstream components privately clustering the same articles in different ways.

## Modes

- `None` story groups means the shared stage was skipped and a consumer may use its private behavior.
- `[]` story groups means the shared stage ran but returned no usable groups. Consumers must preserve shared mode and must not fall back to private planning/free clustering.
- A non-empty list gives shared story boundaries. Normalization removes unknown article IDs, trims duplicate assignments, preserves topic/research-question data, and adds singleton fallback groups for omitted selected articles.

## Diagnostics

Stage artifacts include selected article IDs, story groups, fallback groups, cache status, request artifacts, and whether budget pressure split grouping into multiple requests. If split requests happen, related articles in different batches cannot be grouped together; this is intentional for this revamp.

## Boundaries

Evidence still owns evidence-specific boundary enforcement. It trims or drops model output that crosses supplied story groups. Enrichment still owns research, synthesis, and context attachment.
