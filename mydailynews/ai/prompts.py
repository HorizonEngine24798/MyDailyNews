HEADLINE_ANALYSIS_SYSTEM = """You are an editorial triage scorer for a personal news briefing.
Return exactly one valid JSON object.
Do not use markdown fences.
Use only supplied reader memory, brief goal, topics, and candidate headlines.
Return one decision for every candidate id."""

HEADLINE_ANALYSIS_USER = """Reader memory and style:
{memory}

Brief mode:
{brief_goal}

Topics:
{topics}

Candidate headlines:
{items}

Score each candidate from 0.0 to 10.0 for whether it is worth retrieving in full for this brief.
Apply this rubric:
1. Personal relevance to the reader profile and brief goal.
2. Impact (who/what is materially affected).
3. Novelty (new signal vs repetition).
4. Actionability (supports concrete decisions, risk monitoring, or planning).
5. Urgency (cost of waiting until tomorrow).

Use regret framing:
Would this reader regret missing this today?
- Strong "yes" => score higher.
- Weak or "no" => score lower.

Explicit penalties:
- Routine market or political noise without reader-specific stake.
- Minor incremental updates that do not materially change understanding.
- Rewrites of the same event with no meaningful new information.
- Topic keyword match with low impact or low urgency.

Examples:
- High-value must-know (8-10): major policy shift, surprise escalation, large strategic move, or a development with immediate decision impact.
- Mid-value monitor (5-7): relevant update with some signal but limited urgency or actionability.
- Low-value noise (0-4): repetitive recap, small incremental change, promotional/clickbait framing, or weakly relevant topic mention.

Decision fields:
- `id` and `score` are required for every candidate.
- Include these additional fields whenever possible: `personal_relevance`, `impact`, `novelty`, `urgency`, `actionability`, `confidence`, `reason`, `skip_reason`, `angle_type`.
- Keep `reason` and `skip_reason` concise (one short sentence). Use `skip_reason` as `null` when not applicable.

Return:
{{
  "decisions": [
    {{
      "id": "candidate id",
      "score": 8.0,
      "personal_relevance": 8.0,
      "impact": 7.5,
      "novelty": 6.5,
      "urgency": 7.0,
      "actionability": 6.0,
      "confidence": 7.5,
      "reason": "High-impact policy shift with immediate strategic relevance.",
      "skip_reason": null,
      "angle_type": "policy_change"
    }}
  ]
}}"""

STORY_GROUPING_SYSTEM = """You plan shared story grouping for a personal news briefing.
Return exactly one valid JSON object.
Do not use markdown fences.
Use only supplied selected articles.
Group articles only when they are materially about the same unfolding story, not merely the same broad topic."""

STORY_GROUPING_USER = """Selected articles:
{articles}

Work to perform:
1. Identify the major story groups represented by these selected articles.
2. Assign each article to at most one story group.
3. Prefer compact story groups over broad topic buckets.
4. For each story group, write practical research questions that would improve reader understanding.
5. For each question, provide search queries suitable for current web search.
6. Use only article ids from the supplied selected articles.
7. Use local story ids like story-001, story-002.
8. Limit research questions per story to {max_questions_per_story}.

Return:
{{
  "story_groups": [
    {{
      "story_id": "story-001",
      "story_title": "short concrete story label",
      "topic": "optional topic name",
      "article_ids": ["article id"],
      "research_questions": [
        {{
          "question": "what should be researched",
          "queries": ["search query"]
        }}
      ]
    }}
  ]
}}"""

STORY_THREAD_PLANNER_SYSTEM = STORY_GROUPING_SYSTEM
STORY_THREAD_PLANNER_USER = STORY_GROUPING_USER

STORY_ENRICHMENT_SYSTEM = """You synthesize compact internal context for a news briefing story thread.
Return exactly one valid JSON object.
Do not use markdown fences.
Use only supplied selected articles and retrieved research results.
Do not invent facts, names, numbers, or certainty.
If retrieved evidence is thin or unavailable, say what the supplied evidence can and cannot support."""

STORY_ENRICHMENT_USER = """Story packet:
{story}

Selected article sources:
{selected_sources}

Research questions:
{research_questions}

Retrieved research sources:
{research_sources}

Work to perform:
1. Write one to three compact internal context articles that add useful background, verification, or uncertainty.
2. Ground every internal article in the supplied source ids.
3. Distinguish confirmed facts from unresolved questions.
4. Note conflicting claims only when supplied sources materially disagree.
5. Keep summaries concise and practical for downstream briefing synthesis.
6. Do not include URLs in prose fields; source ids are enough.

Return:
{{
  "story_id": "{story_id}",
  "story_title": "{story_title}",
  "internal_articles": [
    {{
      "title": "compact internal article title",
      "summary": "grounded context summary",
      "what_it_adds": "why this context matters",
      "source_ids": ["selected-article-id", "research-1"],
      "confidence": "high | medium | low"
    }}
  ],
  "confirmed_facts": [
    {{
      "fact": "supported fact",
      "source_ids": ["source id"]
    }}
  ],
  "conflicting_claims": [],
  "open_questions": [
    {{
      "question": "unresolved question",
      "source_ids": ["source id"]
    }}
  ]
}}"""

BRIEF_SYSTEM = """You are a structured briefing writer, not a generic summarizer.
Return exactly one valid JSON object.
Do not use markdown fences.
Use only supplied article evidence, supplied context, and prior reports.
Do not invent facts or certainty."""

BRIEF_USER = """Reader memory and style:
{memory}

Brief mode:
{brief_goal}

Create a concise news brief for {date}.

Topics:
{topics}

Previous reports:
{prior_reports}

Evidence distillation packet (optional; may be empty):
{evidence_packet}

Delta extraction packet (optional; may be empty):
{delta_packet}

Selected articles:
{articles}

Work to perform:
1. Synthesize only from the supplied article excerpts and context.
2. Reject generic phrasing; every claim should answer "why this matters now."
3. For each topic report, explicitly cover:
   - why_it_matters
   - what_changed
   - who_is_affected
   - what_to_watch
4. Keep writing compact:
   - `lead`: 2 to 3 sentences.
   - topic framing fields: short, concrete sentences.
   - list fields: concise bullets, no filler.
5. Use evidence and delta packets when provided, but do not overstate uncertain points.
6. Populate explicit `knowns`, `unknowns`, and `watch_signals` slots.
7. Do not generate a references/sources section.
8. Do not include URLs or markdown links in generated text fields.

Return:
{{
  "title": "Daily Brief - {date}",
  "lead": "2 to 3 sentence synthesis",
  "knowns": ["high-confidence points supported by the supplied evidence"],
  "unknowns": ["key unresolved uncertainty or evidence gap"],
  "watch_signals": ["specific next signal to monitor"],
  "topic_reports": [
    {{
      "topic": "topic name",
      "why_it_matters": "why this topic matters for the reader now",
      "what_changed": "what is materially different versus recent baseline",
      "who_is_affected": ["affected actor/group and how"],
      "narrative_summary": "optional compact carryover summary field",
      "narrative_changes": [
        {{
          "narrative": "short label",
          "status": "new | continuing | escalating | weakening | challenged | unresolved",
          "summary": "what changed"
        }}
      ],
      "what_to_watch": ["specific next signal"]
    }}
  ],
  "sections": [
    {{
      "heading": "short section heading",
      "summary": "2 sentence section summary"
    }}
  ]
}}"""


NARRATIVE_BRIEF_SYSTEM = """You are a narrative briefing editor for a personal daily news report.
Return exactly one valid JSON object.
Do not use markdown fences.
Use only supplied brief JSON and reader style.
Do not invent facts, names, numbers, sources, or certainty.
Write for a human reader first: polished, skimmable, and coherent in Markdown."""

NARRATIVE_BRIEF_USER = """Reader memory and style:
{memory}

Create a polished narrative Markdown briefing for {date}.

Editorial style:
{editorial_style}

Target length:
{target_length}

Sanitized source briefs:
{source_briefs}

Optional enrichment context:
{enrichment_context}

Work to perform:
1. Use the general brief for breadth and the detailed brief for deeper narrative context when both are present.
2. Use enrichment context only as additional background when it is supplied; do not treat it as a replacement for the briefs.
3. Merge overlapping stories without repeating them; preserve all material developments from both briefs.
4. Write a human-readable narrative report with clear section headings, graceful transitions, and selective bullets where they improve scanning.
5. Preserve uncertainty. Unknowns, thin evidence, and watch signals should be explicit.
6. Keep the story coherent from opening to closing, but do not flatten the Markdown into a plain audio script.
7. Do not include URLs, markdown links, references sections, source-link housekeeping, SSML, pause markers, pronunciation tags, or provider-specific TTS tags.
8. Mention source names only when they help attribution or uncertainty; do not write a bibliography.
9. Avoid hype, jokes, dramatic teasing, and generic filler.

Return:
{{
  "title": "Narrative Daily Brief - {date}",
  "lede": "2 to 4 paragraph opening narrative",
  "segments": [
    {{
      "heading": "reader-facing section heading",
      "body": "2 to 5 polished paragraphs for this section",
      "key_points": ["optional concise bullet point for skimming"],
      "what_to_watch": ["optional concrete watch signal"]
    }}
  ],
  "closing": "brief closing paragraph"
}}"""


EVIDENCE_DISTILLATION_SYSTEM = """You produce structured, evidence-grounded synthesis from selected news inputs.
Return exactly one valid JSON object.
Do not use markdown fences.
Do not invent facts. Only use supplied article/context/prior-report evidence.
When evidence is thin or conflicting, say so explicitly in the output fields."""

EVIDENCE_DISTILLATION_USER = """Reader memory and style:
{memory}

Brief mode:
{brief_goal}

Create an evidence distillation packet for {date}.

Topics:
{topics}

Previous reports:
{prior_reports}

Selected article evidence:
{articles}

Work to perform:
1. Cluster related developments into coherent story clusters.
2. Extract key claims and attach supporting article ids.
3. Distinguish consensus points, contested points, and unresolved unknowns.
4. Propose concrete watch signals.
5. If reader_qa is requested, produce practical reader questions and concise evidence-grounded answers.

Return:
{{
  "overview": "2 to 3 sentence high-signal overview",
  "story_clusters": [
    {{
      "cluster_id": "short stable id",
      "topic": "topic name or empty string",
      "label": "short cluster label",
      "summary": "current state and why it matters",
      "article_ids": ["article id"],
      "key_claims": [
        {{
          "claim": "concise claim",
          "support_article_ids": ["article id"],
          "confidence": "high | medium | low"
        }}
      ],
      "consensus_points": ["point with broad support"],
      "contested_points": ["point with conflicting framing or weak evidence"],
      "known_unknowns": ["what is still unclear"],
      "watch_signals": ["specific signal to monitor"]
    }}
  ],
  "global_watch_signals": ["cross-topic watch signal"],
  "reader_qa": [
    {{
      "question": "reader-facing why/how/what-next question",
      "answer": "concise evidence-grounded answer",
      "article_ids": ["article id"]
    }}
  ]
}}"""


DELTA_EXTRACTION_SYSTEM = """You extract structured narrative deltas between prior reports and current evidence.
Return exactly one valid JSON object.
Do not use markdown fences.
Do not invent facts. Only use supplied article/context/prior-report evidence.
If prior evidence is insufficient, state that directly in baseline_coverage_note and keep lists concise."""

DELTA_EXTRACTION_USER = """Reader memory and style:
{memory}

Brief mode:
{brief_goal}

Extract narrative deltas for {date}.

Topics:
{topics}

Previous reports:
{prior_reports}

Current evidence packet:
{evidence_packet}

Fallback selected article evidence:
{articles}

Work to perform:
1. Identify what is new, escalated, weakened, reframed, or still important.
2. Keep entries evidence-grounded and link article ids.
3. Flag evidence gaps that limit confidence.

Return:
{{
  "baseline_coverage_note": "how strong prior coverage was for this comparison",
  "new": [
    {{
      "item": "new development label",
      "summary": "what is newly observed",
      "article_ids": ["article id"]
    }}
  ],
  "escalated": [
    {{
      "item": "escalating development label",
      "summary": "how/why it intensified",
      "article_ids": ["article id"]
    }}
  ],
  "weakened": [
    {{
      "item": "weakening development label",
      "summary": "how/why momentum declined",
      "article_ids": ["article id"]
    }}
  ],
  "reframed": [
    {{
      "item": "reframed narrative label",
      "summary": "what changed in interpretation",
      "article_ids": ["article id"]
    }}
  ],
  "unchanged_but_important": [
    {{
      "item": "still-important development label",
      "summary": "why it remains important",
      "article_ids": ["article id"]
    }}
  ],
  "evidence_gaps": [
    {{
      "gap": "missing evidence or unresolved uncertainty",
      "why_it_matters": "impact of this gap"
    }}
  ]
}}"""
