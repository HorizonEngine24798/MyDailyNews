HEADLINE_ANALYSIS_SYSTEM = """You score news headlines for usefulness.
Return exactly one valid JSON object.
Do not use markdown fences.
Base the score only on the supplied reader preferences, brief goal, topics, and candidate headlines."""

HEADLINE_ANALYSIS_USER = """Reader memory and style:
{memory}

Brief mode:
{brief_goal}

Topics:
{topics}

Candidate headlines:
{items}

Score each candidate for how useful it is to retrieve and read in full for this brief.
Higher scores should favor importance, relevance, freshness, and likely explanatory value.
Return one decision for every candidate id.

Return:
{{
  "decisions": [
    {{
      "id": "candidate id",
      "score": 8.0
    }}
  ]
}}"""

BRIEF_SYSTEM = """You write compact, high-signal news briefs from selected articles.
Return exactly one valid JSON object.
Do not use markdown fences.
Base the synthesis only on supplied article evidence, supplied context, and prior reports."""

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
2. Prefer what changed, what matters, and what remains uncertain.
3. Keep the writing compact and readable.
4. Use evidence and delta packets when provided, but do not overstate uncertain points.
5. Populate explicit `knowns`, `unknowns`, and `watch_signals` slots.
6. Do not generate a references/sources section.
7. Do not include URLs or markdown links in generated text fields.

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
      "narrative_summary": "current state of the topic",
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
