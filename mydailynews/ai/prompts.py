HEADLINE_ANALYSIS_SYSTEM = """You are an editorial triage scorer for a personal news briefing.
Return exactly one valid JSON object.
Do not use markdown fences.
Use only supplied reader memory, brief goal, topics, and candidate headlines.
Return one decision for every candidate id."""

GENERIC_JSON_RETRY_USER = """Retry instruction: your previous answer could not be parsed as one valid JSON object.
Return exactly one JSON object only.
Do not include markdown fences, explanations, or trailing text.

Original request:
{user}"""

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
- Routine high-volume coverage without reader-specific stake.
- Minor incremental updates that do not materially change understanding.
- Rewrites of the same event with no meaningful new information.
- Topic keyword match with low impact or low urgency.

Examples:
- High-value must-know (8-10): a consequential state change with immediate impact on this reader's interests or decisions.
- Mid-value monitor (5-7): relevant update with some signal but limited urgency or actionability.
- Low-value noise (0-4): repetitive recap, small incremental change, promotional/clickbait framing, or weakly relevant topic mention.

Decision fields:
- Return exactly these fields for every candidate: `id`, `score`, `personal_relevance`, `impact`, `novelty`, `urgency`, `actionability`, `confidence`, `angle_type`.
- Use a short snake_case label for `angle_type`.

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
      "angle_type": "material_state_change"
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
2. {article_disposition_instruction}
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
      "disposition": "group | singleton | misc",
      "topic": "optional topic name",
      "article_ids": ["article id"],
      "research_questions": [
        {{
          "question": "what should be researched",
          "queries": ["search query"]
        }}
      ]
    }}
  ],
  "article_dispositions": [
    {{"article_id": "article id", "disposition": "group | singleton | misc", "story_id": "story-001"}}
  ]
}}"""

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

Coverage recall packet (optional; may be empty):
{recall_packet}

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
6. Use coverage guidance to avoid making recently dominant continuing stories the core narrative unless today's supplied evidence shows a material new phase. If a story remains important but repetitive, cover it compactly and leave room for other material developments.
7. Populate explicit `knowns`, `unknowns`, and `watch_signals` slots.
8. Do not generate a references/sources section.
9. Do not include URLs or markdown links in generated text fields.

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
          "status": "new | continuing | material_update | status_change | correction | resolved | reframed | unchanged | uncertain",
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

Optional cross-source perspectives context:
{perspectives_context}

Coverage recall packet (optional; may be empty):
{recall_packet}

Permitted claim-context markers (optional; may be empty):
{claim_cards}

Work to perform:
1. Use the general brief for breadth and the detailed brief for deeper narrative context when both are present.
2. Use enrichment context only as additional background when it is supplied.
3. Use perspectives context to preserve material shared facts, framing differences, qualifications, and coverage limitations. Do not turn every framing observation into prose.
4. Merge overlapping stories without repeating them; preserve all material developments from both briefs.
5. Use coverage guidance to keep recently dominant continuing stories proportionate unless the supplied briefs show a material new phase.
6. Write a human-readable narrative report with clear section headings, graceful transitions, and selective bullets where they improve scanning.
7. Preserve uncertainty. Unknowns, thin evidence, and watch signals should be explicit.
8. Keep the story coherent from opening to closing.
9. Do not include URLs, markdown links, references sections, source-link housekeeping, SSML, pause markers, pronunciation tags, or provider-specific TTS tags.
10. Mention source names only when they help attribution or uncertainty.
11. Avoid hype, jokes, dramatic teasing, and generic filler.
12. Whenever prose materially relies on a permitted claim, place that claim's exact marker such as <<1>> immediately after the sentence or paragraph. Do not state a permitted claim without its marker. Do not invent markers, alter their numbers, or reproduce card contents solely because a marker exists.

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
2. Extract atomic key claims: one proposition per claim, with supporting article ids. Collapse paraphrases that assert the same proposition.
3. Record claimant, claim type, and origin article ids only when supplied evidence establishes them. A claimant is the actor, institution, document, or dataset making the assertion, not the outlet repeating it. Origin ids must also be support ids; prefer empty fields to guesses. Use `other` only when factual, numerical, causal, forecast, or attribution genuinely does not fit.
4. Distinguish consensus points, contested points, and unresolved unknowns.
5. Propose concrete watch signals.
6. If reader_qa is requested, produce practical reader questions and concise evidence-grounded answers.

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
          "claim": "one atomic proposition",
          "claimant": "actor, institution, document, dataset, or empty string",
          "claim_type": "factual | numerical | causal | forecast | attribution | other",
          "support_article_ids": ["article id"],
          "origin_article_ids": ["article id that contains the original statement or record"],
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
Compare complete propositions in context, including attribution, negation, quantity, and time.
Never infer a transition label from the presence of an action word or phrase alone.
If prior evidence is insufficient, state that directly in baseline_coverage_note and keep lists concise."""


DELTA_DECISION_SYSTEM = """You classify story identity and source-backed change.
Return exactly one valid JSON object matching the supplied schema.
Do not invent facts, write markdown, or add editorial sections.
Compare complete propositions in context, including attribution, negation, quantity, and time.
Never infer a transition label from the presence of an action word or phrase alone.
Use only current evidence and the bounded candidate baselines."""

DELTA_EXTRACTION_USER = """Reader profile and style:
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

Story memory (bounded and story-specific; candidate history, not proof of identity):
{story_memory}

Fallback selected article evidence:
{articles}

Work to perform:
1. For every current story thread, decide whether it is the same concrete story, a related theme, distinct, or uncertain relative to the supplied baselines.
2. If it is the same story, use a domain-neutral change label: material_update, status_change, correction, resolved, incremental, reframed, unchanged, or uncertain. Use escalated/weakened only when direction is explicitly supported. Copy the matched baseline's supplied story_key into prior_story_key.
3. Keep entries evidence-grounded and link article ids. Never infer same-story identity from topic overlap alone. A same-story unchanged item should be omitted unless it is critical safety information; a non-material continuation should be a continuing bullet, not a full report.
4. Use uncertain when the baseline is weak or evidence conflicts; do not suppress uncertain stories.
5. Flag evidence gaps that limit confidence.
6. For each decision, cite the minimal supplied current and prior claim IDs in current_evidence_ids
   and prior_evidence_ids. A first observation has no prior IDs.
7. Emit claim_relations edges for every cited current/prior comparison. For each edge classify the
   relation as equivalent, supports, adds_detail, contradicts, supersedes, temporal_successor,
   context_only, or uncertain, and assess entailment in both directions as yes, no, or uncertain.
8. Put an ID in superseded_prior_evidence_ids only when the current evidence actually replaces that
   exact prior proposition. Copy IDs exactly; never construct or guess them.

Return one object matching the supplied JSON schema. Include every required
top-level key, using empty arrays when a category has no items. Emit exactly
one story_decisions entry per current story thread. Keep every prose value to
at most 16 words. Omit optional knowns, unknowns, and watch_signals unless they
are essential. Do not duplicate a decision into a change-category list unless
that category adds useful information."""


DELTA_DECISION_USER = """Date: {date}

Profile priorities:
{profile}

Current source evidence:
{current}

Candidate prior-story baselines:
{baselines}

Classify every current article id (or evidence-cluster article id) exactly once.
- same_story requires the same concrete event or tracked process, not topic overlap.
- Copy the matching baseline story_key into prior_story_key; otherwise use an empty string.
- A first observation is distinct_story + new + full_report.
- For same_story choose the evidence-backed change: material_update, status_change, correction,
  resolved, incremental, reframed, unchanged, or uncertain.
- same_story + unchanged should be omitted. A non-material continuation should be a continuing_bullet.
- Never omit an uncertain or materially changed story.
- Cite the minimal supplied current and prior claim IDs in current_evidence_ids and
  prior_evidence_ids. A first observation has no prior IDs.
- Emit claim_relations for cited pairs. Each edge uses equivalent, supports, adds_detail,
  contradicts, supersedes, temporal_successor, context_only, or uncertain, plus yes/no/uncertain
  entailment in both directions.
- Put an ID in superseded_prior_evidence_ids only when current evidence replaces that exact prior
  proposition. Copy every ID exactly; never construct or guess IDs.
- Keep summary to at most 12 words."""


PERSPECTIVES_PLANNER_SYSTEM = """You plan bounded broad story retrieval and focused claim verification using English queries.
Return exactly one valid JSON object.
Do not use markdown fences.
Do not summarize articles, judge bias, rate source quality, invent feed URLs, or broaden a concrete event into a generic topic."""

PERSPECTIVES_PLANNER_USER = """Create retrieval plans for {date}.

Planner input:
{data}

Work to perform:
1. Return exactly one plan for every supplied story.
2. Write 3 to 5 distinct English queries per story. Vary wording and angle, not language.
3. Every query must combine a stable named entity with a specific event or action term. Add location, institution, or time anchors when they distinguish the event; do not broaden it into a generic topic.
4. Select story-relevant countries and regions from tag_options.
5. Provide anchor groups: stable entities plus event/action terms. Synonyms are welcome; exact headline wording is not required.
6. Provide up to three story_loci: places where the event physically occurs or territory directly affected. Exclude publisher origins, company headquarters, actor nationality, and countries that are merely interested. Use [] for a non-geographic story. Return semantic labels only; do not invent coordinates.
7. Do not return source_id values. Source selection happens later from target_tags.
8. Do not leave countries or regions empty.
9. Select at most two consequential, currently checkable supplied claims per story. Prefer claims where sources disagree, materially qualify scope/timing/cause, repeat one origin without independent support, or where verification could change the narrative. A routine date, count, sequence number, or other clear-cut fact is not a useful target merely because it is numerical or temporal; select it only when evidence conflicts or the exact value materially changes the conclusion.
10. Give each selected claim at most two targeted queries: one for primary/origin evidence and one for independent/counterevidence when applicable. Use only supplied claim_id values. Repetition is not independent verification.

Return:
{{
  "plans": [
    {{
      "story_id": "story-001",
      "queries": [
        "short canonical event query",
        "relevant adjacent queries that may produce related news results"
      ],
      "anchor_groups": [
        {{"kind": "entity", "terms": ["primary entity", "common alias"]}},
        {{"kind": "event", "terms": ["event term", "meaningful synonym"]}}
      ],
      "story_loci": [
        {{
          "label": "place where the event occurs",
          "country": "ISO country code, or empty for a cross-border area",
          "kind": "event_site",
          "confidence": "high",
          "reason": "why this is an event location rather than a reporting origin"
        }}
      ],
      "target_tags": {{
        "countries": ["US", "GB"],
        "regions": ["north_america"]
      }},
      "verification_targets": [
        {{
          "claim_id": "supplied-claim-id",
          "importance_reason": "why resolving this changes the story",
          "required_evidence_types": ["primary", "independent"],
          "queries": [
            {{"query": "targeted official record query", "evidence_type": "primary"}},
            {{"query": "targeted independent evidence query", "evidence_type": "independent"}}
          ]
        }}
      ]
    }}
  ]
}}"""

PERSPECTIVES_PLANNER_RETRY_USER = """{prompt}

Retry instruction: your previous planner response was valid JSON but missing required target_tags values.
Return a complete replacement JSON object.
Every plan must include at least one countries value and one regions value."""

PERSPECTIVES_FRAMING_SYSTEM = """You compare framing across source countries and outlets using only supplied article records.
Return exactly one valid JSON object.
Do not use markdown fences.
State evidence gaps plainly."""

PERSPECTIVES_FRAMING_USER = """Create a framing comparison for the supplied story.

Supplied data:
{data}

Work to perform:
1. Write a short narrative comparison, not a list of observations: establish the common account, explain the most meaningful difference in emphasis, and say why that difference matters.
2. Use only supplied article records and context_text.
3. Identify shared facts, leading facts, centered actors, agency, causal explanations, certainty or hedging, local stakes, terminology, and meaningful prominence differences.
4. Treat wording-only differences as wording-only.
5. Cite article_ids for every substantive comparison, including the synthesis. Use only article_id values supplied in the records; never invent or paraphrase them. Never put article IDs in prose fields; use only the dedicated article-id fields.
6. State when coverage is thin, metadata-only, or uneven.
7. Do not assign bias, truthfulness, source quality, national quality, language quality, or editorial intent.

Return exactly one item in stories[]:
{{
  "stories": [
    {{
      "story_id": "story-001",
      "synthesis": "2-4 sentence narrative: shared account, key framing difference, and why it matters",
      "synthesis_article_ids": ["article-id-a", "article-id-b"],
      "shared_facts": [
        {{"text": "shared factual point", "article_ids": ["article-id"]}}
      ],
      "country_source_comparison": [
        {{"text": "country or outlet framing difference", "article_ids": ["article-id-a", "article-id-b"]}}
      ],
      "coverage_limitations": ["specific evidence limitation"]
    }}
  ]
}}"""
