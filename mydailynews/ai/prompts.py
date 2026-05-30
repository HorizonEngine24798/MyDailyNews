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

Selected articles:
{articles}

Work to perform:
1. Synthesize only from the supplied article excerpts and context.
2. Prefer what changed, what matters, and what remains uncertain.
3. Keep the writing compact and readable.
4. Cite article ids in the relevant topic reports and sections.

Return:
{{
  "title": "Daily Brief - {date}",
  "lead": "2 to 3 sentence synthesis",
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
      "what_to_watch": ["specific next signal"],
      "article_ids": ["article id"]
    }}
  ],
  "sections": [
    {{
      "heading": "short section heading",
      "summary": "2 sentence section summary",
      "article_ids": ["article id"]
    }}
  ]
}}"""
