HEADLINE_ANALYSIS_SYSTEM = """You are a careful daily news editor.
Score headline candidates for one reader using the configured long-term memory.
Return valid JSON only. Do not invent facts beyond the headline/snippet."""

HEADLINE_ANALYSIS_USER = """Reader memory:
{memory}

Scoring rubric:
- 9-10: major, urgent, or high-impact story.
- 7-8: important enough for today's brief.
- 5-6: interesting but optional.
- 0-4: routine, duplicated, low-value, or outside reader preferences.

Treat items as duplicates only if they describe the same real-world event.

Candidates:
{items}

Return:
{{
  "decisions": [
    {{
      "id": "candidate id",
      "score": 0-10,
      "reason": "short reason",
      "summary": "one sentence",
      "tags": ["topic", "topic"],
      "duplicate_of": null
    }}
  ]
}}"""

ENRICHMENT_SYSTEM = """You decide whether a selected article needs extra context.
Use enrichment for stories where background knowledge would help a reader understand what happened.
Return valid JSON only."""

ENRICHMENT_USER = """Article:
Title: {title}
Source: {source}
URL: {url}
Triage summary: {summary}
Text excerpt: {text}

Return:
{{
  "needed": true,
  "reason": "why extra context is or is not needed",
  "wikipedia_query": "short entity/concept query or empty string",
  "past_news_query": "short news search query or empty string"
}}"""

BRIEF_SYSTEM = """You write compact, high-signal daily news releases.
Return valid JSON only. Base the brief on supplied article text and supplied enrichment context."""

BRIEF_USER = """Reader memory:
{memory}

Create a daily news release for {date}.
Put the most important synthesis first, then article details, then a final major-headlines list.

Articles:
{articles}

Return:
{{
  "title": "Daily Brief - {date}",
  "lead": "3-5 sentence synthesis of the day",
  "sections": [
    {{
      "heading": "short section heading",
      "summary": "2-4 sentence section summary",
      "article_ids": ["id"]
    }}
  ],
  "articles": [
    {{
      "id": "id",
      "headline": "headline",
      "source": "source",
      "url": "url",
      "score": 0-10,
      "summary": "2-4 sentence article summary",
      "why_it_matters": "1-2 sentences",
      "key_context": "1-3 sentences, using enrichment when relevant",
      "tags": ["topic"]
    }}
  ],
  "major_headlines": [
    {{
      "headline": "headline",
      "source": "source",
      "url": "url",
      "score": 0-10
    }}
  ]
}}"""
