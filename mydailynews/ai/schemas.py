from __future__ import annotations

from .base import JSONSchemaSpec


HEADLINE_ANALYSIS_JSON_SCHEMA = JSONSchemaSpec(
    name="headline_analysis",
    schema={
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "score": {"type": "number"},
                        "personal_relevance": {"type": "number"},
                        "impact": {"type": "number"},
                        "novelty": {"type": "number"},
                        "urgency": {"type": "number"},
                        "actionability": {"type": "number"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                        "skip_reason": {"type": ["string", "null"]},
                        "angle_type": {"type": "string"},
                    },
                    "required": ["id", "score"],
                },
            }
        },
        "required": ["decisions"],
    },
)

STORY_GROUPING_JSON_SCHEMA = JSONSchemaSpec(
    name="story_grouping",
    schema={
        "type": "object",
        "properties": {
            "story_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "story_id": {"type": "string"},
                        "story_title": {"type": "string"},
                        "topic": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                        "research_questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "queries": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["question", "queries"],
                            },
                        },
                        "fallback": {"type": "boolean"},
                    },
                    "required": ["story_id", "story_title", "article_ids"],
                },
            }
        },
        "required": ["story_groups"],
    },
)

STORY_THREAD_PLANNER_JSON_SCHEMA = STORY_GROUPING_JSON_SCHEMA

STORY_ENRICHMENT_JSON_SCHEMA = JSONSchemaSpec(
    name="story_enrichment",
    schema={
        "type": "object",
        "properties": {
            "story_id": {"type": "string"},
            "story_title": {"type": "string"},
            "internal_articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "what_it_adds": {"type": "string"},
                        "source_ids": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string"},
                    },
                    "required": ["title", "summary", "what_it_adds", "source_ids", "confidence"],
                },
            },
            "confirmed_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string"},
                        "source_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["fact", "source_ids"],
                },
            },
            "conflicting_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["claim", "source_ids"],
                },
            },
            "open_questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "source_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question", "source_ids"],
                },
            },
        },
        "required": [
            "story_id",
            "story_title",
            "internal_articles",
            "confirmed_facts",
            "conflicting_claims",
            "open_questions",
        ],
    },
)

FINAL_BRIEF_JSON_SCHEMA = JSONSchemaSpec(
    name="final_brief",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "lead": {"type": "string"},
            "topic_reports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "what_changed": {"type": "string"},
                        "who_is_affected": {"type": "array", "items": {"type": "string"}},
                        "narrative_summary": {"type": "string"},
                        "narrative_changes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "narrative": {"type": "string"},
                                    "status": {"type": "string"},
                                    "summary": {"type": "string"},
                                },
                            },
                        },
                        "what_to_watch": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                },
            },
            "knowns": {"type": "array", "items": {"type": "string"}},
            "unknowns": {"type": "array", "items": {"type": "string"}},
            "watch_signals": {"type": "array", "items": {"type": "string"}},
        },
        # Keep minimal required keys for backend tolerance; BriefGenerator
        # normalizes and guarantees knowns/unknowns/watch_signals post-generation.
        "required": ["title", "lead", "topic_reports", "sections"],
    },
)


NARRATIVE_BRIEF_JSON_SCHEMA = JSONSchemaSpec(
    name="narrative_brief",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "lede": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "body": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                        "what_to_watch": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "closing": {"type": "string"},
        },
        "required": ["title", "lede", "segments"],
    },
)


EVIDENCE_DISTILLATION_JSON_SCHEMA = JSONSchemaSpec(
    name="evidence_distillation",
    schema={
        "type": "object",
        "properties": {
            "overview": {"type": "string"},
            "story_clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "topic": {"type": "string"},
                        "label": {"type": "string"},
                        "summary": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                        "key_claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "support_article_ids": {"type": "array", "items": {"type": "string"}},
                                    "confidence": {"type": "string"},
                                },
                                "required": ["claim", "support_article_ids", "confidence"],
                            },
                        },
                        "consensus_points": {"type": "array", "items": {"type": "string"}},
                        "contested_points": {"type": "array", "items": {"type": "string"}},
                        "known_unknowns": {"type": "array", "items": {"type": "string"}},
                        "watch_signals": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "cluster_id",
                        "topic",
                        "label",
                        "summary",
                        "article_ids",
                        "key_claims",
                        "consensus_points",
                        "contested_points",
                        "known_unknowns",
                        "watch_signals",
                    ],
                },
            },
            "global_watch_signals": {"type": "array", "items": {"type": "string"}},
            "reader_qa": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question", "answer", "article_ids"],
                },
            },
        },
        "required": ["overview", "story_clusters", "global_watch_signals", "reader_qa"],
    },
)


DELTA_EXTRACTION_JSON_SCHEMA = JSONSchemaSpec(
    name="delta_extraction",
    schema={
        "type": "object",
        "properties": {
            "baseline_coverage_note": {"type": "string"},
            "new": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "summary": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["item", "summary", "article_ids"],
                },
            },
            "escalated": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "summary": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["item", "summary", "article_ids"],
                },
            },
            "weakened": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "summary": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["item", "summary", "article_ids"],
                },
            },
            "reframed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "summary": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["item", "summary", "article_ids"],
                },
            },
            "unchanged_but_important": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "summary": {"type": "string"},
                        "article_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["item", "summary", "article_ids"],
                },
            },
            "evidence_gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "gap": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                    },
                    "required": ["gap", "why_it_matters"],
                },
            },
        },
        "required": [
            "baseline_coverage_note",
            "new",
            "escalated",
            "weakened",
            "reframed",
            "unchanged_but_important",
            "evidence_gaps",
        ],
    },
)
