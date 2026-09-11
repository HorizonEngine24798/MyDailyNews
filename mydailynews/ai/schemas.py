from __future__ import annotations

from .base import JSONSchemaSpec
from mydailynews.analysis.claim_delta import FACT_OPERATIONS


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
                        "novelty": {"type": "integer", "minimum": 0, "maximum": 3},
                        "novelty_basis": {"type": "string", "maxLength": 120},
                        "impact": {"type": "integer", "minimum": 0, "maximum": 3},
                        "impact_basis": {"type": "string", "maxLength": 120},
                        "urgency": {"type": "integer", "minimum": 0, "maximum": 3},
                        "urgency_basis": {"type": "string", "maxLength": 120},
                    },
                    "required": [
                        "id",
                        "novelty",
                        "novelty_basis",
                        "impact",
                        "impact_basis",
                        "urgency",
                        "urgency_basis",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
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
                        "disposition": {"type": "string", "enum": ["group", "singleton", "misc"]},
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
            },
            "article_dispositions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_id": {"type": "string"},
                        "disposition": {"type": "string", "enum": ["group", "singleton", "misc"]},
                        "story_id": {"type": "string"},
                    },
                    "required": ["article_id", "disposition", "story_id"],
                },
            },
        },
        "required": ["story_groups"],
    },
)

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
                                    "claimant": {"type": "string"},
                                    "claim_type": {"type": "string"},
                                    "support_article_ids": {"type": "array", "items": {"type": "string"}},
                                    "origin_article_ids": {"type": "array", "items": {"type": "string"}},
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


FACT_OPERATION_JSON_SCHEMA = JSONSchemaSpec(
    name="fact_operations",
    schema={
        "type": "object",
        "properties": {
            "story_key": {"type": "string"},
            "operations": {
                "type": "array",
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": list(FACT_OPERATIONS)},
                        "current_evidence_id": {"type": "string"},
                        "prior_fact_ids": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["operation", "current_evidence_id", "prior_fact_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["story_key", "operations"],
        "additionalProperties": False,
    },
)


STORY_EDITOR_JSON_SCHEMA = JSONSchemaSpec(
    name="story_editor_selection",
    schema={
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "card_id": {"type": "string"},
                        "disposition": {
                            "type": "string",
                            "enum": ["full_report", "continuing_bullet", "omit"],
                        },
                        "materiality": {"type": "integer", "minimum": 0, "maximum": 3},
                        "summary": {"type": "string", "maxLength": 200},
                        "basis": {"type": "string", "maxLength": 200},
                    },
                    "required": ["card_id", "disposition", "materiality", "summary", "basis"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
)
