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
                    },
                    "required": ["id", "score"],
                },
            }
        },
        "required": ["decisions"],
    },
)

FINAL_BRIEF_JSON_SCHEMA = JSONSchemaSpec(
    name="final_brief",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "lead": {"type": "string"},
            "topic_reports": {"type": "array", "items": {"type": "object"}},
            "sections": {"type": "array", "items": {"type": "object"}},
            "knowns": {"type": "array", "items": {"type": "string"}},
            "unknowns": {"type": "array", "items": {"type": "string"}},
            "watch_signals": {"type": "array", "items": {"type": "string"}},
        },
        # Keep minimal required keys for backend tolerance; BriefGenerator
        # normalizes and guarantees knowns/unknowns/watch_signals post-generation.
        "required": ["title", "lead", "topic_reports", "sections"],
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
