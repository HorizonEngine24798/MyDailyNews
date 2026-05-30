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
        },
        "required": ["title", "lead", "topic_reports", "sections"],
    },
)
