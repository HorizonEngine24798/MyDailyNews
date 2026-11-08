from __future__ import annotations

import re

from mydailynews.ai.base import JSONSchemaSpec
from mydailynews.pipeline.brief_specs import DEFAULT_BRIEF_NAMES


PERSPECTIVES_REPORT_SCHEMA_VERSION = "claim_led_perspectives_report.v4"
STRUCTURED_BRIEF_NAMES = DEFAULT_BRIEF_NAMES
MAX_CANONICAL_QUERIES = 5
MAX_PLANNER_STORIES_PER_CALL = 4
MAX_ANCHOR_QUERIES = 3
MAX_SELECTED_SOURCES = 12
MAX_ARTICLES_PER_SOURCE = 2
MAX_ARTICLES_FOR_REPORT = MAX_SELECTED_SOURCES * MAX_ARTICLES_PER_SOURCE
MIN_COVERAGE_RELEVANCE_SCORE = 2
ARTICLE_CONTEXT_TOKEN_BUDGET = 500
ARTICLE_CONTEXT_MAX_CHARS = ARTICLE_CONTEXT_TOKEN_BUDGET * 4
ARTICLE_FETCH_MAX_CHARS = ARTICLE_CONTEXT_MAX_CHARS * 4
MAX_CONTEXT_PARAGRAPHS = 4
MEANINGFUL_PARAGRAPH_MIN_CHARS = 240
RELATIONSHIPS = {"originates", "independently_supports", "reports_or_quotes", "qualifies", "disputes", "context_only"}
VERDICTS = {"supported", "mostly_supported", "mixed", "contradicted", "unresolved", "not_checkable_yet"}
EVIDENCE_TYPES = {"primary", "origin", "independent", "counterevidence"}
PERSPECTIVES_EVIDENCE_LIST_SCHEMA = {
    "type": "array",
    "maxItems": MAX_ARTICLES_FOR_REPORT,
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "article_ids": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_ARTICLES_FOR_REPORT},
        },
        "required": ["text", "article_ids"],
        "additionalProperties": False,
    },
}
TAG_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
QUERY_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "amid",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "new",
    "of",
    "on",
    "over",
    "says",
    "the",
    "to",
    "with",
}
COVERAGE_RELEVANCE_WEAK_TOKENS = QUERY_STOPWORDS | {
    "latest",
    "live",
    "news",
    "update",
    "updates",
}
COUNTRY_ALIASES = {
    "u_s": "US",
    "us": "US",
    "u_sa": "US",
    "usa": "US",
    "united_states": "US",
    "america": "US",
    "uk": "GB",
    "u_k": "GB",
    "great_britain": "GB",
    "britain": "GB",
    "united_kingdom": "GB",
    "uae": "AE",
    "u_a_e": "AE",
    "united_arab_emirates": "AE",
    "argentina": "AR",
    "australia": "AU",
    "bangladesh": "BD",
    "brazil": "BR",
    "canada": "CA",
    "france": "FR",
    "germany": "DE",
    "hong_kong": "HK",
    "india": "IN",
    "indonesia": "ID",
    "italy": "IT",
    "japan": "JP",
    "new_zealand": "NZ",
    "nigeria": "NG",
    "pakistan": "PK",
    "qatar": "QA",
    "saudi_arabia": "SA",
    "singapore": "SG",
    "south_africa": "ZA",
    "south_korea": "KR",
    "republic_of_korea": "KR",
    "turkey": "TR",
}
LANGUAGE_ALIASES = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}
REGION_ALIASES = {
    "east_asia": "east_asia",
    "eastern_asia": "east_asia",
    "middle_east": "middle_east",
    "mideast": "middle_east",
    "north_america": "north_america",
    "northern_america": "north_america",
    "latin_america": "latin_america",
    "south_america": "latin_america",
    "south_asia": "south_asia",
    "southern_asia": "south_asia",
}

PERSPECTIVES_PLANNER_SCHEMA = JSONSchemaSpec(
    name="perspectives_query_plan",
    schema={
        "type": "object",
        "properties": {
            "plans": {
                "type": "array",
                "maxItems": MAX_PLANNER_STORIES_PER_CALL,
                "items": {
                    "type": "object",
                    "properties": {
                        "story_id": {"type": "string"},
                        "queries": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": MAX_CANONICAL_QUERIES},
                        "aligned_queries": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "anchor_groups": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string"},
                                    "terms": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                },
                                "required": ["kind", "terms"],
                                "additionalProperties": False,
                            },
                            "maxItems": 4,
                        },
                        "story_loci": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "country": {"type": "string"},
                                    "kind": {"type": "string", "enum": ["event_site", "affected_area"]},
                                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "reason": {"type": "string"},
                                },
                                "required": ["label", "country", "kind", "confidence", "reason"],
                                "additionalProperties": False,
                            },
                            "maxItems": 3,
                        },
                        "target_tags": {
                            "type": "object",
                            "properties": {
                                "countries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "regions": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "languages": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["countries", "regions"],
                            "additionalProperties": False,
                        },
                        "verification_targets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim_id": {"type": "string"},
                                    "importance_reason": {"type": "string"},
                                    "required_evidence_types": {"type": "array", "items": {"type": "string"}},
                                    "queries": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "query": {"type": "string"},
                                                "evidence_type": {"type": "string"},
                                            },
                                            "required": ["query", "evidence_type"],
                                        },
                                        "maxItems": 2,
                                    },
                                },
                                "required": ["claim_id", "importance_reason", "required_evidence_types", "queries"],
                            },
                            "maxItems": 2,
                        },
                    },
                    "required": ["story_id", "queries", "story_loci", "target_tags", "verification_targets"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plans"],
        "additionalProperties": False,
    },
)
MULTILINGUAL_QUERY_ALIGNMENT_SCHEMA = JSONSchemaSpec(
    name="multilingual_query_alignment",
    schema={
        "type": "object",
        "properties": {
            "aligned_queries": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "required": ["aligned_queries"],
        "additionalProperties": False,
    },
)
PERSPECTIVES_VERIFIER_SCHEMA = JSONSchemaSpec(
    name="perspectives_claim_verification",
    schema={
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "verdict": {"type": "string"},
            "verdict_scope": {"type": "string"},
            "supporting_evidence": {"type": "array", "items": {"type": "object"}},
            "contradicting_evidence": {"type": "array", "items": {"type": "object"}},
            "reasoning_summary": {"type": "string"},
            "source_independence": {"type": "string"},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "what_would_change_the_verdict": {"type": "string"},
        },
        "required": [
            "claim_id", "verdict", "verdict_scope", "supporting_evidence", "contradicting_evidence",
            "reasoning_summary", "source_independence", "limitations", "what_would_change_the_verdict",
        ],
    },
)
PERSPECTIVES_CLAIM_SYNTHESIS_SCHEMA = JSONSchemaSpec(
    name="perspectives_claim_led_synthesis",
    schema={
        "type": "object",
        "properties": {
            "story_id": {"type": "string"},
            "framing_report": {
                "type": "object",
                "properties": {
                    "synthesis": {"type": "string"},
                    "synthesis_article_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": MAX_ARTICLES_FOR_REPORT,
                    },
                    "shared_facts": PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
                    "repetition_without_independent_support": PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
                    "verified_or_independently_supported_claims": PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
                    "qualified_disputed_or_unresolved_claims": PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
                    "country_source_comparison": PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
                    "coverage_limitations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": MAX_ARTICLES_FOR_REPORT,
                    },
                },
                "required": [
                    "synthesis",
                    "synthesis_article_ids",
                    "shared_facts",
                    "repetition_without_independent_support",
                    "verified_or_independently_supported_claims",
                    "qualified_disputed_or_unresolved_claims",
                    "country_source_comparison",
                    "coverage_limitations",
                ],
                "additionalProperties": False,
            },
            "claim_perspectives": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "coverage": {
                            "type": "array",
                            "maxItems": MAX_ARTICLES_FOR_REPORT,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "article_id": {"type": "string"},
                                    "relationship": {"type": "string", "enum": sorted(RELATIONSHIPS)},
                                    "evidence_basis": {"type": "string"},
                                    "explanation": {"type": "string"},
                                },
                                "required": ["article_id", "relationship", "evidence_basis", "explanation"],
                                "additionalProperties": False,
                            },
                        },
                        "synthesis": {"type": "string"},
                        "coverage_limitations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_ARTICLES_FOR_REPORT,
                        },
                        "card": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "reporting_summary": {"type": "string"},
                                "evidence_check": {"type": "string"},
                                "qualification": {"type": "string"},
                                "limitations": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "required": ["claim_id", "coverage", "synthesis", "coverage_limitations", "card"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["story_id", "framing_report", "claim_perspectives"],
        "additionalProperties": False,
    },
)
PERSPECTIVES_FRAMING_SCHEMA = JSONSchemaSpec(
    name="perspectives_framing_report",
    schema={
        "type": "object",
        "properties": {
            "stories": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "story_id": {"type": "string"},
                        "synthesis": {"type": "string"},
                        "synthesis_article_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_ARTICLES_FOR_REPORT,
                        },
                        "shared_facts": PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
                        "country_source_comparison": PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
                        "coverage_limitations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_ARTICLES_FOR_REPORT,
                        },
                    },
                    "required": [
                        "story_id",
                        "synthesis",
                        "synthesis_article_ids",
                        "shared_facts",
                        "country_source_comparison",
                        "coverage_limitations",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["stories"],
        "additionalProperties": False,
    },
)
