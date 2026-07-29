from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from mydailynews.ai.base import JSONSchemaSpec
from mydailynews.ai.prompts import (
    PERSPECTIVES_FRAMING_SYSTEM,
    PERSPECTIVES_FRAMING_USER,
    PERSPECTIVES_PLANNER_RETRY_USER,
    PERSPECTIVES_PLANNER_SYSTEM,
    PERSPECTIVES_PLANNER_USER,
)
from mydailynews.ai.token_budget import resolve_client_token_budget
from mydailynews.app.models import BriefOutput, NewsCandidate, PerspectivesReportOutput
from mydailynews.briefing.output import write_json
from mydailynews.common.parallel import ordered_parallel_map
from mydailynews.common.utils import canonical_article_url, compact_json, normalize_whitespace, stable_id, utc_now
from mydailynews.common.warnings import extend_warnings
from mydailynews.domain.article_identity import article_aliases_for_candidate
from mydailynews.perspectives.sources import load_source_registry, match_source_by_domain, source_domain_map
from mydailynews.retrieval.article import ArticleRetriever
from mydailynews.retrieval.ddg import DuckDuckGoSearchRetriever
from mydailynews.retrieval.gdelt import GdeltDocRetriever
from mydailynews.retrieval.gnews import GNewsRetriever
from mydailynews.retrieval.registry_rss import RegistryRssRetriever


PERSPECTIVES_REPORT_SCHEMA_VERSION = "claim_led_perspectives_report.v4"
STRUCTURED_BRIEF_NAMES = ("general", "detailed")
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
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
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
    "ai",
    "eu",
    "uk",
    "year",
    "years",
    "global",
    "latest",
    "live",
    "news",
    "rule",
    "rules",
    "ruling",
    "supreme",
    "court",
    "update",
    "updates",
    "us",
    "world",
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


def run_perspectives_report(
    orchestrator,
    *,
    date: str,
    outputs: List[BriefOutput] | None = None,
    enrichment_json_path: str = "",
    allow_disk_fallback: bool = True,
) -> PerspectivesReportOutput | None:
    config = getattr(orchestrator.config, "perspectives_report", None)
    if not bool(getattr(config, "enabled", False)):
        orchestrator.warnings.append("perspectives_report: module is disabled by config; skipped.")
        orchestrator.debug.set_metric("module.perspectives_report.status", "skipped_disabled")
        return None

    warnings: List[str] = []
    output_dir = Path(orchestrator.config.output_dir)
    inputs = collect_perspectives_inputs(
        output_dir=output_dir,
        date=date,
        source_outputs=list(outputs or []),
        enrichment_json_path=enrichment_json_path,
        allow_disk_fallback=allow_disk_fallback,
    )
    extend_warnings(warnings, inputs["warnings"])
    if not inputs["articles"]:
        warnings.append(f"perspectives_report: no same-day enrichment or brief inputs were available for {date}.")
        extend_warnings(orchestrator.warnings, warnings)
        orchestrator.debug.set_metric("module.perspectives_report.status", "skipped_no_inputs")
        return None

    source_registry, registry_warnings = _load_coverage_registry()
    extend_warnings(warnings, registry_warnings)
    enabled_sources = [source for source in source_registry if bool(source.get("enabled", True))]

    _phase(orchestrator, "Planning perspectives coverage...")
    plans_by_story = plan_perspectives_queries(
        orchestrator,
        date=date,
        inputs=inputs,
        config=config,
        source_registry=enabled_sources,
        warnings=warnings,
    )

    _phase(orchestrator, "Retrieving perspectives coverage...")
    coverage_by_story = collect_global_coverage(
        orchestrator,
        inputs=inputs,
        config=config,
        warnings=warnings,
        plans_by_story=plans_by_story,
        source_registry=enabled_sources,
    )

    _phase(orchestrator, "Retrieving bounded claim evidence...")
    verification_documents, verification_diagnostics = collect_verification_documents(
        orchestrator,
        inputs=inputs,
        config=config,
        plans_by_story=plans_by_story,
        warnings=warnings,
    )

    _phase(orchestrator, "Verifying selected claims...")
    verification_by_claim = verify_selected_claims(
        orchestrator,
        inputs=inputs,
        plans_by_story=plans_by_story,
        documents_by_claim=verification_documents,
        warnings=warnings,
    )

    _phase(orchestrator, "Synthesizing claim-led perspectives...")
    framing_by_story = build_framing_comparisons(
        orchestrator,
        inputs=inputs,
        coverage_by_story=coverage_by_story,
        plans_by_story=plans_by_story,
        warnings=warnings,
        verification_by_claim=verification_by_claim,
        verification_documents=verification_documents,
    )

    _phase(orchestrator, "Writing perspectives report...")
    orchestrator.debug.set_metric("module.perspectives_report.status", "running")
    payload = build_perspectives_report_payload(
        date=date,
        inputs=inputs,
        config=config,
        warnings=warnings,
        plans_by_story=plans_by_story,
        coverage_by_story=coverage_by_story,
        framing_by_story=framing_by_story,
        verification_documents=verification_documents,
        verification_diagnostics=verification_diagnostics,
        output_dir=output_dir,
    )
    markdown_path, json_path = write_perspectives_report_outputs(output_dir, date, payload)
    _record_perspectives_report_artifact(orchestrator, payload=payload, markdown_path=markdown_path, json_path=json_path)

    metadata = payload.get("metadata", {})
    orchestrator.debug.set_metric("module.perspectives_report.status", "completed")
    orchestrator.debug.log(
        "perspectives_report.module",
        "complete",
        markdown=markdown_path,
        json=json_path,
        stories=metadata.get("story_count", 0),
        coverage_articles=metadata.get("coverage_article_count", 0),
        coverage_countries=metadata.get("coverage_source_country_count", 0),
    )
    extend_warnings(orchestrator.warnings, warnings)
    return PerspectivesReportOutput(
        name="perspectives_report",
        markdown_path=str(markdown_path),
        json_path=str(json_path),
        story_count=int(metadata.get("story_count", 0) or 0),
        coverage_article_count=int(metadata.get("coverage_article_count", 0) or 0),
        country_count=int(metadata.get("coverage_source_country_count", 0) or 0),
        warnings=warnings,
    )


def collect_perspectives_inputs(
    *,
    output_dir: Path,
    date: str,
    source_outputs: List[BriefOutput] | None = None,
    enrichment_json_path: str = "",
    allow_disk_fallback: bool = True,
) -> Dict[str, Any]:
    warnings: List[str] = []
    brief_payloads: List[Dict[str, Any]] = []
    brief_articles: List[Dict[str, Any]] = []
    source_briefs: List[str] = []
    outputs_by_name = {
        str(output.name or "").strip().lower(): output
        for output in source_outputs or []
        if str(output.name or "").strip().lower() in STRUCTURED_BRIEF_NAMES
    }
    for brief_name in STRUCTURED_BRIEF_NAMES:
        current_output = outputs_by_name.get(brief_name)
        path = Path(current_output.json_path) if current_output and str(current_output.json_path or "").strip() else None
        if path is None and allow_disk_fallback:
            path = output_dir / f"{date}_{brief_name}_brief.json"
        if path is None or not path.exists():
            continue
        payload = _load_json(path, warnings, f"{brief_name} brief")
        if not payload:
            continue
        brief_payloads.append(payload)
        source_briefs.append(brief_name)
        for item in _as_list(payload.get("selected_articles")):
            article = _article_from_selected(item)
            if article["id"] or article["source"]:
                article["source_briefs"] = sorted({*article["source_briefs"], brief_name})
                brief_articles.append(article)

    explicit_path = str(enrichment_json_path or "").strip()
    if explicit_path:
        enrichment_path = Path(explicit_path)
    elif allow_disk_fallback:
        enrichment_path = output_dir / f"{date}_enrichment.json"
    else:
        enrichment_path = None

    if enrichment_path is not None and enrichment_path.exists():
        payload = _load_json(enrichment_path, warnings, "enrichment")
        if payload:
            articles = [_article_from_selected(item) for item in _as_list(payload.get("selected_articles"))]
            articles = [article for article in articles if article["id"] or article["source"]]
            articles = _dedupe_articles([*articles, *brief_articles])
            stories = _stories_from_enrichment(payload, articles)
            _attach_claims_to_stories(stories, brief_payloads, warnings)
            return {
                "mode": "enrichment",
                "path": str(enrichment_path),
                "source_briefs": _unique_text([*_string_list(payload.get("source_briefs")), *source_briefs]),
                "articles": articles,
                "stories": stories,
                "warnings": warnings,
            }
    elif explicit_path:
        warnings.append(f"perspectives_report: enrichment JSON does not exist: {enrichment_path}.")

    articles = _dedupe_articles(brief_articles)
    stories = _fallback_stories(articles)
    _attach_claims_to_stories(stories, brief_payloads, warnings)
    return {
        "mode": "briefs",
        "path": "",
        "source_briefs": source_briefs,
        "articles": articles,
        "stories": stories,
        "warnings": warnings,
    }


def plan_perspectives_queries(
    orchestrator,
    *,
    date: str,
    inputs: Dict[str, Any],
    config: Any,
    source_registry: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    stories = [story for story in _as_list(inputs.get("stories")) if isinstance(story, dict)]
    failed = {_story_identity(story): _empty_plan(story, "planner_failed", ["planner was not run"]) for story in stories}
    ai_client = getattr(orchestrator, "summary_ai_client", None)
    if ai_client is None:
        warnings.append("perspectives_report: planner failed because no summary AI client is available.")
        return failed
    if not source_registry:
        warnings.append("perspectives_report: planner skipped because the active source registry is empty.")
        return failed

    budget = resolve_client_token_budget(ai_client, output_tokens=2400)
    batches = [stories[index : index + MAX_PLANNER_STORIES_PER_CALL] for index in range(0, len(stories), MAX_PLANNER_STORIES_PER_CALL)]
    plans: Dict[str, Dict[str, Any]] = {}
    for batch_index, batch_stories in enumerate(batches, start=1):
        prompt = _planner_prompt(
            date=date,
            inputs={**inputs, "stories": batch_stories},
            source_registry=source_registry,
            config=config,
        )
        batch_label = "perspectives planner" if len(batches) == 1 else f"perspectives planner batch {batch_index}/{len(batches)}"
        last_plans = {_story_identity(story): failed[_story_identity(story)] for story in batch_stories}
        last_warnings: List[str] = []
        for attempt in range(1, 3):
            user_prompt = prompt if attempt == 1 else PERSPECTIVES_PLANNER_RETRY_USER.format(prompt=prompt)
            try:
                raw = ai_client.complete_json(
                    PERSPECTIVES_PLANNER_SYSTEM,
                    user_prompt,
                    label=batch_label,
                    max_new_tokens=budget.output_tokens,
                    input_token_limit=budget.input_tokens,
                    json_schema=PERSPECTIVES_PLANNER_SCHEMA,
                )
            except Exception as exc:
                warnings.append(f"perspectives_report: {batch_label} failed after AI error: {type(exc).__name__}: {exc}")
                break
            last_warnings = []
            last_plans = _validate_planner_response(
                raw,
                stories=batch_stories,
                source_registry=source_registry,
                warnings=last_warnings,
            )
            if not _planner_needs_tag_retry(last_plans):
                break
        extend_warnings(warnings, last_warnings)
        plans.update(last_plans)
    return plans


def collect_global_coverage(
    orchestrator,
    *,
    inputs: Dict[str, Any],
    config: Any,
    warnings: List[str],
    plans_by_story: Dict[str, Dict[str, Any]] | None = None,
    source_registry: List[Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    max_records = int(getattr(config, "coverage_max_records_per_story", MAX_ARTICLES_FOR_REPORT) or 0)
    if max_records <= 0:
        warnings.append("perspectives_report: coverage_max_records_per_story is 0; coverage search skipped.")
        return {}

    if source_registry is None:
        source_registry, registry_warnings = _load_coverage_registry()
        extend_warnings(warnings, registry_warnings)
    source_registry = [source for source in source_registry if bool(source.get("enabled", True))]
    source_domains = source_domain_map(source_registry)
    source_by_id = {str(source.get("source_id") or ""): source for source in source_registry}
    plans_by_story = plans_by_story or {}

    phase = getattr(getattr(orchestrator, "reporter", None), "phase", None)
    retriever_kwargs = {
        "user_agent": str(getattr(orchestrator.config, "user_agent", "MyDailyNews/1.0")),
        "http_cache": getattr(orchestrator, "discovery_cache", getattr(orchestrator, "http_cache", None)),
        "http_cache_mode": str(getattr(getattr(orchestrator.config, "cache", None), "discovery_mode", "cache_first")),
        "debug": getattr(orchestrator, "debug", None),
        "progress_sink": phase if callable(phase) else None,
    }
    gdelt_retriever = GdeltDocRetriever(**retriever_kwargs)
    registry_rss_retriever = RegistryRssRetriever(source_registry, **retriever_kwargs)
    gnews_key = str(getattr(config, "gnews_api_key", "") or "").strip()
    gnews_retriever = GNewsRetriever(api_key=gnews_key, **retriever_kwargs) if gnews_key else None
    if gnews_retriever is None:
        warnings.append("perspectives_report: GNews is unavailable because gnews_api_key is not configured.")
    article_retriever = ArticleRetriever(
        str(getattr(orchestrator.config, "user_agent", "MyDailyNews/1.0")),
        ARTICLE_FETCH_MAX_CHARS,
        http_cache=getattr(orchestrator, "http_cache", None),
        debug=getattr(orchestrator, "debug", None),
    )

    articles_by_id = {str(article.get("id")): article for article in _as_list(inputs.get("articles")) if isinstance(article, dict) and article.get("id")}
    coverage: Dict[str, Dict[str, Any]] = {}
    for story in _as_list(inputs.get("stories")):
        if not isinstance(story, dict):
            continue
        story_id = _story_identity(story)
        plan = plans_by_story.get(story_id) or _empty_plan(story, "planner_failed", ["missing planner output"])
        coverage_story = dict(story)
        seed_articles = [articles_by_id[item] for item in _story_article_ids(story) if item in articles_by_id]
        coverage_story["_seed_text"] = _article_summary(seed_articles)
        coverage[story_id] = _collect_story_coverage(
            story=coverage_story,
            plan=plan,
            config=config,
            source_registry=source_registry,
            source_by_id=source_by_id,
            source_domains=source_domains,
            registry_rss_retriever=registry_rss_retriever,
            gdelt_retriever=gdelt_retriever,
            gnews_retriever=gnews_retriever,
            article_retriever=article_retriever,
            article_text_cache=getattr(orchestrator, "article_text_cache", None),
            max_article_workers=int(getattr(getattr(orchestrator.config, "runtime", None), "max_article_workers", 1) or 1),
            debug=getattr(orchestrator, "debug", None),
            warnings=warnings,
        )
    return coverage


def collect_verification_documents(
    orchestrator,
    *,
    inputs: Dict[str, Any],
    config: Any,
    plans_by_story: Dict[str, Dict[str, Any]],
    warnings: List[str],
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    documents: Dict[str, List[Dict[str, Any]]] = {}
    diagnostics: Dict[str, Any] = {
        "logical_queries": 0,
        "provider_requests": 0,
        "provider_request_counts": {},
        "cache_hits": 0,
        "document_fetches": 0,
        "retained_documents": 0,
        "claims": {},
    }
    claim_count = sum(
        1
        for story in _as_list(inputs.get("stories"))
        if isinstance(story, dict)
        for claim in _as_list(story.get("claims"))
        if isinstance(claim, dict)
    )
    diagnostics["input_claims"] = claim_count
    if not claim_count:
        diagnostics["status"] = "no_claims"
        return documents, diagnostics
    if not bool(getattr(config, "verification_enabled", True)):
        diagnostics["status"] = "verification_not_requested"
        diagnostics["reason"] = "disabled"
        return documents, diagnostics

    per_story = min(2, max(0, int(getattr(config, "verification_claims_per_story", 2) or 0)))
    run_limit = min(12, max(0, int(getattr(config, "verification_claims_per_run", 12) or 0)))
    query_limit = min(2, max(0, int(getattr(config, "verification_queries_per_claim", 2) or 0)))
    document_limit = min(4, max(0, int(getattr(config, "verification_documents_per_claim", 4) or 0)))
    if not run_limit or not query_limit or not document_limit:
        diagnostics["status"] = "verification_not_requested"
        diagnostics["reason"] = "budget_zero"
        return documents, diagnostics

    ddg = DuckDuckGoSearchRetriever(
        str(getattr(orchestrator.config, "user_agent", "MyDailyNews/1.0")),
        http_cache=getattr(orchestrator, "discovery_cache", getattr(orchestrator, "http_cache", None)),
        debug=getattr(orchestrator, "debug", None),
    )
    retriever_kwargs = {
        "user_agent": str(getattr(orchestrator.config, "user_agent", "MyDailyNews/1.0")),
        "http_cache": getattr(orchestrator, "discovery_cache", getattr(orchestrator, "http_cache", None)),
        "http_cache_mode": str(getattr(getattr(orchestrator.config, "cache", None), "discovery_mode", "cache_first")),
        "debug": getattr(orchestrator, "debug", None),
    }
    gnews_key = str(getattr(config, "gnews_api_key", "") or "").strip()
    news_retriever = GNewsRetriever(api_key=gnews_key, **retriever_kwargs) if gnews_key else GdeltDocRetriever(**retriever_kwargs)
    news_provider = "gnews" if gnews_key else "gdelt_doc"
    article_retriever = ArticleRetriever(
        str(getattr(orchestrator.config, "user_agent", "MyDailyNews/1.0")),
        ARTICLE_FETCH_MAX_CHARS,
        http_cache=getattr(orchestrator, "http_cache", None),
        debug=getattr(orchestrator, "debug", None),
    )
    selected_count = 0
    selected_claim_ids: set[str] = set()
    for story in _as_list(inputs.get("stories")):
        if not isinstance(story, dict):
            continue
        story_id = _story_identity(story)
        targets = _as_list((plans_by_story.get(story_id) or {}).get("verification_targets"))[:per_story]
        claims = {str(claim.get("claim_id") or ""): claim for claim in _as_list(story.get("claims")) if isinstance(claim, dict)}
        for target in targets:
            if not isinstance(target, dict):
                continue
            claim_id = str(target.get("claim_id") or "")
            if claim_id not in claims:
                continue
            if claim_id in selected_claim_ids:
                target["status"] = "selected_shared"
                diagnostics["claims"].setdefault(claim_id, {"status": "ready", "story_id": story_id})["duplicate_targets_collapsed"] = True
                continue
            if selected_count >= run_limit:
                diagnostics["claims"][claim_id] = {"status": "not_checked_budget", "story_id": story_id}
                target["status"] = "not_checked_budget"
                continue
            selected_count += 1
            selected_claim_ids.add(claim_id)
            target["status"] = "selected"
            rows: List[Dict[str, Any]] = []
            queries = _as_list(target.get("queries"))[:query_limit]
            for query_item in queries:
                if not isinstance(query_item, dict):
                    continue
                query = str(query_item.get("query") or "").strip()
                evidence_type = str(query_item.get("evidence_type") or "").strip().lower()
                if not query or evidence_type not in EVIDENCE_TYPES:
                    continue
                diagnostics["logical_queries"] += 1
                diagnostics["provider_requests"] += 1
                routed_provider = news_provider if evidence_type in {"independent", "counterevidence"} else "general_web"
                diagnostics["provider_request_counts"][routed_provider] = int(diagnostics["provider_request_counts"].get(routed_provider, 0)) + 1
                try:
                    if evidence_type in {"independent", "counterevidence"}:
                        found, search_warnings = news_retriever.search(
                            query,
                            timespan_days=int(getattr(config, "coverage_timespan_days", 7) or 7),
                            max_records=max(document_limit * 2, document_limit),
                            source_countries=[],
                        )
                        extend_warnings(warnings, search_warnings)
                    else:
                        found = ddg.search(query, max(document_limit * 2, document_limit))
                except Exception as exc:
                    warnings.append(f"perspectives_report: verification search failed for {claim_id!r} ({type(exc).__name__}: {exc}).")
                    continue
                for result in found:
                    is_news = isinstance(result, dict)
                    result_url = str(result.get("canonical_url") or result.get("url") or "") if is_news else result.url
                    result_title = str(result.get("title") or "") if is_news else result.title
                    result_source = str(result.get("source_name") or result.get("domain") or "") if is_news else result.source
                    result_snippet = str(result.get("snippet") or "") if is_news else result.snippet
                    url = canonical_article_url(result_url) or result_url
                    rows.append(
                        {
                            "article_id": f"verification-{stable_id(url, result_title)}",
                            "provider": news_provider if is_news else "general_web",
                            "url": result_url,
                            "canonical_url": url,
                            "domain": _hostname(url),
                            "source_name": result_source,
                            "source_country": str(result.get("source_country") or "") if is_news else "",
                            "source_language": str(result.get("source_language") or "") if is_news else "",
                            "source_key": result_source,
                            "title": result_title,
                            "snippet": result_snippet,
                            "published_at": str(result.get("published_at") or "") if is_news else "",
                            "retrieval_query": query,
                            "retrieval_query_type": "verification",
                            "evidence_type": evidence_type,
                            "source_role": "primary_or_origin_candidate" if evidence_type in {"primary", "origin"} else "independent_or_counterevidence_candidate",
                            "independent_of_claimant": evidence_type in {"independent", "counterevidence"},
                        }
                    )
            retained = _dedupe_coverage_rows(rows)[:document_limit]
            claim_story = dict(story)
            claim_story["summary"] = f"{story.get('summary', '')} {claims[claim_id].get('claim', '')}".strip()
            fetched = _fetch_selected_article_contexts(
                retained,
                story=claim_story,
                article_retriever=article_retriever,
                article_text_cache=getattr(orchestrator, "article_text_cache", None),
                max_workers=int(getattr(getattr(orchestrator.config, "runtime", None), "max_article_workers", 1) or 1),
                warnings=warnings,
            )
            claim_documents = []
            for row in fetched:
                if row.get("context_source") == "article_cache":
                    diagnostics["cache_hits"] += 1
                else:
                    diagnostics["document_fetches"] += 1
                claim_documents.append(
                    {
                        "document_id": str(row.get("article_id") or ""),
                        "title": str(row.get("title") or ""),
                        "source": str(row.get("source_name") or row.get("domain") or ""),
                        "url": str(row.get("canonical_url") or row.get("url") or ""),
                        "evidence_type": str(row.get("evidence_type") or ""),
                        "source_role": str(row.get("source_role") or ""),
                        "independent_of_claimant": bool(row.get("independent_of_claimant")),
                        "context_status": str(row.get("context_status") or ""),
                        "context_text": str(row.get("context_text") or ""),
                    }
                )
            documents[claim_id] = claim_documents
            diagnostics["retained_documents"] += len(claim_documents)
            diagnostics["claims"][claim_id] = {
                "status": "ready" if any(item.get("context_text") for item in claim_documents) else "retrieval_empty",
                "story_id": story_id,
                "queries": len(queries),
                "retained_documents": len(claim_documents),
            }
    if not selected_count:
        diagnostics["status"] = "verification_not_requested"
        diagnostics["reason"] = "no_targets"
    elif not diagnostics["logical_queries"]:
        diagnostics["status"] = "verification_failed"
        diagnostics["reason"] = "no_valid_queries"
    elif not diagnostics["retained_documents"]:
        diagnostics["status"] = "verification_failed"
        diagnostics["reason"] = "no_documents"
    else:
        diagnostics["status"] = "verification_completed"
    return documents, diagnostics


def verify_selected_claims(
    orchestrator,
    *,
    inputs: Dict[str, Any],
    plans_by_story: Dict[str, Dict[str, Any]],
    documents_by_claim: Dict[str, List[Dict[str, Any]]],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    claims = {
        str(claim.get("claim_id") or ""): claim
        for story in _as_list(inputs.get("stories"))
        if isinstance(story, dict)
        for claim in _as_list(story.get("claims"))
        if isinstance(claim, dict)
    }
    selected = {
        str(target.get("claim_id") or ""): str(target.get("status") or "selected")
        for plan in plans_by_story.values()
        for target in _as_list(plan.get("verification_targets"))
        if isinstance(target, dict)
    }
    results: Dict[str, Dict[str, Any]] = {}
    ai_client = getattr(orchestrator, "summary_ai_client", None)
    for claim_id, selection_status in selected.items():
        claim = claims.get(claim_id)
        documents = [item for item in documents_by_claim.get(claim_id, []) if str(item.get("context_text") or "").strip()]
        if selection_status == "not_checked_budget":
            results[claim_id] = _empty_verification("not_checked_budget")
            continue
        if not claim or not documents:
            results[claim_id] = _empty_verification("retrieval_empty")
            continue
        if ai_client is None:
            results[claim_id] = _empty_verification("no_ai_client")
            continue
        budget = resolve_client_token_budget(ai_client, output_tokens=1400)
        prompt = compact_json(
            {
                "claim": claim,
                "documents": [
                    {key: value for key, value in document.items() if key != "url"}
                    for document in documents
                ],
                "instructions": (
                    "Assess only this atomic claim against retrieved evidence. Publication count is not proof. "
                    "A claimant-controlled document verifies attribution but not unrelated external facts. "
                    "Use only supplied document_id values and one allowed verdict. State the evidence scope. "
                    "Return claim_id, verdict, verdict_scope, supporting_evidence and contradicting_evidence items "
                    "with document_id/evidence_basis/explanation, reasoning_summary, source_independence, limitations, "
                    "and what_would_change_the_verdict. Allowed verdicts: supported, mostly_supported, mixed, "
                    "contradicted, unresolved, not_checkable_yet."
                ),
            }
        )
        try:
            raw = ai_client.complete_json(
                "You are a focused claim verifier. Return JSON only and do not claim truth beyond retrieved evidence.",
                prompt,
                label="perspectives claim verifier",
                max_new_tokens=budget.output_tokens,
                input_token_limit=budget.input_tokens,
                json_schema=PERSPECTIVES_VERIFIER_SCHEMA,
            )
        except Exception as exc:
            warnings.append(f"perspectives_report: verifier failed for {claim_id!r} ({type(exc).__name__}: {exc}).")
            results[claim_id] = _empty_verification("verifier_failed")
            continue
        results[claim_id] = _normalize_verification(raw, claim_id, documents, warnings)
    return results


def _empty_verification(status: str) -> Dict[str, Any]:
    return {
        "selected": True,
        "status": status,
        "verdict": "unresolved",
        "verdict_scope": "no_usable_retrieved_evidence",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "reasoning_summary": "Focused verification did not produce usable evidence.",
        "source_independence": "not established",
        "limitations": [status.replace("_", " ")],
        "what_would_change_the_verdict": "Relevant primary or independent evidence.",
    }


def _normalize_verification(
    raw: Dict[str, Any],
    claim_id: str,
    documents: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    allowed_ids = {str(item.get("document_id") or "") for item in documents}
    if str(raw.get("claim_id") or "") != claim_id:
        warnings.append(f"perspectives_report: verifier returned an unknown claim_id for {claim_id!r}; rejected.")
        return _empty_verification("invalid_claim_id")
    verdict = str(raw.get("verdict") or "").strip().lower()
    scope = normalize_whitespace(str(raw.get("verdict_scope") or ""))
    if verdict not in VERDICTS or not scope:
        warnings.append(f"perspectives_report: verifier returned an invalid verdict or scope for {claim_id!r}; rejected.")
        return _empty_verification("invalid_verdict")

    def evidence_items(value: Any) -> List[Dict[str, Any]]:
        output = []
        for item in _as_list(value):
            if not isinstance(item, dict):
                continue
            document_id = str(item.get("document_id") or "")
            if document_id not in allowed_ids:
                if document_id:
                    warnings.append(f"perspectives_report: verifier used unknown document_id {document_id!r}; discarded.")
                continue
            output.append(
                {
                    "document_id": document_id,
                    "evidence_basis": normalize_whitespace(str(item.get("evidence_basis") or "")),
                    "explanation": normalize_whitespace(str(item.get("explanation") or "")),
                }
            )
        return output

    supporting = evidence_items(raw.get("supporting_evidence"))
    contradicting = evidence_items(raw.get("contradicting_evidence"))
    if verdict in {"supported", "contradicted"} and not (supporting if verdict == "supported" else contradicting):
        warnings.append(f"perspectives_report: verifier verdict {verdict!r} lacked cited evidence for {claim_id!r}; downgraded to unresolved.")
        verdict = "unresolved"
    return {
        "selected": True,
        "status": "checked",
        "verdict": verdict,
        "verdict_scope": scope,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "reasoning_summary": normalize_whitespace(str(raw.get("reasoning_summary") or "")),
        "source_independence": normalize_whitespace(str(raw.get("source_independence") or "")),
        "limitations": _text_items(raw.get("limitations")),
        "what_would_change_the_verdict": normalize_whitespace(str(raw.get("what_would_change_the_verdict") or "")),
    }


def build_framing_comparisons(
    orchestrator,
    *,
    inputs: Dict[str, Any],
    coverage_by_story: Dict[str, Dict[str, Any]],
    plans_by_story: Dict[str, Dict[str, Any]],
    warnings: List[str],
    verification_by_claim: Dict[str, Dict[str, Any]] | None = None,
    verification_documents: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    verification_by_claim = verification_by_claim or {}
    verification_documents = verification_documents or {}
    story_payloads = []
    empty_reports: Dict[str, Dict[str, Any]] = {}
    for story in _as_list(inputs.get("stories")):
        if not isinstance(story, dict):
            continue
        story_id = _story_identity(story)
        coverage = coverage_by_story.get(story_id, {})
        articles = _framing_articles_payload(_as_list(coverage.get("coverage_articles")))
        if not articles:
            empty_reports[story_id] = _empty_framing_report("No retrieved articles were available for this story.")
            continue
        target_by_claim = {
            str(target.get("claim_id") or ""): target
            for target in _as_list(plans_by_story.get(story_id, {}).get("verification_targets"))
            if isinstance(target, dict)
        }
        focused_claims = [
            {
                **claim,
                "importance_reason": str(
                    target_by_claim[str(claim.get("claim_id") or "")].get("importance_reason") or ""
                ),
            }
            for claim in _as_list(story.get("claims"))
            if isinstance(claim, dict) and str(claim.get("claim_id") or "") in target_by_claim
        ]
        story_payloads.append(
            {
                "story_id": story_id,
                "title": str(story.get("story_title") or story.get("title") or ""),
                "summary": str(story.get("summary") or ""),
                "articles": articles,
                "coverage_limitations": list(coverage.get("coverage_quality", {}).get("thin_reasons", [])),
                "claims": focused_claims,
                "confirmed_facts": _as_list(story.get("confirmed_facts")),
                "conflicting_claims": _as_list(story.get("conflicting_claims")),
                "open_questions": _as_list(story.get("open_questions")),
            }
        )

    if not story_payloads:
        return empty_reports

    ai_client = getattr(orchestrator, "summary_ai_client", None)
    if ai_client is None:
        warnings.append("perspectives_report: framing comparison skipped because no summary AI client is available.")
        for story in story_payloads:
            empty_reports[story["story_id"]] = _empty_framing_report("Framing comparison skipped because no AI client was available.")
        return empty_reports

    framing_reports = dict(empty_reports)
    budget = resolve_client_token_budget(ai_client)
    for story_payload in story_payloads:
        story_id = story_payload["story_id"]
        if not any(article.get("context_text") for article in story_payload["articles"]):
            framing_reports[story_id] = _empty_framing_report("Retrieved articles had metadata only; framing comparison was not attempted.")
            continue
        try:
            claim_led = bool(story_payload.get("claims"))
            if claim_led:
                claim_ids = {str(claim.get("claim_id") or "") for claim in story_payload["claims"] if isinstance(claim, dict)}
                story_payload["verification"] = {
                    claim_id: verification_by_claim[claim_id]
                    for claim_id in claim_ids
                    if claim_id in verification_by_claim
                }
                story_payload["verification_document_metadata"] = {
                    claim_id: [
                        {key: value for key, value in document.items() if key not in {"url", "context_text"}}
                        for document in verification_documents.get(claim_id, [])
                    ]
                    for claim_id in claim_ids
                    if claim_id in verification_documents
                }
            raw = ai_client.complete_json(
                "You synthesize claim-led perspectives from supplied claims, retrieved reporting, and scoped verification. Return JSON only. Never treat repetition as proof or infer national opinion.",
                _claim_synthesis_prompt(ai_client, story_payload, budget.input_tokens) if claim_led else _framing_prompt([story_payload]),
                label="perspectives claim-led synthesis" if claim_led else "perspectives framing report",
                max_new_tokens=budget.output_tokens,
                input_token_limit=budget.input_tokens,
                json_schema=PERSPECTIVES_CLAIM_SYNTHESIS_SCHEMA if claim_led else PERSPECTIVES_FRAMING_SCHEMA,
            )
        except Exception as exc:
            warnings.append(f"perspectives_report: framing comparison failed for {story_id!r} after AI error: {type(exc).__name__}: {exc}")
            framing_reports[story_id] = _empty_framing_report("Framing comparison unavailable after AI error.")
            continue
        known_ids = [str(article.get("article_id") or "") for article in story_payload["articles"]]
        if story_payload.get("claims"):
            framing_reports[story_id] = _normalize_claim_synthesis_response(
                raw,
                story=story_payload,
                known_article_ids=known_ids,
                verification_by_claim=verification_by_claim,
                verification_documents=verification_documents,
                warnings=warnings,
            )
        else:
            framing_reports.update(
                _normalize_framing_response(
                    raw,
                    known_story_ids=[story_id],
                    known_article_ids={story_id: known_ids},
                    warnings=warnings,
                )
            )
    return framing_reports


def build_perspectives_report_payload(
    *,
    date: str,
    inputs: Dict[str, Any],
    config: Any,
    warnings: List[str],
    plans_by_story: Dict[str, Dict[str, Any]] | None = None,
    coverage_by_story: Dict[str, Dict[str, Any]] | None = None,
    framing_by_story: Dict[str, Dict[str, Any]] | None = None,
    verification_documents: Dict[str, List[Dict[str, Any]]] | None = None,
    verification_diagnostics: Dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    articles_by_id = {article["id"]: article for article in inputs["articles"] if article.get("id")}
    plans_by_story = plans_by_story or {}
    coverage_by_story = coverage_by_story or {}
    framing_by_story = framing_by_story or {}
    verification_documents = verification_documents or {}
    stories = []
    seed_source_keys: set[str] = set()
    for story in _as_list(inputs.get("stories")):
        if not isinstance(story, dict):
            continue
        story_id = _story_identity(story)
        story_articles = [articles_by_id[item] for item in _story_article_ids(story) if item in articles_by_id]
        if not story_articles:
            continue
        report = _story_report(
            story=story,
            articles=story_articles,
            plan=plans_by_story.get(story_id),
            coverage=coverage_by_story.get(story_id),
            framing=framing_by_story.get(story_id),
            verification_documents=verification_documents,
        )
        stories.append(report)
        for source in report["seed_sources"]:
            seed_source_keys.add(source["source_key"])

    coverage_article_count = sum(int(story.get("coverage_counts", {}).get("articles", 0) or 0) for story in stories)
    coverage_countries = {
        country
        for story in stories
        for country in (story.get("coverage_counts", {}).get("source_countries", {}) or {}).keys()
        if country != "unknown"
    }
    coverage_languages = {
        language
        for story in stories
        for language in (story.get("coverage_counts", {}).get("languages", {}) or {}).keys()
        if language != "unknown"
    }
    provider_counts: Counter[str] = Counter()
    for story in stories:
        for provider, count in (story.get("coverage_provider_counts") or {}).items():
            provider_counts[str(provider)] += int(count or 0)

    card_count = sum(len(_as_list(story.get("claim_context_cards"))) for story in stories)
    evidence_claim_count = sum(
        len(_as_list(story.get("claims")))
        for story in _as_list(inputs.get("stories"))
        if isinstance(story, dict)
    )

    return {
        "schema": PERSPECTIVES_REPORT_SCHEMA_VERSION,
        "date": date,
        "generated_at": utc_now().isoformat(),
        "config": {
            "lookback_days": int(getattr(config, "coverage_timespan_days", 7) or 7),
            "max_selected_sources_per_story": MAX_SELECTED_SOURCES,
            "max_canonical_queries_per_story": MAX_CANONICAL_QUERIES,
            "requested_max_articles_per_story": max(0, int(getattr(config, "coverage_max_records_per_story", MAX_ARTICLES_FOR_REPORT) or 0)),
            "max_articles_for_report": _effective_article_limit(config),
            "max_articles_per_source": MAX_ARTICLES_PER_SOURCE,
            "max_records_per_provider_request": _effective_article_limit(config),
            "article_context_token_budget": ARTICLE_CONTEXT_TOKEN_BUDGET,
            "max_article_context_chars": ARTICLE_CONTEXT_MAX_CHARS,
            "meaningful_paragraph_min_chars": MEANINGFUL_PARAGRAPH_MIN_CHARS,
            "gnews_enabled": bool(str(getattr(config, "gnews_api_key", "") or "").strip()),
            "verification_enabled": bool(getattr(config, "verification_enabled", True)),
            "verification_claims_per_story": min(2, int(getattr(config, "verification_claims_per_story", 2) or 0)),
            "verification_claims_per_run": min(12, int(getattr(config, "verification_claims_per_run", 12) or 0)),
            "verification_queries_per_claim": min(2, int(getattr(config, "verification_queries_per_claim", 2) or 0)),
            "verification_documents_per_claim": min(4, int(getattr(config, "verification_documents_per_claim", 4) or 0)),
        },
        "input": {
            "mode": inputs.get("mode", ""),
            "path": inputs.get("path", ""),
            "source_briefs": inputs.get("source_briefs", []),
            "article_count": len(inputs.get("articles", [])),
        },
        "metadata": {
            "story_count": len(stories),
            "seed_article_count": len(inputs.get("articles", [])),
            "seed_source_count": len(seed_source_keys),
            "coverage_article_count": coverage_article_count,
            "coverage_source_country_count": len(coverage_countries),
            "coverage_language_count": len(coverage_languages),
            "coverage_provider_counts": _sorted_counter(provider_counts),
        },
        "stories": stories,
        "evidence_diagnostics": {
            "status": "claims_available" if evidence_claim_count else "no_claims",
            "claims": evidence_claim_count,
        },
        "verification_diagnostics": verification_diagnostics or {},
        "claim_card_diagnostics": {
            "status": "cards_produced" if card_count else "no_cards_produced",
            "cards": card_count,
        },
        "warnings": _unique_text(warnings),
    }


def write_perspectives_report_outputs(output_dir: Path, date: str, payload: Dict[str, Any]) -> tuple[Path, Path]:
    markdown_path = output_dir / f"{date}_perspectives_report.md"
    json_path = output_dir / f"{date}_perspectives_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_perspectives_report_markdown(payload), encoding="utf-8")
    write_json(json_path, payload)
    return markdown_path, json_path


def render_perspectives_report_markdown(payload: Dict[str, Any]) -> str:
    date = str(payload.get("date", "") or "")
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    lines = [
        f"# Perspectives Report - {date}",
        "",
        f"_Stories: {metadata.get('story_count', 0)} | Coverage articles: {metadata.get('coverage_article_count', 0)} | Source countries: {metadata.get('coverage_source_country_count', 0)}_",
        "",
    ]
    for story in _as_list(payload.get("stories")):
        if not isinstance(story, dict):
            continue
        title = str(story.get("story_title") or story.get("story_id") or "Story")
        lines.extend([f"## {title}", ""])
        plan = story.get("planner", {}) if isinstance(story.get("planner"), dict) else {}
        coverage_quality = story.get("coverage_quality", {}) if isinstance(story.get("coverage_quality"), dict) else {}
        lines.append(
            f"_Planner: {plan.get('status', 'unknown')} | Coverage: {coverage_quality.get('status', 'unknown')} | Providers: {_count_text(story.get('coverage_provider_counts') or {}) or 'none'}_"
        )
        lines.append("")
        framing = story.get("framing_report", {}) if isinstance(story.get("framing_report"), dict) else {}
        synthesis = normalize_whitespace(str(framing.get("synthesis") or ""))
        if synthesis:
            lines.extend(["### Bottom line", synthesis, ""])
        _append_evidence_section(lines, "What coverage agrees on", framing.get("shared_facts"))
        _append_evidence_section(lines, "Repeated claims without independent support", framing.get("repetition_without_independent_support"))
        _append_evidence_section(lines, "Claims supported by retrieved evidence", framing.get("verified_or_independently_supported_claims"))
        _append_evidence_section(lines, "Qualified, disputed, or unresolved claims", framing.get("qualified_disputed_or_unresolved_claims"))
        _append_evidence_section(lines, "Where the framing differs", framing.get("country_source_comparison"))
        limitations = _string_list(framing.get("coverage_limitations")) + _string_list(coverage_quality.get("thin_reasons"))
        if limitations:
            lines.append("### What the coverage cannot establish")
            for item in _unique_text(limitations):
                lines.append(f"- {item}")
            lines.append("")
    warnings = _string_list(payload.get("warnings"))
    if warnings:
        lines.append("## Diagnostics")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    references = []
    seen: set[str] = set()
    for story in _as_list(payload.get("stories")):
        if not isinstance(story, dict):
            continue
        for article in _as_list(story.get("coverage_articles")):
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "").strip()
            source = str(article.get("source_name") or article.get("domain") or "").strip()
            url = str(article.get("canonical_url") or article.get("url") or "").strip()
            key = url or f"{title}|{source}"
            if not key or key in seen:
                continue
            seen.add(key)
            references.append((title, source, url))
        for document in _as_list(story.get("verification_documents")):
            if not isinstance(document, dict):
                continue
            title = str(document.get("title") or "").strip()
            source = str(document.get("source") or "").strip()
            url = str(document.get("url") or "").strip()
            key = url or f"{title}|{source}"
            if not key or key in seen:
                continue
            seen.add(key)
            references.append((title, source, url))
    if references:
        lines.append("## References")
        for title, source, url in references:
            if title and source:
                lines.append(f"- {title} ({source})")
            elif title:
                lines.append(f"- {title}")
            elif source:
                lines.append(f"- {source}")
            if url:
                lines.append(f"  {url}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _planner_prompt(*, date: str, inputs: Dict[str, Any], source_registry: List[Dict[str, Any]], config: Any) -> str:
    articles_by_id = {article["id"]: article for article in inputs.get("articles", []) if article.get("id")}
    payload = {
        "report_date": date,
        "lookback_days": int(getattr(config, "coverage_timespan_days", 7) or 7),
        "stories": [_story_planner_payload(story, articles_by_id) for story in _as_list(inputs.get("stories")) if isinstance(story, dict)],
        "tag_options": _planner_tag_options(source_registry),
    }
    return PERSPECTIVES_PLANNER_USER.format(date=date, data=compact_json(payload))


def _validate_planner_response(
    raw: Dict[str, Any],
    *,
    stories: List[Dict[str, Any]],
    source_registry: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    plans = {_story_identity(story): _empty_plan(story, "planner_failed", ["missing planner line"]) for story in stories}
    planner_items = _planner_items(raw, warnings)
    if not planner_items:
        warnings.append("perspectives_report: planner returned no plans.")
        return plans

    story_by_id = {_story_identity(story): story for story in stories}
    for plan_number, item in planner_items:
        story_id = str(item.get("story_id") or "").strip()
        if story_id not in story_by_id:
            warnings.append(f"perspectives_report: planner plan {plan_number} used unknown story_id {story_id!r}; discarded.")
            continue
        plans[story_id] = _normalize_planner_item(item, story_by_id[story_id], source_registry)
        if plans[story_id].get("status") != "ok":
            warnings.append(
                f"perspectives_report: planner plan {plan_number} for {story_id!r} was incomplete: "
                + "; ".join(_string_list(plans[story_id].get("diagnostics")))
            )
    return plans


def _planner_items(raw: Dict[str, Any], warnings: List[str]) -> List[tuple[int, Dict[str, Any]]]:
    if not isinstance(raw, dict):
        return []
    raw_plans = raw.get("plans")
    if not isinstance(raw_plans, list):
        return []
    items = []
    for plan_number, item in enumerate(raw_plans, start=1):
        if not isinstance(item, dict):
            warnings.append(f"perspectives_report: planner plan {plan_number} was not an object.")
            continue
        items.append((plan_number, item))
    return items


def _planner_needs_tag_retry(plans: Dict[str, Dict[str, Any]]) -> bool:
    return any(
        str(diagnostic).startswith("missing required target tags") or str(diagnostic) == "no usable target tags"
        for plan in plans.values()
        for diagnostic in _string_list(plan.get("diagnostics"))
    )


def _normalize_planner_item(item: Dict[str, Any], story: Dict[str, Any], source_registry: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_tags = item.get("target_tags") if isinstance(item.get("target_tags"), dict) else {}
    normalized_tags, rejected_tags = _normalize_target_tags(raw_tags, source_registry)
    diagnostics = []
    if any(rejected_tags.values()):
        diagnostics.append(f"unmatched target tags: {compact_json(rejected_tags)}")

    queries = _text_list(item.get("queries"), max_items=MAX_CANONICAL_QUERIES, max_chars=180)
    anchor_groups = _normalize_anchor_groups(item.get("anchor_groups"))
    if not anchor_groups:
        anchor_groups = _seed_anchor_groups(story)
    if not queries:
        diagnostics.append("no usable canonical query")
    story_loci = []
    for raw_locus in _as_list(item.get("story_loci"))[:3]:
        if not isinstance(raw_locus, dict):
            continue
        label = _short_text(raw_locus.get("label"), 100)
        kind = str(raw_locus.get("kind") or "").strip().lower()
        confidence = str(raw_locus.get("confidence") or "").strip().lower()
        if not label or kind not in {"event_site", "affected_area"} or confidence not in {"high", "medium", "low"}:
            continue
        story_loci.append(
            {
                "label": label,
                "country": str(raw_locus.get("country") or "").strip().upper()[:3],
                "kind": kind,
                "confidence": confidence,
                "reason": _short_text(raw_locus.get("reason"), 240),
            }
        )
    missing_tag_groups = [name for name in ("countries", "regions") if not normalized_tags.get(name)]
    if missing_tag_groups:
        diagnostics.append("missing required target tags: " + ", ".join(missing_tag_groups))
    if len(missing_tag_groups) == 2:
        diagnostics.append("no usable target tags")
    known_claims = {str(claim.get("claim_id") or ""): claim for claim in _as_list(story.get("claims")) if isinstance(claim, dict)}
    verification_targets: List[Dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    for raw_target in _as_list(item.get("verification_targets"))[:2]:
        if not isinstance(raw_target, dict):
            continue
        claim_id = str(raw_target.get("claim_id") or "").strip()
        if claim_id not in known_claims or claim_id in seen_claim_ids:
            if claim_id:
                diagnostics.append(f"unknown or duplicate verification claim_id: {claim_id}")
            continue
        target_queries = []
        for raw_query in _as_list(raw_target.get("queries"))[:2]:
            if not isinstance(raw_query, dict):
                continue
            query = _short_text(raw_query.get("query"), 180)
            evidence_type = str(raw_query.get("evidence_type") or "").strip().lower()
            if query and evidence_type in EVIDENCE_TYPES:
                target_queries.append({"query": query, "evidence_type": evidence_type})
        required_evidence_types = [value for value in _string_list(raw_target.get("required_evidence_types")) if value in EVIDENCE_TYPES]
        missing_evidence_types = set(required_evidence_types).difference(
            str(query.get("evidence_type") or "") for query in target_queries
        )
        if missing_evidence_types:
            diagnostics.append(f"verification target {claim_id} lacked required query types: {', '.join(sorted(missing_evidence_types))}")
            continue
        if not target_queries:
            diagnostics.append(f"verification target {claim_id} had no usable targeted query")
            continue
        seen_claim_ids.add(claim_id)
        verification_targets.append(
            {
                "claim_id": claim_id,
                "importance_reason": normalize_whitespace(str(raw_target.get("importance_reason") or "")),
                "required_evidence_types": required_evidence_types,
                "queries": target_queries,
            }
        )
    return {
        "story_id": _story_identity(story),
        "status": "ok" if queries and not missing_tag_groups else "planner_failed",
        "queries": queries,
        "anchor_groups": anchor_groups,
        "story_loci": story_loci,
        "target_tags": {
            "raw": {
                "countries": _string_list(raw_tags.get("countries") if isinstance(raw_tags, dict) else []),
                "regions": _string_list(raw_tags.get("regions") if isinstance(raw_tags, dict) else []),
                "languages": _string_list(raw_tags.get("languages") if isinstance(raw_tags, dict) else []),
            },
            "normalized": normalized_tags,
            "rejected": rejected_tags,
        },
        "diagnostics": diagnostics,
        "verification_targets": verification_targets,
    }


def _collect_story_coverage(
    *,
    story: Dict[str, Any],
    plan: Dict[str, Any],
    config: Any,
    source_registry: List[Dict[str, Any]],
    source_by_id: Dict[str, Dict[str, Any]],
    source_domains: Dict[str, Dict[str, Any]],
    registry_rss_retriever: RegistryRssRetriever,
    gdelt_retriever: GdeltDocRetriever,
    gnews_retriever: GNewsRetriever | None,
    article_retriever: ArticleRetriever,
    warnings: List[str],
    article_text_cache: Any = None,
    max_article_workers: int = 1,
    debug: Any = None,
) -> Dict[str, Any]:
    story_id = _story_identity(story)
    if plan.get("status") != "ok":
        return _coverage_gap(story_id, plan, "planner failed")

    selected_sources, source_diagnostics = _select_sources_for_plan(plan, source_registry, config)
    plan["selected_sources"] = [_source_metadata(source) for source in selected_sources]
    plan["diagnostics"] = _unique_text([*plan.get("diagnostics", []), *source_diagnostics])
    if not selected_sources:
        return _coverage_gap(story_id, plan, "no active sources matched planner tags")

    requests = _retrieval_requests(plan, selected_sources, story=story)
    if not requests:
        return _coverage_gap(story_id, plan, "planner produced no usable retrieval requests for selected sources")

    providers: List[tuple[str, Any]] = [("gdelt_doc", gdelt_retriever), ("registry_rss", registry_rss_retriever)]
    if gnews_retriever is not None:
        providers.append(("gnews", gnews_retriever))

    def worker(provider: tuple[str, Any]) -> Dict[str, Any]:
        return _run_provider_requests(
            provider,
            requests=requests,
            config=config,
            source_by_id=source_by_id,
            source_domains=source_domains,
        )

    provider_results = ordered_parallel_map(providers, max_workers=len(providers), worker=worker, on_exception=_provider_exception)
    all_rows: List[Dict[str, Any]] = []
    provider_statuses: Dict[str, Dict[str, Any]] = {}
    provider_warnings: List[str] = []
    for result in provider_results:
        provider = str(result.get("provider") or "unknown")
        rows = _as_list(result.get("rows"))
        all_rows.extend(row for row in rows if isinstance(row, dict))
        provider_statuses[provider] = {
            "status": str(result.get("status") or "ok"),
            "rows": len(rows),
            "warnings": _unique_text(_string_list(result.get("warnings"))),
        }
        provider_warnings.extend(_string_list(result.get("warnings")))
    if gnews_retriever is None:
        provider_statuses["gnews"] = {
            "status": "unavailable",
            "rows": 0,
            "warnings": ["gnews_api_key is not configured"],
        }
    extend_warnings(warnings, provider_warnings)

    ranked = _rank_coverage_rows(all_rows, story)
    deduped = _dedupe_coverage_rows(ranked)
    relevance_decisions = [
        {
            "article_id": str(row.get("article_id") or ""),
            "accepted": accepted,
            "reason": reason,
            "score": _coverage_relevance_score(row, story),
        }
        for row in deduped
        for accepted, reason in [_coverage_relevance_decision(row, story)]
    ]
    relevant = [row for row, decision in zip(deduped, relevance_decisions) if decision["accepted"]]
    relevance_reason_counts = Counter(
        f"{'accepted' if decision['accepted'] else 'rejected'}:{decision['reason']}"
        for decision in relevance_decisions
    )
    article_limit = _effective_article_limit(config)
    selected = _diverse_article_selection(relevant, max_articles=article_limit)
    articles = _fetch_selected_article_contexts(
        selected,
        story=story,
        article_retriever=article_retriever,
        article_text_cache=article_text_cache,
        max_workers=max_article_workers,
        warnings=warnings,
    )
    funnel_counts = _coverage_funnel_counts(all_rows, deduped, relevant, articles)
    source_yields = _coverage_source_yields(selected_sources, all_rows, articles)
    if debug is not None:
        debug.log("perspectives_report.retrieval", "funnel", story_id=story_id, **funnel_counts["overall"])
        for reason, count in relevance_reason_counts.items():
            debug.increment(f"perspectives.relevance.{reason}", count)
    return {
        "coverage_status": "ok" if articles else "coverage_gap",
        "coverage_gap": "" if articles else "all providers returned no relevant rows",
        "selected_sources": [_source_metadata(source) for source in selected_sources],
        "retrieval_request_count": len(requests),
        "retrieval_queries": _retrieval_query_summary(requests),
        "provider_statuses": provider_statuses,
        "provider_warnings": _unique_text(provider_warnings),
        "funnel_counts": funnel_counts,
        "relevance_diagnostics": {
            "reason_counts": dict(sorted(relevance_reason_counts.items())),
            "rows": relevance_decisions,
        },
        "source_yields": source_yields,
        "effective_limits": {
            "requested_articles_per_story": max(0, int(getattr(config, "coverage_max_records_per_story", 0) or 0)),
            "articles_per_story": article_limit,
            "records_per_provider_request": article_limit,
            "articles_per_source": MAX_ARTICLES_PER_SOURCE,
        },
        "coverage_provider": _coverage_provider_label(articles),
        "coverage_provider_counts": _provider_counts(articles),
        "coverage_articles": articles,
        "coverage_counts": _coverage_counts(articles),
        "coverage_quality": _coverage_quality(articles, config),
        "context_stats": _context_stats(articles),
    }


def _run_provider_requests(
    provider: tuple[str, Any],
    *,
    requests: List[Dict[str, Any]],
    config: Any,
    source_by_id: Dict[str, Dict[str, Any]],
    source_domains: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    provider_name, retriever = provider
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    failed = False
    max_records = _effective_article_limit(config)
    for request in requests:
        try:
            search_kwargs = {
                "timespan_days": int(getattr(config, "coverage_timespan_days", 7) or 7),
                "max_records": max_records,
                "source_countries": request.get("source_countries", []),
            }
            if provider_name == "registry_rss":
                search_kwargs["source_ids"] = request.get("source_ids", [])
            found, request_warnings = retriever.search(
                request["query"],
                **search_kwargs,
            )
        except Exception as exc:
            failed = True
            warnings.append(f"{provider_name}: failed for query {request['query']!r} ({type(exc).__name__}: {exc})")
            break
        warnings.extend(_string_list(request_warnings))
        rows.extend(
            _tag_retrieval_row(row, request=request, provider_name=provider_name, source_by_id=source_by_id, source_domains=source_domains)
            for row in _as_list(found)
            if isinstance(row, dict)
        )
        if provider_name == "gdelt_doc" and bool(getattr(retriever, "rate_limited", False)):
            break
    return {
        "provider": provider_name,
        "status": "failed" if failed else "warning" if warnings else "ok",
        "rows": rows,
        "warnings": _unique_text(warnings),
    }


def _provider_exception(position: int, item: tuple[str, Any], exc: Exception) -> Dict[str, Any]:
    provider_name = item[0] if item else f"provider_{position}"
    return {
        "provider": provider_name,
        "status": "failed",
        "rows": [],
        "warnings": [f"{provider_name}: failed ({type(exc).__name__}: {exc})"],
    }


def _tag_retrieval_row(
    row: Dict[str, Any],
    *,
    request: Dict[str, Any],
    provider_name: str,
    source_by_id: Dict[str, Dict[str, Any]],
    source_domains: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    current = dict(row)
    current["provider"] = str(current.get("provider") or provider_name)
    url = str(current.get("canonical_url") or current.get("url") or "")
    canonical = canonical_article_url(url)
    domain = _hostname(canonical or str(current.get("url") or ""))
    source_id = str(current.get("source_id") or "")
    registry_source = source_by_id.get(source_id) or match_source_by_domain(source_domains, domain)
    if registry_source:
        source_id = str(registry_source.get("source_id") or source_id)
        current["source_name"] = str(current.get("source_name") or registry_source.get("name") or "")
        current["source_country"] = str(current.get("source_country") or registry_source.get("country") or "").upper()
        current["source_language"] = str(current.get("source_language") or registry_source.get("language") or "").lower()
    current["source_id"] = source_id
    current["canonical_url"] = canonical or str(current.get("url") or "")
    current["domain"] = domain or str(current.get("domain") or "")
    current["title"] = normalize_whitespace(str(current.get("title") or ""))
    current["snippet"] = normalize_whitespace(str(current.get("snippet") or ""))
    current["feed_content"] = normalize_whitespace(str(current.get("feed_content") or ""))
    current["feed_summary"] = normalize_whitespace(str(current.get("feed_summary") or ""))
    current["published_at"] = str(current.get("published_at") or "")
    current["source_country"] = str(current.get("source_country") or "").upper()
    current["source_language"] = str(current.get("source_language") or "").lower()
    current["source_name"] = str(current.get("source_name") or current["domain"] or "Unknown source")
    current["article_id"] = str(current.get("article_id") or stable_id(current["provider"], current["canonical_url"], current["title"]))
    current["provider_id"] = str(current.get("provider_id") or current["article_id"])
    current["source_key"] = str(current.get("source_key") or "|".join([current["provider"], source_id, current["source_name"], current["source_country"], current["source_language"]]))
    current["retrieval_query"] = request["query"]
    current["retrieval_query_type"] = request["query_type"]
    current["retrieval_source_id"] = request.get("source_id", "")
    current["retrieval_anchor_groups"] = request.get("anchor_groups", [])
    current["exact_duplicate_of"] = ""
    return current


def _select_sources_for_plan(plan: Dict[str, Any], sources: List[Dict[str, Any]], config: Any) -> tuple[List[Dict[str, Any]], List[str]]:
    tags = plan.get("target_tags", {}).get("normalized", {}) if isinstance(plan.get("target_tags"), dict) else {}
    matches = _source_matches(sources, tags)
    diagnostics: List[str] = []
    if not matches:
        fallback_tags, fallback_diag = _config_fallback_tags(config, sources)
        diagnostics.extend(fallback_diag)
        matches = _source_matches(sources, fallback_tags)
        if matches:
            diagnostics.append("controlled fallback used config coverage tags after planner tags matched no sources")
    selected = _budget_source_matches(matches, MAX_SELECTED_SOURCES)
    if len(matches) > len(selected):
        diagnostics.append(f"source budget kept {len(selected)} of {len(matches)} matching sources")
    return [match["source"] for match in selected], diagnostics


def _source_matches(sources: List[Dict[str, Any]], tags: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    countries = {str(item).upper() for item in tags.get("countries", [])}
    regions = {str(item).lower() for item in tags.get("regions", [])}
    matches: List[Dict[str, Any]] = []
    for source in sources:
        if not bool(source.get("enabled", True)):
            continue
        source_country = str(source.get("country") or "").upper()
        source_regions = {str(region or "").lower() for region in source.get("regions") or []}
        matched = {
            "country": source_country in countries,
            "region": bool(source_regions.intersection(regions)),
        }
        if any(matched.values()):
            matches.append({"source": source, "matched": matched})
    return matches


def _budget_source_matches(matches: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    def key(match: Dict[str, Any]) -> tuple[int, int, str, str]:
        matched = match["matched"]
        source = match["source"]
        return (
            -int(bool(matched.get("country"))),
            -int(bool(matched.get("region"))),
            str(source.get("country") or ""),
            str(source.get("source_id") or ""),
        )

    ordered = sorted(matches, key=key)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_countries: set[str] = set()
    for match in ordered:
        source = match["source"]
        source_id = str(source.get("source_id") or "")
        country = str(source.get("country") or "")
        if source_id in selected_ids or country in seen_countries:
            continue
        selected.append(match)
        selected_ids.add(source_id)
        seen_countries.add(country)
        if len(selected) >= limit:
            return selected
    for match in ordered:
        source_id = str(match["source"].get("source_id") or "")
        if source_id in selected_ids:
            continue
        selected.append(match)
        selected_ids.add(source_id)
        if len(selected) >= limit:
            return selected
    return selected


def _retrieval_requests(
    plan: Dict[str, Any],
    selected_sources: List[Dict[str, Any]],
    *,
    story: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    countries = _unique_text([str(source.get("country") or "").upper() for source in selected_sources])
    source_ids = _unique_text([str(source.get("source_id") or "") for source in selected_sources])
    story_title_tokens = _coverage_match_tokens(str((story or {}).get("story_title") or ""))
    story_summary_tokens = _coverage_match_tokens(str((story or {}).get("summary") or ""))
    requests: List[Dict[str, Any]] = []
    seen_canonical: set[str] = set()
    for query in _text_list(plan.get("queries"), max_items=MAX_CANONICAL_QUERIES, max_chars=180):
        if query in seen_canonical:
            continue
        query_tokens = _coverage_match_tokens(query)
        if (
            (story_title_tokens or story_summary_tokens)
            and not story_title_tokens.intersection(query_tokens)
            and len(story_summary_tokens.intersection(query_tokens)) < 3
        ):
            plan.setdefault("diagnostics", []).append(
                f"rejected canonical query with insufficient story overlap: {query}"
            )
            continue
        seen_canonical.add(query)
        requests.append(
            {
                "query": query,
                "query_type": "canonical",
                "anchor_groups": plan.get("anchor_groups", []),
                "source_id": "",
                "source_ids": source_ids,
                "source_countries": countries,
            }
        )
    for query in _anchor_queries(plan.get("anchor_groups")):
        if query in seen_canonical:
            continue
        seen_canonical.add(query)
        requests.append(
            {
                "query": query,
                "query_type": "anchor",
                "anchor_groups": plan.get("anchor_groups", []),
                "source_id": "",
                "source_ids": source_ids,
                "source_countries": countries,
            }
        )
    return requests


def _anchor_queries(anchor_groups: Any) -> List[str]:
    groups = _normalize_anchor_groups(anchor_groups)
    queries: List[str] = []
    for index, left_group in enumerate(groups):
        for right_group in groups[index + 1 :]:
            for left in left_group["terms"][:2]:
                for right in right_group["terms"][:2]:
                    query = normalize_whitespace(f"{left} {right}")
                    if query and query not in queries:
                        queries.append(query)
                    if len(queries) >= MAX_ANCHOR_QUERIES:
                        return queries
    if not queries and groups:
        terms = groups[0]["terms"]
        for index, left in enumerate(terms):
            for right in terms[index + 1 :]:
                query = normalize_whitespace(f"{left} {right}")
                if query and query not in queries:
                    queries.append(query)
                if len(queries) >= MAX_ANCHOR_QUERIES:
                    return queries
    return queries


def _rank_coverage_rows(rows: List[Dict[str, Any]], story: Dict[str, Any]) -> List[Dict[str, Any]]:
    def score(row: Dict[str, Any]) -> tuple[int, int, str]:
        available_text = str(row.get("feed_content") or row.get("context_text") or row.get("snippet") or "")
        return _coverage_relevance_score(row, story), len(available_text), str(row.get("published_at") or "")

    return sorted(rows, key=score, reverse=True)


def _relevant_coverage_rows(rows: List[Dict[str, Any]], story: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [row for row in rows if _coverage_relevance_passes(row, story)]


def _coverage_relevance_score(row: Dict[str, Any], story: Dict[str, Any]) -> int:
    # ponytail: weighted lexical anchors; add semantic reranking only after this misses real matches.
    story_tokens = _coverage_match_tokens(
        " ".join([str(story.get("story_title") or ""), str(story.get("summary") or ""), str(story.get("_seed_text") or "")])
    )
    query_tokens = _coverage_match_tokens(str(row.get("retrieval_query") or ""))
    title_tokens = _coverage_match_tokens(str(row.get("title") or ""))
    evidence_tokens = _coverage_match_tokens(" ".join([str(row.get("title") or ""), str(row.get("snippet") or ""), str(row.get("context_text") or "")]))
    query_title_hits = len(title_tokens.intersection(query_tokens))
    query_evidence_hits = len(evidence_tokens.intersection(query_tokens))
    seed_title_hits = len(title_tokens.intersection(story_tokens))
    seed_evidence_hits = len(evidence_tokens.intersection(story_tokens))
    anchor_hits = _anchor_group_hits(row.get("retrieval_anchor_groups"), evidence_tokens)
    return query_title_hits * 3 + query_evidence_hits + seed_title_hits * 2 + seed_evidence_hits + anchor_hits * 4


def _coverage_relevance_passes(row: Dict[str, Any], story: Dict[str, Any]) -> bool:
    return _coverage_relevance_decision(row, story)[0]


def _coverage_relevance_decision(row: Dict[str, Any], story: Dict[str, Any]) -> tuple[bool, str]:
    query_tokens = _coverage_match_tokens(str(row.get("retrieval_query") or ""))
    title_tokens = _coverage_match_tokens(str(row.get("title") or ""))
    evidence_tokens = _coverage_match_tokens(" ".join([str(row.get("title") or ""), str(row.get("snippet") or ""), str(row.get("context_text") or "")]))
    query_title_hits = len(title_tokens.intersection(query_tokens))
    query_evidence_hits = len(evidence_tokens.intersection(query_tokens))
    story_tokens = _coverage_match_tokens(
        " ".join([str(story.get("story_title") or ""), str(story.get("summary") or ""), str(story.get("_seed_text") or "")])
    )
    seed_evidence_hits = len(evidence_tokens.intersection(story_tokens))
    anchor_hits = _anchor_group_hits(row.get("retrieval_anchor_groups"), evidence_tokens)
    if anchor_hits >= 2:
        return True, "multiple_anchor_groups"
    if anchor_hits and query_evidence_hits >= 2:
        return True, "anchor_and_query_overlap"
    if query_title_hits >= 2:
        return True, "query_title_overlap"
    if query_evidence_hits >= 3:
        return True, "query_evidence_overlap"
    if seed_evidence_hits >= 2:
        return True, "seed_story_overlap"
    return False, "insufficient_event_overlap"


def _anchor_group_hits(value: Any, evidence_tokens: set[str]) -> int:
    hits = 0
    for group in value if isinstance(value, list) else []:
        if not isinstance(group, dict):
            continue
        terms = _string_list(group.get("terms"))
        if any(_coverage_match_tokens(term).intersection(evidence_tokens) for term in terms):
            hits += 1
    return hits


def _dedupe_coverage_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # ponytail: quadratic near-duplicate scan is bounded by retrieval limits; index signatures if that ceiling grows.
    selected: List[Dict[str, Any]] = []
    seen_urls: Dict[str, str] = {}
    for row in rows:
        current = dict(row)
        article_id = str(current.get("article_id") or stable_id(str(current.get("provider") or ""), str(current.get("canonical_url") or ""), str(current.get("title") or "")))
        current["article_id"] = article_id
        url_key = str(current.get("canonical_url") or current.get("url") or "").strip().lower()
        duplicate_of = seen_urls.get(url_key) if url_key else ""
        if not duplicate_of:
            duplicate_of = next(
                (
                    str(existing.get("article_id") or "")
                    for existing in selected
                    if _near_duplicate_coverage(current, existing)
                ),
                "",
            )
        if duplicate_of:
            current["exact_duplicate_of"] = duplicate_of
            continue
        if url_key:
            seen_urls[url_key] = article_id
        selected.append(current)
    return selected


def _near_duplicate_coverage(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_title = normalize_name(left.get("title"))
    right_title = normalize_name(right.get("title"))
    if left_title and right_title and SequenceMatcher(None, left_title, right_title).ratio() >= 0.92:
        return True
    left_text = normalize_name(left.get("feed_content") or left.get("snippet"))
    right_text = normalize_name(right.get("feed_content") or right.get("snippet"))
    return min(len(left_text), len(right_text)) >= 160 and SequenceMatcher(None, left_text, right_text).ratio() >= 0.94


def _diverse_article_selection(rows: List[Dict[str, Any]], *, max_articles: int) -> List[Dict[str, Any]]:
    if max_articles <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()

    def add(row: Dict[str, Any]) -> bool:
        article_id = str(row.get("article_id") or "")
        source = str(row.get("source_id") or row.get("domain") or row.get("source_name") or row.get("source_key") or "")
        if article_id in selected_ids or source_counts[source] >= MAX_ARTICLES_PER_SOURCE:
            return False
        selected.append(row)
        selected_ids.add(article_id)
        source_counts[source] += 1
        country_counts[_known_or_unknown(row.get("source_country"))] += 1
        return len(selected) >= max_articles

    for row in rows:
        country = _known_or_unknown(row.get("source_country"))
        if country_counts[country] == 0 and add(row):
            return selected
    for row in rows:
        if add(row):
            return selected
    return selected


def _fetch_selected_article_contexts(
    articles: List[Dict[str, Any]],
    *,
    story: Dict[str, Any],
    article_retriever: ArticleRetriever,
    article_text_cache: Any,
    max_workers: int,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    if not articles:
        return []

    def fetch(article: Dict[str, Any]) -> tuple[str, str, str, str]:
        candidate = _coverage_candidate(article)
        aliases = article_aliases_for_candidate(candidate)
        cached = article_text_cache.get_by_aliases(aliases) if article_text_cache else None
        if cached:
            return (
                str(cached.get("article_text") or ""),
                str(cached.get("extraction_status") or "ok"),
                str(cached.get("resolved_url") or cached.get("url") or candidate.url),
                "article_cache",
            )
        if not candidate.url:
            return "", "missing_url", "", "article_page"
        text, status, resolved_url = article_retriever.fetch_text_with_url(candidate.url)
        if article_text_cache:
            article_text_cache.store(
                candidate=candidate,
                aliases=aliases,
                article_text=text,
                extraction_status=status,
                resolved_url=resolved_url or candidate.url,
            )
        return text, status, resolved_url or candidate.url, "article_page"

    def failed(_index: int, article: Dict[str, Any], exc: Exception) -> tuple[str, str, str, str]:
        warnings.append(
            f"perspectives_report: article fetch failed for {article.get('article_id') or article.get('url')!r} "
            f"({type(exc).__name__}: {exc})."
        )
        return "", "worker_exception", str(article.get("url") or ""), "article_page"

    results = ordered_parallel_map(
        articles,
        min(len(articles), max(1, int(max_workers or 1))),
        fetch,
        on_exception=failed,
    )
    output: List[Dict[str, Any]] = []
    for article, (text, status, resolved_url, fetch_source) in zip(articles, results):
        current = dict(article)
        current["body"] = text
        current["fetch_status"] = status
        current["fetched_url"] = resolved_url
        current["fetch_source"] = fetch_source
        output.append(_with_article_context(current, story=story))
    return output


def _coverage_candidate(article: Dict[str, Any]) -> NewsCandidate:
    return NewsCandidate(
        id=str(article.get("article_id") or ""),
        source=str(article.get("source_name") or article.get("domain") or ""),
        category="perspectives",
        title=str(article.get("title") or ""),
        url=str(article.get("canonical_url") or article.get("url") or ""),
        snippet=str(article.get("snippet") or ""),
        published_at=None,
        metadata={"canonical_url": str(article.get("canonical_url") or "")},
    )


def _with_article_context(article: Dict[str, Any], *, story: Dict[str, Any] | None = None) -> Dict[str, Any]:
    current = dict(article)
    candidates = (
        ("fetched_article", current.get("body"), current.get("fetch_source") or "article_page"),
        ("feed_content", current.get("feed_content"), "feed_content"),
        ("feed_summary", current.get("feed_summary"), "feed_summary"),
        ("provider_snippet", current.get("context_text") or current.get("snippet"), "provider_snippet"),
    )
    for status, raw, source in candidates:
        excerpt = _relevant_context_excerpt(str(raw or ""), story or {}, current)
        if not excerpt:
            continue
        current["context_text"] = excerpt
        current["context_status"] = status
        current["context_source"] = str(source)
        current["context_length_chars"] = len(excerpt)
        return current
    current["context_text"] = ""
    current["context_status"] = "unavailable"
    current["context_source"] = "title_only"
    current["context_length_chars"] = 0
    return current


def _relevant_context_excerpt(text: str, story: Dict[str, Any], article: Dict[str, Any]) -> str:
    paragraphs = _context_paragraphs(text)
    if not paragraphs:
        return ""
    target_tokens = _coverage_match_tokens(
        " ".join(
            [
                str(story.get("story_title") or ""),
                str(story.get("summary") or ""),
                str(story.get("_seed_text") or ""),
                str(article.get("title") or ""),
                str(article.get("retrieval_query") or ""),
            ]
        )
    )
    ranked = sorted(
        enumerate(paragraphs),
        key=lambda item: (len(_coverage_match_tokens(item[1]).intersection(target_tokens)), len(item[1])),
        reverse=True,
    )[:MAX_CONTEXT_PARAGRAPHS]
    chosen = [paragraphs[index] for index in sorted(index for index, _paragraph in ranked)]
    excerpt = ""
    for paragraph in chosen:
        remaining = ARTICLE_CONTEXT_MAX_CHARS - len(excerpt) - (2 if excerpt else 0)
        if remaining <= 0:
            break
        excerpt += ("\n\n" if excerpt else "") + paragraph[:remaining]
    return excerpt


def _context_paragraphs(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    paragraphs = [normalize_whitespace(item) for item in re.split(r"(?:\r?\n){2,}", cleaned)]
    paragraphs = [item for item in paragraphs if item]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = [normalize_whitespace(item) for item in re.split(r"(?<=[.!?])\s+", paragraphs[0]) if normalize_whitespace(item)]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > 650:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
        if len(current) >= MEANINGFUL_PARAGRAPH_MIN_CHARS:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks or paragraphs


def _framing_prompt(story_payloads: List[Dict[str, Any]]) -> str:
    return PERSPECTIVES_FRAMING_USER.format(data=compact_json({"stories": story_payloads}))


def _claim_synthesis_prompt(ai_client: Any, story: Dict[str, Any], input_token_limit: int) -> str:
    instructions = """Analyze only the supplied focus claims. Relationships: originates, independently_supports, reports_or_quotes, qualifies, disputes, context_only. For each claim, return coverage rows only for articles that address it or provide material context; omit unrelated articles. Use each article at most once per claim. Syndication and repeated attribution are not independent support. Return the supplied story_id, a framing_report, and one claim perspective for every supplied focus claim. Claimant, origin, and importance metadata are supplied upstream; do not recreate them. A concise card is optional and should appear only when context would materially help. Do not return URLs. Never put article IDs in prose fields; use only the dedicated article-id fields. Preserve framing analysis beyond verdicts: prominence, actors, agency, terminology, local stakes, hedging, and established omissions.

Output shape:
{"story_id":"supplied id","framing_report":{"synthesis":"prose","synthesis_article_ids":["article id"],"shared_facts":[{"text":"fact","article_ids":["article id"]}],"repetition_without_independent_support":[],"verified_or_independently_supported_claims":[],"qualified_disputed_or_unresolved_claims":[],"country_source_comparison":[],"coverage_limitations":[]},"claim_perspectives":[{"claim_id":"supplied claim id","coverage":[{"article_id":"article id","relationship":"reports_or_quotes","evidence_basis":"basis","explanation":"explanation"}],"synthesis":"claim-level prose","coverage_limitations":[],"card":{"title":"claim title","reporting_summary":"independent contribution versus repetition","evidence_check":"scoped verification result","qualification":"material qualification","limitations":"unresolved point"}}]}"""
    payload = dict(story)
    limit = int(input_token_limit)
    for context_chars in (ARTICLE_CONTEXT_MAX_CHARS, 1200, 700, 360, 180):
        payload["articles"] = [
            {**article, "context_text": _short_text(article.get("context_text"), context_chars)}
            for article in _as_list(story.get("articles"))
            if isinstance(article, dict)
        ]
        prompt = f"{instructions}\n\nSupplied data:\n{compact_json(payload)}"
        estimate = getattr(ai_client, "estimate_tokens", None)
        if not callable(estimate) or int(estimate(prompt)) <= int(limit * 0.95):
            return prompt
    return prompt


def _normalize_claim_synthesis_response(
    raw: Dict[str, Any],
    *,
    story: Dict[str, Any],
    known_article_ids: List[str],
    verification_by_claim: Dict[str, Dict[str, Any]],
    verification_documents: Dict[str, List[Dict[str, Any]]],
    warnings: List[str],
) -> Dict[str, Any]:
    story_id = str(story.get("story_id") or "")
    if str(raw.get("story_id") or "") != story_id:
        warnings.append(f"perspectives_report: claim synthesis returned an unknown story_id for {story_id!r}; rejected.")
        report = _empty_framing_report("Claim-led synthesis returned an invalid story id.")
        report["claim_perspectives"] = []
        return report
    valid_articles = set(known_article_ids)
    article_metadata = {str(item.get("article_id") or ""): item for item in _as_list(story.get("articles")) if isinstance(item, dict)}
    known_claims = {str(item.get("claim_id") or ""): item for item in _as_list(story.get("claims")) if isinstance(item, dict)}
    raw_framing = raw.get("framing_report") if isinstance(raw.get("framing_report"), dict) else {}
    framing = {
        "synthesis": normalize_whitespace(str(raw_framing.get("synthesis") or "")),
        "synthesis_article_ids": _validated_article_ids(raw_framing.get("synthesis_article_ids"), valid_articles, story_id, warnings),
        "shared_facts": _evidence_items(raw_framing.get("shared_facts"), known_article_ids=valid_articles, story_id=story_id, warnings=warnings),
        "repetition_without_independent_support": _evidence_items(raw_framing.get("repetition_without_independent_support"), known_article_ids=valid_articles, story_id=story_id, warnings=warnings),
        "verified_or_independently_supported_claims": _evidence_items(raw_framing.get("verified_or_independently_supported_claims"), known_article_ids=valid_articles, story_id=story_id, warnings=warnings),
        "qualified_disputed_or_unresolved_claims": _evidence_items(raw_framing.get("qualified_disputed_or_unresolved_claims"), known_article_ids=valid_articles, story_id=story_id, warnings=warnings),
        "country_source_comparison": _evidence_items(raw_framing.get("country_source_comparison"), known_article_ids=valid_articles, story_id=story_id, warnings=warnings),
        "coverage_limitations": _text_items(raw_framing.get("coverage_limitations")),
    }
    claim_perspectives = []
    seen_claims: set[str] = set()
    for item in _as_list(raw.get("claim_perspectives")):
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id") or "")
        if claim_id not in known_claims or claim_id in seen_claims:
            if claim_id:
                warnings.append(f"perspectives_report: synthesis used unknown or duplicate claim_id {claim_id!r}; discarded.")
            continue
        coverage = []
        covered_ids: set[str] = set()
        for relation in _as_list(item.get("coverage")):
            if not isinstance(relation, dict):
                warnings.append(f"perspectives_report: claim {claim_id!r} used a non-object relationship row; ignored.")
                continue
            article_id = str(relation.get("article_id") or "")
            relationship = str(relation.get("relationship") or "")
            if article_id not in valid_articles or relationship not in RELATIONSHIPS or article_id in covered_ids:
                warnings.append(f"perspectives_report: claim {claim_id!r} used an unknown, duplicate, or invalid article relationship; ignored.")
                continue
            covered_ids.add(article_id)
            coverage.append(
                {
                    "article_id": article_id,
                    "relationship": relationship,
                    "evidence_basis": normalize_whitespace(str(relation.get("evidence_basis") or "")),
                    "explanation": normalize_whitespace(str(relation.get("explanation") or "")),
                }
            )
        not_covered = [article_id for article_id in known_article_ids if article_id and article_id not in covered_ids]
        claim = known_claims[claim_id]
        claimant = normalize_whitespace(str(claim.get("claimant") or ""))
        claimants = (
            [
                {
                    "name": claimant,
                    "role": "originator",
                    "evidence_basis": "upstream evidence packet",
                    "article_ids": _string_list(claim.get("origin_article_ids")),
                }
            ]
            if claimant
            else []
        )
        card = _trusted_claim_card(
            item.get("card"),
            claim_id=claim_id,
            claim=claim,
            coverage=coverage,
            article_metadata=article_metadata,
            verification=verification_by_claim.get(claim_id),
            verification_documents=verification_documents.get(claim_id, []),
        )
        seen_claims.add(claim_id)
        claim_perspectives.append(
            {
                "claim_id": claim_id,
                "claim": str(claim.get("claim") or ""),
                "claim_type": str(claim.get("claim_type") or "other"),
                "importance_reason": normalize_whitespace(str(claim.get("importance_reason") or "")),
                "claimants": claimants,
                "coverage": coverage,
                "not_covered_article_ids": not_covered,
                "verification": verification_by_claim.get(claim_id, {"selected": False, "status": "not_selected"}),
                "synthesis": normalize_whitespace(str(item.get("synthesis") or "")),
                "coverage_limitations": _text_items(item.get("coverage_limitations")),
                "card": card,
            }
        )
    omitted_claims = set(known_claims).difference(seen_claims)
    if omitted_claims:
        warnings.append(f"perspectives_report: synthesis omitted {len(omitted_claims)} focus claim(s) for {story_id!r}.")
    framing["claim_perspectives"] = claim_perspectives
    return framing


def _trusted_claim_card(
    raw_card: Any,
    *,
    claim_id: str,
    claim: Dict[str, Any],
    coverage: List[Dict[str, Any]],
    article_metadata: Dict[str, Dict[str, Any]],
    verification: Dict[str, Any] | None,
    verification_documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(raw_card, dict) or not any(normalize_whitespace(str(raw_card.get(key) or "")) for key in ("who_says", "reporting_summary", "evidence_check", "qualification", "limitations")):
        return {}
    source_ids = _unique_text(
        [
            str(item.get("article_id") or "")
            for item in coverage
            if str(item.get("relationship") or "") != "context_only"
        ]
    )
    sources = []
    for article_id in source_ids:
        article = article_metadata.get(article_id, {})
        url = str(article.get("url") or "")
        sources.append(
            {
                "article_id": article_id,
                "outlet": str(article.get("source_name") or ""),
                "headline": str(article.get("title") or ""),
                "url": url if urlparse(url).scheme in {"http", "https"} else "",
            }
        )
    document_by_id = {str(item.get("document_id") or ""): item for item in verification_documents}
    evidence_ids = _unique_text(
        [
            str(item.get("document_id") or "")
            for key in ("supporting_evidence", "contradicting_evidence")
            for item in _as_list((verification or {}).get(key))
            if isinstance(item, dict)
        ]
    )
    for document_id in evidence_ids:
        document = document_by_id.get(document_id, {})
        url = str(document.get("url") or "")
        sources.append(
            {
                "article_id": document_id,
                "outlet": str(document.get("source") or ""),
                "headline": str(document.get("title") or ""),
                "url": url if urlparse(url).scheme in {"http", "https"} else "",
            }
        )
    card = {
        "claim_id": claim_id,
        "claim": str(claim.get("claim") or ""),
        "title": normalize_whitespace(str(raw_card.get("title") or claim.get("claim") or "")),
        "who_says": normalize_whitespace(str(claim.get("claimant") or raw_card.get("who_says") or "")),
        "reporting_summary": normalize_whitespace(str(raw_card.get("reporting_summary") or "")),
        "evidence_check": normalize_whitespace(str(raw_card.get("evidence_check") or "")),
        "verification_verdict": str((verification or {}).get("verdict") or "not_checked"),
        "verdict_scope": str((verification or {}).get("verdict_scope") or "not_checked"),
        "qualification": normalize_whitespace(str(raw_card.get("qualification") or "")),
        "limitations": normalize_whitespace(str(raw_card.get("limitations") or "")),
        "sources": sources,
    }
    score = _claim_card_score(claim, coverage, verification, card)
    card["editorial_score"] = score
    return card


def _claim_card_score(
    claim: Dict[str, Any],
    coverage: List[Dict[str, Any]],
    verification: Dict[str, Any] | None,
    card: Dict[str, Any],
) -> int:
    relationships = [str(item.get("relationship") or "") for item in coverage]
    status = str((verification or {}).get("status") or "")
    verdict = str((verification or {}).get("verdict") or "")
    score = 0
    if "disputes" in relationships:
        score += 8
    if "qualifies" in relationships:
        score += 6
    if "independently_supports" in relationships:
        score += 4
    if "originates" in relationships:
        score += 2
    if status == "checked":
        score += {
            "contradicted": 7,
            "mixed": 7,
            "unresolved": 5,
            "mostly_supported": 4,
            "supported": 1,
        }.get(verdict, 0)
    if str(claim.get("claim_type") or "") in {"causal", "forecast", "attribution"}:
        score += 2
    elif str(claim.get("claim_type") or "") == "numerical":
        score += 1
    if card.get("qualification"):
        score += 1
    if card.get("limitations"):
        score += 1
    if relationships.count("reports_or_quotes") >= 2 and "independently_supports" not in relationships:
        score += 1
    if relationships and set(relationships) <= {"reports_or_quotes", "context_only"} and status != "checked":
        score -= 3
    if relationships and set(relationships) == {"context_only"}:
        score -= 4
    return score


def _visible_claim_cards(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        [claim for claim in claims if isinstance(claim.get("card"), dict) and claim.get("card")],
        key=lambda claim: int(claim["card"].get("editorial_score", 0) or 0),
        reverse=True,
    )
    visible: List[Dict[str, Any]] = []
    normalized_claims: List[str] = []
    for claim in ranked:
        card = claim["card"]
        if int(card.get("editorial_score", 0) or 0) < 3:
            continue
        normalized = normalize_name(card.get("claim") or card.get("title"))
        # ponytail: quadratic comparison is bounded by the small per-story claim set.
        if any(SequenceMatcher(None, normalized, previous).ratio() >= 0.82 for previous in normalized_claims):
            continue
        visible.append(card)
        normalized_claims.append(normalized)
    return visible


def _normalize_framing_response(
    raw: Dict[str, Any],
    *,
    known_story_ids: List[str],
    known_article_ids: Dict[str, List[str]],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    known = set(known_story_ids)
    output = {story_id: _empty_framing_report("Model did not return a report for this story.") for story_id in known_story_ids}
    for item in _as_list(raw.get("stories") if isinstance(raw, dict) else []):
        if not isinstance(item, dict):
            continue
        story_id = str(item.get("story_id") or "").strip()
        if story_id not in known:
            if story_id:
                warnings.append(f"perspectives_report: framing response used unknown story_id {story_id!r}; discarded.")
            continue
        valid_article_ids = {article_id for article_id in known_article_ids.get(story_id, []) if article_id}
        synthesis = _strip_known_article_ids(
            normalize_whitespace(str(item.get("synthesis") or "")),
            valid_article_ids,
        )
        shared_facts = _evidence_items(
            item.get("shared_facts"), known_article_ids=valid_article_ids, story_id=story_id, warnings=warnings
        )
        comparisons = _evidence_items(
            item.get("country_source_comparison"), known_article_ids=valid_article_ids, story_id=story_id, warnings=warnings
        )
        for evidence_item in [*shared_facts, *comparisons]:
            evidence_item["text"] = _strip_known_article_ids(
                str(evidence_item.get("text") or ""), valid_article_ids
            )
        output[story_id] = {
            "synthesis": synthesis,
            "synthesis_article_ids": _validated_article_ids(item.get("synthesis_article_ids"), valid_article_ids, story_id, warnings),
            "shared_facts": shared_facts,
            "country_source_comparison": comparisons,
            "coverage_limitations": [
                _strip_known_article_ids(text, valid_article_ids)
                for text in _text_items(item.get("coverage_limitations"))
            ],
        }
    return output


def _strip_known_article_ids(text: str, article_ids: set[str]) -> str:
    cleaned = str(text or "")
    for article_id in sorted(article_ids, key=len, reverse=True):
        cleaned = cleaned.replace(f"article_id:{article_id}", "").replace(article_id, "")
    cleaned = re.sub(r"\[\s*(?:,\s*)*\]", "", cleaned)
    return normalize_whitespace(re.sub(r"\s+([,.;:!?])", r"\1", cleaned))


def _story_report(
    *,
    story: Dict[str, Any],
    articles: List[Dict[str, Any]],
    plan: Dict[str, Any] | None,
    coverage: Dict[str, Any] | None,
    framing: Dict[str, Any] | None,
    verification_documents: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    coverage = coverage or _coverage_gap(_story_identity(story), plan or _empty_plan(story, "planner_failed", []), "coverage was not run")
    plan = plan or _empty_plan(story, "planner_failed", [])
    seed_sources = _seed_sources(articles)
    framing = framing or _empty_framing_report("Framing report unavailable.")
    claim_perspectives = _as_list(framing.get("claim_perspectives"))
    public_framing = {key: value for key, value in framing.items() if key != "claim_perspectives"}
    story_claim_ids = {str(claim.get("claim_id") or "") for claim in _as_list(story.get("claims")) if isinstance(claim, dict)}
    return {
        "story_id": _story_identity(story),
        "story_title": str(story.get("story_title") or story.get("title") or "Story"),
        "summary": str(story.get("summary") or ""),
        "entities": _string_list(story.get("entities")),
        "seed_article_ids": [article["id"] for article in articles],
        "seed_sources": seed_sources,
        "story_loci": plan.get("story_loci", []),
        "planner": _public_plan(plan),
        "coverage_status": coverage.get("coverage_status", "coverage_gap"),
        "coverage_gap": coverage.get("coverage_gap", ""),
        "selected_sources": coverage.get("selected_sources", []),
        "retrieval_request_count": coverage.get("retrieval_request_count", 0),
        "retrieval_queries": coverage.get("retrieval_queries", []),
        "provider_statuses": coverage.get("provider_statuses", {}),
        "funnel_counts": coverage.get("funnel_counts", {}),
        "relevance_diagnostics": coverage.get("relevance_diagnostics", {}),
        "source_yields": coverage.get("source_yields", []),
        "effective_limits": coverage.get("effective_limits", {}),
        "coverage_provider": coverage.get("coverage_provider", ""),
        "coverage_provider_counts": coverage.get("coverage_provider_counts", {}),
        "coverage_articles": coverage.get("coverage_articles", []),
        "coverage_counts": coverage.get("coverage_counts", _coverage_counts([])),
        "coverage_quality": coverage.get("coverage_quality", _coverage_quality([], None)),
        "context_stats": coverage.get("context_stats", _context_stats([])),
        "verification_documents": [
            document
            for claim_id in story_claim_ids
            for document in (verification_documents or {}).get(claim_id, [])
        ],
        "claim_perspectives": claim_perspectives,
        "claim_context_cards": _visible_claim_cards(claim_perspectives),
        "framing_report": public_framing,
    }


def _public_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(plan.get("status") or ""),
        "queries": _string_list(plan.get("queries")),
        "anchor_groups": plan.get("anchor_groups", []),
        "story_loci": plan.get("story_loci", []),
        "target_tags": plan.get("target_tags", {}),
        "selected_sources": plan.get("selected_sources", []),
        "diagnostics": _unique_text(_string_list(plan.get("diagnostics"))),
        "verification_targets": _as_list(plan.get("verification_targets")),
    }


def _coverage_gap(story_id: str, plan: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "coverage_status": "coverage_gap",
        "coverage_gap": reason,
        "selected_sources": [],
        "retrieval_request_count": 0,
        "retrieval_queries": [],
        "provider_statuses": {},
        "funnel_counts": _coverage_funnel_counts([], [], [], []),
        "source_yields": [],
        "effective_limits": {},
        "coverage_provider": "",
        "coverage_provider_counts": {},
        "coverage_articles": [],
        "coverage_counts": _coverage_counts([]),
        "coverage_quality": {
            "status": "gap",
            "source_country_count": 0,
            "language_count": 0,
            "source_count": 0,
            "thin_reasons": [reason],
        },
        "context_stats": _context_stats([]),
        "plan_status": plan.get("status", ""),
        "story_id": story_id,
    }


def _effective_article_limit(config: Any) -> int:
    requested = max(0, int(getattr(config, "coverage_max_records_per_story", MAX_ARTICLES_FOR_REPORT) or 0))
    return min(requested, MAX_SELECTED_SOURCES * MAX_ARTICLES_PER_SOURCE)


def _coverage_funnel_counts(
    raw: List[Dict[str, Any]],
    deduplicated: List[Dict[str, Any]],
    relevant: List[Dict[str, Any]],
    final: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stages = {
        "raw": raw,
        "deduplicated": deduplicated,
        "relevant": relevant,
        "final": final,
    }
    overall = {name: len(rows) for name, rows in stages.items()}
    overall.update(
        {
            "duplicates_removed": overall["raw"] - overall["deduplicated"],
            "relevance_rejects": overall["deduplicated"] - overall["relevant"],
            "per_source_rejects": overall["relevant"] - overall["final"],
        }
    )
    providers = sorted({str(row.get("provider") or "unknown") for rows in stages.values() for row in rows})
    return {
        "overall": overall,
        "by_provider": {
            provider: {
                name: sum(1 for row in rows if str(row.get("provider") or "unknown") == provider)
                for name, rows in stages.items()
            }
            for provider in providers
        },
    }


def _coverage_source_yields(
    selected_sources: List[Dict[str, Any]],
    raw: List[Dict[str, Any]],
    final: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_counts = Counter(str(row.get("source_id") or "") for row in raw)
    final_counts = Counter(str(row.get("source_id") or "") for row in final)
    return [
        {
            "source_id": str(source.get("source_id") or ""),
            "source_name": str(source.get("name") or ""),
            "raw": int(raw_counts.get(str(source.get("source_id") or ""), 0)),
            "final": int(final_counts.get(str(source.get("source_id") or ""), 0)),
        }
        for source in selected_sources
    ]


def _coverage_quality(articles: List[Dict[str, Any]], config: Any) -> Dict[str, Any]:
    countries = {_known_or_unknown(article.get("source_country")) for article in articles if _known_or_unknown(article.get("source_country")) != "unknown"}
    languages = {_known_or_unknown(article.get("source_language")) for article in articles if _known_or_unknown(article.get("source_language")) != "unknown"}
    sources = {_known_or_unknown(article.get("source_id") or article.get("source_key") or article.get("source_name")) for article in articles}
    min_countries = int(getattr(config, "minimum_source_countries", 4) or 0) if config is not None else 4
    reasons: List[str] = []
    if not articles:
        reasons.append("no coverage found")
    if len(countries) < min_countries:
        reasons.append(f"fewer than {min_countries} source countries")
    if articles and not any(str(article.get("context_text") or "").strip() for article in articles):
        reasons.append("metadata-only coverage")
    return {
        "status": "ok" if not reasons else "thin",
        "source_country_count": len(countries),
        "language_count": len(languages),
        "source_count": len(sources),
        "thin_reasons": reasons,
    }


def _coverage_counts(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    countries = Counter(_known_or_unknown(article.get("source_country")) for article in articles)
    languages = Counter(_known_or_unknown(article.get("source_language")) for article in articles)
    domains = Counter(_known_or_unknown(article.get("domain")) for article in articles)
    statuses = Counter(_known_or_unknown(article.get("context_status")) for article in articles)
    return {
        "articles": len(articles),
        "source_countries": _sorted_counter(countries),
        "languages": _sorted_counter(languages),
        "domains": _sorted_counter(domains),
        "context_statuses": _sorted_counter(statuses),
    }


def _context_stats(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = _coverage_counts(articles)["context_statuses"]
    lengths = [int(article.get("context_length_chars") or len(str(article.get("context_text") or ""))) for article in articles]
    return {
        "available_articles": len(articles),
        "fetched_article": int(statuses.get("fetched_article", 0) or 0),
        "feed_content": int(statuses.get("feed_content", 0) or 0),
        "feed_summary": int(statuses.get("feed_summary", 0) or 0),
        "provider_snippet": int(statuses.get("provider_snippet", 0) or 0),
        "unavailable": int(statuses.get("unavailable", 0) or 0),
        "total_context_chars": sum(lengths),
        "minimum_context_chars": min(lengths, default=0),
        "maximum_context_chars": max(lengths, default=0),
    }


def _provider_counts(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    return _sorted_counter(Counter(str(article.get("provider") or "unknown") for article in articles))


def _coverage_provider_label(articles: List[Dict[str, Any]]) -> str:
    return "+".join(sorted({str(article.get("provider") or "") for article in articles if str(article.get("provider") or "")}))


def _framing_articles_payload(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for article in articles:
        payload.append(
            {
                "article_id": str(article.get("article_id") or ""),
                "provider": str(article.get("provider") or ""),
                "title": _short_text(article.get("title"), 220),
                "source_id": str(article.get("source_id") or ""),
                "source_name": _short_text(article.get("source_name"), 120),
                "source_country": str(article.get("source_country") or ""),
                "source_language": str(article.get("source_language") or ""),
                "published_at": str(article.get("published_at") or ""),
                "url": str(article.get("canonical_url") or article.get("url") or ""),
                "context_status": str(article.get("context_status") or ""),
                "context_source": str(article.get("context_source") or ""),
                "context_length_chars": int(article.get("context_length_chars") or 0),
                "context_text": _short_text(article.get("context_text"), ARTICLE_CONTEXT_MAX_CHARS),
            }
        )
    return payload


def _empty_framing_report(reason: str = "") -> Dict[str, Any]:
    return {
        "synthesis": "",
        "synthesis_article_ids": [],
        "shared_facts": [],
        "repetition_without_independent_support": [],
        "verified_or_independently_supported_claims": [],
        "qualified_disputed_or_unresolved_claims": [],
        "country_source_comparison": [],
        "coverage_limitations": [reason] if reason else [],
    }


def _normalize_target_tags(raw_tags: Dict[str, Any], sources: List[Dict[str, Any]]) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    countries = {str(source.get("country") or "").upper() for source in sources if source.get("country")}
    languages = {str(source.get("language") or "").lower() for source in sources if source.get("language")}
    regions = {str(region or "").lower() for source in sources for region in source.get("regions") or [] if str(region or "").strip()}
    normalized_countries, rejected_countries = _normalize_tag_values(_string_list(raw_tags.get("countries")), countries, COUNTRY_ALIASES, upper=True)
    normalized_languages, rejected_languages = _normalize_tag_values(_string_list(raw_tags.get("languages")), languages, LANGUAGE_ALIASES)
    normalized_regions, rejected_regions = _normalize_tag_values(_string_list(raw_tags.get("regions")), regions, REGION_ALIASES)
    return (
        {
            "countries": normalized_countries,
            "languages": normalized_languages,
            "regions": normalized_regions,
        },
        {
            "countries": rejected_countries,
            "languages": rejected_languages,
            "regions": rejected_regions,
        },
    )


def _config_fallback_tags(config: Any, sources: List[Dict[str, Any]]) -> tuple[Dict[str, List[str]], List[str]]:
    raw = {
        "countries": _string_list(getattr(config, "coverage_scope", [])),
        "languages": [],
        "regions": _string_list(getattr(config, "coverage_regions", [])),
    }
    if not any(raw.values()):
        return {"countries": [], "languages": [], "regions": []}, []
    normalized, rejected = _normalize_target_tags(raw, sources)
    diagnostics = ["config coverage tags had unmatched values: " + compact_json(rejected)] if any(rejected.values()) else []
    return normalized, diagnostics


def _normalize_tag_values(
    values: List[str],
    known_values: set[str],
    aliases: Dict[str, str],
    *,
    upper: bool = False,
) -> tuple[List[str], List[str]]:
    known_map = {_tag_key(value): (value.upper() if upper else value.lower()) for value in known_values if value}
    alias_map = {key: (value.upper() if upper else value.lower()) for key, value in aliases.items()}
    output: List[str] = []
    rejected: List[str] = []
    for value in values:
        key = _tag_key(value)
        match = alias_map.get(key) or known_map.get(key) or _fuzzy_tag_match(key, {**known_map, **alias_map})
        if match:
            if match not in output:
                output.append(match)
        else:
            rejected.append(value)
    return output, rejected


def _fuzzy_tag_match(key: str, candidates: Dict[str, str]) -> str:
    if not key:
        return ""
    scored = sorted(
        ((SequenceMatcher(None, key, candidate).ratio(), candidate) for candidate in candidates),
        reverse=True,
    )
    if not scored or scored[0][0] < 0.9:
        return ""
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.04:
        return ""
    return candidates[scored[0][1]]


def _tag_key(value: Any) -> str:
    return TAG_RE.sub("_", str(value or "").strip().lower()).strip("_")


def _source_metadata(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": str(source.get("source_id") or ""),
        "name": str(source.get("name") or ""),
        "country": str(source.get("country") or "").upper(),
        "language": str(source.get("language") or "").lower(),
        "regions": _string_list(source.get("regions")),
        "category": str(source.get("category") or ""),
        "tags": _string_list(source.get("tags")),
    }


def _planner_tag_options(sources: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    enabled = [source for source in sources if bool(source.get("enabled", True))]
    return {
        "countries": sorted({str(source.get("country") or "").upper() for source in enabled if source.get("country")}),
        "regions": sorted({str(region or "").lower() for source in enabled for region in source.get("regions") or [] if str(region or "").strip()}),
    }


def _seed_anchor_groups(story: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = _text_list(story.get("entities"), max_items=6, max_chars=100)
    return [{"kind": "entity", "terms": entities}] if entities else []


def _story_planner_payload(story: Dict[str, Any], articles_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    seed_articles = [articles_by_id[article_id] for article_id in _story_article_ids(story) if article_id in articles_by_id]
    return {
        "story_id": _story_identity(story),
        "title": str(story.get("story_title") or story.get("title") or ""),
        "summary": _short_text(story.get("summary") or _article_summary(seed_articles), 800),
        "entities": _string_list(story.get("entities")),
        "seed_headlines": [_short_text(article.get("headline"), 180) for article in seed_articles[:8]],
        "seed_sources": _unique_text([str(article.get("source") or "") for article in seed_articles if article.get("source")])[:8],
        "claims": [
            {
                "claim_id": str(claim.get("claim_id") or ""),
                "claim": _short_text(claim.get("claim"), 320),
                "claimant": _short_text(claim.get("claimant"), 120),
                "claim_type": str(claim.get("claim_type") or "other"),
                "confidence": str(claim.get("confidence") or ""),
            }
            for claim in _as_list(story.get("claims"))
            if isinstance(claim, dict)
        ],
        "confirmed_facts": _as_list(story.get("confirmed_facts")),
        "conflicting_claims": _as_list(story.get("conflicting_claims")),
        "open_questions": _as_list(story.get("open_questions")),
    }


def _seed_sources(articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    grouped: Dict[str, Dict[str, str]] = {}
    for article in articles:
        source = str(article.get("source") or "Unknown source")
        key = normalize_name(source) or "unknown"
        grouped.setdefault(key, {"source_key": key, "source": source, "article_count": 0})
        grouped[key]["article_count"] = str(int(grouped[key]["article_count"]) + 1)
    return list(grouped.values())


def _retrieval_query_summary(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "query": request["query"],
            "query_type": request["query_type"],
            "source_id": request.get("source_id", ""),
            "countries": request.get("source_countries", []),
        }
        for request in requests[:80]
    ]


def _append_evidence_section(lines: List[str], title: str, items: Any) -> None:
    evidence = _evidence_items(items)
    if not evidence:
        return
    lines.append(f"### {title}")
    for item in evidence:
        lines.append(f"- {item['text']}")
    lines.append("")


def _evidence_items(
    value: Any,
    *,
    known_article_ids: set[str] | None = None,
    story_id: str = "",
    warnings: List[str] | None = None,
) -> List[Dict[str, Any]]:
    items = value if isinstance(value, list) else [value] if value else []
    output: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            text = normalize_whitespace(str(item.get("text") or item.get("summary") or item.get("point") or item.get("claim") or ""))
            article_ids = _validated_article_ids(
                item.get("article_ids") or item.get("evidence_article_ids"),
                known_article_ids,
                story_id,
                warnings,
            )
        else:
            text = normalize_whitespace(str(item or ""))
            article_ids = []
        if text:
            output.append({"text": text, "article_ids": article_ids})
    return output


def _validated_article_ids(
    value: Any,
    known_article_ids: set[str] | None,
    story_id: str,
    warnings: List[str] | None,
) -> List[str]:
    article_ids = _string_list(value)
    if known_article_ids is None:
        return article_ids
    unknown = [article_id for article_id in article_ids if article_id not in known_article_ids]
    if unknown and warnings is not None:
        warnings.append(f"perspectives_report: framing response used unknown article_ids for {story_id!r}: {', '.join(unknown)}")
    return [article_id for article_id in article_ids if article_id in known_article_ids]


def _text_items(value: Any) -> List[str]:
    items = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    output: List[str] = []
    for item in items:
        raw = item.get("text") if isinstance(item, dict) else item
        text = normalize_whitespace(str(raw or ""))
        if text:
            output.append(text)
    return output


def _load_coverage_registry() -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        return load_source_registry(), []
    except Exception as exc:
        return [], [f"perspectives_report: failed to load source registry ({type(exc).__name__}: {exc})."]


def _load_json(path: Path, warnings: List[str], label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        warnings.append(f"perspectives_report: failed to load {label} JSON ({type(exc).__name__}: {exc}).")
        return {}
    return payload if isinstance(payload, dict) else {}


def _stories_from_enrichment(payload: Dict[str, Any], articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    known_ids = {article["id"] for article in articles}
    stories: List[Dict[str, Any]] = []
    for thread in _as_list(payload.get("story_threads")):
        if not isinstance(thread, dict):
            continue
        status = str(thread.get("status", "") or "").strip().lower()
        disposition = str(thread.get("disposition", "") or "").strip().lower()
        if status == "skipped_misc" or disposition == "misc":
            continue
        article_ids = [str(item) for item in _as_list(thread.get("article_ids")) if str(item) in known_ids]
        if article_ids:
            stories.append(
                {
                    "story_id": str(thread.get("story_id", "") or ""),
                    "story_title": str(thread.get("story_title", "") or thread.get("label", "") or "Story"),
                    "summary": _thread_summary(thread),
                    "entities": _string_list(thread.get("entities")),
                    "article_ids": article_ids,
                    "confirmed_facts": _as_list(thread.get("confirmed_facts")),
                    "conflicting_claims": _as_list(thread.get("conflicting_claims")),
                    "open_questions": _as_list(thread.get("open_questions")),
                }
            )
    return stories or _fallback_stories(articles)


def _attach_claims_to_stories(
    stories: List[Dict[str, Any]],
    brief_payloads: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    candidates: List[tuple[set[str], Dict[str, Any]]] = []
    for payload in brief_payloads:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        evidence = analysis.get("evidence_packet") if isinstance(analysis.get("evidence_packet"), dict) else {}
        for cluster in _as_list(evidence.get("story_clusters")):
            if not isinstance(cluster, dict):
                continue
            boundary = set(_string_list(cluster.get("article_ids")))
            for raw_claim in _as_list(cluster.get("key_claims")):
                if not isinstance(raw_claim, dict):
                    continue
                claim_text = normalize_whitespace(str(raw_claim.get("claim") or ""))
                support_ids = [item for item in _string_list(raw_claim.get("support_article_ids")) if item in boundary]
                if not claim_text or not support_ids:
                    continue
                origin_ids = [item for item in _string_list(raw_claim.get("origin_article_ids")) if item in support_ids]
                claim = {
                    "claim_id": f"claim-{stable_id(claim_text.lower())}",
                    "claim": claim_text,
                    "claimant": normalize_whitespace(str(raw_claim.get("claimant") or "")),
                    "claim_type": normalize_whitespace(str(raw_claim.get("claim_type") or "other")) or "other",
                    "support_article_ids": support_ids,
                    "origin_article_ids": origin_ids,
                    "confidence": normalize_whitespace(str(raw_claim.get("confidence") or "")),
                }
                candidates.append((boundary | set(support_ids), claim))

    matched_ids: set[str] = set()
    for story in stories:
        story_ids = set(_story_article_ids(story))
        by_key: Dict[str, Dict[str, Any]] = {}
        for boundary, claim in candidates:
            overlap = story_ids.intersection(boundary).intersection(claim["support_article_ids"])
            if not overlap:
                continue
            key = normalize_name(claim["claim"])
            current = by_key.get(key)
            if current is None:
                current = dict(claim)
                current["support_article_ids"] = []
                current["origin_article_ids"] = []
                by_key[key] = current
            current["support_article_ids"] = _unique_text([*current["support_article_ids"], *[item for item in claim["support_article_ids"] if item in story_ids]])
            current["origin_article_ids"] = _unique_text([*current["origin_article_ids"], *[item for item in claim["origin_article_ids"] if item in story_ids]])
            matched_ids.add(claim["claim_id"])
        story["claims"] = list(by_key.values())
    unmatched = {claim["claim_id"] for _boundary, claim in candidates}.difference(matched_ids)
    if unmatched:
        warnings.append(f"perspectives_report: discarded {len(unmatched)} evidence claim(s) with no supporting-article overlap.")


def _thread_summary(thread: Dict[str, Any]) -> str:
    direct = _short_text(thread.get("summary"), 900)
    if direct:
        return direct
    claims = []
    for item in _as_list(thread.get("key_claims"))[:4]:
        if isinstance(item, dict):
            claims.append(str(item.get("claim") or item.get("summary") or ""))
        else:
            claims.append(str(item or ""))
    if not claims:
        for item in _as_list(thread.get("internal_articles"))[:3]:
            if isinstance(item, dict):
                claims.append(str(item.get("summary") or item.get("title") or ""))
    return _short_text(" ".join(claims), 900)


def _fallback_stories(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for article in articles:
        key = article.get("story_family_key") or article.get("story_key") or article.get("topic") or article["id"]
        group = grouped.setdefault(
            str(key),
            {
                "story_id": str(key),
                "story_title": article.get("topic") or article.get("headline") or "Selected Articles",
                "summary": "",
                "entities": [],
                "article_ids": [],
            },
        )
        group["article_ids"].append(article["id"])
    return list(grouped.values())


def _article_from_selected(item: Any) -> Dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    article_id = str(item.get("id") or candidate.get("id") or decision.get("candidate_id") or "").strip()
    headline = str(item.get("headline") or item.get("title") or candidate.get("title") or "").strip()
    source = str(item.get("source") or candidate.get("source") or "").strip()
    url = str(item.get("url") or candidate.get("url") or "").strip()
    if not article_id:
        article_id = normalize_name(f"{source} {headline}") or normalize_name(url)
    return {
        "id": article_id,
        "headline": headline,
        "source": source,
        "url": url,
        "snippet": str(item.get("snippet") or candidate.get("snippet") or ""),
        "published_at": str(item.get("published_at") or candidate.get("published_at") or ""),
        "topic": str(item.get("topic") or decision.get("topic") or metadata.get("topic_name") or "").strip(),
        "story_key": str(item.get("story_key") or metadata.get("memory_story_key") or "").strip(),
        "story_family_key": str(item.get("story_family_key") or metadata.get("memory_story_family_key") or "").strip(),
        "source_briefs": _string_list(metadata.get("source_briefs") or item.get("source_briefs")),
    }


def _dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for article in articles:
        existing = by_id.get(article["id"])
        if existing is None:
            by_id[article["id"]] = article
            continue
        existing["source_briefs"] = sorted({*existing.get("source_briefs", []), *article.get("source_briefs", [])})
    return list(by_id.values())


def _article_summary(articles: List[Dict[str, Any]]) -> str:
    parts = [str(article.get("snippet") or article.get("headline") or "") for article in articles[:5]]
    return _short_text(" ".join(parts), 800)


def _empty_plan(story: Dict[str, Any], status: str, diagnostics: List[str]) -> Dict[str, Any]:
    return {
        "story_id": _story_identity(story),
        "status": status,
        "queries": [],
        "anchor_groups": [],
        "story_loci": [],
        "target_tags": {"raw": {"countries": [], "languages": [], "regions": []}, "normalized": {"countries": [], "languages": [], "regions": []}, "rejected": {"countries": [], "languages": [], "regions": []}},
        "selected_sources": [],
        "verification_targets": [],
        "diagnostics": diagnostics,
    }


def _record_perspectives_report_artifact(orchestrator, *, payload: Dict[str, Any], markdown_path: Path, json_path: Path) -> None:
    stage_payload_builder = getattr(orchestrator, "_stage_payload", None)
    record_stage_artifact = getattr(orchestrator, "_record_stage_artifact", None)
    if not callable(stage_payload_builder) or not callable(record_stage_artifact):
        return
    record_stage_artifact(
        stage="perspectives_report",
        brief_name="pipeline",
        payload=stage_payload_builder(
            stage="perspectives_report",
            brief_name="pipeline",
            summary={
                "stories": payload.get("metadata", {}).get("story_count", 0),
                "coverage_articles": payload.get("metadata", {}).get("coverage_article_count", 0),
                "coverage_countries": payload.get("metadata", {}).get("coverage_source_country_count", 0),
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
                "warnings": len(payload.get("warnings", [])),
            },
            next_stage_input={
                "perspectives_report": payload,
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
            },
        ),
    )


def _phase(orchestrator, message: str) -> None:
    phase = getattr(getattr(orchestrator, "reporter", None), "phase", None)
    if callable(phase):
        phase(message)


def _story_article_ids(story: Dict[str, Any]) -> List[str]:
    return [str(item) for item in story.get("article_ids", [])]


def _story_identity(story: Dict[str, Any]) -> str:
    return str(story.get("story_id", "") or normalize_name(story.get("story_title", "")) or "story")


def normalize_name(value: Any) -> str:
    return " ".join(WORD_RE.findall(str(value or "").lower()))


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(value or "") if len(token) > 1 and token.lower() not in QUERY_STOPWORDS}


def _coverage_match_tokens(value: str) -> set[str]:
    return {token for token in _tokens(value) if token not in COVERAGE_RELEVANCE_WEAK_TOKENS}


def _hostname(url: str) -> str:
    parsed = urlparse(str(url or ""))
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _known_or_unknown(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _short_text(value: Any, max_chars: int) -> str:
    text = normalize_whitespace(str(value or ""))
    return text[: max(1, int(max_chars))] if text else ""


def _count_text(counts: Dict[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in sorted(counts.items()) if value)


def _sorted_counter(counter: Counter[str]) -> Dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items()) if value}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _text_list(value: Any, *, max_items: int, max_chars: int) -> List[str]:
    return _unique_text([_short_text(item, max_chars) for item in _string_list(value)])[:max_items]


def _normalize_anchor_groups(value: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        terms = _text_list(item.get("terms"), max_items=8, max_chars=80)
        if terms:
            output.append({"kind": _short_text(item.get("kind") or "anchor", 24), "terms": terms})
        if len(output) >= 4:
            break
    return output


def _unique_text(values: List[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output
