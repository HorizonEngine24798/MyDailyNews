from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from mydailynews.app.models import BriefOutput, NewsCandidate, PerspectivesReportOutput
from mydailynews.common.utils import canonical_article_url, stable_id
from mydailynews.common.warnings import extend_warnings
from mydailynews.perspectives.coverage import (
    _anchor_group_hits,
    _anchor_queries,
    _budget_source_matches,
    _collect_story_coverage,
    _context_paragraphs,
    _context_stats,
    _coverage_candidate,
    _coverage_counts,
    _coverage_funnel_counts,
    _coverage_gap,
    _coverage_provider_label,
    _coverage_quality,
    _coverage_relevance_decision,
    _coverage_relevance_passes,
    _coverage_relevance_score,
    _coverage_source_yields,
    _dedupe_coverage_rows,
    _diverse_article_selection,
    _effective_article_limit,
    _fetch_selected_article_contexts,
    _near_duplicate_coverage,
    _provider_counts,
    _provider_exception,
    _rank_coverage_rows,
    _relevant_context_excerpt,
    _relevant_coverage_rows,
    _retrieval_query_summary,
    _retrieval_requests,
    _run_provider_requests,
    _select_sources_for_plan,
    _source_matches,
    _substantial_token_overlap,
    _tag_retrieval_row,
    _with_article_context,
)
from mydailynews.perspectives.planning import (
    _article_from_selected,
    _attach_claims_to_stories,
    _config_fallback_tags,
    _dedupe_articles,
    _fallback_stories,
    _fuzzy_tag_match,
    _load_coverage_registry,
    _load_json,
    _normalize_planner_item,
    _normalize_tag_values,
    _normalize_target_tags,
    _planner_items,
    _planner_needs_tag_retry,
    _planner_prompt,
    _planner_tag_options,
    _seed_anchor_groups,
    _seed_sources,
    _source_metadata,
    _stories_from_enrichment,
    _story_planner_payload,
    _tag_key,
    _thread_summary,
    _validate_planner_response,
    collect_perspectives_inputs,
    plan_perspectives_queries,
)
from mydailynews.perspectives.rendering import (
    _append_evidence_section,
    _article_summary,
    _as_list,
    _count_text,
    _coverage_match_tokens,
    _empty_plan,
    _evidence_items,
    _hostname,
    _known_or_unknown,
    _normalize_anchor_groups,
    _short_text,
    _sorted_counter,
    _story_article_ids,
    _story_identity,
    _string_list,
    _text_items,
    _text_list,
    _tokens,
    _unique_text,
    _validated_article_ids,
    normalize_name,
    render_perspectives_report_markdown,
    write_perspectives_report_outputs,
)
from mydailynews.perspectives.schemas import (
    ARTICLE_CONTEXT_MAX_CHARS,
    ARTICLE_CONTEXT_TOKEN_BUDGET,
    ARTICLE_FETCH_MAX_CHARS,
    COUNTRY_ALIASES,
    COVERAGE_RELEVANCE_WEAK_TOKENS,
    EVIDENCE_TYPES,
    LANGUAGE_ALIASES,
    MAX_ANCHOR_QUERIES,
    MAX_ARTICLES_FOR_REPORT,
    MAX_ARTICLES_PER_SOURCE,
    MAX_CANONICAL_QUERIES,
    MAX_CONTEXT_PARAGRAPHS,
    MAX_PLANNER_STORIES_PER_CALL,
    MAX_SELECTED_SOURCES,
    MEANINGFUL_PARAGRAPH_MIN_CHARS,
    MIN_COVERAGE_RELEVANCE_SCORE,
    PERSPECTIVES_CLAIM_SYNTHESIS_SCHEMA,
    PERSPECTIVES_EVIDENCE_LIST_SCHEMA,
    PERSPECTIVES_FRAMING_SCHEMA,
    PERSPECTIVES_PLANNER_SCHEMA,
    PERSPECTIVES_REPORT_SCHEMA_VERSION,
    PERSPECTIVES_VERIFIER_SCHEMA,
    QUERY_STOPWORDS,
    REGION_ALIASES,
    RELATIONSHIPS,
    STRUCTURED_BRIEF_NAMES,
    TAG_RE,
    VERDICTS,
)
from mydailynews.perspectives.sources import load_source_registry, match_source_by_domain, source_domain_map
from mydailynews.perspectives.synthesis import (
    _claim_card_score,
    _claim_synthesis_prompt,
    _empty_framing_report,
    _empty_verification,
    _framing_articles_payload,
    _framing_prompt,
    _normalize_claim_synthesis_response,
    _normalize_framing_response,
    _normalize_verification,
    _public_plan,
    _story_report,
    _strip_known_article_ids,
    _trusted_claim_card,
    _visible_claim_cards,
    build_framing_comparisons,
    build_perspectives_report_payload,
    verify_selected_claims,
)
from mydailynews.retrieval.article import ArticleRetriever
from mydailynews.retrieval.ddg import DuckDuckGoSearchRetriever
from mydailynews.retrieval.gdelt import GdeltDocRetriever
from mydailynews.retrieval.gnews import GNewsRetriever
from mydailynews.retrieval.registry_rss import RegistryRssRetriever


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

    try:
        source_registry = load_source_registry()
        registry_warnings: List[str] = []
    except Exception as exc:
        source_registry = []
        registry_warnings = [f"perspectives_report: failed to load source registry ({type(exc).__name__}: {exc})."]
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
        try:
            source_registry = load_source_registry()
        except Exception as exc:
            source_registry = []
            warnings.append(f"perspectives_report: failed to load source registry ({type(exc).__name__}: {exc}).")
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


def _phase(orchestrator, message: str) -> None:
    phase = getattr(getattr(orchestrator, "reporter", None), "phase", None)
    if callable(phase):
        phase(message)
