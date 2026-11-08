from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re
from typing import Any, Dict, List

from mydailynews.app.models import NewsCandidate
from mydailynews.common.parallel import ordered_parallel_map
from mydailynews.common.utils import canonical_article_url, normalize_whitespace, stable_id
from mydailynews.common.warnings import extend_warnings
from mydailynews.domain.article_identity import article_aliases_for_candidate
from mydailynews.perspectives.planning import _config_fallback_tags, _source_metadata
from mydailynews.perspectives.rendering import (
    _as_list,
    _coverage_match_tokens,
    _hostname,
    _known_or_unknown,
    _normalize_anchor_groups,
    _sorted_counter,
    _story_identity,
    _string_list,
    _text_list,
    _unique_text,
    normalize_name,
)
from mydailynews.perspectives.schemas import (
    ARTICLE_CONTEXT_MAX_CHARS,
    MAX_ANCHOR_QUERIES,
    MAX_ARTICLES_FOR_REPORT,
    MAX_ARTICLES_PER_SOURCE,
    MAX_CANONICAL_QUERIES,
    MAX_CONTEXT_PARAGRAPHS,
    MAX_SELECTED_SOURCES,
    MEANINGFUL_PARAGRAPH_MIN_CHARS,
)
from mydailynews.perspectives.query_alignment import (
    align_query_with_assist,
    expand_multilingual_anchor_groups,
    normalize_language_code,
    resolve_country_language,
)
from mydailynews.perspectives.sources import match_source_by_domain
from mydailynews.retrieval.article import ArticleRetriever
from mydailynews.retrieval.gdelt import GdeltDocRetriever
from mydailynews.retrieval.gnews import GNewsRetriever
from mydailynews.retrieval.google_news_rss import GoogleNewsRssCoverageRetriever
from mydailynews.retrieval.registry_rss import RegistryRssRetriever


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
    google_news_rss_retriever: GoogleNewsRssCoverageRetriever | None = None,
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

    align_target_languages = bool(getattr(config, "cross_language_alignment", False) or plan.get("align_queries", False))
    requests = _retrieval_requests(plan, selected_sources, story=story, align_target_languages=align_target_languages)
    if not requests:
        return _coverage_gap(story_id, plan, "planner produced no usable retrieval requests for selected sources")

    providers: List[tuple[str, Any]] = [("gdelt_doc", gdelt_retriever), ("registry_rss", registry_rss_retriever)]
    if gnews_retriever is not None:
        providers.append(("gnews", gnews_retriever))
    if google_news_rss_retriever is not None:
        providers.append(("google_news_rss", google_news_rss_retriever))

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
            target_lang = request.get("target_language")
            if target_lang and target_lang != "en":
                search_kwargs["source_languages"] = [target_lang]
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
    align_target_languages: bool = False,
) -> List[Dict[str, Any]]:
    countries = _unique_text([str(source.get("country") or "").upper() for source in selected_sources])
    source_ids = _unique_text([str(source.get("source_id") or "") for source in selected_sources])
    story_title_tokens = _coverage_match_tokens(str((story or {}).get("story_title") or ""))
    story_summary_tokens = _coverage_match_tokens(str((story or {}).get("summary") or ""))

    do_align = align_target_languages or bool(plan.get("align_queries"))
    target_langs = (
        {
            normalize_language_code(source.get("language") or resolve_country_language(source.get("country")))
            for source in selected_sources
        }
        - {"en", ""}
        if do_align
        else set()
    )

    raw_anchor_groups = plan.get("anchor_groups", [])
    anchor_groups = expand_multilingual_anchor_groups(raw_anchor_groups, list(target_langs)) if target_langs else raw_anchor_groups

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
                "anchor_groups": anchor_groups,
                "source_id": "",
                "source_ids": source_ids,
                "source_countries": countries,
            }
        )
        if do_align:
            for lang in target_langs:
                lang_sources = [
                    s for s in selected_sources
                    if normalize_language_code(s.get("language") or resolve_country_language(s.get("country"))) == lang
                ]
                lang_countries = _unique_text([str(s.get("country") or "").upper() for s in lang_sources])
                lang_source_ids = _unique_text([str(s.get("source_id") or "") for s in lang_sources])

                # Use model-provided aligned queries if present in the plan; otherwise apply assist lookup
                plan_aligned_list = (plan.get("aligned_queries") or {}).get(lang)
                if plan_aligned_list and isinstance(plan_aligned_list, list):
                    candidate_aligned = plan_aligned_list
                else:
                    candidate_aligned = [align_query_with_assist(query, lang)]

                for aligned in candidate_aligned:
                    if aligned and aligned != query and aligned not in seen_canonical:
                        seen_canonical.add(aligned)
                        requests.append(
                            {
                                "query": aligned,
                                "query_type": "aligned_canonical",
                                "anchor_groups": anchor_groups,
                                "source_id": "",
                                "source_ids": lang_source_ids,
                                "source_countries": lang_countries,
                                "target_language": lang,
                            }
                        )
    for query in _anchor_queries(anchor_groups):
        if query in seen_canonical:
            continue
        seen_canonical.add(query)
        requests.append(
            {
                "query": query,
                "query_type": "anchor",
                "anchor_groups": anchor_groups,
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
    query_evidence_hits = len(evidence_tokens.intersection(query_tokens))
    story_tokens = _coverage_match_tokens(
        " ".join([str(story.get("story_title") or ""), str(story.get("summary") or ""), str(story.get("_seed_text") or "")])
    )
    anchor_hits = _anchor_group_hits(row.get("retrieval_anchor_groups"), evidence_tokens)
    if anchor_hits >= 2:
        return True, "multiple_anchor_groups"
    if anchor_hits and query_evidence_hits >= 2:
        return True, "anchor_and_query_overlap"
    if _substantial_token_overlap(query_tokens, title_tokens, min_hits=2, min_fraction=0.5):
        return True, "query_title_overlap"
    if _substantial_token_overlap(query_tokens, evidence_tokens, min_hits=3, min_fraction=0.45):
        return True, "query_evidence_overlap"
    if _substantial_token_overlap(story_tokens, evidence_tokens, min_hits=2, min_fraction=0.45):
        return True, "seed_story_overlap"

    story_text = " ".join([str(story.get("story_title") or ""), str(story.get("summary") or "")])
    row_text = " ".join([str(row.get("title") or ""), str(row.get("snippet") or "")])
    if story_text and row_text:
        try:
            from mydailynews.memory.multilingual_embeddings import calculate_text_similarity
            sim = calculate_text_similarity(story_text, row_text)
            if sim >= 0.85:
                return True, "multilingual_semantic_match"
        except Exception:
            pass

    return False, "insufficient_event_overlap"


def _substantial_token_overlap(
    left: set[str],
    right: set[str],
    *,
    min_hits: int,
    min_fraction: float,
) -> bool:
    if not left or not right:
        return False
    hits = len(left.intersection(right))
    containment = hits / max(1, min(len(left), len(right)))
    return hits >= max(1, int(min_hits)) and containment >= float(min_fraction)


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
