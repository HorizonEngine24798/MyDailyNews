from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from mydailynews.ai.prompts import PERSPECTIVES_FRAMING_USER
from mydailynews.ai.token_budget import resolve_client_token_budget
from mydailynews.common.utils import compact_json, normalize_whitespace, utc_now
from mydailynews.perspectives.coverage import (
    _context_stats,
    _coverage_counts,
    _coverage_gap,
    _coverage_quality,
    _effective_article_limit,
)
from mydailynews.perspectives.grounding import verify_synthesis_grounding
from mydailynews.perspectives.planning import _seed_sources
from mydailynews.perspectives.rendering import (
    _as_list,
    _empty_plan,
    _evidence_items,
    _short_text,
    _sorted_counter,
    _story_article_ids,
    _story_identity,
    _string_list,
    _text_items,
    _unique_text,
    _validated_article_ids,
    normalize_name,
)
from mydailynews.perspectives.schemas import (
    ARTICLE_CONTEXT_MAX_CHARS,
    ARTICLE_CONTEXT_TOKEN_BUDGET,
    MAX_ARTICLES_FOR_REPORT,
    MAX_ARTICLES_PER_SOURCE,
    MAX_CANONICAL_QUERIES,
    MAX_SELECTED_SOURCES,
    MEANINGFUL_PARAGRAPH_MIN_CHARS,
    PERSPECTIVES_CLAIM_SYNTHESIS_SCHEMA,
    PERSPECTIVES_FRAMING_SCHEMA,
    PERSPECTIVES_REPORT_SCHEMA_VERSION,
    PERSPECTIVES_VERIFIER_SCHEMA,
    RELATIONSHIPS,
    VERDICTS,
)


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
        coverage_limits = list(coverage.get("coverage_quality", {}).get("thin_reasons", []))
        distinct_countries = {str(a.get("source_country") or "").upper() for a in articles if str(a.get("source_country") or "").strip()}
        if len(distinct_countries) < 2:
            single_country_note = "Coverage is concentrated in a single national jurisdiction; multi-country framing contrast cannot be reliably established."
            if single_country_note not in coverage_limits:
                coverage_limits.append(single_country_note)

        story_payloads.append(
            {
                "story_id": story_id,
                "title": str(story.get("story_title") or story.get("title") or ""),
                "summary": str(story.get("summary") or ""),
                "articles": articles,
                "coverage_limitations": coverage_limits,
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
        articles_by_id = {str(article.get("article_id") or ""): article for article in story_payload["articles"] if isinstance(article, dict)}
        if story_payload.get("claims"):
            normalized_report = _normalize_claim_synthesis_response(
                raw,
                story=story_payload,
                known_article_ids=known_ids,
                verification_by_claim=verification_by_claim,
                verification_documents=verification_documents,
                warnings=warnings,
            )
            framing_reports[story_id] = verify_synthesis_grounding(
                normalized_report,
                story=story_payload,
                articles_by_id=articles_by_id,
                warnings=warnings,
            )
        else:
            normalized_reports = _normalize_framing_response(
                raw,
                known_story_ids=[story_id],
                known_article_ids={story_id: known_ids},
                warnings=warnings,
            )
            for sid, rep in normalized_reports.items():
                framing_reports[sid] = verify_synthesis_grounding(
                    rep,
                    story=story_payload,
                    articles_by_id=articles_by_id,
                    warnings=warnings,
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
        "grounding_diagnostics": framing.get("grounding_diagnostics", {}),
    }


def _public_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(plan.get("status") or ""),
        "queries": _string_list(plan.get("queries")),
        "aligned_queries": plan.get("aligned_queries", {}),
        "anchor_groups": plan.get("anchor_groups", []),
        "story_loci": plan.get("story_loci", []),
        "target_tags": plan.get("target_tags", {}),
        "selected_sources": plan.get("selected_sources", []),
        "diagnostics": _unique_text(_string_list(plan.get("diagnostics"))),
        "verification_targets": _as_list(plan.get("verification_targets")),
    }


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
