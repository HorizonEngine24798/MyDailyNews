from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from mydailynews.common.utils import normalize_whitespace
from mydailynews.perspectives.query_alignment import normalize_language_code, resolve_country_language


COUNTRY_MENTION_MAP: Dict[str, str] = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "american": "US",
    "uk": "GB",
    "britain": "GB",
    "british": "GB",
    "united kingdom": "GB",
    "france": "FR",
    "french": "FR",
    "germany": "DE",
    "german": "DE",
    "japan": "JP",
    "japanese": "JP",
    "china": "CN",
    "chinese": "CN",
    "russia": "RU",
    "russian": "RU",
    "spain": "ES",
    "spanish": "ES",
    "italy": "IT",
    "italian": "IT",
    "canada": "CA",
    "canadian": "CA",
    "australia": "AU",
    "australian": "AU",
    "india": "IN",
    "indian": "IN",
}

DEPENDENT_EVIDENCE_MARKERS = {
    "spokesperson said",
    "spokesperson told",
    "according to",
    "quoted",
    "citing",
    "cited",
    "statement released",
    "press release",
    "in an interview",
    "told reporters",
}


def verify_synthesis_grounding(
    framing_report: Dict[str, Any],
    *,
    story: Dict[str, Any],
    articles_by_id: Dict[str, Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    if not isinstance(framing_report, dict):
        return framing_report

    report = dict(framing_report)
    corrections: List[str] = []
    known_article_ids: Set[str] = set(articles_by_id.keys())
    story_id = str(story.get("story_id") or "")

    # 1. Citation Grounding: Check synthesis_article_ids
    raw_synthesis_ids = list(report.get("synthesis_article_ids") or [])
    valid_synthesis_ids = [aid for aid in raw_synthesis_ids if aid in known_article_ids]
    if len(raw_synthesis_ids) != len(valid_synthesis_ids):
        dropped = set(raw_synthesis_ids) - set(valid_synthesis_ids)
        corrections.append(f"Pruned ungrounded synthesis_article_ids: {', '.join(sorted(dropped))}")
        warnings.append(f"perspectives_report: dropped hallucinated synthesis_article_ids for {story_id!r}: {', '.join(sorted(dropped))}")
    if not valid_synthesis_ids and known_article_ids:
        # Fall back to top known articles
        valid_synthesis_ids = list(sorted(known_article_ids))[:2]
        corrections.append("Auto-populated empty synthesis_article_ids with story articles")
    report["synthesis_article_ids"] = valid_synthesis_ids

    # 2. Evidence Items Grounding: shared_facts, country_source_comparison, etc.
    total_evidence_citations = 0
    valid_evidence_citations = 0

    for key in (
        "shared_facts",
        "country_source_comparison",
        "repetition_without_independent_support",
        "verified_or_independently_supported_claims",
        "qualified_disputed_or_unresolved_claims",
    ):
        items = list(report.get(key) or [])
        grounded_items: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_c = dict(item)
            c_ids = list(item_c.get("article_ids") or [])
            total_evidence_citations += len(c_ids)
            valid_c_ids = [aid for aid in c_ids if aid in known_article_ids]
            valid_evidence_citations += len(valid_c_ids)
            if len(c_ids) != len(valid_c_ids):
                dropped = set(c_ids) - set(valid_c_ids)
                corrections.append(f"Pruned invalid article_ids from {key}: {', '.join(sorted(dropped))}")
            item_c["article_ids"] = valid_c_ids
            grounded_items.append(item_c)
        report[key] = grounded_items

    # 3. Country Attribution Grounding in country_source_comparison
    country_grounding_passed = True
    for item in report.get("country_source_comparison", []):
        text_lower = str(item.get("text") or "").lower()
        item_article_ids = list(item.get("article_ids") or [])
        referenced_countries = _extract_referenced_countries(text_lower)
        if referenced_countries and item_article_ids:
            cited_countries = {
                str(articles_by_id[aid].get("source_country") or "").upper()
                for aid in item_article_ids
                if aid in articles_by_id
            }
            # Check if referenced countries overlap with cited article countries
            missing_grounding = referenced_countries - cited_countries
            if missing_grounding and not cited_countries.intersection(referenced_countries):
                country_grounding_passed = False
                notice = f"Attribution to {', '.join(sorted(missing_grounding))} lacks direct primary coverage from that country."
                corrections.append(f"Annotated ungrounded country attribution: {notice}")
                warnings.append(f"perspectives_report: ungrounded country attribution for {story_id!r}: {notice}")
                current_limits = list(report.get("coverage_limitations") or [])
                if notice not in current_limits:
                    current_limits.append(notice)
                report["coverage_limitations"] = current_limits

    # 4. Independent Support Grounding in claim_perspectives
    claim_perspectives = list(report.get("claim_perspectives") or [])
    grounded_perspectives: List[Dict[str, Any]] = []
    independent_support_grounded = True

    for cp in claim_perspectives:
        if not isinstance(cp, dict):
            continue
        cp_c = dict(cp)
        claimants = {str(c.get("name") or "").lower() for c in cp_c.get("claimants", []) if isinstance(c, dict)}
        coverage_rows = list(cp_c.get("coverage") or [])
        grounded_coverage: List[Dict[str, Any]] = []
        for cov in coverage_rows:
            if not isinstance(cov, dict):
                continue
            cov_c = dict(cov)
            aid = str(cov_c.get("article_id") or "")
            if aid not in known_article_ids:
                corrections.append(f"Dropped unknown article {aid!r} from claim coverage")
                continue
            article = articles_by_id[aid]
            relationship = str(cov_c.get("relationship") or "")
            evidence_basis = str(cov_c.get("evidence_basis") or "").lower()
            explanation = str(cov_c.get("explanation") or "").lower()
            combined_eval = f"{evidence_basis} {explanation}"

            # If marked independently_supports, verify it is truly independent
            if relationship == "independently_supports":
                source_name = str(article.get("source_name") or "").lower()
                is_claimant_source = any(cl in source_name for cl in claimants if cl)
                has_dependent_marker = any(marker in combined_eval for marker in DEPENDENT_EVIDENCE_MARKERS)
                if is_claimant_source or has_dependent_marker:
                    cov_c["relationship"] = "reports_or_quotes"
                    independent_support_grounded = False
                    corrections.append(f"Demoted article {aid!r} from independently_supports to reports_or_quotes (dependent attribution)")
            grounded_coverage.append(cov_c)
        cp_c["coverage"] = grounded_coverage
        grounded_perspectives.append(cp_c)
    report["claim_perspectives"] = grounded_perspectives

    # 5. Multi-Country Coverage Verification
    distinct_countries = {
        str(article.get("source_country") or "").upper()
        for article in articles_by_id.values()
        if str(article.get("source_country") or "").strip()
    } - {"UNKNOWN", ""}
    if len(distinct_countries) < 2:
        limitation_text = "Multi-country perspectives are limited; coverage is concentrated in single-country sources, so divergent national framing cannot be confirmed."
        limits = list(report.get("coverage_limitations") or [])
        if not any("Multi-country perspectives are limited" in item or "single-country" in item for item in limits):
            limits.append(limitation_text)
            corrections.append("Enforced single-country limitation warning")
        report["coverage_limitations"] = limits

    # 6. Compute Grounding Score
    citation_validity = (
        1.0
        if total_evidence_citations == 0
        else valid_evidence_citations / max(1, total_evidence_citations)
    )
    grounding_score = round(
        0.5 * citation_validity
        + 0.3 * (1.0 if country_grounding_passed else 0.5)
        + 0.2 * (1.0 if independent_support_grounded else 0.7),
        2,
    )

    report["grounding_diagnostics"] = {
        "citation_validity_rate": citation_validity,
        "country_grounding_passed": country_grounding_passed,
        "independent_support_grounded": independent_support_grounded,
        "grounding_score": grounding_score,
        "corrections": corrections,
    }
    return report


def _extract_referenced_countries(text: str) -> Set[str]:
    referenced: Set[str] = set()
    for phrase, country_code in COUNTRY_MENTION_MAP.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            referenced.add(country_code)
    return referenced
