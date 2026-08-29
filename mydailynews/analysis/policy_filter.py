from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List

from mydailynews.app.models import PriorReport, SelectedArticle
from mydailynews.domain.candidate_annotations import candidate_memory_annotation


_DELTA_ENTRY_KEYS = ("new", "escalated", "weakened", "reframed", "unchanged_but_important")


def filter_delta_packet_for_articles(
    packet: Dict[str, Any] | None,
    *,
    allowed_article_ids: Iterable[str],
    omitted_count: int,
) -> Dict[str, Any]:
    """Remove suppressed story content from the final writer's delta context."""

    if not isinstance(packet, dict) or not packet:
        return {}
    if omitted_count <= 0:
        return packet
    allowed = {str(value or "").strip() for value in allowed_article_ids if str(value or "").strip()}
    output: Dict[str, Any] = {
        "baseline_coverage_note": "Writer context was filtered after unchanged-story suppression.",
        "evidence_gaps": [],
        "writer_policy": {
            "suppressed_unchanged_story_count": max(0, int(omitted_count)),
            "instruction": "Suppressed story content is intentionally absent; do not reconstruct it from prior reports.",
        },
    }
    for key in _DELTA_ENTRY_KEYS:
        output[key] = _rows_wholly_with_allowed_ids(packet.get(key, []), allowed, id_key="article_ids")
    output["story_decisions"] = _rows_wholly_with_allowed_ids(
        packet.get("story_decisions", []),
        allowed,
        id_key="article_ids",
    )
    if packet.get("deterministic_scaffold"):
        output["deterministic_scaffold"] = True
    if packet.get("deterministic_policy_version"):
        output["deterministic_policy_version"] = str(packet["deterministic_policy_version"])
    return output


def filter_evidence_packet_for_articles(
    packet: Dict[str, Any] | None,
    *,
    allowed_article_ids: Iterable[str],
    omitted_count: int,
) -> Dict[str, Any]:
    """Keep only evidence clusters wholly attributable to writer-visible articles."""

    if not isinstance(packet, dict) or not packet:
        return {}
    if omitted_count <= 0:
        return packet
    allowed = {str(value or "").strip() for value in allowed_article_ids if str(value or "").strip()}
    clusters: List[Dict[str, Any]] = []
    rows = packet.get("story_clusters", [])
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        article_ids = _string_ids(raw.get("article_ids", []))
        # A mixed cluster can contain synthesized text from a suppressed article;
        # dropping it is safer than trying to redact prose after generation.
        if not article_ids or not set(article_ids).issubset(allowed):
            continue
        cluster = dict(raw)
        cluster["article_ids"] = article_ids
        claims: List[Dict[str, Any]] = []
        raw_claims = raw.get("key_claims", [])
        for claim_raw in raw_claims if isinstance(raw_claims, list) else []:
            if not isinstance(claim_raw, dict):
                continue
            support_ids = _string_ids(claim_raw.get("support_article_ids", []))
            origin_ids = _string_ids(claim_raw.get("origin_article_ids", []))
            referenced = set(support_ids).union(origin_ids)
            if referenced and not referenced.issubset(allowed):
                continue
            claim = dict(claim_raw)
            claim["support_article_ids"] = support_ids
            if "origin_article_ids" in claim:
                claim["origin_article_ids"] = origin_ids
            claims.append(claim)
        cluster["key_claims"] = claims
        clusters.append(cluster)

    reader_qa: List[Dict[str, Any]] = []
    rows = packet.get("reader_qa", [])
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        article_ids = _string_ids(raw.get("article_ids", []))
        if article_ids and set(article_ids).issubset(allowed):
            item = dict(raw)
            item["article_ids"] = article_ids
            reader_qa.append(item)
    return {
        "overview": "Evidence context was filtered after unchanged-story suppression.",
        "story_clusters": clusters,
        "global_watch_signals": [],
        "reader_qa": reader_qa,
    }


def filter_prior_reports_for_articles(
    reports: List[PriorReport],
    *,
    selected: List[SelectedArticle],
    omitted_count: int,
) -> List[PriorReport]:
    """Expose only structured history for stories still visible to the writer.

    Legacy report summaries are free-form and cannot be reliably redacted. Once
    a story is suppressed, those summaries are therefore removed and a report is
    retained only when one of its keyed rows belongs to a visible story.
    """

    if omitted_count <= 0:
        return reports
    allowed_story_keys = {
        annotation.story_key
        for article in selected
        for annotation in [candidate_memory_annotation(article.candidate)]
        if annotation is not None and annotation.story_key
    }
    if not allowed_story_keys:
        return []

    filtered: List[PriorReport] = []
    for report in reports:
        major_headlines = _major_rows_with_allowed_story_keys(
            report.major_headlines,
            allowed_story_keys,
        )
        story_baselines = _rows_with_allowed_story_keys(
            report.story_baselines,
            allowed_story_keys,
        )
        if not major_headlines and not story_baselines:
            continue
        filtered.append(
            replace(
                report,
                title=f"Prior report {report.date}",
                path="",
                summary="",
                topics=[],
                major_headlines=major_headlines,
                story_baselines=story_baselines,
            )
        )
    return filtered


def _rows_wholly_with_allowed_ids(
    value: Any,
    allowed: set[str],
    *,
    id_key: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, dict):
            continue
        article_ids = _string_ids(raw.get(id_key, []))
        # Delta prose can summarize every ID in a row. Retaining a mixed row and
        # merely deleting one ID would leave the suppressed article's prose.
        if not article_ids or not set(article_ids).issubset(allowed):
            continue
        item = dict(raw)
        item[id_key] = article_ids
        output.append(item)
    return output


def _string_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _rows_with_allowed_story_keys(value: Any, allowed: set[str]) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        dict(row)
        for row in value
        if isinstance(row, dict) and str(row.get("story_key", "") or "").strip() in allowed
    ]


def _major_rows_with_allowed_story_keys(value: Any, allowed: set[str]) -> List[Dict[str, Any]]:
    safe_fields = {"story_key", "headline", "title", "source", "url", "topic"}
    return [
        {key: item for key, item in row.items() if key in safe_fields}
        for row in (value if isinstance(value, list) else [])
        if isinstance(row, dict) and str(row.get("story_key", "") or "").strip() in allowed
    ]
