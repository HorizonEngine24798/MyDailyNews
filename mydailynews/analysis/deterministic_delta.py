from __future__ import annotations

import re
from typing import Any, Dict, List

from mydailynews.app.models import PriorReport, SelectedArticle


def _tokenize_delta_text(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "latest",
        "today",
        "news",
        "major",
        "about",
        "after",
        "amid",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if token not in stop
    }


def _prior_headline_items(prior_reports: List[PriorReport], max_reports: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for report in prior_reports[: max(1, max_reports)]:
        major = report.major_headlines if isinstance(report.major_headlines, list) else []
        if not major:
            title = str(report.title or "").strip()
            if title:
                items.append(
                    {
                        "headline": title,
                        "report_id": report.id,
                        "report_date": report.date,
                        "tokens": _tokenize_delta_text(title),
                    }
                )
            continue
        for row in major[:8]:
            if not isinstance(row, dict):
                continue
            headline = str(row.get("headline") or row.get("title") or "").strip()
            if not headline:
                continue
            items.append(
                {
                    "headline": headline,
                    "report_id": report.id,
                    "report_date": report.date,
                    "tokens": _tokenize_delta_text(headline),
                }
            )
    return items


def _best_overlap(current_tokens: set[str], prior_items: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any] | None]:
    best_score = 0.0
    best_item: Dict[str, Any] | None = None
    if not current_tokens:
        return best_score, best_item
    for item in prior_items:
        prior_tokens = item.get("tokens", set())
        if not prior_tokens:
            continue
        overlap = len(current_tokens.intersection(prior_tokens))
        if overlap <= 0:
            continue
        score = overlap / max(1, min(len(current_tokens), len(prior_tokens)))
        if score > best_score:
            best_score = score
            best_item = item
    return best_score, best_item


def _delta_entry(article: SelectedArticle, summary: str) -> Dict[str, Any]:
    return {
        "item": str(article.candidate.title or "")[:100],
        "summary": summary[:180],
        "article_ids": [str(article.candidate.id)],
    }


def build_deterministic_delta_scaffold(
    selected: List[SelectedArticle],
    prior_reports: List[PriorReport],
    *,
    max_prior_reports: int = 3,
) -> Dict[str, Any]:
    if not selected:
        return {}

    prior_items = _prior_headline_items(prior_reports, max_prior_reports)
    coverage_note = (
        "No prior reports available; deterministic scaffold highlights only the current-run story shape."
        if not prior_reports
        else (
            f"Compared {len(selected)} current selected article(s) against "
            f"{len(prior_items)} prior headline anchor(s) from {min(len(prior_reports), max_prior_reports)} report(s)."
        )
    )

    escalated_terms = {
        "escalates",
        "escalation",
        "surge",
        "spike",
        "attack",
        "expands",
        "tightens",
        "sanctions",
        "deadline",
        "warning",
    }
    weakened_terms = {
        "decline",
        "drops",
        "drop",
        "eases",
        "eased",
        "ceasefire",
        "cools",
        "paused",
        "delay",
        "delayed",
    }

    new_items: List[Dict[str, Any]] = []
    escalated: List[Dict[str, Any]] = []
    weakened: List[Dict[str, Any]] = []
    reframed: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []

    overlap_count = 0
    for article in selected:
        title = str(article.candidate.title or "").strip()
        snippet = str(article.candidate.snippet or "").strip()
        current_tokens = _tokenize_delta_text(f"{title} {snippet}")
        overlap_score, prior_match = _best_overlap(current_tokens, prior_items)
        prior_label = ""
        prior_date = ""
        if prior_match:
            prior_label = str(prior_match.get("headline", "")).strip()
            prior_date = str(prior_match.get("report_date", "")).strip()
        if overlap_score >= 0.58 and prior_match:
            overlap_count += 1
            has_escalation = bool(current_tokens.intersection(escalated_terms))
            has_weakening = bool(current_tokens.intersection(weakened_terms))
            if has_escalation and not has_weakening:
                escalated.append(
                    _delta_entry(
                        article,
                        f"Likely continuation of prior coverage ({prior_date}: {prior_label}) with signs of escalation.",
                    )
                )
            elif has_weakening and not has_escalation:
                weakened.append(
                    _delta_entry(
                        article,
                        f"Likely continuation of prior coverage ({prior_date}: {prior_label}) with signs of easing.",
                    )
                )
            else:
                unchanged.append(
                    _delta_entry(
                        article,
                        f"Likely continuation of prior coverage ({prior_date}: {prior_label}).",
                    )
                )
            continue
        if overlap_score >= 0.34 and prior_match:
            overlap_count += 1
            reframed.append(
                _delta_entry(
                    article,
                    f"Partially overlaps prior coverage ({prior_date}: {prior_label}) but appears reframed.",
                )
            )
            continue
        new_items.append(
            _delta_entry(
                article,
                "No strong headline-level overlap found in prior report anchors.",
            )
        )

    evidence_gaps: List[Dict[str, Any]] = []
    if not prior_reports:
        evidence_gaps.append(
            {
                "gap": "No prior reports available for direct delta comparison.",
                "why_it_matters": "Change classification is approximate without a baseline brief.",
            }
        )
    elif not prior_items:
        evidence_gaps.append(
            {
                "gap": "Prior reports lacked reusable major-headline anchors.",
                "why_it_matters": "Deterministic overlap had to fall back to limited text anchors.",
            }
        )
    elif overlap_count == 0:
        evidence_gaps.append(
            {
                "gap": "No strong deterministic overlap between current and prior headline anchors.",
                "why_it_matters": "Stories may be genuinely new or token overlap may miss semantic continuation.",
            }
        )

    return {
        "baseline_coverage_note": coverage_note,
        "new": new_items[:6],
        "escalated": escalated[:5],
        "weakened": weakened[:5],
        "reframed": reframed[:5],
        "unchanged_but_important": unchanged[:6],
        "evidence_gaps": evidence_gaps[:4],
        "deterministic_scaffold": True,
    }
