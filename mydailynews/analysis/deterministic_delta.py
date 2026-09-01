from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from mydailynews.app.models import PriorReport, SelectedArticle
from mydailynews.analysis.claim_delta import (
    CLAIM_DELTA_POLICY_VERSION,
    assess_claim_comparison,
    build_claim_comparison,
    current_claim_evidence,
    prior_claim_evidence,
    request_from_claim_delta_payload,
    validate_semantic_decision,
)
from mydailynews.domain.candidate_annotations import candidate_memory_annotation
from mydailynews.domain.text_similarity import compare_token_sets, normalized_word_text, word_tokens
from mydailynews.memory.story_keys import STOPWORDS, story_identity_for_candidate
from mydailynews.common.utils import datetime_to_iso


DETERMINISTIC_DELTA_POLICY_VERSION = "structural-evidence.v3"


@dataclass(frozen=True)
class DeterministicChangeAssessment:
    relationship: str
    change_type: str
    materiality: float
    confidence: float
    disposition: str
    reason: str


def _tokenize_delta_text(text: str) -> set[str]:
    return set(word_tokens(text, stopwords=STOPWORDS, min_alpha_chars=2, keep_numbers=True))


def _prior_headline_items(
    prior_reports: List[PriorReport],
    max_reports: int,
    story_memory: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_item(
        *,
        headline: str,
        story_key: str = "",
        report_id: str = "",
        report_date: str = "",
    ) -> None:
        normalized_headline = " ".join(str(headline or "").split()).strip()
        if not normalized_headline:
            return
        key = (str(story_key or "").strip(), normalized_word_text(normalized_headline))
        if key in seen:
            return
        seen.add(key)
        items.append(
            {
                "headline": normalized_headline,
                "story_key": str(story_key or "").strip(),
                "report_id": str(report_id or "").strip(),
                "report_date": str(report_date or "").strip(),
                "tokens": _tokenize_delta_text(normalized_headline),
            }
        )

    stories = story_memory.get("stories", []) if isinstance(story_memory, dict) else []
    if isinstance(stories, list):
        for story in stories:
            if not isinstance(story, dict):
                continue
            baselines = story.get("prior_baselines", [])
            if not isinstance(baselines, list):
                continue
            for baseline in baselines[: max(1, max_reports)]:
                if not isinstance(baseline, dict):
                    continue
                add_item(
                    headline=str(baseline.get("title") or baseline.get("last_delta_summary") or ""),
                    story_key=str(baseline.get("story_key") or ""),
                    report_id=str(baseline.get("last_report_id") or ""),
                    report_date=str(baseline.get("last_seen") or baseline.get("last_material_change_date") or ""),
                )

    for report in prior_reports[: max(1, max_reports)]:
        major = report.major_headlines if isinstance(report.major_headlines, list) else []
        if not major:
            add_item(headline=str(report.title or ""), report_id=report.id, report_date=report.date)
            continue
        for row in major[:8]:
            if not isinstance(row, dict):
                continue
            add_item(
                headline=str(row.get("headline") or row.get("title") or ""),
                story_key=str(row.get("story_key") or ""),
                report_id=report.id,
                report_date=report.date,
            )
    return items


def _best_prior_match(
    current_tokens: set[str],
    prior_items: Iterable[Dict[str, Any]],
    *,
    current_story_key: str,
) -> tuple[float, Dict[str, Any] | None]:
    best_score = 0.0
    best_item: Dict[str, Any] | None = None
    for item in prior_items:
        prior_tokens = item.get("tokens", set())
        if not isinstance(prior_tokens, set) or not prior_tokens:
            continue
        similarity = compare_token_sets(current_tokens, prior_tokens)
        score = similarity.confidence
        item_story_key = str(item.get("story_key", "") or "").strip()
        if current_story_key and item_story_key == current_story_key and not similarity.numeric_conflict:
            score = max(score, 0.8)
        if score > best_score:
            best_score = score
            best_item = item
    return best_score, best_item


def assess_lexical_change(
    current_title: str,
    prior_title: str | None,
    *,
    story_key_match: bool = False,
) -> DeterministicChangeAssessment:
    """Make only the conclusions that lexical evidence can safely support.

    Semantic direction (for example, whether an unfamiliar event strengthened,
    weakened, corrected, or resolved a story) is deliberately not inferred from
    words. A local model or a structured source fact must supply that judgment.
    """

    if not str(prior_title or "").strip():
        return DeterministicChangeAssessment(
            relationship="distinct_story",
            change_type="new",
            materiality=1.0,
            confidence=0.8,
            disposition="full_report",
            reason="No prior story anchor was available.",
        )

    current_tokens = _tokenize_delta_text(current_title)
    prior_tokens = _tokenize_delta_text(str(prior_title or ""))
    similarity = compare_token_sets(current_tokens, prior_tokens)
    exact_text = normalized_word_text(current_title) == normalized_word_text(prior_title)
    equivalent_tokens = bool(current_tokens) and current_tokens == prior_tokens
    near_duplicate = (
        similarity.containment >= 0.92
        and similarity.left_novelty <= 0.1
        and similarity.right_novelty <= 0.1
        and not similarity.numeric_conflict
    )

    if (exact_text or equivalent_tokens or near_duplicate) and story_key_match:
        return DeterministicChangeAssessment(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.98 if exact_text or equivalent_tokens else 0.9,
            disposition="omit",
            reason="The current headline is a lexical duplicate of the prior anchor.",
        )

    if exact_text or equivalent_tokens or near_duplicate:
        return DeterministicChangeAssessment(
            relationship="same_story",
            change_type="unchanged",
            materiality=0.0,
            confidence=0.7,
            disposition="uncertain",
            reason="The headline repeats a prior anchor, but durable story identity was not confirmed.",
        )

    if similarity.numeric_conflict:
        return DeterministicChangeAssessment(
            relationship="uncertain",
            change_type="uncertain",
            materiality=0.5,
            confidence=0.55,
            disposition="uncertain",
            reason="The headlines overlap but contain conflicting numeric identity signals.",
        )

    if story_key_match or similarity.confidence >= 0.58:
        return DeterministicChangeAssessment(
            relationship="same_story",
            change_type="uncertain",
            materiality=0.5,
            confidence=max(0.6, min(0.85, similarity.confidence)),
            disposition="uncertain",
            reason="The lexical evidence supports story continuity but not a semantic change label.",
        )

    if similarity.confidence >= 0.34:
        return DeterministicChangeAssessment(
            relationship="related_theme",
            change_type="uncertain",
            materiality=0.5,
            confidence=similarity.confidence,
            disposition="uncertain",
            reason="The headlines share a theme, but deterministic evidence cannot establish identity.",
        )

    return DeterministicChangeAssessment(
        relationship="distinct_story",
        change_type="new",
        materiality=1.0,
        confidence=max(0.65, 1.0 - similarity.confidence),
        disposition="full_report",
        reason="No strong lexical link to the prior anchor was found.",
    )


def _delta_entry(article: SelectedArticle, summary: str) -> Dict[str, Any]:
    return {
        "item": str(article.candidate.title or ""),
        "summary": summary,
        "article_ids": [str(article.candidate.id)],
    }


def build_deterministic_delta_scaffold(
    selected: List[SelectedArticle],
    prior_reports: List[PriorReport],
    *,
    max_prior_reports: int = 3,
    story_memory: Dict[str, Any] | None = None,
    story_store: Any | None = None,
) -> Dict[str, Any]:
    if not selected:
        return {}

    prior_items = _prior_headline_items(prior_reports, max_prior_reports, story_memory)
    coverage_note = (
        "No prior story anchors available; current stories are treated as new."
        if not prior_items
        else (
            f"Compared {len(selected)} current selected article(s) against "
            f"{len(prior_items)} prior story anchor(s)."
        )
    )

    new_items: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []
    escalated: List[Dict[str, Any]] = []
    reframed: List[Dict[str, Any]] = []
    story_decisions: List[Dict[str, Any]] = []
    uncertain_count = 0

    for article in selected:
        title = str(article.candidate.title or "").strip()
        annotation = candidate_memory_annotation(article.candidate)
        identity = story_identity_for_candidate(article.candidate)
        story_key = annotation.story_key if annotation and annotation.story_key else identity.story_key
        story_context = _story_context_for_article(story_memory, str(article.candidate.id))
        baselines = story_context.get("prior_baselines", []) if story_context else []
        candidate_baselines = (
            [baseline for baseline in baselines if isinstance(baseline, dict)]
            if isinstance(baselines, list)
            else []
        )
        current_evidence = current_claim_evidence(
            article_id=str(article.candidate.id),
            title=title,
            text=article.article_text or article.candidate.snippet,
            source_name=str(article.candidate.source or ""),
            source_url=str(article.candidate.url or ""),
            published_at=datetime_to_iso(article.candidate.published_at),
        )
        prior_evidence = [
            claim
            for baseline in candidate_baselines
            for claim in prior_claim_evidence(baseline, max_claims=8)
        ][:24]
        comparison = build_claim_comparison(current_evidence, prior_evidence)
        claim_delta = assess_claim_comparison(comparison)
        prior_story_key = claim_delta.prior_story_key
        relationship = claim_delta.relationship
        change_type = claim_delta.change_type
        materiality = claim_delta.materiality
        confidence = claim_delta.confidence
        disposition = claim_delta.disposition
        reason = claim_delta.summary
        summary = claim_delta.summary

        if change_type == "new":
            new_items.append(_delta_entry(article, summary))
        elif change_type == "unchanged":
            unchanged.append(_delta_entry(article, summary))
        elif change_type in {"correction", "reframed"}:
            reframed.append(_delta_entry(article, summary))
        elif change_type in {"material_update", "status_change", "resolved"}:
            escalated.append(_delta_entry(article, summary))
        else:
            uncertain_count += 1

        story_decisions.append(
            {
                "story_key": story_key,
                "article_ids": [str(article.candidate.id)],
                "prior_story_key": prior_story_key,
                "relationship": relationship,
                "change_type": change_type,
                "materiality": materiality,
                "confidence": confidence,
                "disposition": disposition,
                "summary": summary,
                "bullet": title,
                "reason": reason,
                "knowns": list(claim_delta.added_claims or claim_delta.repeated_claims),
                "unknowns": [],
                "watch_signals": [],
                "claim_delta": {
                    "policy_version": CLAIM_DELTA_POLICY_VERSION,
                    "decision_basis": claim_delta.decision_basis,
                    "requires_semantic_inference": claim_delta.requires_semantic_inference,
                    "added_claims": list(claim_delta.added_claims),
                    "repeated_claims": list(claim_delta.repeated_claims),
                    "superseded_claims": list(claim_delta.superseded_claims),
                    "current_claims": list(claim_delta.current_claims),
                    "prior_claims": list(claim_delta.prior_claims),
                    "exact_alignments": list(claim_delta.exact_alignments),
                    "current_evidence_ids": list(claim_delta.current_evidence_ids),
                    "prior_evidence_ids": list(claim_delta.prior_evidence_ids),
                    "superseded_prior_evidence_ids": list(
                        claim_delta.superseded_prior_evidence_ids
                    ),
                    "claim_relations": list(claim_delta.claim_relations),
                },
            }
        )

    evidence_gaps: List[Dict[str, Any]] = []
    if not prior_items:
        evidence_gaps.append(
            {
                "gap": "No prior story anchors were available for comparison.",
                "why_it_matters": "The fallback can identify current stories but cannot measure change from a baseline.",
            }
        )
    if uncertain_count:
        evidence_gaps.append(
            {
                "gap": f"{uncertain_count} story decision(s) require semantic change analysis.",
                "why_it_matters": "Structural evidence establishes neither materiality nor the direction of change.",
            }
        )

    return {
        "baseline_coverage_note": coverage_note,
        "new": new_items,
        "escalated": escalated,
        "weakened": [],
        "reframed": reframed,
        "unchanged_but_important": unchanged,
        "story_decisions": story_decisions,
        "evidence_gaps": evidence_gaps,
        "deterministic_scaffold": True,
        "deterministic_policy_version": DETERMINISTIC_DELTA_POLICY_VERSION,
        "claim_delta_policy_version": CLAIM_DELTA_POLICY_VERSION,
    }


def merge_claim_delta_with_model(
    claim_packet: Dict[str, Any] | None,
    model_packet: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Accept semantic model decisions only when they cite supplied evidence.

    Code remains authoritative for first observations and exact repetition.
    Every other transition is semantic: a backend may propose it, but the
    proposal must cite current and prior claim IDs, select a supplied candidate
    story, and obey conservative visibility constraints. Invalid proposals
    fail open as visible ``uncertain`` decisions.
    """

    claims = dict(claim_packet) if isinstance(claim_packet, dict) else {}
    model = dict(model_packet) if isinstance(model_packet, dict) else {}
    if not claims:
        return model

    model_by_article = _decision_rows_by_article(model.get("story_decisions", []))
    output_rows: List[Dict[str, Any]] = []
    model_used = 0
    model_rejected = 0
    structural_decisions = 0
    awaiting_inference = 0
    for claim_row in claims.get("story_decisions", []):
        if not isinstance(claim_row, dict):
            continue
        article_ids = claim_row.get("article_ids", [])
        article_id = str(article_ids[0] or "") if isinstance(article_ids, list) and article_ids else ""
        claim_delta = claim_row.get("claim_delta", {})
        basis = (
            str(claim_delta.get("decision_basis", "") or "").strip()
            if isinstance(claim_delta, dict) else ""
        )
        is_structural_decision = basis in {"first_observation", "exact_repetition"}
        model_row = model_by_article.get(article_id)
        if is_structural_decision:
            output_rows.append(dict(claim_row))
            structural_decisions += 1
            continue

        request = request_from_claim_delta_payload(claim_delta)
        if model_row is None or request is None:
            retained = dict(claim_row)
            retained["semantic_validation"] = {
                "policy_version": CLAIM_DELTA_POLICY_VERSION,
                "status": "not_attempted" if model_row is None else "rejected",
                "errors": [] if model_row is None else ["claim evidence contract is missing"],
            }
            output_rows.append(retained)
            awaiting_inference += 1
            if model_row is not None:
                model_rejected += 1
            continue

        validated, validation_errors = validate_semantic_decision(model_row, request)
        if validated is None:
            retained = dict(claim_row)
            retained["semantic_validation"] = {
                "policy_version": CLAIM_DELTA_POLICY_VERSION,
                "status": "rejected",
                "errors": validation_errors[:6],
            }
            output_rows.append(retained)
            model_rejected += 1
            awaiting_inference += 1
            continue

        adopted = dict(claim_row)
        adopted.update(
            {
                "prior_story_key": validated.prior_story_key,
                "relationship": validated.relationship,
                "change_type": validated.change_type,
                "materiality": validated.materiality,
                "confidence": validated.confidence,
                "disposition": validated.disposition,
                "summary": validated.summary,
                "reason": str(model_row.get("reason", "") or validated.summary),
            }
        )
        adopted["article_ids"] = [article_id]
        adopted["semantic_validation"] = {
            "policy_version": CLAIM_DELTA_POLICY_VERSION,
            "status": "accepted",
            "current_evidence_ids": list(validated.current_evidence_ids),
            "prior_evidence_ids": list(validated.prior_evidence_ids),
            "superseded_prior_evidence_ids": list(
                validated.superseded_prior_evidence_ids
            ),
            "claim_relations": [relation.payload() for relation in validated.claim_relations],
        }
        current_text = {
            str(item.get("claim_id", "") or ""): str(item.get("text", "") or "")
            for item in claim_delta.get("current_claims", [])
            if isinstance(item, dict)
        }
        prior_text = {
            str(item.get("claim_id", "") or ""): str(item.get("text", "") or "")
            for item in claim_delta.get("prior_claims", [])
            if isinstance(item, dict)
        }
        semantic_claim_delta = dict(claim_delta)
        semantic_claim_delta.update(
            {
                "decision_basis": "semantic_inference",
                "requires_semantic_inference": False,
                "current_evidence_ids": list(validated.current_evidence_ids),
                "prior_evidence_ids": list(validated.prior_evidence_ids),
                "superseded_prior_evidence_ids": list(
                    validated.superseded_prior_evidence_ids
                ),
                "claim_relations": [
                    relation.payload() for relation in validated.claim_relations
                ],
                "superseded_claims": [
                    prior_text[claim_id]
                    for claim_id in validated.superseded_prior_evidence_ids
                    if claim_id in prior_text
                ],
            }
        )
        adopted["claim_delta"] = semantic_claim_delta
        model_knowns = model_row.get("knowns", [])
        adopted["knowns"] = (
            list(model_knowns)
            if isinstance(model_knowns, list) and model_knowns
            else [
                current_text[claim_id]
                for claim_id in validated.current_evidence_ids
                if claim_id in current_text
            ][:6]
        )
        adopted["unknowns"] = list(model_row.get("unknowns", [])) if isinstance(model_row.get("unknowns"), list) else []
        adopted["watch_signals"] = list(model_row.get("watch_signals", [])) if isinstance(model_row.get("watch_signals"), list) else []
        output_rows.append(adopted)
        model_used += 1

    claims["story_decisions"] = output_rows
    claims["semantic_engine"] = {
        "policy_version": CLAIM_DELTA_POLICY_VERSION,
        "model_used_for_uncertain": model_used,
        "model_decisions_rejected": model_rejected,
        "structural_decisions": structural_decisions,
        "awaiting_semantic_inference": awaiting_inference,
    }
    if model_rejected or awaiting_inference:
        gaps = claims.setdefault("evidence_gaps", [])
        if isinstance(gaps, list):
            gaps.append({
                "gap": f"{awaiting_inference} decision(s) lack accepted semantic claim comparison.",
                "why_it_matters": "They remain visible and cannot update the story state as confirmed transitions.",
            })
    return claims


def _decision_rows_by_article(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            continue
        article_ids = row.get("article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for article_id in article_ids:
            key = str(article_id or "").strip()
            if key:
                output[key] = row
    return output


def _story_context_for_article(story_memory: Dict[str, Any] | None, article_id: str) -> Dict[str, Any]:
    stories = story_memory.get("stories", []) if isinstance(story_memory, dict) else []
    if not isinstance(stories, list):
        return {}
    for story in stories:
        if not isinstance(story, dict):
            continue
        article_ids = story.get("current_article_ids", [])
        if isinstance(article_ids, list) and article_id in {str(value or "") for value in article_ids}:
            return story
    return {}
