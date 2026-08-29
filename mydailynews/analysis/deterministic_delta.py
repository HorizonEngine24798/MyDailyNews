from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from mydailynews.app.models import PriorReport, SelectedArticle
from mydailynews.domain.candidate_annotations import candidate_memory_annotation
from mydailynews.domain.text_similarity import compare_token_sets, normalized_word_text, word_tokens
from mydailynews.memory.story_keys import STOPWORDS, story_identity_for_candidate


DETERMINISTIC_DELTA_POLICY_VERSION = "lexical-conservative.v2"


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
    story_decisions: List[Dict[str, Any]] = []
    uncertain_count = 0

    for article in selected:
        title = str(article.candidate.title or "").strip()
        current_tokens = _tokenize_delta_text(title)
        annotation = candidate_memory_annotation(article.candidate)
        identity = story_identity_for_candidate(article.candidate)
        story_key = annotation.story_key if annotation and annotation.story_key else identity.story_key
        _, prior_match = _best_prior_match(current_tokens, prior_items, current_story_key=story_key)
        prior_title = str(prior_match.get("headline", "") or "").strip() if prior_match else ""
        prior_story_key = str(prior_match.get("story_key", "") or "").strip() if prior_match else ""
        assessment = assess_lexical_change(
            title,
            prior_title,
            story_key_match=bool(story_key and prior_story_key and story_key == prior_story_key),
        )
        summary = assessment.reason
        if prior_match:
            prior_date = str(prior_match.get("report_date", "") or "").strip()
            anchor = f"{prior_date}: {prior_title}" if prior_date else prior_title
            summary = f"{assessment.reason} Prior anchor: {anchor}."

        if assessment.change_type == "new":
            new_items.append(_delta_entry(article, summary))
        elif assessment.change_type == "unchanged":
            unchanged.append(_delta_entry(article, summary))
        else:
            uncertain_count += 1

        story_decisions.append(
            {
                "story_key": story_key,
                "article_ids": [str(article.candidate.id)],
                "prior_story_key": prior_story_key,
                "relationship": assessment.relationship,
                "change_type": assessment.change_type,
                "materiality": assessment.materiality,
                "confidence": assessment.confidence,
                "disposition": assessment.disposition,
                "summary": summary,
                "bullet": title,
                "reason": assessment.reason,
                "knowns": [],
                "unknowns": [],
                "watch_signals": [],
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
                "why_it_matters": "Lexical overlap establishes neither materiality nor the direction of change.",
            }
        )

    return {
        "baseline_coverage_note": coverage_note,
        "new": new_items,
        "escalated": [],
        "weakened": [],
        "reframed": [],
        "unchanged_but_important": unchanged,
        "story_decisions": story_decisions,
        "evidence_gaps": evidence_gaps,
        "deterministic_scaffold": True,
        "deterministic_policy_version": DETERMINISTIC_DELTA_POLICY_VERSION,
    }
