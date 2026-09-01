from __future__ import annotations

from typing import Any, Dict, Iterable, List

from mydailynews.app.models import PriorReport, SelectedArticle
from mydailynews.common.utils import datetime_to_iso
from mydailynews.domain.candidate_annotations import candidate_memory_annotation
from mydailynews.memory.coverage import CoverageMemoryStore
from mydailynews.memory.story_retrieval import StoryCandidateMatch
from mydailynews.memory.story_reranker import StoryCandidateReranker, rerank_story_candidates
from mydailynews.memory.story_store import StoryStore, story_baseline_payload


def build_story_memory_context(
    *,
    selected: List[SelectedArticle],
    story_groups: Iterable[Any] | None,
    story_store: StoryStore | None,
    coverage_store: CoverageMemoryStore | None,
    prior_reports: List[PriorReport],
    date: str,
    coverage_window_days: int = 10,
    max_baselines_per_story: int = 3,
    candidate_reranker: StoryCandidateReranker | None = None,
    reranker_acceptance_threshold: float = 0.5,
    reranker_hard_rejection: bool = False,
) -> Dict[str, Any]:
    """Build the bounded story-memory view shared by analysis and briefing."""
    by_article_id = {article.candidate.id: article for article in selected}
    groups = list(story_groups or [])
    current_groups: List[tuple[str, str, List[SelectedArticle]]] = []
    claimed: set[str] = set()

    for group in groups:
        article_ids = [str(value or "").strip() for value in getattr(group, "article_ids", [])]
        articles = [by_article_id[item] for item in article_ids if item in by_article_id]
        if not articles:
            continue
        key = _story_key(articles[0], getattr(group, "story_id", ""))
        current_groups.append((key, str(getattr(group, "story_title", "") or ""), articles))
        claimed.update(article.candidate.id for article in articles)

    for article in selected:
        if article.candidate.id not in claimed:
            current_groups.append((_story_key(article, ""), "", [article]))

    stories = [
        _story_context(
            key=key,
            group_title=group_title,
            articles=articles,
            story_store=story_store,
            coverage_store=coverage_store,
            prior_reports=prior_reports,
            date=date,
            coverage_window_days=coverage_window_days,
            max_baselines_per_story=max_baselines_per_story,
            candidate_reranker=candidate_reranker,
            reranker_acceptance_threshold=reranker_acceptance_threshold,
            reranker_hard_rejection=reranker_hard_rejection,
        )
        for key, group_title, articles in current_groups
    ]
    return {
        "schema_version": 2,
        "as_of_date": str(date or ""),
        "coverage_window_days": max(0, int(coverage_window_days)),
        "stories": stories,
    }


def _story_context(
    *,
    key: str,
    group_title: str,
    articles: List[SelectedArticle],
    story_store: StoryStore | None,
    coverage_store: CoverageMemoryStore | None,
    prior_reports: List[PriorReport],
    date: str,
    coverage_window_days: int,
    max_baselines_per_story: int,
    candidate_reranker: StoryCandidateReranker | None,
    reranker_acceptance_threshold: float,
    reranker_hard_rejection: bool,
) -> Dict[str, Any]:
    representative = articles[0]
    annotation = candidate_memory_annotation(representative.candidate)
    baselines: List[Dict[str, Any]] = []
    if story_store is not None:
        matches = _story_matches(
            articles,
            story_store=story_store,
            limit=max_baselines_per_story,
            reranker=candidate_reranker,
            reranker_acceptance_threshold=reranker_acceptance_threshold,
            reranker_hard_rejection=reranker_hard_rejection,
        )
        for match in matches:
            coverage = (
                coverage_store.recent_records(
                    story_key=match.record.story_key,
                    as_of_date=date,
                    window_days=coverage_window_days,
                    limit=4,
                )
                if coverage_store is not None
                else []
            )
            baselines.append(
                {
                    **story_baseline_payload(match, max_facts=4),
                    "recent_coverage": [_coverage_payload(item) for item in coverage],
                }
            )

    # Prior-report rows are a memory-disabled fallback only. When StoryStore is
    # active every baseline must be an explicitly retrieved candidate so an
    # unscored report row cannot bypass the identity gate.
    if story_store is None:
        current_family = annotation.story_family_key if annotation else ""
        prior_baselines = _prior_report_baselines(
            prior_reports,
            {item["story_key"] for item in baselines},
            current_key=key,
            current_family=current_family,
        )
        if prior_baselines:
            for baseline in prior_baselines:
                if not any(item.get("story_key") == baseline.get("story_key") for item in baselines):
                    baselines.append(baseline)
                elif baseline.get("last_delta_summary"):
                    for item in baselines:
                        if item.get("story_key") == baseline.get("story_key") and not item.get("last_delta_summary"):
                            item.update({field: value for field, value in baseline.items() if value})

    return {
        "story_key": key,
        "story_family_key": annotation.story_family_key if annotation else "",
        "current_title": group_title or (annotation.story_title if annotation else "") or representative.candidate.title,
        "current_article_ids": [article.candidate.id for article in articles],
        "current_articles": [_article_payload(article) for article in articles[:8]],
        "current_memory": _annotation_payload(annotation),
        "prior_baselines": baselines[: max(0, int(max_baselines_per_story))],
    }


def _story_matches(
    articles: List[SelectedArticle],
    *,
    story_store: StoryStore,
    limit: int,
    reranker: StoryCandidateReranker | None = None,
    reranker_acceptance_threshold: float = 0.5,
    reranker_hard_rejection: bool = False,
) -> List[StoryCandidateMatch]:
    by_story_key: Dict[str, StoryCandidateMatch] = {}
    for article in articles:
        matches = story_store.candidate_stories(
            article.candidate,
            source_text=article.article_text or article.candidate.snippet,
            limit=limit,
        )
        matches = rerank_story_candidates(
            article.candidate,
            matches,
            reranker,
            source_text=article.article_text or article.candidate.snippet,
            acceptance_threshold=reranker_acceptance_threshold,
            reject_below_threshold=reranker_hard_rejection,
        )
        article.candidate.metadata["memory_prior_story_candidates"] = [
            match.metadata() for match in matches
        ]
        for match in matches:
            previous = by_story_key.get(match.record.story_key)
            if previous is None or match.score > previous.score:
                by_story_key[match.record.story_key] = match
    output = sorted(
        by_story_key.values(),
        key=lambda match: (match.score, match.record.last_seen, match.record.story_key),
        reverse=True,
    )
    return output[: max(0, min(3, int(limit)))]


def _coverage_payload(item: Any) -> Dict[str, Any]:
    return {
        "date": item.date,
        "brief_name": item.brief_name,
        "prominence": item.prominence,
        "title": item.title[:140],
        "article_ids": item.article_ids[:4],
    }


def _article_payload(article: SelectedArticle) -> Dict[str, Any]:
    annotation = candidate_memory_annotation(article.candidate)
    return {
        "id": article.candidate.id,
        "headline": article.candidate.title[:180],
        "source": article.candidate.source[:100],
        "published_at": datetime_to_iso(article.candidate.published_at),
        "score": round(float(article.decision.score), 4),
        "excerpt": (article.article_text or article.candidate.snippet)[:420],
        "memory": _annotation_payload(annotation),
    }


def _annotation_payload(annotation: Any) -> Dict[str, Any]:
    if annotation is None:
        return {}
    return {
        "story_key": annotation.story_key,
        "story_family_key": annotation.story_family_key,
        "story_title": annotation.story_title,
        "match_confidence": round(float(annotation.match_confidence), 4),
        "recent_coverage_count": int(annotation.recent_coverage_count),
        "recent_lead_count": int(annotation.recent_lead_count),
        "covered_yesterday": bool(annotation.covered_yesterday),
        "change_type": annotation.change_type,
        "materiality": round(float(annotation.materiality), 4),
        "today_policy": annotation.today_policy,
        "reason": annotation.reason,
    }


def _prior_report_baselines(
    prior_reports: List[PriorReport],
    known_keys: set[str],
    *,
    current_key: str,
    current_family: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for report in prior_reports:
        rows = getattr(report, "story_baselines", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("story_key", "") or "").strip()
            family = str(row.get("story_family_key", "") or "").strip()
            if not key or key in known_keys or (key != current_key and (not current_family or family != current_family)):
                continue
            output.append(
                {
                    "story_key": key,
                    "story_family_key": str(row.get("story_family_key", "") or "").strip(),
                    "title": str(row.get("title", "") or "").strip()[:140],
                    "last_seen": report.date,
                    "last_material_change_date": str(row.get("last_material_change_date", "") or "").strip(),
                    "last_change_type": str(row.get("change_type", "") or "").strip(),
                    "last_delta_summary": str(row.get("summary", "") or row.get("bullet", "") or "").strip()[:400],
                    "knowns": _string_list(row.get("knowns", []), 6, 180),
                    "unknowns": _string_list(row.get("unknowns", []), 6, 180),
                    "watch_signals": _string_list(row.get("watch_signals", []), 6, 180),
                    "last_disposition": str(row.get("disposition", "") or "").strip(),
                    "last_report_id": report.id,
                    "lexical_match_confidence": 0.0,
                    "recent_coverage": [],
                }
            )
    return output


def _string_list(value: Any, max_items: int, max_chars: int) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = " ".join(str(item or "").split()).strip()[:max_chars]
        if text and text not in output:
            output.append(text)
        if len(output) >= max_items:
            break
    return output


def _story_key(article: SelectedArticle, fallback: str) -> str:
    annotation = candidate_memory_annotation(article.candidate)
    return (annotation.story_key if annotation and annotation.story_key else str(fallback or "").strip()) or f"article:{article.candidate.id}"
