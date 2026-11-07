from __future__ import annotations

from typing import Any, Dict, Iterable, List

from mydailynews.app.models import PriorReport, SelectedArticle
from mydailynews.common.utils import datetime_to_iso
from mydailynews.domain.candidate_annotations import candidate_memory_annotation
from mydailynews.memory.coverage import CoverageMemoryStore
from mydailynews.memory.story_retrieval import (
    DEFAULT_CANDIDATE_THRESHOLD,
    StoryCandidateMatch,
    retrieve_story_candidates,
    source_signals,
)
from mydailynews.memory.story_store import (
    MATCH_CONFIDENCE_THRESHOLD,
    StoryRecord,
    StoryStore,
    source_facts_for_article,
    story_baseline_payload,
)


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
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
) -> Dict[str, Any]:
    """Build the bounded story-memory view shared by analysis and briefing."""
    by_article_id = {article.candidate.id: article for article in selected}
    groups = list(story_groups or [])
    current_groups: List[tuple[str, str, List[SelectedArticle]]] = []
    claimed: set[str] = set()
    matches_by_article_id: Dict[str, List[StoryCandidateMatch]] = {}
    if story_store is not None:
        for article in selected:
            matches_by_article_id[article.candidate.id] = _article_prior_matches(
                article,
                story_store=story_store,
                limit=max_baselines_per_story,
                candidate_threshold=candidate_threshold,
                as_of_date=date,
            )

    for group in groups:
        article_ids = [str(value or "").strip() for value in getattr(group, "article_ids", [])]
        articles = [by_article_id[item] for item in article_ids if item in by_article_id]
        if not articles:
            continue
        current_groups.extend(
            _partition_story_group(
                group,
                articles,
                matches_by_article_id=matches_by_article_id,
                candidate_threshold=candidate_threshold,
                as_of_date=date,
            )
        )
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
            candidate_threshold=candidate_threshold,
            precomputed_matches=matches_by_article_id,
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
    candidate_threshold: float,
    precomputed_matches: Dict[str, List[StoryCandidateMatch]] | None = None,
) -> Dict[str, Any]:
    representative = articles[0]
    annotation = candidate_memory_annotation(representative.candidate)
    baselines: List[Dict[str, Any]] = []
    if story_store is not None:
        matches = _story_matches(
            articles,
            story_store=story_store,
            limit=max_baselines_per_story,
            candidate_threshold=candidate_threshold,
            as_of_date=date,
            precomputed_matches=precomputed_matches,
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
    # unscored report row cannot bypass the bounded comparison set.
    if story_store is None:
        current_family = annotation.story_family_key if annotation else ""
        prior_baselines = _prior_report_baselines(
            prior_reports,
            {item["story_key"] for item in baselines},
            current_key=key,
            current_family=current_family,
            as_of_date=date,
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


def _article_prior_matches(
    article: SelectedArticle,
    *,
    story_store: StoryStore,
    limit: int,
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
    as_of_date: str = "",
) -> List[StoryCandidateMatch]:
    matches = story_store.candidate_stories(
        article.candidate,
        source_text=article.article_text or article.candidate.snippet,
        limit=max(3, int(limit) * 2),
        min_score=candidate_threshold,
    )
    if as_of_date:
        cutoff = str(as_of_date)[:10]
        matches = [
            match
            for match in matches
            if not str(match.record.last_seen or "")[:10]
            or str(match.record.last_seen or "")[:10] < cutoff
        ]
    article.candidate.metadata["memory_prior_story_candidates"] = [
        match.metadata() for match in matches
    ]
    return matches


def _story_matches(
    articles: List[SelectedArticle],
    *,
    story_store: StoryStore,
    limit: int,
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
    as_of_date: str = "",
    precomputed_matches: Dict[str, List[StoryCandidateMatch]] | None = None,
) -> List[StoryCandidateMatch]:
    by_story_key: Dict[str, StoryCandidateMatch] = {}
    for article in articles:
        if precomputed_matches is not None and article.candidate.id in precomputed_matches:
            matches = precomputed_matches[article.candidate.id]
        else:
            matches = _article_prior_matches(
                article,
                story_store=story_store,
                limit=limit,
                candidate_threshold=candidate_threshold,
                as_of_date=as_of_date,
            )
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


def _partition_story_group(
    group: Any,
    articles: List[SelectedArticle],
    *,
    matches_by_article_id: Dict[str, List[StoryCandidateMatch]],
    candidate_threshold: float,
    as_of_date: str,
) -> List[tuple[str, str, List[SelectedArticle]]]:
    group_title = str(getattr(group, "story_title", "") or "").strip()
    group_topic = str(getattr(group, "topic", "") or "").strip()
    group_story_id = str(getattr(group, "story_id", "") or "").strip()
    if len(articles) <= 1:
        return [(_story_key(articles[0], group_story_id), group_title, articles)]

    clusters: List[List[SelectedArticle]] = []
    for article in articles:
        best_cluster: List[SelectedArticle] | None = None
        best_score = -1.0
        for cluster in clusters:
            cohesion = _cluster_compatibility_score(
                article,
                cluster,
                matches_by_article_id=matches_by_article_id,
                candidate_threshold=candidate_threshold,
                as_of_date=as_of_date,
            )
            if cohesion is not None and cohesion > best_score:
                best_score = cohesion
                best_cluster = cluster
        if best_cluster is not None:
            best_cluster.append(article)
        else:
            clusters.append([article])

    if len(clusters) == 1:
        return [(_story_key(articles[0], group_story_id), group_title, articles)]

    is_generic_topic_title = bool(
        group_title
        and group_topic
        and group_title.casefold() == group_topic.casefold()
    )
    output: List[tuple[str, str, List[SelectedArticle]]] = []
    for cluster in clusters:
        lead = cluster[0]
        cluster_key = _story_key(lead, "")
        if len(cluster) > 1 and lead.candidate.id == articles[0].candidate.id and not is_generic_topic_title:
            cluster_title = group_title
        else:
            cluster_title = ""
        output.append((cluster_key, cluster_title, cluster))
    return output


def _cluster_compatibility_score(
    article: SelectedArticle,
    cluster: List[SelectedArticle],
    *,
    matches_by_article_id: Dict[str, List[StoryCandidateMatch]],
    candidate_threshold: float,
    as_of_date: str,
) -> float | None:
    article_matches = matches_by_article_id.get(article.candidate.id, [])
    article_all_keys = {match.record.story_key for match in article_matches}
    article_strong_keys = {
        match.record.story_key
        for match in article_matches
        if float(match.score) >= MATCH_CONFIDENCE_THRESHOLD
    }
    min_cohesion = max(0.20, float(candidate_threshold or DEFAULT_CANDIDATE_THRESHOLD))
    best_pair_score = 0.0

    for member in cluster:
        member_matches = matches_by_article_id.get(member.candidate.id, [])
        member_all_keys = {match.record.story_key for match in member_matches}
        member_strong_keys = {
            match.record.story_key
            for match in member_matches
            if float(match.score) >= MATCH_CONFIDENCE_THRESHOLD
        }
        if (
            article_strong_keys
            and member_strong_keys
            and article_all_keys.isdisjoint(member_all_keys)
        ):
            return None

        if _story_key(article, "") == _story_key(member, ""):
            best_pair_score = max(best_pair_score, 1.0)
            continue

        shared_prior_keys = article_all_keys & member_all_keys
        shared_prior_score = max(
            (
                min(float(am.score), float(mm.score))
                for am in article_matches
                for mm in member_matches
                if am.record.story_key in shared_prior_keys
            ),
            default=0.0,
        )
        direct_score = _pairwise_article_story_score(
            article,
            member,
            as_of_date=as_of_date,
        )
        unshared_strong_baseline = bool(
            (article_strong_keys and not (article_strong_keys & member_all_keys))
            or (member_strong_keys and not (member_strong_keys & article_all_keys))
        )
        required_score = (
            MATCH_CONFIDENCE_THRESHOLD
            if unshared_strong_baseline and not shared_prior_keys
            else min_cohesion
        )
        pair_score = max(shared_prior_score, direct_score)
        if pair_score >= required_score:
            best_pair_score = max(best_pair_score, pair_score)

    return best_pair_score if best_pair_score > 0.0 else None


def _pairwise_article_story_score(
    left: SelectedArticle,
    right: SelectedArticle,
    *,
    as_of_date: str = "",
) -> float:
    left_record = _transient_story_record(left, as_of_date=as_of_date)
    right_record = _transient_story_record(right, as_of_date=as_of_date)
    forward = retrieve_story_candidates(
        right.candidate,
        [left_record],
        source_text=right.article_text or right.candidate.snippet,
        limit=1,
        min_score=0.0,
    )
    backward = retrieve_story_candidates(
        left.candidate,
        [right_record],
        source_text=left.article_text or left.candidate.snippet,
        limit=1,
        min_score=0.0,
    )
    forward_score = float(forward[0].score) if forward else 0.0
    backward_score = float(backward[0].score) if backward else 0.0
    return max(forward_score, backward_score)


def _transient_story_record(article: SelectedArticle, *, as_of_date: str = "") -> StoryRecord:
    candidate = article.candidate
    annotation = candidate_memory_annotation(candidate)
    source_text = article.article_text or candidate.snippet
    signals = source_signals(candidate, source_text=source_text)
    facts = source_facts_for_article(
        article,
        observed_at=as_of_date,
        user_visible=False,
    )
    key = _story_key(article, "")
    title = (annotation.story_title if annotation and annotation.story_title else "") or candidate.title
    aliases = [candidate.title]
    if title and title not in aliases:
        aliases.append(title)
    return StoryRecord(
        story_key=key,
        story_family_key=annotation.story_family_key if annotation else "",
        title=str(title or "")[:180],
        topic=str(article.decision.topic or candidate.metadata.get("topic_name", "") or "")[:120],
        tokens=list(signals.all_tokens[:24]),
        aliases=aliases,
        entity_tokens=list(signals.entity_tokens[:32]),
        event_tokens=list(signals.event_tokens[:40]),
        number_tokens=list(signals.number_tokens[:20]),
        first_seen=str(as_of_date or ""),
        last_seen=str(as_of_date or ""),
        status="active",
        source_document_ids=[str(candidate.id)],
        facts=facts,
        active_fact_ids=[fact.fact_id for fact in facts],
    )


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
    as_of_date: str = "",
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for report in prior_reports:
        if as_of_date and str(report.date or "")[:10] >= str(as_of_date)[:10]:
            continue
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
