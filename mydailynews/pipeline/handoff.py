from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

from mydailynews.app.models import (
    CandidateAnnotations,
    ContextSource,
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    PriorReport,
    ProfileMatchAnnotation,
    SelectedArticle,
    SelectionAnnotation,
    TopicConfig,
)
from mydailynews.domain.article_identity import article_aliases_for_candidate
from mydailynews.pipeline.stage_artifacts import to_jsonable


BRIEF_HANDOFF_SCHEMA_VERSION = "brief_handoff.v1"
STRUCTURED_BRIEF_NAMES = ("general", "detailed")


@dataclass
class HandoffLoadResult:
    path: str
    payload: Dict[str, Any]
    selected_articles: List[SelectedArticle]
    warnings: List[str]


def handoff_path(output_dir: Path | str, date: str, brief_name: str) -> Path:
    return Path(output_dir) / "handoff" / f"{date}_{brief_name}_handoff.json"


def write_brief_handoff(
    *,
    output_dir: Path | str,
    date: str,
    brief_name: str,
    json_path: Path | str,
    markdown_path: Path | str,
    topics: List[TopicConfig],
    prior_reports: List[PriorReport],
    brief_goal: str,
    filtering: FilteringConfig,
    selected_articles: List[SelectedArticle],
) -> Path:
    path = handoff_path(output_dir, date, brief_name)
    payload = {
        "schema_version": BRIEF_HANDOFF_SCHEMA_VERSION,
        "date": date,
        "brief_name": brief_name,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_brief": {
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        },
        "topics": [to_jsonable(asdict(topic)) for topic in topics],
        "prior_reports": [to_jsonable(asdict(report)) for report in prior_reports],
        "brief_goal": str(brief_goal or ""),
        "filtering": to_jsonable(asdict(filtering)),
        "selected_articles": [
            selected_article_to_handoff_payload(
                article,
                source_brief=brief_name,
                source_json_path=str(json_path),
            )
            for article in selected_articles
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_brief_handoff(path: Path | str) -> HandoffLoadResult:
    path_obj = Path(path)
    warnings: List[str] = []
    payload = json.loads(path_obj.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Brief handoff is not a JSON object: {path_obj}")
    schema = str(payload.get("schema_version", "")).strip()
    if schema != BRIEF_HANDOFF_SCHEMA_VERSION:
        raise ValueError(f"Unsupported handoff schema '{schema}' in {path_obj}")
    selected_payloads = payload.get("selected_articles", [])
    if not isinstance(selected_payloads, list):
        raise ValueError(f"Brief handoff selected_articles is not a list: {path_obj}")
    selected: List[SelectedArticle] = []
    for index, raw in enumerate(selected_payloads):
        if not isinstance(raw, dict):
            warnings.append(f"{path_obj}: selected_articles[{index}] is not an object; skipped.")
            continue
        try:
            selected.append(selected_article_from_payload(raw))
        except Exception as exc:
            warnings.append(f"{path_obj}: selected_articles[{index}] failed to load ({type(exc).__name__}: {exc}).")
    return HandoffLoadResult(
        path=str(path_obj),
        payload=payload,
        selected_articles=selected,
        warnings=warnings,
    )


def selected_article_to_handoff_payload(
    article: SelectedArticle,
    *,
    source_brief: str = "",
    source_json_path: str = "",
) -> Dict[str, Any]:
    candidate = to_jsonable(asdict(article.candidate))
    decision = to_jsonable(asdict(article.decision))
    return {
        "candidate": candidate,
        "decision": decision,
        "selection_reason_code": article.selection_reason_code,
        "selection_rank_score": article.selection_rank_score,
        "selection_rank_mode": article.selection_rank_mode,
        "article_text": article.article_text,
        "extraction_status": article.extraction_status,
        "enrichment_needed": article.enrichment_needed,
        "enrichment_reason": article.enrichment_reason,
        "context_sources": [to_jsonable(asdict(source)) for source in article.context_sources],
        "source_trace": {
            "brief_name": source_brief,
            "source_json_path": source_json_path,
        },
    }


def selected_article_from_payload(payload: Dict[str, Any]) -> SelectedArticle:
    candidate = _candidate_from_payload(_require_dict(payload.get("candidate"), "candidate"))
    decision = _decision_from_payload(_require_dict(payload.get("decision"), "decision"), candidate.id)
    article = SelectedArticle(
        candidate=candidate,
        decision=decision,
        selection_reason_code=str(payload.get("selection_reason_code", "") or ""),
        selection_rank_score=_float(payload.get("selection_rank_score"), decision.selection_rank_score),
        selection_rank_mode=str(payload.get("selection_rank_mode", "") or decision.selection_rank_mode or "score"),
        article_text=str(payload.get("article_text", "") or ""),
        extraction_status=str(payload.get("extraction_status", "") or "pending"),
        enrichment_needed=bool(payload.get("enrichment_needed", False)),
        enrichment_reason=str(payload.get("enrichment_reason", "") or ""),
        context_sources=[
            _context_source_from_payload(raw)
            for raw in payload.get("context_sources", [])
            if isinstance(raw, dict)
        ],
    )
    _merge_source_trace(article, payload.get("source_trace"))
    return article


def selected_articles_from_brief_json(
    brief_payload: Dict[str, Any],
    *,
    brief_name: str,
    json_path: Path | str,
    article_text_cache: Any | None = None,
) -> List[SelectedArticle]:
    selected: List[SelectedArticle] = []
    for raw in brief_payload.get("selected_articles", []):
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("id", "") or "").strip()
        headline = str(raw.get("headline", "") or raw.get("title", "") or "").strip()
        url = str(raw.get("url", "") or "").strip()
        source = str(raw.get("source", "") or "").strip()
        if not candidate_id and not (headline or url):
            continue
        if not candidate_id:
            candidate_id = f"brief-{len(selected) + 1}"
        snippet = str(raw.get("snippet", "") or headline)
        candidate = NewsCandidate(
            id=candidate_id,
            source=source,
            category=str(raw.get("category", "") or "brief"),
            title=headline,
            url=url,
            snippet=snippet,
            published_at=_parse_datetime(raw.get("published_at")),
            tags=_string_list(raw.get("tags", [])),
            metadata={
                "topic_name": str(raw.get("topic", "") or ""),
                "rehydrated_from_brief": True,
                "source_json_path": str(json_path),
            },
        )
        cached_text = _article_text_from_cache(article_text_cache, candidate)
        text = cached_text or snippet or headline
        decision = HeadlineDecision(
            candidate_id=candidate.id,
            score=_float(raw.get("score"), 0.0),
            topic=str(raw.get("topic", "") or ""),
            selection_reason_code=str(raw.get("selection_reason_code", "") or ""),
            selection_rank_score=_float(raw.get("selection_rank_score"), _float(raw.get("score"), 0.0)),
            selection_rank_mode=str(raw.get("selection_rank_mode", "") or "score"),
        )
        article = SelectedArticle(
            candidate=candidate,
            decision=decision,
            selection_reason_code=decision.selection_reason_code,
            selection_rank_score=decision.selection_rank_score,
            selection_rank_mode=decision.selection_rank_mode,
            article_text=text,
            extraction_status="ok" if cached_text else "degraded_brief_json",
        )
        _merge_source_trace(article, {"brief_name": brief_name, "source_json_path": str(json_path)})
        selected.append(article)
    return selected


def load_brief_json(path: Path | str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Brief JSON is not an object: {path}")
    return payload


def _candidate_from_payload(payload: Dict[str, Any]) -> NewsCandidate:
    return NewsCandidate(
        id=str(payload.get("id", "") or ""),
        source=str(payload.get("source", "") or ""),
        category=str(payload.get("category", "") or ""),
        title=str(payload.get("title", "") or ""),
        url=str(payload.get("url", "") or ""),
        snippet=str(payload.get("snippet", "") or ""),
        published_at=_parse_datetime(payload.get("published_at")),
        tags=_string_list(payload.get("tags", [])),
        metadata=dict(payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}),
        annotations=_annotations_from_payload(payload.get("annotations")),
    )


def _decision_from_payload(payload: Dict[str, Any], fallback_candidate_id: str) -> HeadlineDecision:
    return HeadlineDecision(
        candidate_id=str(payload.get("candidate_id", "") or fallback_candidate_id),
        score=_float(payload.get("score"), 0.0),
        topic=str(payload.get("topic", "") or ""),
        personal_relevance=_float(payload.get("personal_relevance"), 5.0),
        impact=_float(payload.get("impact"), 5.0),
        novelty=_float(payload.get("novelty"), 5.0),
        urgency=_float(payload.get("urgency"), 5.0),
        actionability=_float(payload.get("actionability"), 5.0),
        confidence=_float(payload.get("confidence"), 5.0),
        reason=str(payload.get("reason", "") or ""),
        skip_reason=payload.get("skip_reason"),
        angle_type=str(payload.get("angle_type", "") or ""),
        selection_reason_code=str(payload.get("selection_reason_code", "") or ""),
        selection_rank_score=_float(payload.get("selection_rank_score"), 0.0),
        selection_rank_mode=str(payload.get("selection_rank_mode", "") or "score"),
    )


def _context_source_from_payload(payload: Dict[str, Any]) -> ContextSource:
    return ContextSource(
        id=str(payload.get("id", "") or ""),
        parent_article_id=str(payload.get("parent_article_id", "") or ""),
        kind=str(payload.get("kind", "") or ""),
        title=str(payload.get("title", "") or ""),
        source=str(payload.get("source", "") or ""),
        url=str(payload.get("url", "") or ""),
        summary=str(payload.get("summary", "") or ""),
        items=list(payload.get("items", []) if isinstance(payload.get("items"), list) else []),
    )


def _annotations_from_payload(value: Any) -> CandidateAnnotations:
    if not isinstance(value, dict):
        return CandidateAnnotations()
    profile = value.get("profile_match")
    selection = value.get("selection")
    return CandidateAnnotations(
        profile_match=_profile_match_from_payload(profile) if isinstance(profile, dict) else None,
        selection=_selection_annotation_from_payload(selection) if isinstance(selection, dict) else None,
    )


def _profile_match_from_payload(value: Dict[str, Any]) -> ProfileMatchAnnotation:
    return ProfileMatchAnnotation(
        source_preferred=bool(value.get("source_preferred", False)),
        source_avoided=bool(value.get("source_avoided", False)),
        geo_match=bool(value.get("geo_match", False)),
        wants_match_count=int(value.get("wants_match_count", 0) or 0),
        avoid_match_count=int(value.get("avoid_match_count", 0) or 0),
        beat_weight_sum=_float(value.get("beat_weight_sum"), 0.0),
        geo_matches=_string_list(value.get("geo_matches", [])),
        wants_matches=_string_list(value.get("wants_matches", [])),
        avoid_matches=_string_list(value.get("avoid_matches", [])),
        beat_matches=_string_list(value.get("beat_matches", [])),
    )


def _selection_annotation_from_payload(value: Dict[str, Any]) -> SelectionAnnotation:
    return SelectionAnnotation(
        reason_code=str(value.get("reason_code", "") or ""),
        skip_reason=str(value.get("skip_reason", "") or ""),
        rank_score=_float(value.get("rank_score"), 0.0),
        rank_mode=str(value.get("rank_mode", "") or "score"),
    )


def _merge_source_trace(article: SelectedArticle, source_trace: Any) -> None:
    if not isinstance(source_trace, dict):
        return
    brief_name = str(source_trace.get("brief_name", "") or "").strip()
    source_json_path = str(source_trace.get("source_json_path", "") or "").strip()
    metadata = article.candidate.metadata
    source_briefs = list(metadata.get("source_briefs", []) if isinstance(metadata.get("source_briefs"), list) else [])
    if brief_name and brief_name not in source_briefs:
        source_briefs.append(brief_name)
    metadata["source_briefs"] = source_briefs
    if source_json_path:
        paths = list(metadata.get("source_json_paths", []) if isinstance(metadata.get("source_json_paths"), list) else [])
        if source_json_path not in paths:
            paths.append(source_json_path)
        metadata["source_json_paths"] = paths


def _article_text_from_cache(article_text_cache: Any | None, candidate: NewsCandidate) -> str:
    if article_text_cache is None:
        return ""
    get_by_aliases = getattr(article_text_cache, "get_by_aliases", None)
    if not callable(get_by_aliases):
        return ""
    record = get_by_aliases(article_aliases_for_candidate(candidate))
    if not isinstance(record, dict):
        return ""
    return str(record.get("article_text", "") or "").strip()


def _require_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
