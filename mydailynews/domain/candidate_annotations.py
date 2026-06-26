from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mydailynews.common.booleans import parse_bool
from mydailynews.app.models import (
    CandidateAnnotations,
    NewsCandidate,
    ProfileMatchAnnotation,
    SelectionAnnotation,
)


PROFILE_MATCH_METADATA_KEYS = (
    "user_source_preferred",
    "user_source_avoided",
    "user_geo_match",
    "user_wants_match_count",
    "user_avoid_match_count",
    "user_beats_match_weight",
    "user_geo_matches",
    "user_wants_matches",
    "user_avoid_matches",
)
SELECTION_METADATA_KEYS = (
    "selection_reason_code",
    "selection_skip_reason",
    "selection_rank_score",
    "selection_rank_mode",
)


def candidate_annotations(candidate: NewsCandidate) -> CandidateAnnotations:
    annotations = getattr(candidate, "annotations", None)
    if isinstance(annotations, CandidateAnnotations):
        return annotations
    normalized = CandidateAnnotations()
    if isinstance(annotations, Mapping):
        normalized.profile_match = _profile_match_annotation_from_raw(annotations.get("profile_match"))
        normalized.selection = _selection_annotation_from_raw(annotations.get("selection"))
    setattr(candidate, "annotations", normalized)
    return normalized


def candidate_profile_match_annotation(candidate: NewsCandidate) -> ProfileMatchAnnotation | None:
    typed = candidate_annotations(candidate).profile_match
    if typed is not None:
        return _normalize_profile_match_annotation(typed)
    return profile_match_annotation_from_metadata(candidate.metadata)


def profile_match_annotation_from_metadata(metadata: Mapping[str, Any]) -> ProfileMatchAnnotation | None:
    if not any(key in metadata for key in PROFILE_MATCH_METADATA_KEYS):
        return None
    geo_matches = _string_list(metadata.get("user_geo_matches", []), max_items=16)
    wants_matches = _string_list(metadata.get("user_wants_matches", []), max_items=16)
    avoid_matches = _string_list(metadata.get("user_avoid_matches", []), max_items=16)
    return ProfileMatchAnnotation(
        source_preferred=_to_bool(metadata.get("user_source_preferred", False)),
        source_avoided=_to_bool(metadata.get("user_source_avoided", False)),
        geo_match=_to_bool(metadata.get("user_geo_match", bool(geo_matches))),
        wants_match_count=max(0, _to_int(metadata.get("user_wants_match_count", len(wants_matches)), 0)),
        avoid_match_count=max(0, _to_int(metadata.get("user_avoid_match_count", len(avoid_matches)), 0)),
        beat_weight_sum=round(_to_float(metadata.get("user_beats_match_weight", 0.0), 0.0), 3),
        geo_matches=geo_matches,
        wants_matches=wants_matches,
        avoid_matches=avoid_matches,
    )


def set_profile_match_annotation(
    candidate: NewsCandidate,
    *,
    source_preferred: bool,
    source_avoided: bool,
    geo_matches: Sequence[Any],
    wants_matches: Sequence[Any],
    avoid_matches: Sequence[Any],
    beat_matches: Sequence[Any] | None = None,
    beat_weight_sum: float = 0.0,
) -> ProfileMatchAnnotation:
    geo_values = _string_list(geo_matches, max_items=16)
    wants_values = _string_list(wants_matches, max_items=16)
    avoid_values = _string_list(avoid_matches, max_items=16)
    beat_values = _string_list(beat_matches or [], max_items=16)
    annotation = ProfileMatchAnnotation(
        source_preferred=parse_bool(source_preferred, default=False, field_name="profile_match.source_preferred"),
        source_avoided=parse_bool(source_avoided, default=False, field_name="profile_match.source_avoided"),
        geo_match=bool(geo_values),
        wants_match_count=len(wants_values),
        avoid_match_count=len(avoid_values),
        beat_weight_sum=round(float(beat_weight_sum or 0.0), 3),
        geo_matches=geo_values,
        wants_matches=wants_values,
        avoid_matches=avoid_values,
        beat_matches=beat_values,
    )
    candidate_annotations(candidate).profile_match = annotation
    candidate.metadata["user_source_preferred"] = annotation.source_preferred
    candidate.metadata["user_source_avoided"] = annotation.source_avoided
    candidate.metadata["user_geo_match"] = annotation.geo_match
    candidate.metadata["user_wants_match_count"] = annotation.wants_match_count
    candidate.metadata["user_avoid_match_count"] = annotation.avoid_match_count
    candidate.metadata["user_beats_match_weight"] = annotation.beat_weight_sum
    _write_optional_list(candidate.metadata, "user_geo_matches", annotation.geo_matches[:3])
    _write_optional_list(candidate.metadata, "user_wants_matches", annotation.wants_matches[:4])
    _write_optional_list(candidate.metadata, "user_avoid_matches", annotation.avoid_matches[:4])
    return annotation


def candidate_selection_annotation(candidate: NewsCandidate) -> SelectionAnnotation | None:
    typed = candidate_annotations(candidate).selection
    if typed is not None:
        return _normalize_selection_annotation(typed)
    return selection_annotation_from_metadata(candidate.metadata)


def selection_annotation_from_metadata(metadata: Mapping[str, Any]) -> SelectionAnnotation | None:
    if not any(key in metadata for key in SELECTION_METADATA_KEYS):
        return None
    return SelectionAnnotation(
        reason_code=str(metadata.get("selection_reason_code", "") or "").strip(),
        skip_reason=str(metadata.get("selection_skip_reason", "") or "").strip(),
        rank_score=_to_float(metadata.get("selection_rank_score", 0.0), 0.0),
        rank_mode=str(metadata.get("selection_rank_mode", "score") or "score").strip() or "score",
    )


def reset_selection_annotation(candidate: NewsCandidate, *, rank_score: float, rank_mode: str) -> SelectionAnnotation:
    annotation = SelectionAnnotation(
        rank_score=round(float(rank_score or 0.0), 4),
        rank_mode=str(rank_mode or "score").strip() or "score",
    )
    candidate_annotations(candidate).selection = annotation
    candidate.metadata["selection_rank_score"] = annotation.rank_score
    candidate.metadata["selection_rank_mode"] = annotation.rank_mode
    candidate.metadata.pop("selection_reason_code", None)
    candidate.metadata.pop("selection_skip_reason", None)
    return annotation


def set_selection_skip_annotation(candidate: NewsCandidate, code: str) -> SelectionAnnotation:
    current = candidate_selection_annotation(candidate) or SelectionAnnotation()
    annotation = SelectionAnnotation(
        reason_code=str(code or "").strip(),
        skip_reason=str(code or "").strip(),
        rank_score=float(current.rank_score or 0.0),
        rank_mode=str(current.rank_mode or "score") or "score",
    )
    candidate_annotations(candidate).selection = annotation
    candidate.metadata["selection_skip_reason"] = annotation.skip_reason
    candidate.metadata.pop("selection_reason_code", None)
    return annotation


def set_selection_selected_annotation(candidate: NewsCandidate, code: str) -> SelectionAnnotation:
    current = candidate_selection_annotation(candidate) or SelectionAnnotation()
    annotation = SelectionAnnotation(
        reason_code=str(code or "").strip(),
        skip_reason="",
        rank_score=float(current.rank_score or 0.0),
        rank_mode=str(current.rank_mode or "score") or "score",
    )
    candidate_annotations(candidate).selection = annotation
    candidate.metadata["selection_reason_code"] = annotation.reason_code
    candidate.metadata.pop("selection_skip_reason", None)
    return annotation


def _normalize_profile_match_annotation(annotation: ProfileMatchAnnotation) -> ProfileMatchAnnotation:
    return ProfileMatchAnnotation(
        source_preferred=parse_bool(annotation.source_preferred, default=False, field_name="profile_match.source_preferred"),
        source_avoided=parse_bool(annotation.source_avoided, default=False, field_name="profile_match.source_avoided"),
        geo_match=parse_bool(annotation.geo_match, default=False, field_name="profile_match.geo_match"),
        wants_match_count=max(0, _to_int(annotation.wants_match_count, 0)),
        avoid_match_count=max(0, _to_int(annotation.avoid_match_count, 0)),
        beat_weight_sum=round(_to_float(annotation.beat_weight_sum, 0.0), 3),
        geo_matches=_string_list(annotation.geo_matches, max_items=16),
        wants_matches=_string_list(annotation.wants_matches, max_items=16),
        avoid_matches=_string_list(annotation.avoid_matches, max_items=16),
        beat_matches=_string_list(annotation.beat_matches, max_items=16),
    )


def _normalize_selection_annotation(annotation: SelectionAnnotation) -> SelectionAnnotation:
    return SelectionAnnotation(
        reason_code=str(annotation.reason_code or "").strip(),
        skip_reason=str(annotation.skip_reason or "").strip(),
        rank_score=_to_float(annotation.rank_score, 0.0),
        rank_mode=str(annotation.rank_mode or "score").strip() or "score",
    )


def _profile_match_annotation_from_raw(value: Any) -> ProfileMatchAnnotation | None:
    if isinstance(value, ProfileMatchAnnotation):
        return _normalize_profile_match_annotation(value)
    if not isinstance(value, Mapping):
        return None
    return ProfileMatchAnnotation(
        source_preferred=_to_bool(value.get("source_preferred", False)),
        source_avoided=_to_bool(value.get("source_avoided", False)),
        geo_match=_to_bool(value.get("geo_match", False)),
        wants_match_count=max(0, _to_int(value.get("wants_match_count", 0), 0)),
        avoid_match_count=max(0, _to_int(value.get("avoid_match_count", 0), 0)),
        beat_weight_sum=round(_to_float(value.get("beat_weight_sum", 0.0), 0.0), 3),
        geo_matches=_string_list(value.get("geo_matches", []), max_items=16),
        wants_matches=_string_list(value.get("wants_matches", []), max_items=16),
        avoid_matches=_string_list(value.get("avoid_matches", []), max_items=16),
        beat_matches=_string_list(value.get("beat_matches", []), max_items=16),
    )


def _selection_annotation_from_raw(value: Any) -> SelectionAnnotation | None:
    if isinstance(value, SelectionAnnotation):
        return _normalize_selection_annotation(value)
    if not isinstance(value, Mapping):
        return None
    return SelectionAnnotation(
        reason_code=str(value.get("reason_code", "") or "").strip(),
        skip_reason=str(value.get("skip_reason", "") or "").strip(),
        rank_score=_to_float(value.get("rank_score", 0.0), 0.0),
        rank_mode=str(value.get("rank_mode", "score") or "score").strip() or "score",
    )


def _string_list(values: Any, *, max_items: int) -> list[str]:
    if isinstance(values, str):
        iterable = [values]
    elif isinstance(values, Sequence):
        iterable = list(values)
    else:
        return []
    output: list[str] = []
    for value in iterable:
        text = str(value or "").strip()
        if not text:
            continue
        output.append(text)
        if len(output) >= max_items:
            break
    return output


def _write_optional_list(metadata: dict[str, Any], key: str, values: list[str]) -> None:
    if values:
        metadata[key] = values
        return
    metadata.pop(key, None)


def _to_bool(value: Any) -> bool:
    return parse_bool(value, default=False, field_name="metadata boolean")


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
