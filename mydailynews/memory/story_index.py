from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as date_type, timedelta
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mydailynews.app.models import MemoryAnnotation, NewsCandidate, SelectedArticle
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.memory.story_keys import (
    StoryIdentity,
    slugify_text,
    story_identity_for_candidate,
    token_overlap_confidence,
)


STORY_INDEX_SCHEMA_VERSION = 2
MATCH_CONFIDENCE_THRESHOLD = 0.58
STORY_STATUSES = {"active", "stale"}
IDENTITY_WEAK_TOKENS = {
    "ai",
    "business",
    "center",
    "centers",
    "data",
    "economy",
    "global",
    "market",
    "markets",
    "model",
    "models",
    "policy",
    "technology",
    "world",
}


@dataclass(frozen=True)
class StoryIndexRecord:
    story_key: str
    story_family_key: str
    title: str
    topic: str
    tokens: List[str]
    first_seen: str
    last_seen: str
    status: str = "active"
    last_material_change_date: str = ""
    last_change_type: str = ""
    last_delta_summary: str = ""
    last_knowns: List[str] = field(default_factory=list)
    last_unknowns: List[str] = field(default_factory=list)
    last_watch_signals: List[str] = field(default_factory=list)
    last_disposition: str = ""
    last_report_id: str = ""


class StoryIndexStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records: List[StoryIndexRecord] | None = None

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "StoryIndexStore":
        return cls(Path(state_dir) / "story_index.json")

    def records(self) -> List[StoryIndexRecord]:
        if self._records is None:
            self._records = self._read_records()
        return list(self._records)

    def match_candidate(self, candidate: NewsCandidate) -> StoryIdentity:
        base = story_identity_for_candidate(candidate)
        best: tuple[float, StoryIndexRecord] | None = None
        for record in self.records():
            confidence = _candidate_record_confidence(base, record)
            if confidence < MATCH_CONFIDENCE_THRESHOLD:
                continue
            if best is None or (confidence, record.last_seen) > (best[0], best[1].last_seen):
                best = (confidence, record)
        if best is None:
            return base
        confidence, record = best
        return StoryIdentity(
            story_key=record.story_key,
            story_family_key=record.story_family_key,
            story_title=record.title,
            tokens=base.tokens,
            match_confidence=confidence,
        )

    def candidate_baselines(self, candidate: NewsCandidate, *, limit: int = 3) -> List[tuple[float, StoryIndexRecord]]:
        """Return likely prior records; semantic identity remains delta's job."""
        base = story_identity_for_candidate(candidate)
        matches: List[tuple[float, StoryIndexRecord]] = []
        for record in self.records():
            confidence = _candidate_record_confidence(base, record)
            if confidence >= MATCH_CONFIDENCE_THRESHOLD:
                matches.append((confidence, record))
        matches.sort(key=lambda item: (item[0], item[1].last_seen, item[1].story_key), reverse=True)
        return matches[: max(0, int(limit))]

    def update_selected(
        self,
        *,
        selected: List[SelectedArticle],
        date: str,
        story_groups: Iterable[Any] | None = None,
        delta_packet: Dict[str, Any] | None = None,
        stale_after_days: int = 7,
        retention_days: int = 30,
    ) -> List[StoryIndexRecord]:
        existing: Dict[str, StoryIndexRecord] = {record.story_key: record for record in self.records()}
        group_by_article = _story_group_by_article_id(story_groups or [])
        decisions = _story_decisions_by_key(delta_packet or {})
        updated: Dict[str, StoryIndexRecord] = dict(existing)

        for article in selected:
            annotation = candidate_memory_annotation(article.candidate)
            if annotation is None or not annotation.story_key:
                continue
            group = group_by_article.get(article.candidate.id)
            if group is not None:
                annotation = _annotation_with_group(annotation, group)
                set_memory_annotation(article.candidate, annotation)
            candidate_identity = story_identity_for_candidate(article.candidate)
            previous = updated.get(annotation.story_key)
            tokens = _merged_tokens(
                previous.tokens if previous else [],
                annotation.story_key.split("-"),
                candidate_identity.tokens,
            )
            title = annotation.story_title or (previous.title if previous else "") or article.candidate.title
            topic = str(article.decision.topic or article.candidate.metadata.get("topic_name", "") or "")
            updated[annotation.story_key] = StoryIndexRecord(
                story_key=annotation.story_key,
                story_family_key=annotation.story_family_key or (previous.story_family_key if previous else ""),
                title=title[:140],
                topic=topic[:120],
                tokens=tokens[:18],
                first_seen=previous.first_seen if previous else str(date or ""),
                last_seen=str(date or ""),
                status="active",
                **_semantic_baseline_fields(previous, decisions.get(annotation.story_key), date),
            )

        records = _refresh_lifecycle_records(
            sorted(updated.values(), key=lambda item: item.story_key),
            as_of_date=date,
            stale_after_days=stale_after_days,
            retention_days=retention_days,
            prune=True,
        )
        self._records = records
        self._write_records(records)
        return records

    def refresh_lifecycle(
        self,
        *,
        as_of_date: str,
        stale_after_days: int = 7,
        retention_days: int = 30,
        prune: bool = False,
    ) -> List[StoryIndexRecord]:
        records = _refresh_lifecycle_records(
            self.records(),
            as_of_date=as_of_date,
            stale_after_days=stale_after_days,
            retention_days=retention_days,
            prune=prune,
        )
        self._records = records
        self._write_records(records)
        return records

    def _read_records(self) -> List[StoryIndexRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        raw_stories = payload.get("stories", [])
        if not isinstance(raw_stories, list):
            return []
        records: List[StoryIndexRecord] = []
        for raw in raw_stories:
            if not isinstance(raw, dict):
                continue
            record = _record_from_payload(raw)
            if record is not None:
                records.append(record)
        return records

    def _write_records(self, records: List[StoryIndexRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STORY_INDEX_SCHEMA_VERSION,
            "stories": [asdict(record) for record in records],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_record_confidence(base: StoryIdentity, record: StoryIndexRecord) -> float:
    if record.story_key == base.story_key:
        return 1.0
    confidence = token_overlap_confidence(base.tokens, record.tokens)
    left = set(base.story_key.split("-")).difference(IDENTITY_WEAK_TOKENS)
    right = set(record.story_key.split("-")).difference(IDENTITY_WEAK_TOKENS)
    overlap = left.intersection(right)
    if len(overlap) >= 2 and len(overlap) / max(1, min(len(left), len(right))) >= 0.4:
        confidence = max(confidence, 0.62)
    return confidence


def _record_from_payload(raw: Dict[str, Any]) -> StoryIndexRecord | None:
    story_key = str(raw.get("story_key", "") or "").strip()
    if not story_key:
        return None
    tokens = raw.get("tokens", [])
    if not isinstance(tokens, list):
        tokens = []
    status = str(raw.get("status", "active") or "active").strip().lower() or "active"
    if status not in STORY_STATUSES:
        status = "active"
    return StoryIndexRecord(
        story_key=story_key,
        story_family_key=str(raw.get("story_family_key", "") or "").strip(),
        title=str(raw.get("title", "") or "").strip(),
        topic=str(raw.get("topic", "") or "").strip(),
        tokens=[str(item) for item in tokens if str(item).strip()][:24],
        first_seen=str(raw.get("first_seen", "") or "").strip(),
        last_seen=str(raw.get("last_seen", "") or "").strip(),
        status=status,
        last_material_change_date=str(raw.get("last_material_change_date", "") or "").strip(),
        last_change_type=str(raw.get("last_change_type", "") or "").strip(),
        last_delta_summary=str(raw.get("last_delta_summary", "") or "").strip()[:400],
        last_knowns=_string_list(raw.get("last_knowns", []), max_items=6, max_chars=180),
        last_unknowns=_string_list(raw.get("last_unknowns", []), max_items=6, max_chars=180),
        last_watch_signals=_string_list(raw.get("last_watch_signals", []), max_items=6, max_chars=180),
        last_disposition=str(raw.get("last_disposition", "") or "").strip(),
        last_report_id=str(raw.get("last_report_id", "") or "").strip(),
    )


def _refresh_lifecycle_records(
    records: List[StoryIndexRecord],
    *,
    as_of_date: str,
    stale_after_days: int,
    retention_days: int,
    prune: bool,
) -> List[StoryIndexRecord]:
    as_of = _parse_date(as_of_date)
    if as_of is None:
        return sorted(records, key=lambda item: item.story_key)
    stale_cutoff = as_of - timedelta(days=max(0, int(stale_after_days)))
    prune_cutoff = as_of - timedelta(days=max(0, int(retention_days)))
    refreshed: List[StoryIndexRecord] = []
    for record in records:
        last_seen = _parse_date(record.last_seen)
        if prune and last_seen is not None and last_seen < prune_cutoff:
            continue
        status = "stale" if last_seen is not None and last_seen < stale_cutoff else "active"
        refreshed.append(
            StoryIndexRecord(
                story_key=record.story_key,
                story_family_key=record.story_family_key,
                title=record.title,
                topic=record.topic,
                tokens=record.tokens,
                first_seen=record.first_seen,
                last_seen=record.last_seen,
                status=status,
                last_material_change_date=record.last_material_change_date,
                last_change_type=record.last_change_type,
                last_delta_summary=record.last_delta_summary,
                last_knowns=record.last_knowns,
                last_unknowns=record.last_unknowns,
                last_watch_signals=record.last_watch_signals,
                last_disposition=record.last_disposition,
                last_report_id=record.last_report_id,
            )
        )
    return sorted(refreshed, key=lambda item: item.story_key)


def _parse_date(value: str) -> date_type | None:
    try:
        return date_type.fromisoformat(str(value or "").strip())
    except ValueError:
        return None


def _story_group_by_article_id(story_groups: Iterable[Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for group in story_groups:
        article_ids = getattr(group, "article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for article_id in article_ids:
            key = str(article_id or "").strip()
            if key and key not in output:
                output[key] = group
    return output


def _annotation_with_group(annotation: MemoryAnnotation, group: Any) -> MemoryAnnotation:
    title = str(getattr(group, "story_title", "") or annotation.story_title).strip()
    topic = str(getattr(group, "topic", "") or "").strip()
    family = annotation.story_family_key
    if topic:
        family = slugify_text(topic, max_tokens=4) or family
    return MemoryAnnotation(
        story_key=annotation.story_key,
        story_family_key=family,
        story_title=title or annotation.story_title,
        match_confidence=annotation.match_confidence,
        recent_coverage_count=annotation.recent_coverage_count,
        recent_lead_count=annotation.recent_lead_count,
        covered_yesterday=annotation.covered_yesterday,
        change_type=annotation.change_type,
        materiality=annotation.materiality,
        score_adjustment=annotation.score_adjustment,
        today_policy=annotation.today_policy,
        reason=annotation.reason,
    )


def _merged_tokens(*groups: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for token in group:
            normalized = str(token or "").strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
    return output


def _story_decisions_by_key(delta_packet: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = delta_packet.get("story_decisions", []) if isinstance(delta_packet, dict) else []
    if not isinstance(rows, list):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("story_key", "") or "").strip()
        if key:
            output[key] = row
    return output


def _semantic_baseline_fields(
    previous: StoryIndexRecord | None,
    decision: Dict[str, Any] | None,
    date: str,
) -> Dict[str, Any]:
    if not decision:
        return {
            "last_material_change_date": previous.last_material_change_date if previous else "",
            "last_change_type": previous.last_change_type if previous else "",
            "last_delta_summary": previous.last_delta_summary if previous else "",
            "last_knowns": list(previous.last_knowns) if previous else [],
            "last_unknowns": list(previous.last_unknowns) if previous else [],
            "last_watch_signals": list(previous.last_watch_signals) if previous else [],
            "last_disposition": previous.last_disposition if previous else "",
            "last_report_id": previous.last_report_id if previous else "",
        }
    change_type = str(decision.get("change_type", "") or "").strip()
    material = change_type in {"new", "escalated", "weakened", "reframed"}
    return {
        "last_material_change_date": str(date or "") if material else (previous.last_material_change_date if previous else ""),
        "last_change_type": change_type or (previous.last_change_type if previous else ""),
        "last_delta_summary": str(decision.get("summary", "") or decision.get("bullet", "") or "").strip()[:400]
        or (previous.last_delta_summary if previous else ""),
        "last_knowns": _string_list(decision.get("knowns", previous.last_knowns if previous else []), max_items=6, max_chars=180),
        "last_unknowns": _string_list(decision.get("unknowns", previous.last_unknowns if previous else []), max_items=6, max_chars=180),
        "last_watch_signals": _string_list(decision.get("watch_signals", previous.last_watch_signals if previous else []), max_items=6, max_chars=180),
        "last_disposition": str(decision.get("disposition", previous.last_disposition if previous else "") or "").strip(),
        "last_report_id": str(decision.get("prior_report_id", previous.last_report_id if previous else "") or "").strip(),
    }


def _string_list(value: Any, *, max_items: int, max_chars: int) -> List[str]:
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
