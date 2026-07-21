from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date as date_type, timedelta
import json
from pathlib import Path
from typing import Any, Iterable, List

from mydailynews.app.models import SelectedArticle
from mydailynews.domain.candidate_annotations import candidate_memory_annotation


@dataclass(frozen=True)
class CoverageRecord:
    schema_version: int
    date: str
    brief_name: str
    story_key: str
    story_family_key: str
    title: str
    prominence: str
    article_ids: List[str]
    angle: str = ""
    rank_score: float = 0.0


@dataclass(frozen=True)
class CoverageSummary:
    recent_coverage_count: int = 0
    recent_lead_count: int = 0
    covered_yesterday: bool = False
    latest_date: str = ""


class CoverageMemoryStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "CoverageMemoryStore":
        return cls(Path(state_dir) / "coverage_log.jsonl")

    def read_records(self) -> List[CoverageRecord]:
        if not self.path.exists():
            return []
        records: List[CoverageRecord] = []
        for line in self.path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            record = _record_from_payload(raw)
            if record is not None:
                records.append(record)
        return records

    def recent_summary(
        self,
        *,
        story_key: str,
        as_of_date: str,
        window_days: int,
    ) -> CoverageSummary:
        key = str(story_key or "").strip()
        if not key:
            return CoverageSummary()
        as_of = _parse_date(as_of_date)
        if as_of is None:
            return CoverageSummary()
        start = as_of - timedelta(days=max(0, int(window_days)))
        yesterday = as_of - timedelta(days=1)
        count = 0
        lead_count = 0
        covered_yesterday = False
        latest: date_type | None = None
        for record in self.read_records():
            if record.story_key != key:
                continue
            record_date = _parse_date(record.date)
            if record_date is None or record_date >= as_of or record_date < start:
                continue
            count += 1
            if record.prominence == "lead":
                lead_count += 1
            if record_date == yesterday:
                covered_yesterday = True
            if latest is None or record_date > latest:
                latest = record_date
        return CoverageSummary(
            recent_coverage_count=count,
            recent_lead_count=lead_count,
            covered_yesterday=covered_yesterday,
            latest_date=latest.isoformat() if latest else "",
        )

    def recent_records(
        self,
        *,
        story_key: str,
        as_of_date: str,
        window_days: int,
        limit: int = 6,
    ) -> List[CoverageRecord]:
        key = str(story_key or "").strip()
        as_of = _parse_date(as_of_date)
        if not key or as_of is None:
            return []
        start = as_of - timedelta(days=max(0, int(window_days)))
        records = [
            record
            for record in self.read_records()
            if record.story_key == key
            and (record_date := _parse_date(record.date)) is not None
            and start <= record_date < as_of
        ]
        records.sort(key=lambda record: (record.date, record.brief_name), reverse=True)
        return records[: max(0, int(limit))]

    def write_records(self, records: Iterable[CoverageRecord]) -> None:
        incoming = [record for record in records if record.story_key and record.date and record.brief_name]
        if not incoming:
            return
        by_key = {
            (record.date, record.brief_name, record.story_key): record
            for record in self.read_records()
        }
        for record in incoming:
            by_key[(record.date, record.brief_name, record.story_key)] = record
        ordered = sorted(
            by_key.values(),
            key=lambda item: (item.date, item.brief_name, item.story_key),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) for record in ordered]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def prune(self, *, as_of_date: str, retention_days: int) -> int:
        as_of = _parse_date(as_of_date)
        if as_of is None:
            return 0
        cutoff = as_of - timedelta(days=max(0, int(retention_days)))
        kept: List[CoverageRecord] = []
        removed = 0
        for record in self.read_records():
            record_date = _parse_date(record.date)
            if record_date is not None and record_date < cutoff:
                removed += 1
                continue
            kept.append(record)
        if removed <= 0:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if kept:
            lines = [json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) for record in kept]
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            self.path.write_text("", encoding="utf-8")
        return removed

    def write_selected(self, *, date: str, brief_name: str, selected: List[SelectedArticle]) -> List[CoverageRecord]:
        records = coverage_records_for_selected(date=date, brief_name=brief_name, selected=selected)
        self.write_records(records)
        return records


def coverage_records_for_selected(
    *,
    date: str,
    brief_name: str,
    selected: List[SelectedArticle],
) -> List[CoverageRecord]:
    records: List[CoverageRecord] = []
    for index, article in enumerate(selected):
        annotation = candidate_memory_annotation(article.candidate)
        if annotation is None or not annotation.story_key:
            continue
        prominence = "lead" if index == 0 else "body"
        if prominence != "lead" and annotation.today_policy.startswith("capsule"):
            prominence = "capsule"
        rank_score = float(article.selection_rank_score or article.decision.selection_rank_score or 0.0)
        records.append(
            CoverageRecord(
                schema_version=1,
                date=str(date or ""),
                brief_name=str(brief_name or ""),
                story_key=annotation.story_key,
                story_family_key=annotation.story_family_key,
                title=annotation.story_title or article.candidate.title,
                prominence=prominence,
                article_ids=[article.candidate.id],
                angle=annotation.change_type or article.decision.angle_type,
                rank_score=round(rank_score, 4),
            )
        )
    return records


def _record_from_payload(raw: dict[str, Any]) -> CoverageRecord | None:
    story_key = str(raw.get("story_key", "") or "").strip()
    date = str(raw.get("date", "") or "").strip()
    brief_name = str(raw.get("brief_name", "") or "").strip()
    if not story_key or not date or not brief_name:
        return None
    article_ids = raw.get("article_ids", [])
    if not isinstance(article_ids, list):
        article_ids = []
    return CoverageRecord(
        schema_version=int(raw.get("schema_version", 1) or 1),
        date=date,
        brief_name=brief_name,
        story_key=story_key,
        story_family_key=str(raw.get("story_family_key", "") or "").strip(),
        title=str(raw.get("title", "") or "").strip(),
        prominence=str(raw.get("prominence", "body") or "body").strip() or "body",
        article_ids=[str(item) for item in article_ids if str(item).strip()],
        angle=str(raw.get("angle", "") or "").strip(),
        rank_score=_float(raw.get("rank_score"), 0.0),
    )


def _parse_date(value: str) -> date_type | None:
    try:
        return date_type.fromisoformat(str(value or "").strip())
    except ValueError:
        return None


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
