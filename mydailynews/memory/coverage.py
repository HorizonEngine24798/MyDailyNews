from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date as date_type, timedelta
import json
from pathlib import Path
from typing import Any, Iterable, List

from mydailynews.app.models import SelectedArticle
from mydailynews.common.storage import MEMORY_DATABASE_NAME, claim_migration, open_database, set_metadata
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
    def __init__(self, path: Path | str, *, database_path: Path | str | None = None) -> None:
        self.path = Path(path)
        is_database_path = self.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
        self.database_path = Path(database_path) if database_path is not None else (
            self.path if is_database_path else self.path.with_suffix(".sqlite3")
        )
        self._legacy_path = None if self.path == self.database_path else self.path

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "CoverageMemoryStore":
        root = Path(state_dir)
        return cls(root / "coverage_log.jsonl", database_path=root / MEMORY_DATABASE_NAME)

    def read_records(self, *, database: Any | None = None) -> List[CoverageRecord]:
        if database is None:
            self._import_legacy_once()
            with open_database(self.database_path) as opened:
                return self.read_records(database=opened)
        rows = database.execute(
            "SELECT date, brief_name, story_key, payload "
            "FROM coverage ORDER BY date, brief_name, story_key"
        ).fetchall()
        output: List[CoverageRecord] = []
        for row in rows:
            record = _record_from_payload(_json_object(row["payload"]))
            key = (str(row["date"]), str(row["brief_name"]), str(row["story_key"]))
            if record is None or (record.date, record.brief_name, record.story_key) != key:
                raise ValueError(f"SQLite coverage row has an invalid payload: {key}")
            output.append(record)
        return output

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

    def write_records(
        self,
        records: Iterable[CoverageRecord],
        *,
        database: Any | None = None,
    ) -> None:
        incoming = [record for record in records if record.story_key and record.date and record.brief_name]
        if not incoming:
            return
        if database is None:
            self._import_legacy_once()
            with open_database(self.database_path, immediate=True) as opened:
                self.write_records(incoming, database=opened)
            return
        affected = {(record.date, record.brief_name, record.story_key) for record in incoming}
        existing = [
            record
            for record in self.read_records(database=database)
            if (record.date, record.brief_name, record.story_key) in affected
        ]
        merged = _merge_coverage_records([*existing, *incoming])
        database.executemany(
            "INSERT INTO coverage(date, brief_name, story_key, payload) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(date, brief_name, story_key) DO UPDATE SET payload = excluded.payload",
            [_coverage_row(record) for record in merged],
        )

    def replace_records(
        self,
        records: Iterable[CoverageRecord],
        *,
        database: Any | None = None,
    ) -> None:
        output = _merge_coverage_records(
            record
            for record in records
            if record.story_key and record.date and record.brief_name
        )
        if database is None:
            self._import_legacy_once()
            with open_database(self.database_path, immediate=True) as opened:
                self.replace_records(output, database=opened)
            return
        database.execute("DELETE FROM coverage")
        database.executemany(
            "INSERT INTO coverage(date, brief_name, story_key, payload) VALUES (?, ?, ?, ?)",
            [_coverage_row(record) for record in output],
        )
        set_metadata(database, "migration.coverage.v1")

    def archive_records(
        self,
        records: Iterable[CoverageRecord],
        *,
        archived_at: str,
        database: Any | None = None,
    ) -> int:
        output = list(records)
        if not output:
            return 0
        if database is None:
            self._import_legacy_once()
            with open_database(self.database_path, immediate=True) as opened:
                return self.archive_records(output, archived_at=archived_at, database=opened)
        database.executemany(
            "INSERT INTO coverage_archive(archived_at, payload) VALUES (?, ?)",
            [
                (
                    str(archived_at or ""),
                    json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")),
                )
                for record in output
            ],
        )
        return len(output)

    def read_archive_records(self) -> List[dict[str, Any]]:
        self._import_legacy_once()
        with open_database(self.database_path) as database:
            rows = database.execute(
                "SELECT id, archived_at, payload FROM coverage_archive ORDER BY id"
            ).fetchall()
        output: List[dict[str, Any]] = []
        for row in rows:
            payload = _json_object(row["payload"])
            if not payload:
                raise ValueError(f"SQLite coverage archive row has an invalid payload: {row['id']}")
            output.append({**payload, "archived_at": str(row["archived_at"])})
        return output

    def prune(
        self,
        *,
        as_of_date: str,
        retention_days: int,
        database: Any | None = None,
    ) -> int:
        as_of = _parse_date(as_of_date)
        if as_of is None:
            return 0
        if database is None:
            self._import_legacy_once()
            with open_database(self.database_path, immediate=True) as opened:
                return self.prune(
                    as_of_date=as_of_date,
                    retention_days=retention_days,
                    database=opened,
                )
        cutoff = as_of - timedelta(days=max(0, int(retention_days)))
        kept: List[CoverageRecord] = []
        removed = 0
        for record in self.read_records(database=database):
            record_date = _parse_date(record.date)
            if record_date is not None and record_date < cutoff:
                removed += 1
                continue
            kept.append(record)
        if removed <= 0:
            return 0
        self.replace_records(kept, database=database)
        return removed

    def write_selected(
        self,
        *,
        date: str,
        brief_name: str,
        selected: List[SelectedArticle],
        database: Any | None = None,
    ) -> List[CoverageRecord]:
        records = coverage_records_for_selected(date=date, brief_name=brief_name, selected=selected)
        self.write_records(records, database=database)
        return records

    def _import_legacy_once(self) -> None:
        archive_path = self.path.with_name("coverage_log.archive.jsonl")
        with open_database(self.database_path) as database:
            if not claim_migration(database, "migration.coverage.v1"):
                return
            if self._legacy_path is not None and self._legacy_path.exists():
                records = _merge_coverage_records(_read_legacy_records(self._legacy_path))
                database.executemany(
                    "INSERT OR IGNORE INTO coverage(date, brief_name, story_key, payload) VALUES (?, ?, ?, ?)",
                    [_coverage_row(record) for record in records],
                )
            if archive_path.exists():
                database.executemany(
                    "INSERT INTO coverage_archive(archived_at, payload) VALUES (?, ?)",
                    [
                        (
                            str(raw.pop("archived_at", "") or ""),
                            json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                        )
                        for _, raw in _read_jsonl_objects(archive_path)
                    ],
                )
            set_metadata(database, "migration.coverage.v1")


def coverage_records_for_selected(
    *,
    date: str,
    brief_name: str,
    selected: List[SelectedArticle],
) -> List[CoverageRecord]:
    records: List[CoverageRecord] = []
    for article in selected:
        annotation = candidate_memory_annotation(article.candidate)
        if annotation is None or not annotation.story_key:
            continue
        if annotation.today_policy == "omit":
            continue
        prominence = "lead" if not records else "body"
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


def _coverage_row(record: CoverageRecord) -> tuple[str, str, str, str]:
    return (
        record.date,
        record.brief_name,
        record.story_key,
        json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")),
    )


def _merge_coverage_records(records: Iterable[CoverageRecord]) -> List[CoverageRecord]:
    grouped: dict[tuple[str, str, str], List[CoverageRecord]] = {}
    for record in records:
        grouped.setdefault((record.date, record.brief_name, record.story_key), []).append(record)
    output: List[CoverageRecord] = []
    prominence_rank = {"capsule": 0, "body": 1, "lead": 2}
    for key in sorted(grouped):
        items = grouped[key]
        best = max(items, key=lambda item: item.rank_score)
        prominence = max(
            (item.prominence for item in items),
            key=lambda value: prominence_rank.get(value, 1),
        )
        article_ids = list(dict.fromkeys(article_id for item in items for article_id in item.article_ids))
        output.append(
            CoverageRecord(
                schema_version=max(item.schema_version for item in items),
                date=key[0],
                brief_name=key[1],
                story_key=key[2],
                story_family_key=best.story_family_key or next(
                    (item.story_family_key for item in items if item.story_family_key),
                    "",
                ),
                title=best.title or next((item.title for item in items if item.title), ""),
                prominence=prominence,
                article_ids=article_ids,
                angle=best.angle or next((item.angle for item in items if item.angle), ""),
                rank_score=max(item.rank_score for item in items),
            )
        )
    return output


def _read_legacy_records(path: Path) -> List[CoverageRecord]:
    records: List[CoverageRecord] = []
    for line_number, raw in _read_jsonl_objects(path):
        record = _record_from_payload(raw)
        if record is None:
            raise ValueError(
                f"Legacy coverage file has an invalid record at line {line_number}: {path}"
            )
        records.append(record)
    return records


def _read_jsonl_objects(path: Path) -> List[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Could not read legacy coverage file: {path}") from exc
    output: List[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Legacy coverage file has invalid JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(
                f"Legacy coverage file must contain objects; invalid line {line_number}: {path}"
            )
        output.append((line_number, raw))
    return output


def _json_object(value: Any) -> dict[str, Any]:
    try:
        raw = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


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
