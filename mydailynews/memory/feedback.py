from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

from mydailynews.common.storage import (
    MEMORY_DATABASE_NAME,
    claim_migration,
    metadata_value,
    open_database,
    set_metadata,
)


FEEDBACK_ACTIONS = (
    "too_repetitive",
    "not_relevant",
    "not_interested_in_topic",
    "more_like_this",
)


@dataclass(frozen=True)
class FeedbackEvent:
    schema_version: int
    created_at: str
    action: str
    report_date: str = ""
    brief_name: str = ""
    article_id: str = ""
    story_key: str = ""
    story_family_key: str = ""
    title: str = ""
    source: str = ""
    topic: str = ""
    notes: str = ""


class FeedbackStore:
    def __init__(self, path: Path | str, *, database_path: Path | str | None = None) -> None:
        self.path = Path(path)
        is_database_path = self.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
        self.database_path = Path(database_path) if database_path is not None else (
            self.path if is_database_path else self.path.with_suffix(".sqlite3")
        )
        self._legacy_path = None if self.path == self.database_path else self.path

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "FeedbackStore":
        root = Path(state_dir)
        return cls(root / "feedback_events.jsonl", database_path=root / MEMORY_DATABASE_NAME)

    def read_events(self) -> List[FeedbackEvent]:
        self._import_legacy_once()
        with open_database(self.database_path) as database:
            rows = database.execute("SELECT id, payload FROM feedback ORDER BY id").fetchall()
        output: List[FeedbackEvent] = []
        for row in rows:
            event = _event_from_payload(_json_object(row["payload"]))
            if event is None:
                raise ValueError(f"SQLite feedback row has an invalid payload: {row['id']}")
            output.append(event)
        return output

    def append_event(self, event: FeedbackEvent) -> FeedbackEvent:
        _validate_action(event.action)
        self._import_legacy_once()
        with open_database(self.database_path) as database:
            database.execute(
                "INSERT INTO feedback(payload) VALUES (?)",
                (json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")),),
            )
        return event

    def replace_events(self, events: List[FeedbackEvent]) -> None:
        for event in events:
            _validate_action(event.action)
        with open_database(self.database_path) as database:
            database.execute("DELETE FROM feedback")
            database.executemany(
                "INSERT INTO feedback(payload) VALUES (?)",
                [
                    (json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")),)
                    for event in events
                ],
            )
            set_metadata(database, "migration.feedback.v1")

    def record(
        self,
        *,
        action: str,
        report_date: str = "",
        brief_name: str = "",
        article_id: str = "",
        story_key: str = "",
        story_family_key: str = "",
        title: str = "",
        source: str = "",
        topic: str = "",
        notes: str = "",
        created_at: str = "",
    ) -> FeedbackEvent:
        event = FeedbackEvent(
            schema_version=1,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            action=_validate_action(action),
            report_date=str(report_date or ""),
            brief_name=str(brief_name or ""),
            article_id=str(article_id or ""),
            story_key=str(story_key or ""),
            story_family_key=str(story_family_key or ""),
            title=str(title or "")[:240],
            source=str(source or "")[:120],
            topic=str(topic or "")[:120],
            notes=str(notes or "")[:500],
        )
        return self.append_event(event)

    def counts_by_action(self) -> Dict[str, int]:
        counts = {action: 0 for action in FEEDBACK_ACTIONS}
        for event in self.read_events():
            counts[event.action] = counts.get(event.action, 0) + 1
        return counts

    def migration_stats(self) -> Dict[str, Any]:
        self._import_legacy_once()
        with open_database(self.database_path) as database:
            return {
                "path": str(self.database_path),
                "rows": int(database.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]),
                "invalid_rows": int(metadata_value(database, "migration.feedback.invalid_rows") or 0),
                "line_numbers": json.loads(
                    metadata_value(database, "migration.feedback.invalid_line_numbers") or "[]"
                ),
            }

    def _import_legacy_once(self) -> None:
        with open_database(self.database_path) as database:
            if not claim_migration(database, "migration.feedback.v1"):
                return
            events: List[FeedbackEvent] = []
            invalid_lines: List[int] = []
            if self._legacy_path is not None and self._legacy_path.exists():
                for line_number, line in enumerate(
                    self._legacy_path.read_text(encoding="utf-8-sig").splitlines(),
                    start=1,
                ):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_lines.append(line_number)
                        continue
                    event = _event_from_payload(raw) if isinstance(raw, dict) else None
                    if event is None:
                        invalid_lines.append(line_number)
                    else:
                        events.append(event)
            if database.execute("SELECT 1 FROM feedback LIMIT 1").fetchone() is None:
                database.executemany(
                    "INSERT INTO feedback(payload) VALUES (?)",
                    [
                        (json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")),)
                        for event in events
                    ],
                )
            set_metadata(database, "migration.feedback.invalid_rows", str(len(invalid_lines)))
            set_metadata(
                database,
                "migration.feedback.invalid_line_numbers",
                json.dumps(invalid_lines[:50], separators=(",", ":")),
            )
            set_metadata(database, "migration.feedback.v1")


def _event_from_payload(raw: Dict[str, Any]) -> FeedbackEvent | None:
    action = str(raw.get("action", "") or "").strip()
    if action not in FEEDBACK_ACTIONS:
        return None
    return FeedbackEvent(
        schema_version=int(raw.get("schema_version", 1) or 1),
        created_at=str(raw.get("created_at", "") or "").strip(),
        action=action,
        report_date=str(raw.get("report_date", "") or "").strip(),
        brief_name=str(raw.get("brief_name", "") or "").strip(),
        article_id=str(raw.get("article_id", "") or "").strip(),
        story_key=str(raw.get("story_key", "") or "").strip(),
        story_family_key=str(raw.get("story_family_key", "") or "").strip(),
        title=str(raw.get("title", "") or "").strip(),
        source=str(raw.get("source", "") or "").strip(),
        topic=str(raw.get("topic", "") or "").strip(),
        notes=str(raw.get("notes", "") or "").strip(),
    )


def _validate_action(action: str) -> str:
    normalized = str(action or "").strip().lower()
    if normalized not in FEEDBACK_ACTIONS:
        allowed = ", ".join(FEEDBACK_ACTIONS)
        raise ValueError(f"Unsupported feedback action '{action}'. Allowed actions: {allowed}")
    return normalized


def _json_object(value: Any) -> Dict[str, Any]:
    try:
        raw = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}
