from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List


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
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "FeedbackStore":
        return cls(Path(state_dir) / "feedback_events.jsonl")

    def read_events(self) -> List[FeedbackEvent]:
        if not self.path.exists():
            return []
        events: List[FeedbackEvent] = []
        for line in self.path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            event = _event_from_payload(raw)
            if event is not None:
                events.append(event)
        return events

    def append_event(self, event: FeedbackEvent) -> FeedbackEvent:
        _validate_action(event.action)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")) + "\n")
        return event

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
