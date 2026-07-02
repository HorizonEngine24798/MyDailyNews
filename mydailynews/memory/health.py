from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, List

from mydailynews.memory.feedback import FEEDBACK_ACTIONS


def memory_health_checks(
    *,
    state_dir: Path | str,
    story_index: Iterable[Any],
    coverage_records: Iterable[Any],
    feedback_events: Iterable[Any],
) -> dict[str, Any]:
    stories = list(story_index)
    coverage = list(coverage_records)
    feedback = list(feedback_events)

    story_keys = [_field(record, "story_key") for record in stories]
    story_keys = [key for key in story_keys if key]
    story_key_counts = Counter(story_keys)
    duplicate_keys = sorted(key for key, count in story_key_counts.items() if count > 1)
    stories_missing_last_seen = sorted(
        _field(record, "story_key")
        for record in stories
        if _field(record, "story_key") and not _field(record, "last_seen")
    )

    indexed_story_keys = set(story_keys)
    coverage_without_story = sorted(
        {
            _field(record, "story_key")
            for record in coverage
            if _field(record, "story_key") and _field(record, "story_key") not in indexed_story_keys
        }
    )

    feedback_without_identity = [
        _feedback_label(event)
        for event in feedback
        if not any(_field(event, key) for key in ("article_id", "story_key", "source", "topic"))
    ]

    feedback_stats = feedback_jsonl_stats(Path(state_dir) / "feedback_events.jsonl")
    warnings: list[dict[str, Any]] = []
    if feedback_stats["invalid_rows"]:
        warnings.append(
            _warning(
                "invalid_feedback_jsonl_rows",
                feedback_stats["invalid_rows"],
                _plural(
                    feedback_stats["invalid_rows"],
                    "feedback event row could not be read and was skipped",
                    "feedback event rows could not be read and were skipped",
                ),
                line_numbers=feedback_stats["line_numbers"],
            )
        )
    if duplicate_keys:
        warnings.append(
            _warning(
                "duplicate_story_keys",
                len(duplicate_keys),
                _plural(
                    len(duplicate_keys),
                    "story key appears more than once",
                    "story keys appear more than once",
                ),
                story_keys=duplicate_keys[:20],
            )
        )
    if stories_missing_last_seen:
        warnings.append(
            _warning(
                "story_records_missing_last_seen",
                len(stories_missing_last_seen),
                _plural(
                    len(stories_missing_last_seen),
                    "story record has no last_seen date",
                    "story records have no last_seen date",
                ),
                story_keys=stories_missing_last_seen[:20],
            )
        )
    if coverage_without_story:
        warnings.append(
            _warning(
                "coverage_story_key_missing_from_index",
                len(coverage_without_story),
                _plural(
                    len(coverage_without_story),
                    "coverage story key is absent from the story index",
                    "coverage story keys are absent from the story index",
                ),
                story_keys=coverage_without_story[:20],
            )
        )
    if feedback_without_identity:
        warnings.append(
            _warning(
                "feedback_rows_missing_identity",
                len(feedback_without_identity),
                _plural(
                    len(feedback_without_identity),
                    "feedback event has no article, story, source, or topic identity",
                    "feedback events have no article, story, source, or topic identity",
                ),
                events=feedback_without_identity[:20],
            )
        )

    return {
        "ok": not warnings,
        "warnings": warnings,
        "counts": {
            "invalid_feedback_rows": int(feedback_stats["invalid_rows"]),
            "duplicate_story_keys": len(duplicate_keys),
            "story_records_missing_last_seen": len(stories_missing_last_seen),
            "coverage_story_key_missing_from_index": len(coverage_without_story),
            "feedback_rows_missing_identity": len(feedback_without_identity),
        },
    }


def feedback_jsonl_stats(path: Path | str) -> dict[str, Any]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return {"path": str(jsonl_path), "rows": 0, "invalid_rows": 0, "line_numbers": []}
    rows = 0
    invalid_rows = 0
    line_numbers: List[int] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        rows += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            invalid_rows += 1
            line_numbers.append(line_number)
            continue
        if not isinstance(raw, dict) or str(raw.get("action", "") or "").strip() not in FEEDBACK_ACTIONS:
            invalid_rows += 1
            line_numbers.append(line_number)
    return {
        "path": str(jsonl_path),
        "rows": rows,
        "invalid_rows": invalid_rows,
        "line_numbers": line_numbers[:50],
    }


def _warning(code: str, count: int, message: str, **details: Any) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "warning",
        "count": int(count),
        "message": message,
        **details,
    }


def _field(record: Any, name: str) -> str:
    if isinstance(record, dict):
        value = record.get(name, "")
    else:
        value = getattr(record, name, "")
    return str(value or "").strip()


def _feedback_label(event: Any) -> dict[str, str]:
    return {
        "created_at": _field(event, "created_at"),
        "action": _field(event, "action"),
        "title": _field(event, "title"),
    }


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if int(count) == 1 else plural
