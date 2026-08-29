from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Sequence

from mydailynews.memory.coverage import CoverageMemoryStore
from mydailynews.memory.feedback import FEEDBACK_ACTIONS, FeedbackStore
from mydailynews.memory.story_store import (
    StoryStore,
    merge_story_records,
    story_record_from_payload,
)


MEMORY_REPAIR_BACKUP_DIR = "backups"
REPAIRABLE_MEMORY_FILES = {
    "story_store": "story_store.json",
    "coverage": "coverage_log.jsonl",
    "coverage_archive": "coverage_log.archive.jsonl",
    "feedback": "feedback_events.jsonl",
}


def coverage_row_id(index: int, record: Any) -> str:
    return _row_id("coverage", index, record)


def feedback_row_id(index: int, event: Any) -> str:
    return _row_id("feedback", index, event)


def delete_story_record(
    state_dir: Path | str,
    *,
    story_key: str,
    confirm: bool = False,
) -> Dict[str, Any]:
    _require_confirm(confirm)
    state = Path(state_dir)
    key = _clean(story_key, 160)
    if not key:
        raise ValueError("story_key is required.")

    records = _read_story_records(state)
    kept = [record for record in records if record["story_key"] != key]
    removed = len(records) - len(kept)
    if removed <= 0:
        raise ValueError(f"Story key not found: {key}")

    backup = _create_backup(state, _story_backup_files(state), reason="story_delete")
    _write_story_records(state, kept)
    return {
        "operation": "story_delete",
        "story_key": key,
        "stories_deleted": removed,
        "backup": backup,
    }


def repair_coverage_rows(
    state_dir: Path | str,
    *,
    row_ids: Sequence[str],
    action: str,
    confirm: bool = False,
) -> Dict[str, Any]:
    _require_confirm(confirm)
    state = Path(state_dir)
    normalized_action = _clean(action, 40).lower()
    if normalized_action not in {"delete", "archive"}:
        raise ValueError("Coverage repair action must be delete or archive.")
    records = _read_coverage_records(state)
    selected, kept = _split_rows_by_ids(
        records,
        row_ids=row_ids,
        row_id_func=coverage_row_id,
        label="coverage",
    )

    files = [REPAIRABLE_MEMORY_FILES["coverage"]]
    if normalized_action == "archive":
        files.append(REPAIRABLE_MEMORY_FILES["coverage_archive"])
    backup = _create_backup(state, files, reason=f"coverage_{normalized_action}")
    _write_jsonl_payloads(state / REPAIRABLE_MEMORY_FILES["coverage"], kept)

    archived = 0
    if normalized_action == "archive":
        archive_path = state / REPAIRABLE_MEMORY_FILES["coverage_archive"]
        existing_archive = _read_jsonl_payloads(archive_path)
        archived_at = _now_iso()
        archived_rows = [{**row["row"], "archived_at": archived_at} for row in selected]
        _write_jsonl_payloads(archive_path, existing_archive + archived_rows)
        archived = len(archived_rows)

    return {
        "operation": f"coverage_{normalized_action}",
        "coverage_rows_deleted": len(selected),
        "coverage_rows_archived": archived,
        "backup": backup,
    }


def repair_feedback_events(
    state_dir: Path | str,
    *,
    action: str,
    row_ids: Sequence[str],
    event_patch: Dict[str, Any] | None = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    _require_confirm(confirm)
    state = Path(state_dir)
    normalized_action = _clean(action, 40).lower()
    if normalized_action not in {"delete", "edit"}:
        raise ValueError("Feedback repair action must be delete or edit.")
    events = _read_feedback_events(state)
    selected, kept = _split_rows_by_ids(
        events,
        row_ids=row_ids,
        row_id_func=feedback_row_id,
        label="feedback",
    )

    edited = 0
    if normalized_action == "delete":
        output = kept
    else:
        if len(selected) != 1:
            raise ValueError("Feedback edit requires exactly one row id.")
        if not isinstance(event_patch, dict):
            raise ValueError("Feedback edit requires an event object.")
        selected_row = selected[0]
        selected_id = feedback_row_id(selected_row["_row_index"], selected_row["row"])
        updated = _normalize_feedback_event({**selected_row["row"], **event_patch})
        output = []
        for index, event in enumerate(events):
            if feedback_row_id(index, event) == selected_id:
                output.append(updated)
                edited += 1
            else:
                output.append(event)
    backup = _create_backup(state, [REPAIRABLE_MEMORY_FILES["feedback"]], reason=f"feedback_{normalized_action}")
    _write_jsonl_payloads(state / REPAIRABLE_MEMORY_FILES["feedback"], output)

    return {
        "operation": f"feedback_{normalized_action}",
        "feedback_events_deleted": len(selected) if normalized_action == "delete" else 0,
        "feedback_events_edited": edited,
        "backup": backup,
    }


def merge_stories(
    state_dir: Path | str,
    *,
    source_story_keys: Sequence[str],
    canonical_story: Dict[str, Any] | None,
    confirm: bool = False,
) -> Dict[str, Any]:
    _require_confirm(confirm)
    state = Path(state_dir)
    source_keys = _unique_strings(source_story_keys)
    if len(source_keys) < 2:
        raise ValueError("Story merge requires at least two source story keys.")

    story_records = _read_story_records(state)
    by_key = {record["story_key"]: record for record in story_records}
    missing = [key for key in source_keys if key not in by_key]
    if missing:
        raise ValueError(f"Story key not found: {', '.join(missing)}")

    source_records = [by_key[key] for key in source_keys]
    canonical = _canonical_story_record(source_records, canonical_story or {})
    if canonical["story_key"] not in source_keys and canonical["story_key"] in by_key:
        raise ValueError(f"Canonical story key already exists outside the merge: {canonical['story_key']}")

    merged_records = [
        record
        for record in story_records
        if record["story_key"] not in set(source_keys) and record["story_key"] != canonical["story_key"]
    ]
    merged_records.append(canonical)
    _validate_unique_story_keys(merged_records)

    coverage_records = _read_coverage_records(state)
    rewritten_coverage, coverage_rewrites = _rewrite_story_references(
        coverage_records,
        source_story_keys=source_keys,
        canonical=canonical,
    )
    feedback_events = _read_feedback_events(state)
    rewritten_feedback, feedback_rewrites = _rewrite_story_references(
        feedback_events,
        source_story_keys=source_keys,
        canonical=canonical,
    )

    backup = _create_backup(
        state,
        [
            *_story_backup_files(state),
            REPAIRABLE_MEMORY_FILES["coverage"],
            REPAIRABLE_MEMORY_FILES["feedback"],
        ],
        reason="story_merge",
    )
    _write_story_records(state, merged_records)
    _write_jsonl_payloads(state / REPAIRABLE_MEMORY_FILES["coverage"], rewritten_coverage)
    _write_jsonl_payloads(state / REPAIRABLE_MEMORY_FILES["feedback"], rewritten_feedback)
    return {
        "operation": "story_merge",
        "source_story_keys": source_keys,
        "canonical_story_key": canonical["story_key"],
        "stories_removed": len(source_records),
        "stories_written": len(merged_records),
        "coverage_rows_rewritten": coverage_rewrites,
        "feedback_events_rewritten": feedback_rewrites,
        "backup": backup,
    }


def split_story(
    state_dir: Path | str,
    *,
    source_story_key: str,
    new_story: Dict[str, Any],
    coverage_row_ids: Sequence[str],
    feedback_row_ids: Sequence[str] | None = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    _require_confirm(confirm)
    state = Path(state_dir)
    source_key = _clean(source_story_key, 160)
    if not source_key:
        raise ValueError("source_story_key is required.")
    story_records = _read_story_records(state)
    by_key = {record["story_key"]: record for record in story_records}
    source = by_key.get(source_key)
    if source is None:
        raise ValueError(f"Story key not found: {source_key}")

    if not isinstance(new_story, dict):
        raise ValueError("new_story must be an object.")
    candidate_new_story = {
        "story_family_key": source.get("story_family_key", ""),
        "first_seen": source.get("first_seen", ""),
        "last_seen": source.get("last_seen", ""),
        "status": source.get("status", "active"),
        **new_story,
    }
    new_record = _normalize_story_record(candidate_new_story)
    if new_record["story_key"] == source_key:
        raise ValueError("New story key must differ from the source story key.")
    if new_record["story_key"] in by_key:
        raise ValueError(f"New story key already exists: {new_record['story_key']}")

    coverage_records = _read_coverage_records(state)
    selected_coverage, kept_coverage = _split_rows_by_ids(
        coverage_records,
        row_ids=coverage_row_ids,
        row_id_func=coverage_row_id,
        label="coverage",
    )
    if not selected_coverage:
        raise ValueError("Story split requires at least one coverage row.")
    for row in selected_coverage:
        if row["row"].get("story_key") != source_key:
            raise ValueError("Selected coverage rows must belong to the source story.")

    moved_coverage = []
    for row in selected_coverage:
        moved = dict(row["row"])
        moved["story_key"] = new_record["story_key"]
        moved["story_family_key"] = new_record["story_family_key"]
        if new_record["title"]:
            moved["title"] = new_record["title"]
        moved_coverage.append(moved)
    rewritten_coverage = _merge_rows_by_original_index(kept_coverage, selected_coverage, moved_coverage)

    feedback_ids = list(feedback_row_ids or [])
    feedback_events = _read_feedback_events(state)
    rewritten_feedback = feedback_events
    feedback_rewrites = 0
    if feedback_ids:
        selected_feedback, kept_feedback = _split_rows_by_ids(
            feedback_events,
            row_ids=feedback_ids,
            row_id_func=feedback_row_id,
            label="feedback",
        )
        for row in selected_feedback:
            if row["row"].get("story_key") != source_key:
                raise ValueError("Selected feedback rows must belong to the source story.")
        moved_feedback = []
        for row in selected_feedback:
            moved = dict(row["row"])
            moved["story_key"] = new_record["story_key"]
            moved["story_family_key"] = new_record["story_family_key"]
            moved_feedback.append(moved)
        rewritten_feedback = _merge_rows_by_original_index(kept_feedback, selected_feedback, moved_feedback)
        feedback_rewrites = len(moved_feedback)

    updated_stories = story_records + [new_record]
    _validate_unique_story_keys(updated_stories)

    backup = _create_backup(
        state,
        [
            *_story_backup_files(state),
            REPAIRABLE_MEMORY_FILES["coverage"],
            REPAIRABLE_MEMORY_FILES["feedback"],
        ],
        reason="story_split",
    )
    _write_story_records(state, updated_stories)
    _write_jsonl_payloads(state / REPAIRABLE_MEMORY_FILES["coverage"], rewritten_coverage)
    _write_jsonl_payloads(state / REPAIRABLE_MEMORY_FILES["feedback"], rewritten_feedback)
    return {
        "operation": "story_split",
        "source_story_key": source_key,
        "new_story_key": new_record["story_key"],
        "coverage_rows_rewritten": len(moved_coverage),
        "feedback_events_rewritten": feedback_rewrites,
        "backup": backup,
    }


def _require_confirm(confirm: bool) -> None:
    if confirm is not True:
        raise ValueError("Memory repair requires confirm=true.")


def _story_backup_files(state_dir: Path) -> List[str]:
    candidates = ["story_store.json", "story_index.json", "story_ledger.json"]
    existing = [name for name in candidates if (state_dir / name).exists()]
    return existing or [REPAIRABLE_MEMORY_FILES["story_store"]]


def _row_id(prefix: str, index: int, row: Any) -> str:
    payload = _row_payload(row)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{int(index) + 1}-{digest}"


def _row_payload(row: Any) -> Dict[str, Any]:
    if is_dataclass(row):
        payload = asdict(row)
    elif isinstance(row, dict):
        payload = dict(row)
    else:
        payload = dict(getattr(row, "__dict__", {}) or {})
    payload.pop("_row_index", None)
    return payload


def _read_story_records(state_dir: Path) -> List[Dict[str, Any]]:
    return [asdict(record) for record in StoryStore.from_state_dir(state_dir).records()]


def _write_story_records(state_dir: Path, records: Sequence[Dict[str, Any]]) -> None:
    _validate_unique_story_keys(records)
    normalized = [story_record_from_payload(record) for record in records]
    if any(record is None for record in normalized):
        raise ValueError("Story records require valid story payloads.")
    StoryStore.from_state_dir(state_dir).replace_records(
        [record for record in normalized if record is not None]
    )


def _normalize_story_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    record = story_record_from_payload(raw)
    if record is None:
        raise ValueError("Story records require story_key.")
    return asdict(record)


def _read_coverage_records(state_dir: Path) -> List[Dict[str, Any]]:
    return [asdict(record) for record in CoverageMemoryStore.from_state_dir(state_dir).read_records()]


def _read_feedback_events(state_dir: Path) -> List[Dict[str, Any]]:
    return [asdict(event) for event in FeedbackStore.from_state_dir(state_dir).read_events()]


def _normalize_feedback_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    action = _clean(raw.get("action"), 80).lower()
    if action not in FEEDBACK_ACTIONS:
        raise ValueError(f"Unsupported feedback action: {action}")
    return {
        "schema_version": int(raw.get("schema_version", 1) or 1),
        "created_at": _clean(raw.get("created_at"), 80),
        "action": action,
        "report_date": _clean(raw.get("report_date"), 40),
        "brief_name": _clean(raw.get("brief_name"), 80),
        "article_id": _clean(raw.get("article_id"), 160),
        "story_key": _clean(raw.get("story_key"), 160),
        "story_family_key": _clean(raw.get("story_family_key"), 160),
        "title": _clean(raw.get("title"), 240),
        "source": _clean(raw.get("source"), 120),
        "topic": _clean(raw.get("topic"), 120),
        "notes": _clean(raw.get("notes"), 500),
    }


def _split_rows_by_ids(
    rows: Sequence[Dict[str, Any]],
    *,
    row_ids: Sequence[str],
    row_id_func,
    label: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requested = set(_unique_strings(row_ids))
    if not requested:
        raise ValueError(f"At least one {label} row id is required.")
    selected: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    matched: set[str] = set()
    for index, row in enumerate(rows):
        current_id = row_id_func(index, row)
        wrapped = {"_row_index": index, "row": dict(row)}
        if current_id in requested:
            selected.append(wrapped)
            matched.add(current_id)
        else:
            kept.append(wrapped)
    missing = sorted(requested - matched)
    if missing:
        raise ValueError(f"Unknown {label} row id(s): {', '.join(missing)}")
    return selected, [item["row"] for item in kept]


def _merge_rows_by_original_index(
    kept_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    moved_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_index = {row["_row_index"]: dict(replacement) for row, replacement in zip(selected_rows, moved_rows)}
    kept_iter = iter(kept_rows)
    total = len(kept_rows) + len(selected_rows)
    output: List[Dict[str, Any]] = []
    for index in range(total):
        if index in by_index:
            output.append(by_index[index])
        else:
            output.append(dict(next(kept_iter)))
    return output


def _canonical_story_record(source_records: Sequence[Dict[str, Any]], raw: Dict[str, Any]) -> Dict[str, Any]:
    parsed = [story_record_from_payload(record) for record in source_records]
    valid = [record for record in parsed if record is not None]
    if not valid:
        raise ValueError("Story merge requires valid source records.")
    return asdict(merge_story_records(valid, raw))


def _rewrite_story_references(
    rows: Sequence[Dict[str, Any]],
    *,
    source_story_keys: Sequence[str],
    canonical: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], int]:
    source_set = set(source_story_keys)
    rewritten: List[Dict[str, Any]] = []
    changed = 0
    for row in rows:
        output = dict(row)
        if output.get("story_key") in source_set:
            output["story_key"] = canonical["story_key"]
            output["story_family_key"] = canonical["story_family_key"]
            changed += 1
        rewritten.append(output)
    return rewritten, changed


def _validate_unique_story_keys(records: Sequence[Dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        key = _clean(record.get("story_key"), 160)
        if not key:
            raise ValueError("Story records require story_key.")
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate story key(s): {', '.join(sorted(duplicates))}")


def _read_jsonl_payloads(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_jsonl_payloads(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) for row in rows]
    _write_text_atomic(path, ("\n".join(lines) + "\n") if lines else "")


def _create_backup(state_dir: Path, filenames: Sequence[str], *, reason: str) -> Dict[str, Any]:
    created_at = _now_iso()
    stamp = created_at.replace(":", "").replace("-", "").replace("+0000", "Z").replace(".", "")
    backup_root = state_dir / MEMORY_REPAIR_BACKUP_DIR
    backup_dir = backup_root / stamp
    suffix = 1
    while backup_dir.exists():
        suffix += 1
        backup_dir = backup_root / f"{stamp}-{suffix}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    copied: List[str] = []
    for filename in filenames:
        source = state_dir / filename
        if not source.exists():
            continue
        target = backup_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(filename)
    manifest = {
        "schema_version": 1,
        "created_at": created_at,
        "reason": reason,
        "files": copied,
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "created_at": created_at,
        "path": str(backup_dir),
        "files": copied,
    }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    try:
        os.replace(temp_path, path)
    except PermissionError:
        path.write_text(text, encoding="utf-8")
        try:
            temp_path.unlink()
        except (FileNotFoundError, PermissionError):
            pass


def _clean(value: Any, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def _unique_strings(values: Any) -> List[str]:
    if isinstance(values, str):
        items: Iterable[Any] = [part.strip() for part in values.splitlines()]
    elif isinstance(values, Iterable):
        items = values
    else:
        items = []
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = _clean(item, 160)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _min_nonempty(values: Iterable[str]) -> str:
    items = sorted(_clean(value, 40) for value in values if _clean(value, 40))
    return items[0] if items else ""


def _max_nonempty(values: Iterable[str]) -> str:
    items = sorted(_clean(value, 40) for value in values if _clean(value, 40))
    return items[-1] if items else ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
