from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List

from mydailynews.app.config import load_config
from mydailynews.app.models import UserMemory
from mydailynews.gui.runs import BRIEF_CHOICES, MEMORY_RUN_ACTIONS, RUN_KINDS, GuiRunManager
from mydailynews.memory.cli import export_memory, prune_memory
from mydailynews.memory.config import memory_state_dir
from mydailynews.memory.feedback import FEEDBACK_ACTIONS, FeedbackStore
from mydailynews.memory.health import memory_health_checks
from mydailynews.memory.learned_preferences import LearnedPreferences, LearnedPreferencesStore
from mydailynews.memory.preference_learning import apply_feedback_event
from mydailynews.memory.repair import (
    coverage_row_id,
    delete_story_record,
    feedback_row_id,
    merge_stories,
    repair_coverage_rows,
    repair_feedback_events,
    split_story,
)


REPORT_NAME_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<kind>.+)\.md$")
RECALL_PACKET_NAME_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<brief>.+)$")
AUTOCONFIG_TIMEOUT_DEFAULT_SECONDS = 90
AUTOCONFIG_TIMEOUT_MIN_SECONDS = 5
AUTOCONFIG_TIMEOUT_MAX_SECONDS = 300
LEARNED_NOTES_LIMIT = 2000
MARKDOWN_TITLE_SCAN_LINES = 8
STORY_TITLE_LIMIT = 140
STORY_TOPIC_LIMIT = 120
STORY_TOKEN_LIMIT = 24
EDITABLE_CONFIG_SECTIONS = {
    "ai_summary",
    "ai_final",
    "user_memory",
    "general_topics",
    "general_filtering",
    "topics_to_examine",
    "filtering",
    "memory",
    "enrichment",
    "runtime",
    "narrative_briefing",
    "tts",
    "pipeline",
    "analysis",
    "cache",
    "sources",
}


class GuiDataService:
    def __init__(self, root: Path | str, config_path: Path | str = "config.local.json") -> None:
        self.root = Path(root).resolve()
        raw_config_path = Path(config_path)
        if raw_config_path.is_absolute():
            self.config_path = raw_config_path.resolve()
        else:
            self.config_path = (self.root / raw_config_path).resolve()
        self._require_inside_root(self.config_path)
        self.run_manager = GuiRunManager(root=self.root, config_path=self.config_path)

    def app_state(self) -> Dict[str, Any]:
        config = self._load_config()
        output_dir = self._resolve_project_path(config.output_dir)
        state_dir = self._memory_state_dir(config)
        return {
            "project_root": str(self.root),
            "config_path": str(self.config_path),
            "output_dir": str(output_dir),
            "memory_state_dir": str(state_dir),
            "feedback_actions": list(FEEDBACK_ACTIONS),
            "run_kinds": sorted(RUN_KINDS),
            "run_brief_choices": sorted(BRIEF_CHOICES),
            "run_memory_actions": sorted(MEMORY_RUN_ACTIONS),
        }

    def read_config(self) -> Dict[str, Any]:
        payload = self._read_config_payload()
        stat = self.config_path.stat()
        return {
            "path": str(self.config_path),
            "modified_at": _iso_from_timestamp(stat.st_mtime),
            "config": payload,
        }

    def save_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Config payload must be an object.")
        self._validate_config_payload(payload)
        self._write_json_atomic(self.config_path, payload)
        return self.read_config()

    def save_config_section(self, section: str, payload: Any) -> Dict[str, Any]:
        normalized = str(section or "").strip()
        if normalized not in EDITABLE_CONFIG_SECTIONS:
            raise ValueError(f"Config section is not editable: {section}")
        config = self._read_config_payload()
        config[normalized] = payload
        return self.save_config(config)

    def preview_user_memory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Ground-Truth Profile preview payload must be an object.")
        profile = UserMemory(
            avoided_topics=_string_list(payload.get("avoided_topics")),
            preferred_sources=_string_list(payload.get("preferred_sources")),
            avoided_sources=_string_list(payload.get("avoided_sources")),
            role=str(payload.get("role", "") or ""),
            geography_focus=_string_list(payload.get("geography_focus")),
            time_horizon=str(payload.get("time_horizon", "tactical") or "tactical"),
            beats=_positive_float_map(payload.get("beats")),
            wants=_string_list(payload.get("wants")),
            avoid=_string_list(payload.get("avoid")),
            portfolio_or_stake_notes=str(payload.get("portfolio_or_stake_notes", "") or ""),
            preferred_depth=str(payload.get("preferred_depth", "analytical") or "analytical"),
            briefing_style=str(payload.get("briefing_style", "") or ""),
            custom_instructions=str(payload.get("custom_instructions", "") or ""),
        )
        return {
            "profile": asdict(profile),
            "prompt": profile.to_prompt(),
        }

    def preview_learned_preferences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Learned Preferences preview payload must be an object.")
        preferences = LearnedPreferences(
            schema_version=int(payload.get("schema_version", 1) or 1),
            updated_at=str(payload.get("updated_at", "") or ""),
            preferred_topics=_string_list(payload.get("preferred_topics")),
            avoided_topics=_string_list(payload.get("avoided_topics")),
            preferred_sources=_string_list(payload.get("preferred_sources")),
            avoided_sources=_string_list(payload.get("avoided_sources")),
            topic_weights=_float_map(payload.get("topic_weights")),
            source_weights=_float_map(payload.get("source_weights")),
            notes=str(payload.get("notes", "") or "").strip()[:LEARNED_NOTES_LIMIT],
        )
        return {
            "preferences": asdict(preferences),
            "effective_weights": {
                "topics": _sorted_weight_rows(preferences.topic_weights),
                "sources": _sorted_weight_rows(preferences.source_weights),
            },
        }

    def list_reports(self) -> Dict[str, Any]:
        output_dir = self._output_dir()
        reports: List[Dict[str, Any]] = []
        if output_dir.exists():
            paths = sorted(output_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
            for path in paths:
                parsed = _parse_report_name(path.name)
                stat = path.stat()
                reports.append(
                    {
                        "id": path.name,
                        "filename": path.name,
                        "title": _markdown_title(path),
                        "date": parsed["date"],
                        "kind": parsed["kind"],
                        "brief_name": _brief_name_for_kind(parsed["kind"]),
                        "size_bytes": stat.st_size,
                        "modified_at": _iso_from_timestamp(stat.st_mtime),
                        "has_json": path.with_suffix(".json").exists(),
                    }
                )
        return {"output_dir": str(output_dir), "reports": reports}

    def report_detail(self, report_id: str) -> Dict[str, Any]:
        path = self._report_path(report_id)
        markdown = path.read_text(encoding="utf-8-sig")
        json_path = path.with_suffix(".json")
        payload: Dict[str, Any] = {}
        json_error = ""
        if json_path.exists():
            try:
                loaded = json.loads(json_path.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError as exc:
                json_error = str(exc)
        parsed = _parse_report_name(path.name)
        brief_name = _brief_name_for_kind(parsed["kind"])
        feedback_items = _feedback_items(payload)
        feedback_events = FeedbackStore.from_state_dir(self._memory_state_dir(self._load_config())).read_events()
        feedback_items, feedback_history = _feedback_items_with_history(
            feedback_items,
            feedback_events,
            report_date=parsed["date"],
            brief_name=brief_name,
        )
        return {
            "id": path.name,
            "filename": path.name,
            "title": _brief_title(payload, markdown, path.name),
            "date": parsed["date"],
            "kind": parsed["kind"],
            "brief_name": brief_name,
            "markdown": markdown,
            "json": payload,
            "json_error": json_error,
            "feedback_items": feedback_items,
            "feedback_history": feedback_history,
        }

    def memory_snapshot(self) -> Dict[str, Any]:
        config = self._load_config()
        state_dir = self._memory_state_dir(config)
        payload = export_memory(config, state_dir=state_dir)
        raw_coverage_records = _as_list(payload.get("coverage_records", []))
        raw_story_index = _as_list(payload.get("story_index", []))
        raw_feedback_events = _as_list(payload.get("feedback_events", []))
        learned = payload.get("learned_preferences", {})
        learned_store = LearnedPreferencesStore.from_state_dir(state_dir)
        coverage_counts = Counter(_text(item, "story_key") for item in raw_coverage_records if _text(item, "story_key"))
        story_index = [_story_index_row(item, coverage_counts) for item in raw_story_index]
        coverage_records = [_coverage_row(item, index=index) for index, item in enumerate(raw_coverage_records)]
        feedback_events = [_feedback_event_row(item, index=index) for index, item in enumerate(raw_feedback_events)]
        recall_packets = _recall_packets_summary(state_dir)
        health = memory_health_checks(
            state_dir=state_dir,
            story_index=raw_story_index,
            coverage_records=raw_coverage_records,
            feedback_events=raw_feedback_events,
        )
        learned_summary = _learned_preferences_summary(learned, exists=learned_store.path.exists())
        payload["coverage_records"] = coverage_records
        payload["story_index"] = story_index
        payload["feedback_events"] = feedback_events
        payload["learned_preferences_summary"] = learned_summary
        payload["learned_preferences_file"] = {
            "path": str(learned_store.path),
            "preferences": learned if isinstance(learned, dict) else {},
        }
        payload["recall_packets"] = recall_packets
        payload["health"] = health
        payload["summary"] = {
            "state_dir": str(state_dir),
            "memory_enabled": bool(config.memory.enabled),
            "feedback_enabled": bool(config.memory.feedback_enabled),
            "coverage_records": len(coverage_records),
            "story_index_records": len(story_index),
            "story_index_active": sum(1 for item in story_index if item.get("status") == "active"),
            "story_index_stale": sum(1 for item in story_index if item.get("status") == "stale"),
            "feedback_events": len(feedback_events),
            "feedback_counts": _feedback_counts(feedback_events),
            "learned_preferences_exists": learned_store.path.exists(),
            "learned_preferences_updated_at": learned_summary["updated_at"],
            "recall_packet_count": recall_packets["count"],
            "latest_recall_packet_date": recall_packets["latest"].get("date", ""),
            "health_warnings": len(health["warnings"]),
        }
        payload["story_index_file"] = {
            "schema_version": 1,
            "stories": raw_story_index,
        }
        payload["feedback_actions"] = list(FEEDBACK_ACTIONS)
        return payload

    def prune_memory(self) -> Dict[str, Any]:
        config = self._load_config()
        summary = prune_memory(config, state_dir=self._memory_state_dir(config))
        return {"summary": summary, "memory": self.memory_snapshot()}

    def save_story_index(self, payload: Any) -> Dict[str, Any]:
        normalized = _normalize_story_index_payload(payload)
        config = self._load_config()
        path = self._memory_state_dir(config) / "story_index.json"
        self._write_json_atomic(path, normalized)
        return self.memory_snapshot()

    def repair_memory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Memory repair payload must be an object.")
        config = self._load_config()
        state_dir = self._memory_state_dir(config)
        action = str(payload.get("action", "") or "").strip().lower()
        confirm = payload.get("confirm") is True

        if action == "story_delete":
            result = delete_story_record(
                state_dir,
                story_key=str(payload.get("story_key", "") or ""),
                confirm=confirm,
            )
        elif action in {"coverage_delete", "coverage_archive"}:
            result = repair_coverage_rows(
                state_dir,
                row_ids=_string_list(payload.get("row_ids")),
                action=action.replace("coverage_", ""),
                confirm=confirm,
            )
        elif action in {"feedback_delete", "feedback_edit"}:
            result = repair_feedback_events(
                state_dir,
                action=action.replace("feedback_", ""),
                row_ids=_string_list(payload.get("row_ids")),
                event_patch=payload.get("event") if isinstance(payload.get("event"), dict) else None,
                confirm=confirm,
            )
        elif action == "story_merge":
            result = merge_stories(
                state_dir,
                source_story_keys=_string_list(payload.get("source_story_keys")),
                canonical_story=payload.get("canonical_story") if isinstance(payload.get("canonical_story"), dict) else {},
                confirm=confirm,
            )
        elif action == "story_split":
            result = split_story(
                state_dir,
                source_story_key=str(payload.get("source_story_key", "") or ""),
                new_story=payload.get("new_story") if isinstance(payload.get("new_story"), dict) else {},
                coverage_row_ids=_string_list(payload.get("coverage_row_ids")),
                feedback_row_ids=_string_list(payload.get("feedback_row_ids")),
                confirm=confirm,
            )
        else:
            raise ValueError(f"Unknown memory repair action: {action}")

        return {
            "repair": result,
            "memory": self.memory_snapshot(),
        }

    def learned_preferences(self) -> Dict[str, Any]:
        config = self._load_config()
        store = LearnedPreferencesStore.from_state_dir(self._memory_state_dir(config))
        preferences = store.read()
        return {
            "path": str(store.path),
            "preferences": asdict(preferences),
        }

    def save_learned_preferences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Learned preferences payload must be an object.")
        config = self._load_config()
        store = LearnedPreferencesStore.from_state_dir(self._memory_state_dir(config))
        preferences = LearnedPreferences(
            schema_version=int(payload.get("schema_version", 1) or 1),
            updated_at=datetime.now(timezone.utc).isoformat(),
            preferred_topics=_string_list(payload.get("preferred_topics")),
            avoided_topics=_string_list(payload.get("avoided_topics")),
            preferred_sources=_string_list(payload.get("preferred_sources")),
            avoided_sources=_string_list(payload.get("avoided_sources")),
            topic_weights=_float_map(payload.get("topic_weights")),
            source_weights=_float_map(payload.get("source_weights")),
            notes=str(payload.get("notes", "") or "").strip()[:LEARNED_NOTES_LIMIT],
        )
        store.write(preferences)
        return self.learned_preferences()

    def record_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Feedback payload must be an object.")
        config = self._load_config()
        if not bool(config.memory.feedback_enabled):
            raise ValueError("Feedback is disabled in the active config.")
        report_id = str(payload.get("report_id", "") or "").strip()
        report_info = _parse_report_name(report_id) if report_id else {"date": "", "kind": ""}
        item = payload.get("item", {})
        if not isinstance(item, dict):
            item = {}
        state_dir = self._memory_state_dir(config)
        store = FeedbackStore.from_state_dir(state_dir)
        event = store.record(
            action=str(payload.get("action", "") or ""),
            report_date=str(payload.get("report_date", "") or report_info["date"]),
            brief_name=str(payload.get("brief_name", "") or _brief_name_for_kind(report_info["kind"])),
            article_id=str(item.get("id", "") or payload.get("article_id", "")),
            story_key=str(item.get("story_key", "") or payload.get("story_key", "")),
            story_family_key=str(item.get("story_family_key", "") or payload.get("story_family_key", "")),
            title=str(item.get("title", "") or item.get("headline", "") or payload.get("title", "")),
            source=str(item.get("source", "") or payload.get("source", "")),
            topic=str(item.get("topic", "") or payload.get("topic", "")),
            notes=str(payload.get("notes", "") or ""),
        )
        learned_store = LearnedPreferencesStore.from_state_dir(state_dir)
        preferences = learned_store.read()
        learned_preferences, learned_delta = apply_feedback_event(preferences, event)
        if learned_delta.changed:
            learned_preferences.updated_at = datetime.now(timezone.utc).isoformat()
            learned_store.write(learned_preferences)
        else:
            learned_preferences = preferences
        return {
            "event": asdict(event),
            "counts": store.counts_by_action(),
            "learned_preferences_changed": learned_delta.changed,
            "learned_preference_delta": asdict(learned_delta),
            "learned_preferences": asdict(learned_preferences),
            "learned_preferences_path": str(learned_store.path),
        }

    def run_autoconfig(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = payload or {}
        write_name = str(payload.get("write", "") or "config.recommended.json").strip()
        write_path = self._resolve_project_path(write_name)
        if write_path.suffix.lower() != ".json":
            raise ValueError("Autoconfig output path must be a JSON file.")
        command = [
            sys.executable,
            str(self.root / "tools" / "autoconfig.py"),
            "--config",
            str(self.config_path),
            "--write",
            str(write_path),
            "--no-preference-prompt",
            "--no-download-prompt",
            "--no-server-probe",
            "--print-launch-command",
        ]
        raw_timeout = int(payload.get("timeout_seconds", AUTOCONFIG_TIMEOUT_DEFAULT_SECONDS) or AUTOCONFIG_TIMEOUT_DEFAULT_SECONDS)
        timeout = max(AUTOCONFIG_TIMEOUT_MIN_SECONDS, min(AUTOCONFIG_TIMEOUT_MAX_SECONDS, raw_timeout))
        result = subprocess.run(
            command,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "write_path": str(write_path),
        }

    def list_runs(self) -> Dict[str, Any]:
        return self.run_manager.list_runs()

    def run_detail(self, run_id: str) -> Dict[str, Any]:
        return self.run_manager.get_run(run_id)

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_manager.start_run(payload)

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        return self.run_manager.cancel_run(run_id)

    def _load_config(self):
        return load_config(self.config_path)

    def _read_config_payload(self) -> Dict[str, Any]:
        payload = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Config root must be an object.")
        return payload

    def _validate_config_payload(self, payload: Dict[str, Any]) -> None:
        temp_path = self.config_path.with_name(f"{self.config_path.name}.gui-validate.tmp")
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            load_config(temp_path)
        finally:
            try:
                temp_path.unlink()
            except (FileNotFoundError, PermissionError):
                pass

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        self._require_inside_root(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        temp_path.write_text(text, encoding="utf-8")
        try:
            os.replace(temp_path, path)
        except PermissionError:
            path.write_text(text, encoding="utf-8")
            try:
                temp_path.unlink()
            except (FileNotFoundError, PermissionError):
                pass

    def _output_dir(self) -> Path:
        return self._resolve_project_path(self._load_config().output_dir)

    def _memory_state_dir(self, config: Any) -> Path:
        return self._resolve_project_path(memory_state_dir(config))

    def _resolve_project_path(self, raw_path: Path | str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (self.root / path).resolve()
        self._require_inside_root(resolved)
        return resolved

    def _require_inside_root(self, path: Path) -> None:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Path is outside the project root: {resolved}") from exc

    def _report_path(self, report_id: str) -> Path:
        name = Path(str(report_id or "")).name
        if not name or name != str(report_id) or not name.endswith(".md"):
            raise FileNotFoundError("Report not found.")
        path = (self._output_dir() / name).resolve()
        self._require_inside_root(path)
        if not path.exists():
            raise FileNotFoundError("Report not found.")
        return path


def _parse_report_name(filename: str) -> Dict[str, str]:
    match = REPORT_NAME_RE.match(str(filename or ""))
    if not match:
        return {"date": "", "kind": Path(str(filename or "")).stem}
    return {"date": match.group("date"), "kind": match.group("kind")}


def _brief_name_for_kind(kind: str) -> str:
    normalized = str(kind or "").strip()
    if normalized.endswith("_brief") and normalized in {"general_brief", "detailed_brief"}:
        return normalized[: -len("_brief")]
    return normalized


def _markdown_title(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines()[:MARKDOWN_TITLE_SCAN_LINES]:
            text = line.strip()
            if text.startswith("#"):
                return text.lstrip("#").strip() or path.name
    except OSError:
        pass
    return path.name


def _brief_title(payload: Dict[str, Any], markdown: str, fallback: str) -> str:
    title = str(payload.get("title", "") or "").strip()
    if title:
        return title
    for line in markdown.splitlines()[:MARKDOWN_TITLE_SCAN_LINES]:
        text = line.strip()
        if text.startswith("#"):
            return text.lstrip("#").strip() or fallback
    return fallback


def _feedback_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = payload.get("selected_articles")
    if not isinstance(raw_items, list) or not raw_items:
        raw_items = payload.get("major_headlines")
    if isinstance(raw_items, list) and raw_items:
        return [_feedback_item_from_article(item, index) for index, item in enumerate(raw_items) if isinstance(item, dict)]
    segments = payload.get("segments")
    if isinstance(segments, list):
        items: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            title = str(segment.get("heading", "") or f"Segment {index + 1}").strip()
            items.append(
                {
                    "id": f"segment-{index + 1}",
                    "title": title,
                    "source": "narrative",
                    "url": "",
                    "topic": "",
                    "story_key": "",
                    "story_family_key": "",
                }
            )
        return items
    return []


def _feedback_item_from_article(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    title = str(item.get("headline", "") or item.get("title", "") or f"Item {index + 1}").strip()
    return {
        "id": str(item.get("id", "") or f"item-{index + 1}"),
        "title": title,
        "source": str(item.get("source", "") or ""),
        "url": str(item.get("url", "") or ""),
        "topic": str(item.get("topic", "") or item.get("category", "") or ""),
        "story_key": str(item.get("story_key", "") or ""),
        "story_family_key": str(item.get("story_family_key", "") or ""),
    }


def _feedback_items_with_history(
    items: List[Dict[str, Any]],
    events: List[Any],
    *,
    report_date: str,
    brief_name: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output: List[Dict[str, Any]] = []
    all_history: List[Dict[str, Any]] = []
    for item in items:
        rows = [
            _feedback_event_row(event)
            for event in events
            if _feedback_event_matches_item(event, item, report_date=report_date, brief_name=brief_name)
        ]
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        all_history.extend(rows)
        enriched = dict(item)
        enriched["feedback_history"] = rows
        enriched["feedback_count"] = len(rows)
        enriched["latest_feedback_action"] = rows[0]["action"] if rows else ""
        enriched["latest_feedback_at"] = rows[0]["created_at"] if rows else ""
        output.append(enriched)
    all_history.sort(key=lambda row: row.get("created_at", ""), reverse=True)
    return output, all_history


def _feedback_event_matches_item(event: Any, item: Dict[str, Any], *, report_date: str, brief_name: str) -> bool:
    event_date = _text(event, "report_date")
    event_brief = _text(event, "brief_name")
    if event_date and report_date and event_date != report_date:
        return False
    if event_brief and brief_name and event_brief != brief_name:
        return False
    article_id = _text(event, "article_id")
    if article_id and article_id == str(item.get("id", "") or "").strip():
        return True
    story_key = _text(event, "story_key")
    if story_key and story_key == str(item.get("story_key", "") or "").strip():
        return True
    return False


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _story_index_row(item: Any, coverage_counts: Counter[str]) -> Dict[str, Any]:
    story_key = _text(item, "story_key")
    story_family_key = _text(item, "story_family_key")
    return {
        "story_key": story_key,
        "story_family_key": story_family_key,
        "family": story_family_key,
        "title": _text(item, "title"),
        "topic": _text(item, "topic"),
        "first_seen": _text(item, "first_seen"),
        "last_seen": _text(item, "last_seen"),
        "status": _text(item, "status") or "active",
        "coverage_count": int(coverage_counts.get(story_key, 0)),
    }


def _coverage_row(item: Any, *, index: int) -> Dict[str, Any]:
    story_family_key = _text(item, "story_family_key")
    return {
        "row_id": coverage_row_id(index, item),
        "date": _text(item, "date"),
        "brief_name": _text(item, "brief_name"),
        "title": _text(item, "title"),
        "prominence": _text(item, "prominence"),
        "story_key": _text(item, "story_key"),
        "story_family_key": story_family_key,
        "family": story_family_key,
    }


def _feedback_event_row(item: Any, *, index: int | None = None) -> Dict[str, Any]:
    created_at = _text(item, "created_at")
    row = {
        "created_at": created_at,
        "created_date": created_at[:10],
        "action": _text(item, "action"),
        "title": _text(item, "title"),
        "source": _text(item, "source"),
        "topic": _text(item, "topic"),
        "story_key": _text(item, "story_key"),
        "story_family_key": _text(item, "story_family_key"),
        "article_id": _text(item, "article_id"),
        "report_date": _text(item, "report_date"),
        "brief_name": _text(item, "brief_name"),
        "notes": _text(item, "notes"),
    }
    if index is not None:
        row["row_id"] = feedback_row_id(index, item)
    return row


def _feedback_counts(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {action: 0 for action in FEEDBACK_ACTIONS}
    for event in events:
        action = str(event.get("action", "") or "")
        if action:
            counts[action] = counts.get(action, 0) + 1
    return counts


def _learned_preferences_summary(learned: Any, *, exists: bool) -> Dict[str, Any]:
    if not isinstance(learned, dict):
        learned = {}
    return {
        "exists": bool(exists),
        "updated_at": str(learned.get("updated_at", "") or ""),
        "preferred_topics": len(_as_list(learned.get("preferred_topics"))),
        "avoided_topics": len(_as_list(learned.get("avoided_topics"))),
        "preferred_sources": len(_as_list(learned.get("preferred_sources"))),
        "avoided_sources": len(_as_list(learned.get("avoided_sources"))),
        "topic_weights": len(learned.get("topic_weights", {}) if isinstance(learned.get("topic_weights"), dict) else {}),
        "source_weights": len(learned.get("source_weights", {}) if isinstance(learned.get("source_weights"), dict) else {}),
        "has_notes": bool(str(learned.get("notes", "") or "").strip()),
    }


def _recall_packets_summary(state_dir: Path) -> Dict[str, Any]:
    packet_dir = state_dir / "recall_packets"
    paths = sorted(packet_dir.glob("*.json")) if packet_dir.exists() else []
    if not paths:
        return {"exists": False, "count": 0, "latest": {}}
    latest = max(paths, key=lambda path: path.stat().st_mtime)
    parsed = _parse_recall_packet_name(latest.stem)
    return {
        "exists": True,
        "count": len(paths),
        "latest": {
            "date": parsed["date"],
            "brief_name": parsed["brief_name"],
            "path": str(latest),
        },
    }


def _parse_recall_packet_name(name: str) -> Dict[str, str]:
    match = RECALL_PACKET_NAME_RE.match(str(name or ""))
    if not match:
        return {"date": "", "brief_name": str(name or "")}
    return {"date": match.group("date"), "brief_name": match.group("brief")}


def _text(item: Any, key: str) -> str:
    if isinstance(item, dict):
        value = item.get(key, "")
    else:
        value = getattr(item, key, "")
    return str(value or "").strip()


def _normalize_story_index_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        stories = payload
    elif isinstance(payload, dict):
        stories = payload.get("stories", [])
    else:
        raise ValueError("Story index payload must be an object or list.")
    if not isinstance(stories, list):
        raise ValueError("Story index stories must be a list.")
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in stories:
        if not isinstance(raw, dict):
            continue
        story_key = str(raw.get("story_key", "") or "").strip()
        if not story_key or story_key in seen:
            continue
        seen.add(story_key)
        status = str(raw.get("status", "active") or "active").strip().lower()
        if status not in {"active", "stale"}:
            status = "active"
        normalized.append(
            {
                "story_key": story_key,
                "story_family_key": str(raw.get("story_family_key", "") or "").strip(),
                "title": str(raw.get("title", "") or "").strip()[:STORY_TITLE_LIMIT],
                "topic": str(raw.get("topic", "") or "").strip()[:STORY_TOPIC_LIMIT],
                "tokens": _string_list(raw.get("tokens"))[:STORY_TOKEN_LIMIT],
                "first_seen": str(raw.get("first_seen", "") or "").strip(),
                "last_seen": str(raw.get("last_seen", "") or "").strip(),
                "status": status,
            }
        )
    normalized.sort(key=lambda item: item["story_key"])
    return {"schema_version": 1, "stories": normalized}


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        values = [part.strip() for part in value.splitlines()]
    elif isinstance(value, list):
        values = [str(item or "").strip() for item in value]
    else:
        values = []
    output: List[str] = []
    seen: set[str] = set()
    for text in values:
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _float_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: Dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        try:
            score = float(raw_value)
        except (TypeError, ValueError):
            continue
        output[key] = round(max(-3.0, min(3.0, score)), 4)
    return output


def _positive_float_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: Dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        try:
            score = float(raw_value)
        except (TypeError, ValueError):
            continue
        output[key] = round(max(0.0, min(3.0, score)), 4)
    return output


def _sorted_weight_rows(weights: Dict[str, float]) -> List[Dict[str, Any]]:
    return [
        {"name": name, "weight": weight}
        for name, weight in sorted(weights.items(), key=lambda item: (-abs(float(item[1])), item[0].lower()))
    ]


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
