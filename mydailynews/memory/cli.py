from __future__ import annotations

from dataclasses import asdict
from datetime import date as date_type
import json
from pathlib import Path
import shutil
from typing import Any, Dict

from mydailynews.app.models import AppConfig
from mydailynews.memory.config import memory_state_dir
from mydailynews.memory.coverage import CoverageMemoryStore
from mydailynews.memory.feedback import FeedbackStore
from mydailynews.memory.learned_preferences import LearnedPreferencesStore
from mydailynews.memory.story_index import StoryIndexStore


MEMORY_ACTIONS = ("inspect", "prune", "export", "reset")


def run_memory_command(
    config: AppConfig,
    *,
    action: str,
    export_path: str = "",
    confirm_reset: bool = False,
) -> int:
    normalized = str(action or "").strip().lower()
    if normalized not in MEMORY_ACTIONS:
        print(f"Unknown memory action: {action}")
        return 1

    state_dir = memory_state_dir(config)
    if normalized == "inspect":
        _print_summary(_memory_summary(config, state_dir))
        return 0
    if normalized == "prune":
        summary = prune_memory(config, state_dir=state_dir)
        _print_summary(summary)
        return 0
    if normalized == "export":
        payload = export_memory(config, state_dir=state_dir)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if export_path:
            path = Path(export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
            print(f"Memory export written: {path}")
        else:
            print(rendered)
        return 0
    if not confirm_reset:
        print("Memory reset requires --confirm-memory-reset.")
        return 1
    reset_memory(state_dir)
    print(f"Memory state reset: {state_dir}")
    return 0


def prune_memory(config: AppConfig, *, state_dir: Path) -> Dict[str, Any]:
    memory_config = config.memory
    today = date_type.today().isoformat()
    coverage_store = CoverageMemoryStore.from_state_dir(state_dir)
    story_index_store = StoryIndexStore.from_state_dir(state_dir)
    coverage_pruned = coverage_store.prune(
        as_of_date=today,
        retention_days=int(memory_config.coverage_retention_days),
    )
    story_records = story_index_store.refresh_lifecycle(
        as_of_date=today,
        stale_after_days=int(memory_config.story_stale_after_days),
        retention_days=int(memory_config.story_retention_days),
        prune=True,
    )
    summary = _memory_summary(config, state_dir)
    summary["coverage_records_pruned"] = coverage_pruned
    summary["story_index_records_after_prune"] = len(story_records)
    return summary


def export_memory(config: AppConfig, *, state_dir: Path) -> Dict[str, Any]:
    coverage_store = CoverageMemoryStore.from_state_dir(state_dir)
    story_index_store = StoryIndexStore.from_state_dir(state_dir)
    feedback_store = FeedbackStore.from_state_dir(state_dir)
    learned_store = LearnedPreferencesStore.from_state_dir(state_dir)
    return {
        "schema_version": 1,
        "state_dir": str(state_dir),
        "config": {
            "coverage_window_days": config.memory.coverage_window_days,
            "coverage_retention_days": config.memory.coverage_retention_days,
            "story_stale_after_days": config.memory.story_stale_after_days,
            "story_retention_days": config.memory.story_retention_days,
            "feedback_enabled": config.memory.feedback_enabled,
        },
        "coverage_records": [asdict(record) for record in coverage_store.read_records()],
        "story_index": [asdict(record) for record in story_index_store.records()],
        "feedback_events": [asdict(event) for event in feedback_store.read_events()],
        "learned_preferences": asdict(learned_store.read()),
    }


def reset_memory(state_dir: Path) -> None:
    targets = (
        Path(state_dir) / "coverage_log.jsonl",
        Path(state_dir) / "story_index.json",
        Path(state_dir) / "feedback_events.jsonl",
        Path(state_dir) / "learned_preferences.json",
    )
    for path in targets:
        if path.exists():
            path.unlink()
    recall_packets = Path(state_dir) / "recall_packets"
    if recall_packets.exists():
        shutil.rmtree(recall_packets)


def _memory_summary(config: AppConfig, state_dir: Path) -> Dict[str, Any]:
    coverage_store = CoverageMemoryStore.from_state_dir(state_dir)
    story_index_store = StoryIndexStore.from_state_dir(state_dir)
    feedback_store = FeedbackStore.from_state_dir(state_dir)
    learned_store = LearnedPreferencesStore.from_state_dir(state_dir)
    stories = story_index_store.records()
    feedback_events = feedback_store.read_events()
    learned_path = learned_store.path
    return {
        "state_dir": str(state_dir),
        "memory_enabled": bool(config.memory.enabled),
        "coverage_window_days": config.memory.coverage_window_days,
        "coverage_retention_days": config.memory.coverage_retention_days,
        "story_stale_after_days": config.memory.story_stale_after_days,
        "story_retention_days": config.memory.story_retention_days,
        "feedback_enabled": bool(config.memory.feedback_enabled),
        "coverage_records": len(coverage_store.read_records()),
        "story_index_records": len(stories),
        "story_index_active": sum(1 for record in stories if record.status == "active"),
        "story_index_stale": sum(1 for record in stories if record.status == "stale"),
        "feedback_events": len(feedback_events),
        "feedback_counts": feedback_store.counts_by_action(),
        "learned_preferences_path": str(learned_path),
        "learned_preferences_exists": learned_path.exists(),
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print(f"Memory state: {summary['state_dir']}")
    print(f"Memory enabled: {summary['memory_enabled']}")
    print(
        "Retention: "
        f"coverage {summary['coverage_retention_days']} day(s), "
        f"story stale after {summary['story_stale_after_days']} day(s), "
        f"story retention {summary['story_retention_days']} day(s)"
    )
    print(f"Coverage records: {summary['coverage_records']}")
    print(
        "Story index records: "
        f"{summary['story_index_records']} "
        f"({summary['story_index_active']} active, {summary['story_index_stale']} stale)"
    )
    print(f"Feedback events: {summary['feedback_events']}")
    print(f"Learned preferences: {summary['learned_preferences_path']}")
