from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List


LEARNED_PREFERENCES_SCHEMA_VERSION = 1


@dataclass
class LearnedPreferences:
    schema_version: int = LEARNED_PREFERENCES_SCHEMA_VERSION
    updated_at: str = ""
    preferred_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    avoided_sources: List[str] = field(default_factory=list)
    topic_weights: Dict[str, float] = field(default_factory=dict)
    source_weights: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


class LearnedPreferencesStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "LearnedPreferencesStore":
        return cls(Path(state_dir) / "learned_preferences.json")

    def read(self) -> LearnedPreferences:
        if not self.path.exists():
            return LearnedPreferences()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return LearnedPreferences()
        if not isinstance(raw, dict):
            return LearnedPreferences()
        return _preferences_from_payload(raw)

    def write(self, preferences: LearnedPreferences) -> LearnedPreferences:
        preferences.updated_at = preferences.updated_at or datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(preferences), ensure_ascii=False, indent=2), encoding="utf-8")
        return preferences

    def ensure_exists(self) -> LearnedPreferences:
        preferences = self.read()
        if not self.path.exists():
            preferences.updated_at = datetime.now(timezone.utc).isoformat()
            self.write(preferences)
        return preferences


def _preferences_from_payload(raw: Dict[str, Any]) -> LearnedPreferences:
    return LearnedPreferences(
        schema_version=int(raw.get("schema_version", LEARNED_PREFERENCES_SCHEMA_VERSION) or 1),
        updated_at=str(raw.get("updated_at", "") or "").strip(),
        preferred_topics=_string_list(raw.get("preferred_topics")),
        avoided_topics=_string_list(raw.get("avoided_topics")),
        preferred_sources=_string_list(raw.get("preferred_sources")),
        avoided_sources=_string_list(raw.get("avoided_sources")),
        topic_weights=_float_map(raw.get("topic_weights")),
        source_weights=_float_map(raw.get("source_weights")),
        notes=str(raw.get("notes", "") or "").strip(),
    )


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
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
    for key, raw_score in value.items():
        name = str(key or "").strip()
        if not name:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        output[name] = round(max(-3.0, min(3.0, score)), 4)
    return output
