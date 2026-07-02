from __future__ import annotations

from pathlib import Path

from mydailynews.app.models import AppConfig, MemoryConfig


def memory_enabled(config: MemoryConfig | None) -> bool:
    return bool(config is not None and getattr(config, "enabled", False))


def memory_state_dir(app_config: AppConfig | MemoryConfig) -> Path:
    memory_config = getattr(app_config, "memory", app_config)
    raw = str(getattr(memory_config, "state_dir", "state/memory") or "state/memory").strip()
    return Path(raw or "state/memory")

