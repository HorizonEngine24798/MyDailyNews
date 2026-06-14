from __future__ import annotations

from typing import Any


TRUE_TEXT = {"1", "true", "yes", "y", "on", "enabled", "enable"}
FALSE_TEXT = {"0", "false", "no", "n", "off", "disabled", "disable"}


def parse_bool(value: Any, *, default: bool | None = None, field_name: str = "boolean value") -> bool:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must be a boolean")
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_TEXT:
            return True
        if normalized in FALSE_TEXT:
            return False
        raise ValueError(f"{field_name} must be a boolean")
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be 0, 1, true, or false")
    raise ValueError(f"{field_name} must be a boolean")


def parse_optional_bool(value: Any, *, field_name: str = "boolean value") -> bool | None:
    if value is None:
        return None
    return parse_bool(value, field_name=field_name)

