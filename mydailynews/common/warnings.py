from __future__ import annotations

from typing import Any, Iterable, MutableSequence


def extend_warnings(target: MutableSequence[str], messages: Iterable[Any]) -> None:
    for message in messages:
        text = str(message).strip()
        if text:
            target.append(text)


def extend_prefixed_warnings(target: MutableSequence[str], prefix: str, messages: Iterable[Any]) -> None:
    clean_prefix = str(prefix).strip().rstrip(":")
    for message in messages:
        text = str(message).strip()
        if not text:
            continue
        target.append(f"{clean_prefix}: {text}" if clean_prefix else text)


def prompt_pressure_warning_count(messages: Iterable[Any]) -> int:
    count = 0
    for message in messages:
        text = str(message).lower()
        if "budget" in text or "dropped lower-ranked article" in text:
            count += 1
    return count
