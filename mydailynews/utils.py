from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(*parts: str) -> str:
    raw = "|".join(part or "" for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_html(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text or "")
    return normalize_whitespace(html.unescape(without_tags))


def normalize_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", parsed.query, ""))


def datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    return value.astimezone(timezone.utc).isoformat()


def safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    value = _load_json_object(text or "")
    if value is not None:
        return value

    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    return _load_json_object(match.group(0))


def _load_json_object(text: str) -> Optional[Dict[str, Any]]:
    for strict in (True, False):
        try:
            value = json.loads(text, strict=strict)
            return value if isinstance(value, dict) else None
        except Exception:
            continue
    return None
