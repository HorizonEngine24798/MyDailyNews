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
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except Exception:
            return None
