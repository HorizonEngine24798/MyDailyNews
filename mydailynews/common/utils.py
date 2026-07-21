from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_PARAMS = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid"}


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


def canonical_article_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return normalize_whitespace(str(url or ""))
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
        ],
        doseq=True,
    )
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/") or parsed.path, "", query, ""))


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


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_object(text: str) -> Optional[Dict[str, Any]]:
    for strict in (True, False):
        try:
            value = json.loads(text, strict=strict)
            return value if isinstance(value, dict) else None
        except Exception:
            continue
    return None
