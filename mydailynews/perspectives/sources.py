from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


SOURCE_PACK_VERSION = "rss_sources.v2"
DEFAULT_SOURCE_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "sources.rss.json"
SOURCE_TYPES = {
    "wire",
    "newspaper",
    "broadcaster",
    "state_media",
    "public_media",
    "regional",
    "official",
    "specialized",
}
ID_RE = re.compile(r"^[a-z0-9_]+$")
CODE_RE = re.compile(r"^[a-z]{2}$", re.IGNORECASE)


def load_source_registry() -> List[Dict[str, Any]]:
    payload = json.loads(DEFAULT_SOURCE_REGISTRY_PATH.read_text(encoding="utf-8-sig"))
    sources = payload.get("sources") if isinstance(payload, dict) else payload
    if not isinstance(sources, list):
        raise ValueError("source registry must be a list or an object with a sources list")
    rows = [source for source in sources if isinstance(source, dict)]
    errors = validate_source_registry(rows)
    if errors:
        raise ValueError("source registry validation failed: " + "; ".join(errors[:5]))
    return rows


def validate_source_registry(sources: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    source_ids: set[str] = set()
    feed_urls: set[str] = set()
    for row_number, source in enumerate(sources):
        label = str(source.get("source_id") or f"source[{row_number}]")
        source_id = str(source.get("source_id") or "").strip()
        if not source_id or not ID_RE.match(source_id):
            errors.append(f"{label}: source_id must be lowercase letters, numbers, and underscores")
        elif source_id in source_ids:
            errors.append(f"{label}: duplicate source_id")
        source_ids.add(source_id)

        for field in ("name", "country", "language", "source_type", "homepage_url", "category"):
            if not str(source.get(field) or "").strip():
                errors.append(f"{label}: missing {field}")
        tags = source.get("tags") if isinstance(source.get("tags"), list) else []
        if not any(str(tag or "").strip() for tag in tags):
            errors.append(f"{label}: tags must contain at least one value")
        if source.get("source_type") not in SOURCE_TYPES:
            errors.append(f"{label}: source_type must be one of {', '.join(sorted(SOURCE_TYPES))}")
        if not CODE_RE.match(str(source.get("country") or "")):
            errors.append(f"{label}: country must be a two-letter code")
        if not CODE_RE.match(str(source.get("language") or "")):
            errors.append(f"{label}: language must be a two-letter code")
        for field in ("homepage_url",):
            if not _valid_url(str(source.get(field) or "")):
                errors.append(f"{label}: {field} must be http(s)")

        feeds = source.get("feed_urls") if isinstance(source.get("feed_urls"), list) else []
        sitemaps = source.get("sitemap_urls") if isinstance(source.get("sitemap_urls"), list) else []
        if not feeds and not sitemaps:
            errors.append(f"{label}: feed_urls or sitemap_urls is required")
        for url in [*feeds, *sitemaps]:
            text = str(url or "").strip()
            if not _valid_url(text):
                errors.append(f"{label}: invalid feed/sitemap URL {text!r}")
            elif text in feed_urls:
                errors.append(f"{label}: duplicate feed/sitemap URL {text}")
            feed_urls.add(text)
    return errors


def registry_stats(sources: List[Dict[str, Any]]) -> Dict[str, int]:
    enabled = [source for source in sources if bool(source.get("enabled", True))]
    return {
        "source_count": len(enabled),
        "country_count": len({str(source.get("country") or "").upper() for source in enabled if source.get("country")}),
        "language_count": len({str(source.get("language") or "").lower() for source in enabled if source.get("language")}),
    }


def source_domain_map(sources: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        for url in [source.get("homepage_url"), *(source.get("feed_urls") or []), *(source.get("sitemap_urls") or [])]:
            host = _host(str(url or ""))
            if host:
                output.setdefault(host, source)
    return output


def match_source_by_domain(domain_map: Dict[str, Dict[str, Any]], domain: str) -> Dict[str, Any] | None:
    host = _host(domain if "://" in domain else f"https://{domain}")
    while host:
        match = domain_map.get(host)
        if match is not None:
            return match
        parts = host.split(".", 1)
        host = parts[1] if len(parts) == 2 else ""
    return None


def select_sources(
    sources: List[Dict[str, Any]],
    *,
    countries: List[str] | None = None,
    languages: List[str] | None = None,
    regions: List[str] | None = None,
) -> List[Dict[str, Any]]:
    country_set = {str(item or "").upper() for item in countries or [] if str(item or "").strip()}
    language_set = {str(item or "").lower() for item in languages or [] if str(item or "").strip()}
    region_set = {str(item or "").lower() for item in regions or [] if str(item or "").strip()}
    output = []
    for source in sources:
        if not bool(source.get("enabled", True)):
            continue
        if country_set and str(source.get("country") or "").upper() not in country_set:
            continue
        if language_set and str(source.get("language") or "").lower() not in language_set:
            continue
        source_regions = {str(item or "").lower() for item in source.get("regions") or []}
        if region_set and not region_set.intersection(source_regions):
            continue
        output.append(source)
    return output


def _valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _host(value: str) -> str:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host
