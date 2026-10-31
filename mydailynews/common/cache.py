from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode

import requests

from mydailynews.common.storage import CACHE_DATABASE_NAME, claim_migration, open_database, set_metadata
from mydailynews.diagnostics.debug import DebugLogger


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def retry_after_seconds(headers: Dict[str, Any] | None, default: float, maximum: float = 60.0) -> float:
    value = next((raw for key, raw in (headers or {}).items() if str(key).lower() == "retry-after"), "")
    if not str(value).strip():
        return float(default)
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - _utc_now()).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return float(default)
    return max(0.0, min(float(maximum), seconds))


@dataclass
class CachedHttpResponse:
    url: str
    status_code: int
    body: str
    fetched_at: datetime
    etag: str = ""
    last_modified: str = ""
    content_type: str = ""


class HTTPCache:
    """Minimal SQLite cache for HTTP GET content."""

    def __init__(self, root_dir: str, namespace: str, enabled: bool = True, debug: DebugLogger | None = None) -> None:
        self.enabled = enabled
        self.namespace = str(namespace)
        self.root = Path(root_dir) / "http" / namespace
        self.database_path = Path(root_dir) / CACHE_DATABASE_NAME
        self.debug = debug or DebugLogger(False)
        if self.enabled:
            self._import_legacy_once()

    def prune_older_than_days(self, retention_days: int) -> int:
        if not self.enabled:
            return 0
        days = max(0, int(retention_days))
        if days <= 0:
            return 0
        cutoff = (_utc_now() - timedelta(days=days)).isoformat()
        with open_database(self.database_path) as database:
            cursor = database.execute(
                "DELETE FROM http_cache WHERE namespace = ? AND fetched_at < ?",
                (self.namespace, cutoff),
            )
            removed = max(0, int(cursor.rowcount))
        if removed:
            self.debug.log("cache.http", "pruned", removed=removed, retention_days=days)
        return removed

    def get(self, url: str) -> CachedHttpResponse | None:
        if not self.enabled:
            return None
        record = self._read_record(url)
        if record is None:
            return None
        try:
            response = CachedHttpResponse(
                url=record.get("url", url),
                status_code=int(record.get("status_code", 200)),
                body=str(record.get("body", "")),
                fetched_at=_parse_iso(str(record["fetched_at"])),
                etag=str(record.get("etag", "")),
                last_modified=str(record.get("last_modified", "")),
                content_type=str(record.get("content_type", "")),
            )
        except Exception:
            return None
        return response

    def put(self, url: str, status_code: int, body: str, headers: Dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        headers = headers or {}
        record = {
            "url": url,
            "status_code": int(status_code),
            "body": body or "",
            "fetched_at": _utc_now().isoformat(),
            "etag": headers.get("ETag", ""),
            "last_modified": headers.get("Last-Modified", ""),
            "content_type": headers.get("Content-Type", ""),
        }
        self._write_record(url, record)

    def _read_record(self, url: str) -> Dict[str, Any] | None:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        with open_database(self.database_path) as database:
            row = database.execute(
                "SELECT url, status_code, body, fetched_at, etag, last_modified, content_type "
                "FROM http_cache WHERE namespace = ? AND cache_key = ?",
                (self.namespace, key),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def _write_record(self, url: str, record: Dict[str, Any]) -> None:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        with open_database(self.database_path) as database:
            database.execute(
                "INSERT INTO http_cache(namespace, cache_key, url, status_code, body, fetched_at, "
                "etag, last_modified, content_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(namespace, cache_key) DO UPDATE SET "
                "url=excluded.url, status_code=excluded.status_code, body=excluded.body, "
                "fetched_at=excluded.fetched_at, etag=excluded.etag, "
                "last_modified=excluded.last_modified, content_type=excluded.content_type",
                (
                    self.namespace,
                    key,
                    str(record.get("url", url)),
                    int(record.get("status_code", 200)),
                    str(record.get("body", "")),
                    str(record.get("fetched_at", _utc_now().isoformat())),
                    str(record.get("etag", "")),
                    str(record.get("last_modified", "")),
                    str(record.get("content_type", "")),
                ),
            )

    def _import_legacy_once(self) -> None:
        marker = f"migration.http_cache.{self.namespace}.v1"
        with open_database(self.database_path) as database:
            if not claim_migration(database, marker):
                return
            rows = []
            for path in self.root.glob("*.json"):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(raw, dict) or not raw.get("fetched_at"):
                    continue
                rows.append(
                    (
                        self.namespace,
                        path.stem,
                        str(raw.get("url", "")),
                        int(raw.get("status_code", 200)),
                        str(raw.get("body", "")),
                        str(raw["fetched_at"]),
                        str(raw.get("etag", "")),
                        str(raw.get("last_modified", "")),
                        str(raw.get("content_type", "")),
                    )
                )
            database.executemany(
                "INSERT OR IGNORE INTO http_cache(namespace, cache_key, url, status_code, body, fetched_at, "
                "etag, last_modified, content_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            set_metadata(database, marker)


class JSONCache:
    """Simple persistent JSON-value cache backed by SQLite."""

    def __init__(self, root_dir: str, namespace: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.namespace = str(namespace)
        self.root = Path(root_dir) / "json" / namespace
        self.database_path = Path(root_dir) / CACHE_DATABASE_NAME
        if self.enabled:
            self._import_legacy_once()

    @staticmethod
    def make_key(payload: str) -> str:
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str, max_age_seconds: int | None = None) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        with open_database(self.database_path) as database:
            row = database.execute(
                "SELECT cached_at, payload FROM json_cache WHERE namespace = ? AND cache_key = ?",
                (self.namespace, str(key)),
            ).fetchone()
        if row is None:
            return None
        try:
            cached_at = _parse_iso(str(row["cached_at"]))
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            if max_age_seconds is not None and max_age_seconds >= 0:
                if (_utc_now() - cached_at) > timedelta(seconds=max_age_seconds):
                    return None
            value = json.loads(str(row["payload"]))
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
            return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        with open_database(self.database_path) as database:
            database.execute(
                "INSERT INTO json_cache(namespace, cache_key, cached_at, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(namespace, cache_key) DO UPDATE SET "
                "cached_at=excluded.cached_at, payload=excluded.payload",
                (
                    self.namespace,
                    str(key),
                    _utc_now().isoformat(),
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def prune_older_than_days(self, retention_days: int) -> int:
        if not self.enabled:
            return 0
        days = max(0, int(retention_days))
        if days <= 0:
            return 0
        cutoff = (_utc_now() - timedelta(days=days)).isoformat()
        with open_database(self.database_path) as database:
            cursor = database.execute(
                "DELETE FROM json_cache WHERE namespace = ? AND cached_at < ?",
                (self.namespace, cutoff),
            )
            return max(0, int(cursor.rowcount))

    def _import_legacy_once(self) -> None:
        marker = f"migration.json_cache.{self.namespace}.v1"
        with open_database(self.database_path) as database:
            if not claim_migration(database, marker):
                return
            rows = []
            for path in self.root.glob("*.json"):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(raw, dict):
                    continue
                if "value" in raw and "cached_at" in raw:
                    value = raw.get("value")
                    cached_at = str(raw.get("cached_at", ""))
                else:
                    value = raw
                    cached_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                if not isinstance(value, dict):
                    continue
                rows.append(
                    (
                        self.namespace,
                        path.stem,
                        cached_at,
                        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                    )
                )
            database.executemany(
                "INSERT OR IGNORE INTO json_cache(namespace, cache_key, cached_at, payload) VALUES (?, ?, ?, ?)",
                rows,
            )
            set_metadata(database, marker)


@dataclass
class HTTPFetchResult:
    ok: bool
    status_code: int
    text: str
    headers: Dict[str, str]
    cache_state: str = "network"  # network | fresh_cache | cached_fallback


class CachedHttpClient:
    """HTTP GET helper backed by a local non-expiring cache."""

    CACHE_FIRST = "cache_first"
    NETWORK_FIRST = "network_first"
    NO_CACHE = "no_cache"

    def __init__(
        self,
        user_agent: str,
        cache: HTTPCache | None,
        debug: DebugLogger | None = None,
        cache_mode: str = CACHE_FIRST,
    ) -> None:
        self.user_agent = user_agent
        self.cache = cache
        self.debug = debug or DebugLogger(False)
        self.cache_mode = self._normalize_cache_mode(cache_mode)

    def get_text(
        self,
        url: str,
        *,
        timeout: int = 20,
        allow_redirects: bool = True,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
        cache_mode: str | None = None,
    ) -> HTTPFetchResult:
        mode = self._normalize_cache_mode(cache_mode or self.cache_mode)
        cache_key = self._cache_key(url, params)
        use_cache = self.cache if mode != self.NO_CACHE else None
        cached = use_cache.get(cache_key) if use_cache and mode == self.CACHE_FIRST else None
        if cached is not None:
            return HTTPFetchResult(
                ok=True,
                status_code=cached.status_code,
                text=cached.body,
                headers={},
                cache_state="fresh_cache",
            )

        request_headers = {"User-Agent": self.user_agent}
        request_headers.update(headers or {})

        try:
            response = requests.get(
                url,
                params=params,
                headers=request_headers,
                timeout=timeout,
                allow_redirects=allow_redirects,
            )
        except requests.RequestException:
            fallback = use_cache.get(cache_key) if use_cache and mode == self.NETWORK_FIRST else None
            if fallback is not None:
                return HTTPFetchResult(
                    ok=True,
                    status_code=fallback.status_code,
                    text=fallback.body,
                    headers={},
                    cache_state="cached_fallback",
                )
            return HTTPFetchResult(ok=False, status_code=0, text="", headers={}, cache_state="network")

        if response.status_code >= 400:
            return HTTPFetchResult(
                ok=False,
                status_code=response.status_code,
                text=response.text,
                headers=dict(response.headers),
                cache_state="network",
            )

        body = response.text
        if use_cache:
            use_cache.put(cache_key, response.status_code, body, headers=dict(response.headers))
        return HTTPFetchResult(
            ok=True,
            status_code=response.status_code,
            text=body,
            headers=dict(response.headers),
            cache_state="network",
        )

    @staticmethod
    def _cache_key(url: str, params: Dict[str, Any] | None) -> str:
        if not params:
            return url
        query = urlencode(sorted((str(key), str(value)) for key, value in params.items()))
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{query}"

    @classmethod
    def _normalize_cache_mode(cls, value: str) -> str:
        mode = str(value or cls.CACHE_FIRST).strip().lower()
        if mode in {cls.CACHE_FIRST, cls.NETWORK_FIRST, cls.NO_CACHE}:
            return mode
        return cls.CACHE_FIRST
