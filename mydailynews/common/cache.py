from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock, get_ident
from typing import Any, Dict
from urllib.parse import urlencode

import requests

from mydailynews.diagnostics.debug import DebugLogger


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


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
    """Minimal file cache for HTTP GET content. Records do not expire locally."""

    def __init__(self, root_dir: str, namespace: str, enabled: bool = True, debug: DebugLogger | None = None) -> None:
        self.enabled = enabled
        self.root = Path(root_dir) / "http" / namespace
        self.debug = debug or DebugLogger(False)
        self._lock = RLock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def prune_older_than_days(self, retention_days: int) -> int:
        if not self.enabled:
            return 0
        days = max(0, int(retention_days))
        if days <= 0:
            return 0
        cutoff = _utc_now().timestamp() - (days * 24 * 60 * 60)
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    try:
                        path.unlink()
                    except PermissionError:
                        path.write_text("{}", encoding="utf-8")
                    removed += 1
            except OSError as exc:
                self.debug.log("cache.http", "prune_failed", file=path.name, error=type(exc).__name__)
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

    def _record_path(self, url: str) -> Path:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.root / f"{key}.json"

    def _read_record(self, url: str) -> Dict[str, Any] | None:
        path = self._record_path(url)
        if not path.exists():
            return None
        try:
            with self._lock:
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_record(self, url: str, record: Dict[str, Any]) -> None:
        path = self._record_path(url)
        payload = json.dumps(record, ensure_ascii=False)
        temp_path = path.with_suffix(path.suffix + f".tmp.{get_ident()}")
        with self._lock:
            temp_path.write_text(payload, encoding="utf-8")
            try:
                temp_path.replace(path)
            except PermissionError:
                path.write_text(payload, encoding="utf-8")
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


class JSONCache:
    """Simple persistent JSON value cache keyed by a stable hash."""

    def __init__(self, root_dir: str, namespace: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.root = Path(root_dir) / "json" / namespace
        self._lock = RLock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(payload: str) -> str:
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str, max_age_seconds: int | None = None) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            with self._lock:
                raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(raw, dict) and "value" in raw and "cached_at" in raw:
            try:
                cached_at = _parse_iso(str(raw["cached_at"]))
                if max_age_seconds is not None and max_age_seconds >= 0:
                    if (_utc_now() - cached_at) > timedelta(seconds=max_age_seconds):
                        return None
                value = raw["value"]
                return value if isinstance(value, dict) else None
            except Exception:
                return None
        return raw if isinstance(raw, dict) else None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self.root / f"{key}.json"
        payload = {
            "cached_at": _utc_now().isoformat(),
            "value": value,
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        temp_path = path.with_suffix(path.suffix + f".tmp.{get_ident()}")
        with self._lock:
            temp_path.write_text(serialized, encoding="utf-8")
            try:
                temp_path.replace(path)
            except PermissionError:
                path.write_text(serialized, encoding="utf-8")
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def prune_older_than_days(self, retention_days: int) -> int:
        if not self.enabled:
            return 0
        days = max(0, int(retention_days))
        if days <= 0:
            return 0
        cutoff = _utc_now() - timedelta(days=days)
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                cached_at = self._path_cached_at(path)
                if cached_at is not None:
                    is_stale = cached_at < cutoff
                else:
                    is_stale = path.stat().st_mtime < cutoff.timestamp()
                if path.is_file() and is_stale:
                    try:
                        path.unlink()
                    except PermissionError:
                        path.write_text("{}", encoding="utf-8")
                    removed += 1
            except OSError:
                continue
        return removed

    def _path_cached_at(self, path: Path) -> datetime | None:
        try:
            with self._lock:
                raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("cached_at"):
                cached_at = _parse_iso(str(raw["cached_at"]))
                return cached_at if cached_at.tzinfo else cached_at.replace(tzinfo=timezone.utc)
        except Exception:
            return None
        return None


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

        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
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
                text="",
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
