from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock, get_ident
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests

from .debug import DebugLogger


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

    def is_fresh(self, max_age: Optional[timedelta]) -> bool:
        if max_age is None:
            return True
        return (_utc_now() - self.fetched_at) <= max_age


class HTTPCache:
    """Minimal file cache for HTTP GET content with conditional revalidation metadata."""

    def __init__(self, root_dir: str, namespace: str, enabled: bool = True, debug: DebugLogger | None = None) -> None:
        self.enabled = enabled
        self.root = Path(root_dir) / "http" / namespace
        self.debug = debug or DebugLogger(False)
        self._lock = RLock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def conditional_headers(self, url: str) -> Dict[str, str]:
        cached = self.get(url)
        if cached is None:
            return {}
        headers: Dict[str, str] = {}
        if cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        return headers

    def get(self, url: str, max_age: timedelta | None = None) -> CachedHttpResponse | None:
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
        if not response.is_fresh(max_age):
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

    def touch(self, url: str, headers: Dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        record = self._read_record(url)
        if record is None:
            return
        record["fetched_at"] = _utc_now().isoformat()
        if headers:
            if headers.get("ETag"):
                record["etag"] = headers["ETag"]
            if headers.get("Last-Modified"):
                record["last_modified"] = headers["Last-Modified"]
            if headers.get("Content-Type"):
                record["content_type"] = headers["Content-Type"]
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
            temp_path.replace(path)


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
            temp_path.replace(path)


@dataclass
class HTTPFetchResult:
    ok: bool
    status_code: int
    text: str
    headers: Dict[str, str]
    cache_state: str = "network"  # network | fresh_cache | revalidated | stale_cache


class CachedHttpClient:
    """HTTP GET helper that supports local freshness + conditional requests."""

    def __init__(
        self,
        user_agent: str,
        cache: HTTPCache | None,
        fresh_seconds: int,
        debug: DebugLogger | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.cache = cache
        self.fresh_seconds = max(0, int(fresh_seconds))
        self.debug = debug or DebugLogger(False)

    def get_text(
        self,
        url: str,
        *,
        timeout: int = 20,
        allow_redirects: bool = True,
        params: Dict[str, Any] | None = None,
    ) -> HTTPFetchResult:
        cache_key = self._cache_key(url, params)
        max_age = timedelta(seconds=self.fresh_seconds) if self.fresh_seconds > 0 else None
        cached = self.cache.get(cache_key, max_age=max_age) if self.cache else None
        if cached is not None:
            return HTTPFetchResult(
                ok=True,
                status_code=cached.status_code,
                text=cached.body,
                headers={},
                cache_state="fresh_cache",
            )

        headers = {"User-Agent": self.user_agent}
        if self.cache:
            headers.update(self.cache.conditional_headers(cache_key))

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                allow_redirects=allow_redirects,
            )
        except requests.RequestException:
            stale = self.cache.get(cache_key) if self.cache else None
            if stale is not None:
                return HTTPFetchResult(
                    ok=True,
                    status_code=stale.status_code,
                    text=stale.body,
                    headers={},
                    cache_state="stale_cache",
                )
            return HTTPFetchResult(ok=False, status_code=0, text="", headers={}, cache_state="network")

        if response.status_code == 304 and self.cache:
            stale = self.cache.get(cache_key)
            if stale is not None:
                self.cache.touch(cache_key, headers=dict(response.headers))
                return HTTPFetchResult(
                    ok=True,
                    status_code=304,
                    text=stale.body,
                    headers=dict(response.headers),
                    cache_state="revalidated",
                )

        if response.status_code >= 400:
            return HTTPFetchResult(
                ok=False,
                status_code=response.status_code,
                text="",
                headers=dict(response.headers),
                cache_state="network",
            )

        body = response.text
        if self.cache:
            self.cache.put(cache_key, response.status_code, body, headers=dict(response.headers))
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
