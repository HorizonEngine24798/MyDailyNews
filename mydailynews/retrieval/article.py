from __future__ import annotations

import base64
import json
import re
from typing import Tuple
from urllib.parse import unquote, urlparse

import requests
import trafilatura

from mydailynews.common.cache import CachedHttpClient, HTTPCache
from mydailynews.diagnostics.debug import DebugLogger, safe_url
from mydailynews.common.utils import normalize_whitespace


GOOGLE_NEWS_HOST = "news.google.com"
GOOGLE_NEWS_RESOLVE_ENDPOINT = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
GOOGLE_NEWS_RPC_ID = "Fbv4je"
ARTICLE_OK_MIN_CHARS = 250


class ArticleRetriever:
    def __init__(
        self,
        user_agent: str,
        max_chars: int,
        http_cache: HTTPCache | None = None,
        debug: DebugLogger | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.max_chars = max_chars
        self.debug = debug or DebugLogger(False)
        self._resolved_url_cache: dict[str, str] = {}
        self.http = CachedHttpClient(
            user_agent=user_agent,
            cache=http_cache,
            debug=self.debug,
        )

    def fetch_text_with_url(self, url: str) -> Tuple[str, str, str]:
        self.debug.log("article.fetch", "starting", url=safe_url(url), max_chars=self.max_chars)
        response, effective_url = self._fetch_article_response(url)
        if not response.ok:
            self.debug.log("article.fetch", "http_error", status=response.status_code, url=safe_url(effective_url))
            status = f"http_{response.status_code}" if response.status_code else "request_failed"
            return "", status, effective_url
        try:
            text = trafilatura.extract(
                response.text,
                url=effective_url,
                include_comments=False,
                include_images=False,
                include_links=False,
                include_tables=False,
            )
            cleaned = normalize_whitespace(text or "")
            if len(cleaned) < ARTICLE_OK_MIN_CHARS:
                self.debug.log(
                    "article.fetch",
                    "short_text",
                    chars=len(cleaned),
                    url=safe_url(effective_url),
                    cache=response.cache_state,
                )
                return cleaned[: self.max_chars], "short_text", effective_url
            self.debug.log(
                "article.fetch",
                "complete",
                chars=len(cleaned),
                used_chars=min(len(cleaned), self.max_chars),
                cache=response.cache_state,
            )
            return cleaned[: self.max_chars], "ok", effective_url
        except Exception:
            self.debug.log("article.fetch", "extract_failed", url=safe_url(effective_url))
            return "", "extract_failed", effective_url

    def _fetch_article_response(self, url: str):
        if not self._is_google_news_article_url(url):
            return self.http.get_text(url, timeout=20, allow_redirects=True), url

        resolved_url = self._resolve_google_news_url(url)
        if resolved_url != url:
            response = self.http.get_text(resolved_url, timeout=20, allow_redirects=True)
            return response, resolved_url

        return self.http.get_text(url, timeout=20, allow_redirects=True), url

    def _resolve_google_news_url(self, url: str) -> str:
        cached = self._resolved_url_cache.get(url)
        if cached:
            return cached

        decoded = self._decode_embedded_google_news_url(url)
        if decoded:
            self._resolved_url_cache[url] = decoded
            self.debug.log("article.fetch", "google_news_decoded", url=safe_url(url), resolved=safe_url(decoded))
            return decoded

        wrapper = self.http.get_text(url, timeout=20, allow_redirects=True)
        if not wrapper.ok:
            self.debug.log(
                "article.fetch",
                "google_news_resolve_http_error",
                status=wrapper.status_code,
                url=safe_url(url),
            )
            return url

        resolved = self._resolve_google_news_from_html(url, wrapper.text)
        if resolved:
            self._resolved_url_cache[url] = resolved
            self.debug.log("article.fetch", "google_news_resolved", url=safe_url(url), resolved=safe_url(resolved))
            return resolved

        self.debug.log("article.fetch", "google_news_resolve_failed", url=safe_url(url), cache=wrapper.cache_state)
        return url

    def _resolve_google_news_from_html(self, url: str, html_text: str) -> str:
        article_id = self._google_news_article_id(url)
        signature = self._html_attr(html_text, "data-n-a-sg")
        timestamp = self._html_attr(html_text, "data-n-a-ts")
        if not article_id or not signature or not timestamp:
            return ""

        request_payload = [
            "garturlreq",
            [
                [
                    "en-US",
                    "US",
                    ["FINANCE_TOP_INDICES", "GENESIS_PUBLISHER_SECTION", "WEB_TEST_1_0_0"],
                    None,
                    None,
                    1,
                    1,
                    "US:en",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    False,
                    5,
                ],
                "en-US",
                "US",
                1,
                [2, 3, 4, 8],
                1,
                0,
                "655000234",
                0,
                0,
                None,
                0,
            ],
            article_id,
            timestamp,
            signature,
        ]
        f_req = json.dumps(
            [[[GOOGLE_NEWS_RPC_ID, json.dumps(request_payload, separators=(",", ":")), None, "generic"]]],
            separators=(",", ":"),
        )
        try:
            response = requests.post(
                GOOGLE_NEWS_RESOLVE_ENDPOINT,
                data={"f.req": f_req},
                headers={
                    "User-Agent": self.user_agent,
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                },
                timeout=20,
            )
        except Exception:
            return ""
        if response.status_code >= 400:
            return ""
        return self._extract_google_news_resolved_url(response.text)

    @staticmethod
    def _extract_google_news_resolved_url(response_text: str) -> str:
        json_start = response_text.find("[")
        if json_start < 0:
            return ""
        try:
            payload = json.loads(response_text[json_start:])
        except json.JSONDecodeError:
            return ""
        if not isinstance(payload, list):
            return ""
        for row in payload:
            if not isinstance(row, list) or len(row) < 3:
                continue
            if row[0] != "wrb.fr" or row[1] != GOOGLE_NEWS_RPC_ID or not isinstance(row[2], str):
                continue
            try:
                inner = json.loads(row[2])
            except json.JSONDecodeError:
                continue
            if isinstance(inner, list) and len(inner) >= 2 and inner[0] == "garturlres":
                resolved = str(inner[1] or "").strip()
                if resolved.startswith(("http://", "https://")):
                    return resolved
        return ""

    @classmethod
    def _decode_embedded_google_news_url(cls, url: str) -> str:
        article_id = cls._google_news_article_id(url)
        if not article_id:
            return ""
        try:
            padding = "=" * ((4 - len(article_id) % 4) % 4)
            decoded = base64.urlsafe_b64decode((article_id + padding).encode("ascii"))
        except Exception:
            return ""
        match = re.search(rb"https?://[^\x00-\x20\"<>]+", decoded)
        if not match:
            return ""
        try:
            return unquote(match.group(0).decode("utf-8", errors="ignore")).strip()
        except Exception:
            return ""

    @staticmethod
    def _google_news_article_id(url: str) -> str:
        try:
            parsed = urlparse(url)
        except Exception:
            return ""
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2 or parts[-2] not in {"articles", "read"}:
            return ""
        return parts[-1]

    @staticmethod
    def _html_attr(text: str, name: str) -> str:
        match = re.search(rf'{re.escape(name)}="([^"]+)"', text or "")
        return match.group(1).strip() if match else ""

    @staticmethod
    def _is_google_news_article_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        return parsed.netloc.lower() == GOOGLE_NEWS_HOST and any(
            marker in parsed.path for marker in ("/articles/", "/read/")
        )
