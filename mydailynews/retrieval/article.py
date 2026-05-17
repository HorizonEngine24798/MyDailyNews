from __future__ import annotations

from typing import Tuple

import requests
import trafilatura

from ..utils import normalize_whitespace


class ArticleRetriever:
    def __init__(self, user_agent: str, max_chars: int) -> None:
        self.user_agent = user_agent
        self.max_chars = max_chars

    def fetch_text(self, url: str) -> Tuple[str, str]:
        try:
            response = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=20, allow_redirects=True)
            if response.status_code >= 400:
                return "", f"http_{response.status_code}"
            text = trafilatura.extract(
                response.text,
                url=url,
                include_comments=False,
                include_images=False,
                include_links=False,
                include_tables=False,
            )
            cleaned = normalize_whitespace(text or "")
            if len(cleaned) < 250:
                return cleaned[: self.max_chars], "short_text"
            return cleaned[: self.max_chars], "ok"
        except requests.RequestException:
            return "", "request_failed"
        except Exception:
            return "", "extract_failed"
