from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from html import escape
from typing import Dict, List

from mydailynews.app.models import NewsCandidate, RSSSourceConfig, RunSourceSnapshot
from mydailynews.common.cache import HTTPFetchResult
from mydailynews.evaluation.schema import EvalArcInput, EvalDocument
from mydailynews.scrapers.rss import RSSScraper


@dataclass(frozen=True)
class FixtureFetch:
    date: str
    candidates: List[NewsCandidate]


class FixtureNewsProvider:
    """An offline source adapter with the same candidate/body boundary as retrieval.

    The provider receives only public fixture fields. Gold labels, fact IDs, trap
    tags, and expected decisions never enter the retrieval-facing object.
    """

    def __init__(self, arc: EvalArcInput) -> None:
        self.arc = arc
        self._documents: Dict[str, EvalDocument] = {
            document.id: document
            for day in arc.days
            for document in day.documents
        }

    def fetch(self, date: str) -> FixtureFetch:
        for day in self.arc.days:
            if day.date == date:
                return FixtureFetch(date=date, candidates=[item.to_candidate() for item in day.documents])
        return FixtureFetch(date=date, candidates=[])

    def fetch_via_rss(self, date: str) -> FixtureFetch:
        """Exercise the production RSS parser while preserving fixture IDs.

        RSS normalization intentionally creates its own candidate IDs and uses
        the configured feed as the source. The fixture boundary restores fields
        that are private to the synthetic document after parsing so predictions
        still join cleanly to the answer key.
        """

        documents = next((day.documents for day in self.arc.days if day.date == date), [])
        if not documents:
            return FixtureFetch(date=date, candidates=[])
        published = [
            candidate.published_at
            for document in documents
            for candidate in [document.to_candidate()]
            if candidate.published_at is not None
        ]
        since = min(published, default=datetime.now(timezone.utc)) - timedelta(seconds=1)
        candidates = self.rss_scraper(date).fetch(since)
        by_id = {document.id: document for document in documents}
        normalized: List[NewsCandidate] = []
        for candidate in candidates:
            fixture_id = str(candidate.metadata.get("entry_id", "") or "").strip()
            document = by_id.get(fixture_id)
            if document is None:
                continue
            candidate.id = document.id
            candidate.source = document.source
            candidate.category = document.category
            candidate.tags = list(document.tags)
            candidate.metadata.update(
                {
                    "fixture_source": True,
                    "fixture_transport": "rss",
                }
            )
            normalized.append(candidate)
        return FixtureFetch(date=date, candidates=normalized)

    def article_text(self, candidate_id: str) -> str:
        document = self._documents.get(str(candidate_id or ""))
        return document.body if document is not None else ""

    def snapshot(self, date: str) -> RunSourceSnapshot:
        fetched = self.fetch(date)
        return RunSourceSnapshot(
            fetched_since=min(
                (candidate.published_at for candidate in fetched.candidates if candidate.published_at is not None),
                default=datetime.now(timezone.utc),
            ),
            rss_candidates=list(fetched.candidates),
            merged_candidates=list(fetched.candidates),
            metadata={"provider": "offline_fixture", "arc_id": self.arc.id, "date": date},
        )

    def rss_scraper(self, date: str) -> RSSScraper:
        """Return the real RSS parser wired to an offline fixture transport."""

        feed_url = self.feed_url(date)
        scraper = RSSScraper(
            [
                RSSSourceConfig(
                    name=f"Fixture {self.arc.id}",
                    source_id=f"fixture:{self.arc.id}",
                    url=feed_url,
                    category="synthetic",
                    tags=["offline-eval"],
                )
            ],
            user_agent="MyDailyNews-Evaluation/1.0",
            max_per_source=100,
        )
        scraper.http = FixtureHttpTransport(self)
        return scraper

    def feed_url(self, date: str) -> str:
        return f"https://fixture.invalid/{self.arc.id}/{date}/feed.xml"

    def render_rss(self, date: str) -> str:
        documents = next((day.documents for day in self.arc.days if day.date == date), [])
        items = []
        for document in documents:
            published = document.to_candidate().published_at
            published_text = format_datetime(published) if published is not None else ""
            items.append(
                "<item>"
                f"<guid>{escape(document.id)}</guid>"
                f"<title>{escape(document.title)}</title>"
                f"<link>{escape(document.url)}</link>"
                f"<description>{escape(document.snippet)}</description>"
                f"<pubDate>{escape(published_text)}</pubDate>"
                "</item>"
            )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<rss version=\"2.0\"><channel>"
            f"<title>{escape(self.arc.id)}</title>"
            + "".join(items)
            + "</channel></rss>"
        )


class FixtureHttpTransport:
    """A no-network HTTP seam for RSS and article-body integration tests."""

    def __init__(self, provider: FixtureNewsProvider) -> None:
        self.provider = provider
        self.calls: List[str] = []

    def get_text(self, url: str, **_kwargs) -> HTTPFetchResult:
        self.calls.append(str(url))
        for day in self.provider.arc.days:
            if url == self.provider.feed_url(day.date):
                return HTTPFetchResult(
                    ok=True,
                    status_code=200,
                    text=self.provider.render_rss(day.date),
                    headers={"Content-Type": "application/rss+xml"},
                    cache_state="network",
                )
        document = self.provider._documents.get(str(url))
        if document is None:
            document = next(
                (item for item in self.provider._documents.values() if item.url == str(url)),
                None,
            )
        if document is not None:
            html = f"<html><body><article><h1>{escape(document.title)}</h1><p>{escape(document.body)}</p></article></body></html>"
            return HTTPFetchResult(
                ok=True,
                status_code=200,
                text=html,
                headers={"Content-Type": "text/html"},
                cache_state="network",
            )
        return HTTPFetchResult(
            ok=False,
            status_code=404,
            text="",
            headers={},
            cache_state="network",
        )
