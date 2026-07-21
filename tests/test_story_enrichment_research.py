from __future__ import annotations

from datetime import datetime, timezone
from threading import Barrier, Lock
import unittest

from mydailynews.app.models import HeadlineDecision, NewsCandidate, SelectedArticle
from mydailynews.enrichment.models import ResearchResult
from mydailynews.enrichment.research import StoryResearchCollector
from mydailynews.retrieval.ddg import DDGSearchResult, DuckDuckGoSearchRetriever


PUBLISHED_AT = datetime(2099, 1, 1, tzinfo=timezone.utc)


class FakeSearchRetriever:
    def __init__(self, results_by_query: dict[str, list[DDGSearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.errors: list[str] = []
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[DDGSearchResult]:
        self.calls.append((query, limit))
        return list(self.results_by_query.get(query, []))[:limit]


class FakeArticleRetriever:
    def __init__(self) -> None:
        self.fetched_urls: list[str] = []

    def fetch_text_with_url(self, url: str) -> tuple[str, str, str]:
        self.fetched_urls.append(url)
        return f"Fetched article text for {url}. " * 20, "ok", url


def _selected(candidate_id: str, title: str, *, url: str) -> SelectedArticle:
    candidate = NewsCandidate(
        id=candidate_id,
        source="Selected Source",
        category="general",
        title=title,
        url=url,
        snippet=f"Snippet for {title} with chip supply context.",
        published_at=PUBLISHED_AT,
        metadata={"topic_name": "Technology policy"},
    )
    return SelectedArticle(
        candidate=candidate,
        decision=HeadlineDecision(candidate_id, score=8.0, topic="Technology policy"),
        article_text=f"Full article text about {title}.",
        extraction_status="ok",
    )


class StoryEnrichmentResearchTests(unittest.TestCase):
    def test_page_fetches_are_parallel_ordered_and_failure_isolated(self) -> None:
        class CoordinatedRetriever:
            def __init__(self) -> None:
                self.barrier = Barrier(2)
                self.lock = Lock()
                self.active = 0
                self.max_active = 0

            def fetch_text_with_url(self, url: str) -> tuple[str, str, str]:
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    if url.endswith("/bad"):
                        raise RuntimeError("boom")
                    self.barrier.wait(timeout=2)
                    return f"Text for {url}", "ok", url
                finally:
                    with self.lock:
                        self.active -= 1

        fetcher = CoordinatedRetriever()
        warnings: list[str] = []
        collector = StoryResearchCollector(
            FakeSearchRetriever({}),
            fetcher,
            warning_sink=warnings.append,
            max_fetch_workers=2,
        )
        ranked = [
            ResearchResult(f"r{index}", "query", f"Title {index}", url, "Snippet", "source")
            for index, url in enumerate(
                ["https://example.test/a", "https://example.test/b", "https://example.test/bad"],
                1,
            )
        ]

        results = collector._fetch_ranked_results(
            ranked,
            story_articles=[],
            max_fetched_research_pages_per_story=3,
        )

        self.assertEqual(fetcher.max_active, 2)
        self.assertEqual([result.url for result in results], [item.url for item in ranked])
        self.assertEqual([result.status for result in results], ["ok", "ok", "fetch_exception_RuntimeError"])
        self.assertEqual(len(warnings), 1)

    def test_collector_dedupes_ranks_fetches_and_skips_selected_article_urls(self) -> None:
        selected = _selected(
            "a",
            "Chip export scrutiny expands",
            url="https://selected.example/story-a",
        )
        search = FakeSearchRetriever(
            {
                "chip supply": [
                    DDGSearchResult(
                        query="chip supply",
                        title="Chip supply scrutiny context",
                        url="https://research.example/context",
                        snippet="Chip supply chain context for export scrutiny.",
                        source="research.example",
                    ),
                    DDGSearchResult(
                        query="chip supply",
                        title="Selected copy of chip export scrutiny",
                        url=selected.candidate.url,
                        snippet="The already selected article.",
                        source="selected.example",
                    ),
                    DDGSearchResult(
                        query="chip supply",
                        title="Duplicate chip supply context",
                        url="https://research.example/context",
                        snippet="Duplicate URL should be ignored.",
                        source="research.example",
                    ),
                    DDGSearchResult(
                        query="chip supply",
                        title="Lower overlap report",
                        url="https://other.example/report",
                        snippet="A less relevant result.",
                        source="other.example",
                    ),
                ]
            }
        )
        article_fetcher = FakeArticleRetriever()
        collector = StoryResearchCollector(search, article_fetcher)

        results = collector.collect(
            queries=["chip supply"],
            story_title="Chip supply scrutiny",
            story_articles=[selected],
            search_results_per_query=10,
            max_fetched_research_pages_per_story=1,
        )

        self.assertEqual(search.calls, [("chip supply", 10)])
        self.assertEqual(article_fetcher.fetched_urls, ["https://research.example/context"])
        self.assertEqual([result.id for result in results], ["research-1", "research-2", "research-3"])
        self.assertEqual(
            len([result for result in results if result.url == "https://research.example/context"]),
            1,
        )
        selected_duplicate = next(result for result in results if result.url == selected.candidate.url)
        self.assertEqual(selected_duplicate.status, "selected_article_duplicate")

    def test_ddg_parser_decodes_html_escaped_result_urls_and_snippets(self) -> None:
        html_text = """
        <div class="result">
          <a class="result__a" href="/l/?kh=-1&amp;uddg=https%3A%2F%2Fexample.com%2Fstory%3Fid%3D1">Example &amp; Story</a>
          <div class="result__snippet">Alpha <b>beta</b> &amp; context.</div>
        </div>
        <div class="result">
          <a class="result__a" href="https://direct.example/path">Direct result</a>
          <a class="result__snippet">Direct <span>snippet.</span></a>
        </div>
        <div class="result">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fstory%3Fid%3D1">Duplicate</a>
          <div class="result__snippet">Duplicate result.</div>
        </div>
        """

        results = DuckDuckGoSearchRetriever.parse_html("chip supply", html_text, 10)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].url, "https://example.com/story?id=1")
        self.assertEqual(results[0].title, "Example & Story")
        self.assertEqual(results[0].snippet, "Alpha beta & context.")
        self.assertEqual(results[0].source, "example.com")
        self.assertEqual(results[1].url, "https://direct.example/path")
        self.assertEqual(results[1].snippet, "Direct snippet.")


if __name__ == "__main__":
    unittest.main()
