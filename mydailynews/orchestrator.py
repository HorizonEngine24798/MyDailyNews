from __future__ import annotations

"""Horizon-inspired staged orchestrator for MyDailyNews.

This borrows the architecture of Horizon's fetch -> dedupe -> AI score -> enrich
-> summarize pipeline, but keeps only RSS/news discovery and local-first outputs.

Horizon: https://github.com/Thysrael/Horizon
License: MIT
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, List

from .ai.client import LocalAIClient
from .ai.headline_analyzer import HeadlineAnalyzer
from .brief import BriefGenerator, brief_metadata
from .enrichment import SimpleEnricher
from .models import AppConfig, HeadlineDecision, NewsCandidate, PipelineResult, SelectedArticle
from .output import write_json, write_markdown
from .retrieval.article import ArticleRetriever
from .scrapers.rss import RSSScraper
from .utils import normalize_url, utc_now


class NewsOrchestrator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ai_client = LocalAIClient(config.ai)
        self.headline_analyzer = HeadlineAnalyzer(self.ai_client)
        self.article_retriever = ArticleRetriever(config.user_agent, config.filtering.article_text_max_chars)
        self.enricher = SimpleEnricher(config, self.ai_client)
        self.brief_generator = BriefGenerator(
            self.ai_client,
            config.enrichment.max_context_chars_per_article,
        )

    def run(self) -> PipelineResult:
        since = utc_now() - timedelta(hours=self.config.filtering.time_window_hours)
        candidates = self.fetch_headlines(since)
        unique_candidates = self.merge_url_duplicates(candidates)
        newest_first = sorted(
            unique_candidates,
            key=lambda item: item.published_at or since,
            reverse=True,
        )
        limited_candidates = newest_first[: self.config.filtering.max_candidates_for_ai]
        decisions = self.headline_analyzer.analyze(limited_candidates, self.config.user_memory)
        selected = self.select_articles(limited_candidates, decisions)

        for article in selected:
            article.article_text, article.extraction_status = self.article_retriever.fetch_text(article.candidate.url)
            if not article.article_text:
                article.article_text = article.candidate.snippet
            self.enricher.enrich(article)

        date = utc_now().strftime("%Y-%m-%d")
        brief = self.brief_generator.generate(selected, self.config.user_memory, date)
        brief["metadata"] = brief_metadata(
            date=date,
            model=self.config.ai.model,
            candidate_count=len(unique_candidates),
            selected_count=len(selected),
        )

        output_dir = Path(self.config.output_dir)
        markdown_path = output_dir / f"{date}_brief.md"
        json_path = output_dir / f"{date}_brief.json"
        write_markdown(markdown_path, brief)
        write_json(json_path, brief)

        return PipelineResult(
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            candidate_count=len(unique_candidates),
            selected_count=len(selected),
        )

    def fetch_headlines(self, since) -> List[NewsCandidate]:
        scraper = RSSScraper(
            self.config.rss_sources,
            self.config.user_agent,
            self.config.filtering.max_headlines_per_source,
        )
        return scraper.fetch(since)

    @staticmethod
    def merge_url_duplicates(candidates: List[NewsCandidate]) -> List[NewsCandidate]:
        by_url: Dict[str, NewsCandidate] = {}
        for candidate in candidates:
            key = normalize_url(candidate.url)
            existing = by_url.get(key)
            if not existing or len(candidate.snippet) > len(existing.snippet):
                by_url[key] = candidate
        return list(by_url.values())

    def select_articles(
        self,
        candidates: List[NewsCandidate],
        decisions: Dict[str, HeadlineDecision],
    ) -> List[SelectedArticle]:
        selected: List[SelectedArticle] = []
        seen_duplicate_targets: set[str] = set()
        sorted_candidates = sorted(
            candidates,
            key=lambda item: decisions.get(item.id, HeadlineDecision(item.id, 0, "", "")).score,
            reverse=True,
        )

        for candidate in sorted_candidates:
            decision = decisions.get(candidate.id)
            if not decision:
                continue
            if decision.score < self.config.filtering.headline_score_cutoff:
                continue
            if decision.duplicate_of:
                continue
            if candidate.id in seen_duplicate_targets:
                continue
            seen_duplicate_targets.add(candidate.id)
            selected.append(SelectedArticle(candidate=candidate, decision=decision))
            if len(selected) >= self.config.filtering.max_selected_articles:
                break
        return selected
