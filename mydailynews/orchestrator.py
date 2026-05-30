from __future__ import annotations

"""Horizon-inspired staged orchestrator for MyDailyNews.

This borrows the architecture of Horizon's fetch -> dedupe -> AI score -> enrich
-> summarize pipeline, but keeps only RSS/news discovery and local-first outputs.

Horizon: https://github.com/Thysrael/Horizon
License: MIT
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Dict, List
from urllib.parse import urlparse

from .ai.factory import create_ai_client
from .ai.headline_analyzer import HeadlineAnalyzer
from .cache import HTTPCache, JSONCache
from .brief import BriefGenerator, brief_metadata
from .debug import DebugLogger
from .enrichment import SimpleEnricher
from .models import (
    AppConfig,
    BriefOutput,
    FilteringConfig,
    HeadlineDecision,
    NewsCandidate,
    PipelineResult,
    PriorReport,
    RunSourceSnapshot,
    SelectedArticle,
    TopicConfig,
)
from .output import write_json, write_markdown
from .retrieval.article import ArticleRetriever
from .retrieval.google_news import GoogleNewsQueryRetriever
from .retrieval.reports import PriorReportRetriever
from .scrapers.rss import RSSScraper
from .utils import datetime_to_iso, normalize_url, utc_now


class NewsOrchestrator:
    def __init__(self, config: AppConfig, debug: bool = False) -> None:
        self.config = config
        self.debug = DebugLogger(debug)
        self.summary_ai_client = create_ai_client(config.ai_summary, self.debug)
        self.final_ai_client = create_ai_client(config.ai_final, self.debug)
        # Legacy compatibility alias for tools expecting one client.
        self.ai_client = self.summary_ai_client
        self.http_cache = HTTPCache(
            root_dir=config.cache.dir,
            namespace="shared",
            enabled=config.cache.enabled,
            debug=self.debug,
        )
        self.synth_cache = JSONCache(
            root_dir=config.cache.dir,
            namespace="synth",
            enabled=config.cache.enabled and config.cache.ai_enabled,
        )
        self.google_news_retriever = GoogleNewsQueryRetriever(
            config.google_news_source,
            config.user_agent,
            max_workers=config.runtime.max_http_workers,
            http_cache=self.http_cache,
            cache_fresh_seconds=config.cache.http_fresh_seconds,
            debug=self.debug,
        )
        self.prior_report_retriever = PriorReportRetriever(
            config.prior_reports_source,
            config.output_dir,
            self.debug,
        )
        self.warnings: List[str] = []

    def run(self) -> PipelineResult:
        now = utc_now()
        today = now.date()
        date = today.isoformat()
        self.warnings = []
        general_topics = [topic for topic in self.config.general_topics if topic.enabled]
        detailed_topics = [topic for topic in self.config.topics_to_examine if topic.enabled]
        self.debug.log(
            "pipeline",
            "starting",
            summary_model=self.config.ai_summary.effective_model_label,
            summary_backend=self.config.ai_summary.backend,
            summary_device=self.config.ai_summary.device,
            final_model=self.config.ai_final.effective_model_label,
            final_backend=self.config.ai_final.backend,
            final_device=self.config.ai_final.device,
            sources=len([source for source in self.config.rss_sources if source.enabled]),
            general_topics=len(general_topics),
            detailed_topics=len(detailed_topics),
            enrichment=self.config.enrichment.enabled,
            use_shared_snapshot=self.config.runtime.use_shared_snapshot,
        )
        prior_reports = self.fetch_prior_reports(today)
        snapshot = self._build_snapshot(now, general_topics, detailed_topics)
        outputs = [
            self._run_brief(
                name="general",
                output_suffix="general",
                topics=general_topics,
                filtering=self.config.general_filtering,
                prior_reports=prior_reports,
                now=now,
                date=date,
                snapshot=snapshot,
                brief_goal=(
                    "General daily news pass. Prefer breadth and usefulness over deep specialization. "
                    "Use the lower threshold to fill the brief with the strongest general stories, up to the configured article count. "
                    "Still avoid trivia, gossip, minor sports, and duplicate rewrites."
                ),
            ),
            self._run_brief(
                name="detailed",
                output_suffix="detailed",
                topics=detailed_topics,
                filtering=self.config.filtering,
                prior_reports=prior_reports,
                now=now,
                date=date,
                snapshot=snapshot,
                brief_goal=(
                    "Detailed topic investigation pass. Focus on the configured topics, identify major narratives, "
                    "compare with prior reports, and select sources that can deepen, challenge, or reshape those narratives."
                ),
            ),
        ]
        self.debug.log("pipeline", "complete", outputs=len(outputs), warnings=len(self.warnings))
        return PipelineResult(outputs=outputs, warnings=self.warnings)

    def _run_brief(
        self,
        name: str,
        output_suffix: str,
        topics: List[TopicConfig],
        filtering: FilteringConfig,
        prior_reports: List[PriorReport],
        now,
        date: str,
        snapshot: RunSourceSnapshot | None,
        brief_goal: str,
    ) -> BriefOutput:
        since = now - timedelta(hours=filtering.time_window_hours)
        run_warnings: List[str] = []
        self.debug.log(
            "brief.run",
            "starting",
            name=name,
            topics=len(topics),
            max_candidates=filtering.max_candidates_for_ai,
            ai_batch_size=filtering.max_headlines_per_ai_batch,
            cutoff=filtering.headline_score_cutoff,
            max_selected=filtering.max_selected_articles,
            fill=filtering.fill_selected_articles,
        )

        unique_candidates: List[NewsCandidate]
        raw_candidate_count = 0
        rss_candidate_count = 0
        topic_candidate_count = 0
        if snapshot:
            run_warnings.extend(str(item) for item in snapshot.metadata.get("warnings", []))
            rss_candidates, topic_candidates, unique_candidates = self._snapshot_candidates_for_brief(snapshot, since)
            raw_candidate_count = len(rss_candidates) + len(topic_candidates)
            rss_candidate_count = len(rss_candidates)
            topic_candidate_count = len(topic_candidates)
            self.debug.log(
                "headline.fetch",
                "reused_snapshot",
                brief=name,
                snapshot_since=snapshot.fetched_since,
                raw_candidates=raw_candidate_count,
                rss_candidates=rss_candidate_count,
                topic_candidates=topic_candidate_count,
                unique_candidates=len(unique_candidates),
                prior_reports=len(prior_reports),
            )
        else:
            candidates = self.fetch_headlines(since, filtering.max_headlines_per_source, run_warnings)
            topic_candidates = self.fetch_topic_headlines(topics, since, run_warnings)
            candidates.extend(topic_candidates)
            raw_candidate_count = len(candidates)
            rss_candidate_count = len(candidates) - len(topic_candidates)
            topic_candidate_count = len(topic_candidates)
            self.debug.log(
                "headline.fetch",
                "complete",
                brief=name,
                raw_candidates=raw_candidate_count,
                rss_candidates=rss_candidate_count,
                topic_candidates=topic_candidate_count,
                prior_reports=len(prior_reports),
            )
            unique_candidates = self.merge_url_duplicates(candidates)
            self.debug.log("headline.dedupe", "complete", brief=name, unique_candidates=len(unique_candidates))
        if not unique_candidates:
            run_warnings.append(f"{name}: No live headline candidates were fetched.")
        limited_candidates = self.limit_candidates_for_ai(unique_candidates, topics, filtering, since)
        self.debug.log("headline.limit", "complete", brief=name, candidates_for_ai=len(limited_candidates))

        # Batch size is configurable; smaller values trade speed for reliability on constrained hardware.
        headline_analyzer = HeadlineAnalyzer(
            self.summary_ai_client,
            max(1, int(filtering.max_headlines_per_ai_batch)),
            self.debug,
            cache=self.synth_cache,
            cache_ttl_seconds=self.config.cache.synth_fresh_seconds,
        )
        decisions = headline_analyzer.analyze(
            limited_candidates,
            self.config.user_memory,
            topics,
            brief_goal,
            brief_name=name,
        )
        run_warnings.extend(headline_analyzer.warnings)
        self.debug.log("headline.decisions", "complete", brief=name, decisions=len(decisions))
        selected = self.select_articles(limited_candidates, decisions, topics, filtering)
        self.debug.log("headline.select", "complete", brief=name, selected=len(selected))

        article_retriever = ArticleRetriever(
            self.config.user_agent,
            filtering.article_text_max_chars,
            http_cache=self.http_cache,
            cache_fresh_seconds=self.config.cache.http_fresh_seconds,
            debug=self.debug,
        )
        enricher = SimpleEnricher(
            self.config,
            http_cache=self.http_cache,
            debug=self.debug,
        )
        for article in selected:
            self.debug.log(
                "article",
                "selected",
                brief=name,
                score=article.decision.score,
                topic=article.decision.topic,
                source=article.candidate.source,
                title=article.candidate.title,
            )
        self._populate_article_texts(name, selected, article_retriever, run_warnings)
        enricher.enrich_many(selected, max_workers=self.config.runtime.max_enrichment_workers)

        if self.summary_ai_client is not self.final_ai_client:
            self.summary_ai_client.unload()

        brief_generator = BriefGenerator(
            self.final_ai_client,
            self.config.enrichment.max_context_chars_per_article,
            input_token_limit=self.config.ai_final.max_input_tokens,
            max_new_tokens=self.config.ai_final.max_new_tokens,
            debug=self.debug,
        )
        brief = brief_generator.generate(
            selected,
            self.config.user_memory,
            topics,
            prior_reports,
            brief_goal,
            date,
        )
        self.final_ai_client.unload()
        run_warnings.extend(enricher.warnings)
        run_warnings.extend(brief_generator.warnings)
        brief["metadata"] = brief_metadata(
            date=date,
            model=f"{self.config.ai_summary.backend}:{self.config.ai_summary.effective_model_label} -> "
            f"{self.config.ai_final.backend}:{self.config.ai_final.effective_model_label}",
            candidate_count=len(unique_candidates),
            selected_count=len(selected),
            topics=[topic.name for topic in topics],
            prior_reports_count=len(prior_reports),
            brief_name=name,
            warnings=run_warnings,
        )

        output_dir = Path(self.config.output_dir)
        markdown_path = output_dir / f"{date}_{output_suffix}_brief.md"
        json_path = output_dir / f"{date}_{output_suffix}_brief.json"
        write_markdown(markdown_path, brief)
        write_json(json_path, brief)
        self.warnings.extend(run_warnings)
        self.debug.log("brief.run", "complete", name=name, markdown=markdown_path, json=json_path, warnings=len(run_warnings))

        return BriefOutput(
            name=name,
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            candidate_count=len(unique_candidates),
            selected_count=len(selected),
            warnings=run_warnings,
        )

    def _build_snapshot(self, now, general_topics: List[TopicConfig], detailed_topics: List[TopicConfig]) -> RunSourceSnapshot | None:
        if not self.config.runtime.use_shared_snapshot:
            return None

        general_since = now - timedelta(hours=self.config.general_filtering.time_window_hours)
        detailed_since = now - timedelta(hours=self.config.filtering.time_window_hours)
        snapshot_since = min(general_since, detailed_since)
        max_headlines_per_source = max(
            self.config.general_filtering.max_headlines_per_source,
            self.config.filtering.max_headlines_per_source,
        )
        shared_topics = self._merge_topics_for_snapshot(general_topics, detailed_topics)

        snapshot_warnings: List[str] = []
        rss_candidates = self.fetch_headlines(snapshot_since, max_headlines_per_source, snapshot_warnings)
        topic_candidates = self.fetch_topic_headlines(shared_topics, snapshot_since, snapshot_warnings)
        merged_candidates = self.merge_url_duplicates(rss_candidates + topic_candidates)
        self.debug.log(
            "snapshot",
            "built",
            since=snapshot_since,
            rss_candidates=len(rss_candidates),
            topic_candidates=len(topic_candidates),
            merged_candidates=len(merged_candidates),
            warnings=len(snapshot_warnings),
        )
        return RunSourceSnapshot(
            fetched_since=snapshot_since,
            rss_candidates=rss_candidates,
            topic_candidates=topic_candidates,
            merged_candidates=merged_candidates,
            metadata={
                "warnings": snapshot_warnings,
                "max_headlines_per_source": max_headlines_per_source,
                "topic_count": len(shared_topics),
            },
        )

    @staticmethod
    def _merge_topics_for_snapshot(*topic_groups: List[TopicConfig]) -> List[TopicConfig]:
        merged: List[TopicConfig] = []
        seen: set[str] = set()
        for topics in topic_groups:
            for topic in topics:
                key = "|".join(
                    [
                        topic.name.strip().lower(),
                        topic.description.strip().lower(),
                        ",".join(query.strip().lower() for query in topic.queries if query.strip()),
                    ]
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(topic)
        return merged

    def _snapshot_candidates_for_brief(
        self,
        snapshot: RunSourceSnapshot,
        since,
    ) -> tuple[List[NewsCandidate], List[NewsCandidate], List[NewsCandidate]]:
        rss_candidates = [candidate for candidate in snapshot.rss_candidates if self._candidate_in_window(candidate, since)]
        topic_candidates = [candidate for candidate in snapshot.topic_candidates if self._candidate_in_window(candidate, since)]
        merged_candidates = [candidate for candidate in snapshot.merged_candidates if self._candidate_in_window(candidate, since)]
        return rss_candidates, topic_candidates, merged_candidates

    @staticmethod
    def _candidate_in_window(candidate: NewsCandidate, since) -> bool:
        published_at = candidate.published_at
        latest_iso = str(candidate.metadata.get("merged_latest_published_at", "")).strip()
        if latest_iso:
            try:
                merged_latest = datetime.fromisoformat(latest_iso)
                if merged_latest.tzinfo is not None:
                    if published_at is None or merged_latest > published_at:
                        published_at = merged_latest
            except ValueError:
                pass
        if published_at is None:
            return True
        return published_at >= since

    def _populate_article_texts(
        self,
        brief_name: str,
        selected: List[SelectedArticle],
        article_retriever: ArticleRetriever,
        warnings: List[str],
    ) -> None:
        if not selected:
            return
        worker_count = min(max(1, int(self.config.runtime.max_article_workers)), len(selected))
        self.debug.log("article.fetch", "batch_start", brief=brief_name, selected=len(selected), workers=worker_count)

        if worker_count <= 1:
            for article in selected:
                article.article_text, article.extraction_status = article_retriever.fetch_text(article.candidate.url)
                if not article.article_text:
                    article.article_text = article.candidate.snippet
            self.debug.log("article.fetch", "batch_complete", brief=brief_name, selected=len(selected), workers=worker_count)
            return

        results: dict[int, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(article_retriever.fetch_text, article.candidate.url): index
                for index, article in enumerate(selected)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                article = selected[index]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    warnings.append(f"article fetch {article.candidate.id}: worker_exception={type(exc).__name__}")
                    self.debug.log(
                        "article.fetch",
                        "worker_exception",
                        brief=brief_name,
                        article_id=article.candidate.id,
                        error=type(exc).__name__,
                    )
                    results[index] = ("", "worker_exception")

        for index, article in enumerate(selected):
            text, status = results.get(index, ("", "worker_missing"))
            article.article_text = text or article.candidate.snippet
            article.extraction_status = status
        self.debug.log("article.fetch", "batch_complete", brief=brief_name, selected=len(selected), workers=worker_count)

    def fetch_headlines(self, since, max_headlines_per_source: int, warnings: List[str]) -> List[NewsCandidate]:
        scraper = RSSScraper(
            self.config.rss_sources,
            self.config.user_agent,
            max_headlines_per_source,
            max_workers=self.config.runtime.max_http_workers,
            http_cache=self.http_cache,
            cache_fresh_seconds=self.config.cache.http_fresh_seconds,
            debug=self.debug,
        )
        candidates = scraper.fetch(since)
        warnings.extend(f"RSS: {error}" for error in scraper.errors)
        return candidates

    def fetch_topic_headlines(self, topics, since, warnings: List[str]) -> List[NewsCandidate]:
        candidates = self.google_news_retriever.fetch(topics, since)
        warnings.extend(f"Google News: {error}" for error in self.google_news_retriever.errors)
        return candidates

    def fetch_prior_reports(self, today):
        reports = self.prior_report_retriever.fetch(today)
        self.warnings.extend(f"Prior reports: {error}" for error in self.prior_report_retriever.errors)
        return reports

    @staticmethod
    def merge_url_duplicates(candidates: List[NewsCandidate]) -> List[NewsCandidate]:
        # Borrowed from Horizon's non-LLM cross-source URL merge strategy (MIT):
        # https://github.com/Thysrael/Horizon
        # Group by canonical URL key, keep the richest item, and merge metadata.
        by_url: Dict[str, List[NewsCandidate]] = {}
        for candidate in candidates:
            key = NewsOrchestrator._dedupe_url_key(candidate.url)
            by_url.setdefault(key, []).append(candidate)

        merged: List[NewsCandidate] = []
        for group in by_url.values():
            if len(group) == 1:
                only_item = group[0]
                if "merged_latest_published_at" not in only_item.metadata:
                    only_item.metadata["merged_latest_published_at"] = datetime_to_iso(only_item.published_at)
                merged.append(group[0])
                continue

            primary = max(group, key=lambda item: len(item.snippet or ""))
            merged_sources = sorted({item.source for item in group if item.source})
            latest_published_at = max(
                (item.published_at for item in group if item.published_at is not None),
                default=None,
            )
            merged_tags: List[str] = []
            seen_tags: set[str] = set()

            for item in group:
                for tag in item.tags:
                    if tag and tag not in seen_tags:
                        seen_tags.add(tag)
                        merged_tags.append(tag)

                for meta_key, meta_value in item.metadata.items():
                    if meta_key not in primary.metadata or not primary.metadata[meta_key]:
                        primary.metadata[meta_key] = meta_value

                if item is not primary and len(item.snippet or "") > len(primary.snippet or ""):
                    primary.snippet = item.snippet

            primary.tags = merged_tags
            primary.metadata["merged_sources"] = merged_sources
            primary.metadata["merged_count"] = len(group)
            primary.metadata["merged_latest_published_at"] = datetime_to_iso(latest_published_at)
            merged.append(primary)
        return merged

    @staticmethod
    def _dedupe_url_key(url: str) -> str:
        normalized = normalize_url(url)
        parsed = urlparse(normalized)
        host = parsed.hostname or ""
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/")
        return f"{host}{path}"

    def limit_candidates_for_ai(self, candidates: List[NewsCandidate], topics, filtering: FilteringConfig, since) -> List[NewsCandidate]:
        max_total = filtering.max_candidates_for_ai
        if max_total <= 0:
            return []

        candidates = self._dedupe_similar_titles(candidates)
        scored = self._heuristic_ranked_candidates(candidates, topics, since)
        if not scored:
            return []

        score_by_id = {item.id: score for item, score in scored}
        ranked = [item for item, _ in scored]
        pool_target = min(len(ranked), max_total * 2)
        nonnegative = [item for item in ranked if score_by_id.get(item.id, 0.0) >= 0.0]
        if len(nonnegative) < max_total:
            candidate_pool = ranked[:pool_target]
        else:
            candidate_pool = nonnegative[:pool_target]
        self.debug.log(
            "headline.heuristics",
            "prefilter_complete",
            input=len(candidates),
            pool=len(candidate_pool),
            max_total=max_total,
        )

        selected: List[NewsCandidate] = []
        selected_ids: set[str] = set()
        enabled_topics = [topic for topic in topics if topic.enabled]
        per_topic = max(1, max_total // max(1, len(enabled_topics))) if enabled_topics else 0

        for topic in enabled_topics:
            topic_items = [
                item
                for item in candidate_pool
                if self._candidate_topic_match(item, topic) > 0.0
            ]
            topic_items = self._sort_by_heuristic_then_time(topic_items, score_by_id, since)
            for item in topic_items[:per_topic]:
                if item.id not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item.id)

        remaining = [item for item in candidate_pool if item.id not in selected_ids]
        for item in self._sort_by_heuristic_then_time(remaining, score_by_id, since):
            if len(selected) >= max_total:
                break
            selected.append(item)
            selected_ids.add(item.id)

        return selected[:max_total]

    @staticmethod
    def _sort_by_heuristic_then_time(
        candidates: List[NewsCandidate],
        score_by_id: Dict[str, float],
        fallback_date,
    ) -> List[NewsCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                score_by_id.get(item.id, -999.0),
                item.published_at or fallback_date,
            ),
            reverse=True,
        )

    def _heuristic_ranked_candidates(self, candidates: List[NewsCandidate], topics, since) -> List[tuple[NewsCandidate, float]]:
        scored = [(item, self._candidate_heuristic_score(item, topics, since)) for item in candidates]
        scored.sort(
            key=lambda pair: (
                pair[1],
                pair[0].published_at or since,
            ),
            reverse=True,
        )
        return scored

    def _candidate_heuristic_score(self, item: NewsCandidate, topics, since) -> float:
        score = 0.0
        published_at = item.published_at or since
        age_hours = max(0.0, (utc_now() - published_at).total_seconds() / 3600.0)
        score += max(0.0, 3.0 - 0.07 * age_hours)

        topic_name = str(item.metadata.get("topic_name", "")).strip()
        if topic_name:
            score += 2.0

        if topic_name and self._topic_is_enabled(topics, topic_name):
            score += 1.0

        topic_match = 0.0
        for topic in topics:
            if getattr(topic, "enabled", False):
                topic_match = max(topic_match, self._candidate_topic_match(item, topic))
        score += min(3.0, topic_match * 1.2)

        merged_count = int(item.metadata.get("merged_count", 1) or 1)
        if merged_count > 1:
            score += min(1.5, 0.35 * (merged_count - 1))

        snippet_len = len(item.snippet or "")
        if snippet_len >= 260:
            score += 0.8
        elif snippet_len >= 120:
            score += 0.4
        elif snippet_len < 40:
            score -= 0.4

        title_len = len((item.title or "").strip())
        if 24 <= title_len <= 140:
            score += 0.5
        elif title_len < 12 or title_len > 180:
            score -= 0.8

        lowered_title = (item.title or "").lower()
        if any(needle in lowered_title for needle in ("live updates", "watch live", "photo gallery", "opinion:", "newsletter")):
            score -= 1.0

        preferred_sources = {source.lower() for source in self.config.user_memory.preferred_sources}
        avoided_sources = {source.lower() for source in self.config.user_memory.avoided_sources}
        source_name = (item.source or "").lower()
        if source_name in preferred_sources:
            score += 0.9
        if source_name in avoided_sources:
            score -= 2.5

        return round(score, 4)

    @staticmethod
    def _topic_is_enabled(topics, topic_name: str) -> bool:
        for topic in topics:
            if getattr(topic, "enabled", False) and topic.name == topic_name:
                return True
        return False

    def _candidate_topic_match(self, item: NewsCandidate, topic) -> float:
        item_topic = str(item.metadata.get("topic_name", "")).strip()
        if item_topic and item_topic == topic.name:
            return 1.0

        text = f"{item.title or ''} {item.snippet or ''}".lower()
        text_tokens = set(self._tokenize_for_match(text))
        if not text_tokens:
            return 0.0

        query_tokens = set(self._tokenize_for_match(topic.name))
        query_tokens.update(self._tokenize_for_match(topic.description))
        for query in topic.queries or []:
            query_tokens.update(self._tokenize_for_match(query))
        if not query_tokens:
            return 0.0
        overlap = len(text_tokens.intersection(query_tokens))
        if overlap <= 0:
            return 0.0
        return min(1.0, overlap / max(3, int(len(query_tokens) * 0.12)))

    @staticmethod
    def _tokenize_for_match(text: str) -> List[str]:
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "into",
            "over",
            "under",
            "latest",
            "today",
            "news",
            "major",
            "about",
        }
        tokens = [token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in stop]
        return tokens

    def _dedupe_similar_titles(self, candidates: List[NewsCandidate]) -> List[NewsCandidate]:
        groups: Dict[str, List[NewsCandidate]] = {}
        for item in candidates:
            key = self._title_dedupe_key(item.title)
            if not key:
                groups.setdefault(item.id, []).append(item)
                continue
            groups.setdefault(key, []).append(item)

        deduped: List[NewsCandidate] = []
        removed = 0
        for group in groups.values():
            if len(group) == 1:
                deduped.append(group[0])
                continue
            winner = max(
                group,
                key=lambda candidate: (
                    len(candidate.snippet or ""),
                    candidate.published_at or utc_now(),
                ),
            )
            related_ids = [candidate.id for candidate in group if candidate.id != winner.id]
            if related_ids:
                winner.metadata["headline_dupe_ids"] = related_ids
                winner.metadata["headline_dupe_count"] = len(group)
                removed += len(related_ids)
            deduped.append(winner)
        if removed > 0:
            self.debug.log("headline.heuristics", "title_dedupe", removed=removed, kept=len(deduped))
        return deduped

    @staticmethod
    def _title_dedupe_key(title: str) -> str:
        tokens = re.findall(r"[a-z0-9]{3,}", (title or "").lower())
        if not tokens:
            return ""
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "into",
            "over",
            "under",
            "live",
            "latest",
            "news",
            "update",
            "updates",
            "says",
            "say",
        }
        core = [token for token in tokens if token not in stop]
        if len(core) < 4:
            core = tokens
        return " ".join(core[:10])

    @staticmethod
    def _newest_first(candidates: List[NewsCandidate], fallback_date) -> List[NewsCandidate]:
        return sorted(
            candidates,
            key=lambda item: item.published_at or fallback_date,
            reverse=True,
        )

    def select_articles(
        self,
        candidates: List[NewsCandidate],
        decisions: Dict[str, HeadlineDecision],
        topics: List[TopicConfig],
        filtering: FilteringConfig,
    ) -> List[SelectedArticle]:
        selected: List[SelectedArticle] = []
        seen_duplicate_targets: set[str] = set()
        topic_limits = {
            topic.name: topic.max_selected_articles
            for topic in topics
            if topic.enabled and topic.max_selected_articles is not None
        }
        topic_counts: Dict[str, int] = {}
        sorted_candidates = sorted(
            candidates,
            key=lambda item: decisions.get(item.id, HeadlineDecision(item.id, 0)).score,
            reverse=True,
        )

        def try_select(candidate: NewsCandidate, require_cutoff: bool) -> None:
            if len(selected) >= filtering.max_selected_articles:
                return
            decision = decisions.get(candidate.id)
            if not decision:
                return
            if require_cutoff and decision.score < filtering.headline_score_cutoff:
                return
            if candidate.id in seen_duplicate_targets:
                return
            topic = decision.topic or candidate.metadata.get("topic_name", "")
            topic_limit = topic_limits.get(topic)
            if topic_limit is not None and topic_counts.get(topic, 0) >= int(topic_limit):
                return
            seen_duplicate_targets.add(candidate.id)
            selected.append(SelectedArticle(candidate=candidate, decision=decision))
            if topic:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        for candidate in sorted_candidates:
            try_select(candidate, require_cutoff=True)
            if len(selected) >= filtering.max_selected_articles:
                break

        if filtering.fill_selected_articles and len(selected) < filtering.max_selected_articles:
            for candidate in sorted_candidates:
                try_select(candidate, require_cutoff=False)
                if len(selected) >= filtering.max_selected_articles:
                    break
        return selected
