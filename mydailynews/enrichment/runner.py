from __future__ import annotations

from threading import Lock
from typing import Any

from mydailynews.ai.base import AIClient
from mydailynews.common.cache import HTTPCache, JSONCache
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import AppConfig, ContextSource, SelectedArticle
from mydailynews.retrieval.article import ArticleRetriever
from mydailynews.retrieval.ddg import DuckDuckGoSearchRetriever
from mydailynews.common.utils import stable_id
from mydailynews.enrichment.models import ResearchQuestion, ResearchResult, StoryEnrichment
from mydailynews.enrichment.payloads import (
    clean_text,
    context_story_id,
    queries_for_story,
    selected_article_artifact,
    story_thread_artifact,
)
from mydailynews.enrichment.research import StoryResearchCollector
from mydailynews.enrichment.synthesis import StorySynthesizer
from mydailynews.story_grouping.models import StoryGroup as StoryThread
from mydailynews.story_grouping.normalization import normalize_story_groups
from mydailynews.story_grouping.planner import StoryGroupingPlanner


STORY_CONTEXT_KIND = "story_llm_research_context"


class StoryThreadEnricher:
    def __init__(
        self,
        config: AppConfig,
        http_cache: HTTPCache | None = None,
        debug: DebugLogger | None = None,
        ai_client: AIClient | None = None,
        cache: JSONCache | None = None,
        brief_name: str = "",
        date: str = "",
    ) -> None:
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.ai_client = ai_client
        self.cache = cache
        self.brief_name = brief_name
        self.date = date
        self.warnings: list[str] = []
        self.artifact: dict[str, Any] = {}
        self.story_threads_created = 0
        self.story_threads_enriched = 0
        self.story_threads_skipped = 0
        self._warning_lock = Lock()
        self.search_retriever = DuckDuckGoSearchRetriever(
            config.user_agent,
            http_cache=http_cache,
            debug=self.debug,
        )
        self.research_article_retriever = ArticleRetriever(
            config.user_agent,
            max(1200, int(config.enrichment.max_research_excerpt_chars) * 3),
            http_cache=http_cache,
            debug=self.debug,
        )
        self.research_collector = StoryResearchCollector(
            self.search_retriever,
            self.research_article_retriever,
            warning_sink=self._push_warning,
        )
        self.planner = (
            StoryGroupingPlanner(
                config,
                ai_client,
                cache=cache,
                debug=self.debug,
                warning_sink=self._push_warning,
                brief_name=brief_name,
                date=date,
            )
            if ai_client is not None
            else None
        )
        self.synthesizer = (
            StorySynthesizer(
                config,
                ai_client,
                cache=cache,
                debug=self.debug,
                warning_sink=self._push_warning,
                brief_name=brief_name,
                date=date,
            )
            if ai_client is not None
            else None
        )

    def enrich_many(
        self,
        articles: list[SelectedArticle],
        *,
        story_groups: list[StoryThread] | None = None,
    ) -> None:
        self.warnings = []
        self.story_threads_created = 0
        self.story_threads_enriched = 0
        self.story_threads_skipped = 0
        self.artifact = {
            "mode": self._mode(),
            "brief_name": self.brief_name,
            "date": self.date,
            "selected_articles": [selected_article_artifact(article) for article in articles],
            "planner": {},
            "story_threads": [],
            "warnings": self.warnings,
        }
        if not articles:
            return

        for article in articles:
            article.enrichment_needed = False
            article.enrichment_reason = "Story-thread enrichment did not attach context."

        mode = self._mode()
        if mode == "disabled":
            for article in articles:
                article.enrichment_needed = False
                article.enrichment_reason = "Skipped enrichment: enrichment is disabled."
                self.debug.log("enrichment", "skipped_disabled", article_id=article.candidate.id)
            return
        self._enrich_many_story_llm(articles, story_groups=story_groups)

    def _mode(self) -> str:
        if not self.config.enrichment.enabled:
            return "disabled"
        mode = str(getattr(self.config.enrichment, "mode", "story_llm") or "story_llm").strip().lower()
        if mode in {"disabled", "story_llm"}:
            return mode
        return "story_llm"

    def _enrich_many_story_llm(
        self,
        articles: list[SelectedArticle],
        *,
        story_groups: list[StoryThread] | None = None,
    ) -> None:
        if self.ai_client is None:
            warning = "story LLM enrichment skipped: no AI client was provided"
            self._push_warning(warning)
            self.artifact["planner"] = {"status": "skipped_no_ai_client"}
            for article in articles:
                article.enrichment_needed = False
                article.enrichment_reason = "Skipped story-thread enrichment: no AI client was available."
            self.debug.log("enrichment", "skipped_no_ai_client", articles=len(articles))
            return

        if story_groups is not None:
            story_threads = self._story_threads_from_shared_groups(story_groups, articles)
            self.artifact["planner"] = {
                "status": "shared_story_grouping",
                "story_groups": [story_thread_artifact(story) for story in story_threads],
            }
        elif self.planner is None:
            self.artifact["planner"] = {"status": "skipped_no_ai_client"}
            return
        else:
            planning = self.planner.plan(articles)
            self.artifact["planner"] = planning.artifact
            story_threads = planning.story_threads
        self.story_threads_created = len(story_threads)
        if not story_threads:
            self.artifact["planner"].setdefault("status", "no_story_threads")
            self._push_warning("story LLM enrichment skipped: planner produced no story threads")
            for article in articles:
                article.enrichment_needed = False
                article.enrichment_reason = "Story-thread enrichment found no usable story threads."
            return

        article_by_id = {article.candidate.id: article for article in articles}
        ranked_threads = self._rank_story_threads(story_threads, article_by_id)
        max_threads = max(1, int(self.config.enrichment.max_story_threads))
        skipped_for_cap = ranked_threads[max_threads:]
        self.story_threads_skipped += len(skipped_for_cap)
        for story in skipped_for_cap:
            self.artifact["story_threads"].append(
                {
                    "story_id": story.story_id,
                    "story_title": story.story_title,
                    "article_ids": story.article_ids,
                    "status": "skipped_thread_cap",
                }
            )
            self._mark_story_articles(
                story,
                article_by_id,
                "Skipped story-thread enrichment: thread cap.",
            )

        for story in ranked_threads[:max_threads]:
            enrichment, entry = self._enrich_story(story, article_by_id)
            self.artifact["story_threads"].append(entry)
            if enrichment and enrichment.internal_articles:
                self._attach_story_context(story, enrichment, article_by_id, entry.get("fetched_urls", []))
                self.story_threads_enriched += 1
                continue
            self.story_threads_skipped += 1
            self._mark_story_articles(
                story,
                article_by_id,
                _skip_reason_for_story_status(str(entry.get("status", ""))),
            )

        self.debug.log(
            "enrichment",
            "complete",
            mode="story_llm",
            story_threads_created=self.story_threads_created,
            story_threads_enriched=self.story_threads_enriched,
            story_threads_skipped=self.story_threads_skipped,
        )

    def _rank_story_threads(
        self,
        story_threads: list[StoryThread],
        article_by_id: dict[str, SelectedArticle],
    ) -> list[StoryThread]:
        def key(story: StoryThread) -> tuple[int, int, float, float]:
            articles = [article_by_id[article_id] for article_id in story.article_ids if article_id in article_by_id]
            sources = {str(article.candidate.source or "").strip().lower() for article in articles if article.candidate.source}
            scores = [float(article.decision.score) for article in articles]
            return (
                len(articles),
                len(sources),
                max(scores, default=0.0),
                sum(scores) / max(1, len(scores)),
            )

        return sorted(story_threads, key=key, reverse=True)

    def _story_threads_from_shared_groups(
        self,
        story_groups: list[StoryThread],
        articles: list[SelectedArticle],
    ) -> list[StoryThread]:
        result = normalize_story_groups(
            selected=articles,
            raw_groups=story_groups,
            caller="shared story grouping",
            allow_singleton_fallback=True,
            fallback_questions=_fallback_questions,
            fallback_when_empty_input=False,
        )
        for warning in result.warnings:
            self._push_warning(warning)
        return result.groups

    def _enrich_story(
        self,
        story: StoryThread,
        article_by_id: dict[str, SelectedArticle],
    ) -> tuple[StoryEnrichment | None, dict[str, Any]]:
        story_articles = [article_by_id[article_id] for article_id in story.article_ids if article_id in article_by_id]
        entry: dict[str, Any] = {
            "story_id": story.story_id,
            "story_title": story.story_title,
            "article_ids": list(story.article_ids),
            "fallback": story.fallback,
            "research_questions": [
                {"question": question.question, "queries": question.queries}
                for question in story.research_questions
            ],
            "queries": queries_for_story(story),
            "retrieved_urls": [],
            "fetched_urls": [],
            "synthesis": {},
            "internal_articles": [],
            "warnings": [],
            "status": "pending",
        }
        try:
            results = self._retrieve_research_results(story, story_articles)
            entry["retrieved_urls"] = [
                {
                    "id": result.id,
                    "query": result.query,
                    "title": result.title,
                    "url": result.url,
                    "source": result.source,
                    "status": result.status,
                }
                for result in results
            ]
            entry["fetched_urls"] = [
                {
                    "id": result.id,
                    "url": result.effective_url or result.url,
                    "title": result.title,
                    "status": result.status,
                    "chars": len(result.text or ""),
                }
                for result in results
                if result.status != "search_result"
            ]
            enrichment, synthesis_artifact = self._synthesize_story(story, story_articles, results)
            entry["synthesis"] = synthesis_artifact
            if enrichment is None:
                entry["status"] = "skipped_synthesis"
                return None, entry
            entry["internal_articles"] = enrichment.internal_articles
            entry["status"] = "enriched" if enrichment.internal_articles else "no_internal_articles"
            return enrichment, entry
        except Exception as exc:
            warning = f"story enrichment {story.story_id}: {type(exc).__name__}: {exc}"
            self._push_warning(warning)
            entry["warnings"].append(warning)
            entry["status"] = "failed"
            self.debug.log("enrichment.story", "failed", story_id=story.story_id, error=type(exc).__name__)
            return None, entry

    def _retrieve_research_results(
        self,
        story: StoryThread,
        story_articles: list[SelectedArticle],
    ) -> list[ResearchResult]:
        return self.research_collector.collect(
            queries=queries_for_story(story),
            story_title=story.story_title,
            story_articles=story_articles,
            search_results_per_query=self.config.enrichment.search_results_per_query,
            max_fetched_research_pages_per_story=self.config.enrichment.max_fetched_research_pages_per_story,
        )

    def _synthesize_story(
        self,
        story: StoryThread,
        story_articles: list[SelectedArticle],
        research_results: list[ResearchResult],
    ) -> tuple[StoryEnrichment | None, dict[str, Any]]:
        if self.synthesizer is None:
            return None, {"status": "skipped_no_ai_client"}
        return self.synthesizer.synthesize(story, story_articles, research_results)

    def _attach_story_context(
        self,
        story: StoryThread,
        enrichment: StoryEnrichment,
        article_by_id: dict[str, SelectedArticle],
        fetched_urls: list[dict[str, Any]],
    ) -> None:
        for article_id in story.article_ids:
            article = article_by_id.get(article_id)
            if article is None:
                continue
            article.context_sources = [
                source
                for source in article.context_sources
                if not (source.kind == STORY_CONTEXT_KIND and context_story_id(source) == story.story_id)
            ]
            for internal in enrichment.internal_articles:
                item_payload = {
                    "story_id": story.story_id,
                    "story_title": story.story_title,
                    "internal_article_title": internal["title"],
                    "what_it_adds": internal.get("what_it_adds", ""),
                    "source_ids": internal.get("source_ids", []),
                    "confidence": internal.get("confidence", "medium"),
                    "research_questions": [question.question for question in story.research_questions],
                    "retrieved_urls": fetched_urls[:8],
                    "confirmed_facts": enrichment.confirmed_facts[:8],
                    "conflicting_claims": enrichment.conflicting_claims[:5],
                    "open_questions": enrichment.open_questions[:5],
                }
                summary = internal["summary"]
                if internal.get("what_it_adds"):
                    summary = f"{summary} What it adds: {internal['what_it_adds']}"
                article.context_sources.append(
                    ContextSource(
                        id=stable_id(article.candidate.id, STORY_CONTEXT_KIND, story.story_id, internal["title"]),
                        parent_article_id=article.candidate.id,
                        kind=STORY_CONTEXT_KIND,
                        title=internal["title"],
                        source="LLM story research",
                        url="",
                        summary=summary[:1200],
                        items=[item_payload],
                    )
                )
            if enrichment.internal_articles:
                article.enrichment_needed = True
                article.enrichment_reason = "Added story-thread research context from selected-article grouping."

    def _mark_story_articles(
        self,
        story: StoryThread,
        article_by_id: dict[str, SelectedArticle],
        reason: str,
    ) -> None:
        for article_id in story.article_ids:
            article = article_by_id.get(article_id)
            if article is None or article.enrichment_needed:
                continue
            article.enrichment_reason = reason

    def _push_warning(self, warning: str) -> None:
        if not warning:
            return
        with self._warning_lock:
            self.warnings.append(warning)


def _skip_reason_for_story_status(status: str) -> str:
    if status == "no_internal_articles":
        return "Skipped story-thread enrichment: no internal articles."
    if status == "failed":
        return "Skipped story-thread enrichment: synthesis failed."
    return "Skipped story-thread enrichment: synthesis failed."


def _fallback_questions(story_title: str) -> list[ResearchQuestion]:
    query = clean_text(story_title, 140)
    if not query:
        return []
    return [
        ResearchQuestion(
            question=f"What fresh context helps explain {query}?",
            queries=[query],
        )
    ]
