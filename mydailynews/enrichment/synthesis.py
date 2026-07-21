from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mydailynews.ai.base import AIClient
from mydailynews.ai.prompts import STORY_ENRICHMENT_SYSTEM, STORY_ENRICHMENT_USER
from mydailynews.ai.schemas import STORY_ENRICHMENT_JSON_SCHEMA
from mydailynews.ai.token_budget import TokenBudget, resolve_client_token_budget
from mydailynews.app.models import AppConfig, SelectedArticle
from mydailynews.common.cache import JSONCache
from mydailynews.common.utils import compact_json
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.enrichment.models import (
    ResearchResult,
    STORY_ENRICHMENT_CACHE_VERSION,
    StoryEnrichment,
)
from mydailynews.enrichment.payloads import (
    clean_text,
    confidence,
    fact_list,
    research_sources_payload,
    selected_source_payload,
    story_enrichment_payload,
    string_list,
)
from mydailynews.story_grouping.models import StoryGroup
from mydailynews.story_grouping.payloads import story_group_artifact


@dataclass
class SynthesisPrompt:
    prompt: str
    selected_excerpt_chars: int
    research_excerpt_chars: int
    fetched_pages: int
    estimated_tokens: int
    budget_tokens: int
    max_new_tokens: int
    selected_sources: list[dict[str, Any]]
    research_sources: list[dict[str, Any]]


class StorySynthesizer:
    def __init__(
        self,
        config: AppConfig,
        ai_client: AIClient,
        *,
        cache: JSONCache | None = None,
        debug: DebugLogger | None = None,
        warning_sink: Callable[[str], None] | None = None,
        brief_name: str = "",
        date: str = "",
    ) -> None:
        self.config = config
        self.ai_client = ai_client
        self.cache = cache
        self.debug = debug or DebugLogger(False)
        self.warning_sink = warning_sink or (lambda warning: None)
        self.brief_name = brief_name
        self.date = date

    def synthesize(
        self,
        story: StoryGroup,
        story_articles: list[SelectedArticle],
        research_results: list[ResearchResult],
    ) -> tuple[StoryEnrichment | None, dict[str, Any]]:
        fitted = self.fit_prompt(story, story_articles, research_results)
        if fitted is None:
            warning = f"story synthesis {story.story_id} skipped: prompt could not fit model budget"
            self.warning_sink(warning)
            return None, {"status": "skipped_budget", "warning": warning}

        artifact = {
            "status": "pending",
            "estimated_tokens": fitted.estimated_tokens,
            "budget_tokens": fitted.budget_tokens,
            "selected_excerpt_chars": fitted.selected_excerpt_chars,
            "research_excerpt_chars": fitted.research_excerpt_chars,
            "excerpt_strategy": self.config.enrichment.excerpt_strategy,
            "fetched_pages": fitted.fetched_pages,
            "research_source_count": len(fitted.research_sources),
            "max_new_tokens": fitted.max_new_tokens,
        }
        cache_key = self.cache_key(story, fitted)
        if self.cache:
            cached = self.cache.get(cache_key, max_age_seconds=self.config.enrichment.cache_ttl_seconds)
            if cached is not None:
                self.debug.log("enrichment.synthesis", "cache_hit", story_id=story.story_id)
                enrichment = self.parse_enrichment(cached, story)
                artifact["status"] = "cache_hit"
                return enrichment, artifact

        label = f"story enrichment synthesis {story.story_id}"
        if self.brief_name:
            label = f"{label} ({self.brief_name})"
        try:
            raw = self.ai_client.complete_json(
                STORY_ENRICHMENT_SYSTEM,
                fitted.prompt,
                label=label,
                max_new_tokens=fitted.max_new_tokens,
                input_token_limit=fitted.budget_tokens,
                json_schema=STORY_ENRICHMENT_JSON_SCHEMA,
            )
        except Exception as exc:
            warning = f"{label}: skipped after AI error: {type(exc).__name__}: {exc}"
            self.warning_sink(warning)
            self.debug.log("enrichment.synthesis", "failed", story_id=story.story_id, error=type(exc).__name__)
            artifact["status"] = "failed"
            artifact["warning"] = warning
            return None, artifact

        enrichment = self.parse_enrichment(raw, story)
        if self.cache:
            self.cache.put(cache_key, story_enrichment_payload(enrichment))
        artifact["status"] = "ok"
        artifact["internal_articles"] = enrichment.internal_articles
        return enrichment, artifact

    def fit_prompt(
        self,
        story: StoryGroup,
        story_articles: list[SelectedArticle],
        research_results: list[ResearchResult],
    ) -> SynthesisPrompt | None:
        budget = self._stage_token_budget()
        budget_tokens = budget.input_tokens
        max_new_tokens = budget.output_tokens
        fetched_results = [result for result in research_results if result.text]
        fetched_count = min(
            len(fetched_results),
            max(0, int(self.config.enrichment.max_fetched_research_pages_per_story)),
        )
        selected_excerpt_chars = max(0, int(self.config.enrichment.max_selected_article_excerpt_chars))
        research_excerpt_chars = max(0, int(self.config.enrichment.max_research_excerpt_chars))
        story_terms = _story_terms(story, story_articles)
        selected_sources = [
            selected_source_payload(
                article,
                selected_excerpt_chars,
                excerpt_strategy=self.config.enrichment.excerpt_strategy,
                terms_text=f"{story_terms} {article.candidate.title} {article.candidate.snippet}",
                lead_chars=self.config.enrichment.selected_excerpt_lead_chars,
                window_chars=self.config.enrichment.selected_excerpt_window_chars,
                max_windows=self.config.enrichment.selected_excerpt_max_windows,
            )
            for article in story_articles
        ]
        research_sources = research_sources_payload(
            research_results,
            fetched_count=fetched_count,
            excerpt_chars=research_excerpt_chars,
            search_results_per_query=self.config.enrichment.search_results_per_query,
            excerpt_strategy=self.config.enrichment.excerpt_strategy,
            terms_text=story_terms,
            lead_chars=self.config.enrichment.research_excerpt_lead_chars,
            window_chars=self.config.enrichment.research_excerpt_window_chars,
            max_windows=self.config.enrichment.research_excerpt_max_windows,
        )
        prompt = self._render_prompt(
            story,
            selected_sources=selected_sources,
            research_sources=research_sources,
        )
        estimated_tokens = self._estimate_chat_tokens(STORY_ENRICHMENT_SYSTEM, prompt)
        self.debug.log(
            "enrichment.synthesis",
            "budget_check",
            story_id=story.story_id,
            selected_excerpt_chars=selected_excerpt_chars,
            research_excerpt_chars=research_excerpt_chars,
            fetched_pages=fetched_count,
            estimated_tokens=estimated_tokens,
            budget_tokens=budget_tokens,
            max_new_tokens=max_new_tokens,
        )
        if budget_tokens > 0 and estimated_tokens <= budget_tokens:
            return SynthesisPrompt(
                prompt=prompt,
                selected_excerpt_chars=selected_excerpt_chars,
                research_excerpt_chars=research_excerpt_chars,
                fetched_pages=fetched_count,
                estimated_tokens=estimated_tokens,
                budget_tokens=budget_tokens,
                max_new_tokens=max_new_tokens,
                selected_sources=selected_sources,
                research_sources=research_sources,
            )
        return None

    def cache_key(self, story: StoryGroup, fitted: SynthesisPrompt) -> str:
        fingerprint = {
            "v": STORY_ENRICHMENT_CACHE_VERSION,
            "stage": "story_enrichment_synthesis",
            "brief_name": self.brief_name,
            "date": self.date,
            "backend": getattr(self.ai_client.config, "backend", ""),
            "model": getattr(self.ai_client.config, "effective_model_label", ""),
            "response_format": getattr(self.ai_client.config, "response_format", ""),
            "config": {
                "synthesis_max_input_tokens": self.config.enrichment.synthesis_max_input_tokens,
                "synthesis_max_new_tokens": self.config.enrichment.synthesis_max_new_tokens,
                "max_selected_article_excerpt_chars": self.config.enrichment.max_selected_article_excerpt_chars,
                "max_research_excerpt_chars": self.config.enrichment.max_research_excerpt_chars,
                "excerpt_strategy": self.config.enrichment.excerpt_strategy,
                "selected_excerpt_lead_chars": self.config.enrichment.selected_excerpt_lead_chars,
                "selected_excerpt_window_chars": self.config.enrichment.selected_excerpt_window_chars,
                "selected_excerpt_max_windows": self.config.enrichment.selected_excerpt_max_windows,
                "research_excerpt_lead_chars": self.config.enrichment.research_excerpt_lead_chars,
                "research_excerpt_window_chars": self.config.enrichment.research_excerpt_window_chars,
                "research_excerpt_max_windows": self.config.enrichment.research_excerpt_max_windows,
                "max_fetched_research_pages_per_story": self.config.enrichment.max_fetched_research_pages_per_story,
                "effective_input_budget_tokens": fitted.budget_tokens,
                "effective_max_new_tokens": fitted.max_new_tokens,
            },
            "story": story_group_artifact(story),
            "selected_sources": fitted.selected_sources,
            "research_sources": fitted.research_sources,
        }
        return JSONCache.make_key(compact_json(fingerprint))

    def parse_enrichment(self, raw: dict[str, Any], story: StoryGroup) -> StoryEnrichment:
        internal_articles: list[dict[str, Any]] = []
        for item in raw.get("internal_articles", []):
            if not isinstance(item, dict):
                continue
            title = clean_text(item.get("title"), None)
            summary = clean_text(item.get("summary"), None)
            what_it_adds = clean_text(item.get("what_it_adds"), None)
            if not title or not summary:
                continue
            internal_articles.append(
                {
                    "title": title,
                    "summary": summary,
                    "what_it_adds": what_it_adds,
                    "source_ids": string_list(item.get("source_ids", []), max_items=None, max_chars=None),
                    "confidence": confidence(item.get("confidence")),
                }
            )
        return StoryEnrichment(
            story_id=clean_text(raw.get("story_id"), 80) or story.story_id,
            story_title=clean_text(raw.get("story_title"), None) or story.story_title,
            internal_articles=internal_articles,
            confirmed_facts=fact_list(raw.get("confirmed_facts", []), text_key="fact"),
            conflicting_claims=fact_list(raw.get("conflicting_claims", []), text_key="claim"),
            open_questions=fact_list(raw.get("open_questions", []), text_key="question"),
        )

    def _render_prompt(
        self,
        story: StoryGroup,
        *,
        selected_sources: list[dict[str, Any]],
        research_sources: list[dict[str, Any]],
    ) -> str:
        story_payload = {
            "story_id": story.story_id,
            "story_title": story.story_title,
            "article_ids": story.article_ids,
        }
        research_questions = [
            {"question": question.question, "queries": question.queries}
            for question in story.research_questions
        ]
        return STORY_ENRICHMENT_USER.format(
            story=compact_json(story_payload),
            selected_sources=compact_json(selected_sources),
            research_questions=compact_json(research_questions),
            research_sources=compact_json(research_sources),
            story_id=story.story_id,
            story_title=story.story_title,
        )

    def _estimate_chat_tokens(self, system: str, user: str) -> int:
        return self.ai_client.estimate_tokens(f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n")

    def _stage_token_budget(self) -> TokenBudget:
        return resolve_client_token_budget(
            self.ai_client,
            input_tokens=self.config.enrichment.synthesis_max_input_tokens,
            output_tokens=self.config.enrichment.synthesis_max_new_tokens,
        )


def _story_terms(story: StoryGroup, story_articles: list[SelectedArticle]) -> str:
    question_terms = [
        text
        for question in story.research_questions
        for text in [question.question, *question.queries]
    ]
    return " ".join(
        [story.story_title]
        + [article.candidate.title for article in story_articles]
        + [article.candidate.snippet for article in story_articles]
        + question_terms
    )
