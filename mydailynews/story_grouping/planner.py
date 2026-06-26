from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mydailynews.ai.base import AIClient
from mydailynews.ai.prompts import STORY_GROUPING_SYSTEM, STORY_GROUPING_USER
from mydailynews.ai.schemas import STORY_GROUPING_JSON_SCHEMA
from mydailynews.app.models import AppConfig, SelectedArticle
from mydailynews.common.cache import JSONCache
from mydailynews.common.utils import compact_json
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.story_grouping.models import (
    ResearchQuestion,
    STORY_GROUPING_CACHE_VERSION,
    StoryGroup,
)
from mydailynews.story_grouping.normalization import normalize_story_groups
from mydailynews.story_grouping.payloads import (
    clean_text,
    planner_article_payload,
    story_group_artifact,
    string_list,
)


@dataclass(init=False)
class StoryGroupingResult:
    story_groups: list[StoryGroup]
    artifact: dict[str, Any]

    def __init__(
        self,
        story_groups: list[StoryGroup] | None = None,
        artifact: dict[str, Any] | None = None,
        *,
        story_threads: list[StoryGroup] | None = None,
    ) -> None:
        self.story_groups = list(story_groups if story_groups is not None else story_threads or [])
        self.artifact = artifact or {}

    @property
    def story_threads(self) -> list[StoryGroup]:
        return self.story_groups


StoryPlanningResult = StoryGroupingResult


@dataclass
class PlannerRequest:
    prompt: str
    articles: list[SelectedArticle]
    excerpt_chars: int
    estimated_tokens: int
    budget_tokens: int
    payload: list[dict[str, Any]]


class StoryGroupingPlanner:
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
        self.cache_hits = 0

    def plan(self, articles: list[SelectedArticle]) -> StoryGroupingResult:
        self.cache_hits = 0
        requests = self._build_requests(articles)
        if not requests:
            return StoryGroupingResult(
                story_groups=[],
                artifact={"status": "skipped_budget"},
            )

        raw_groups: list[dict[str, Any]] = []
        request_artifacts: list[dict[str, Any]] = []
        successful_requests = 0
        for index, request in enumerate(requests, start=1):
            request_artifact = {
                "request": index,
                "article_ids": [article.candidate.id for article in request.articles],
                "excerpt_chars": request.excerpt_chars,
                "estimated_tokens": request.estimated_tokens,
                "budget_tokens": request.budget_tokens,
                "status": "pending",
            }
            request_artifacts.append(request_artifact)
            result, cache_hit = self._run_request(request, request_index=index, total_requests=len(requests))
            if result is None:
                request_artifact["status"] = "failed"
                continue
            successful_requests += 1
            request_artifact["status"] = "cache_hit" if cache_hit else "ok"
            response_groups = self._raw_story_groups(result)
            request_artifact["story_groups"] = response_groups
            request_artifact["story_threads"] = response_groups
            for raw in response_groups:
                if isinstance(raw, dict):
                    raw_groups.append(raw)

        groups = self._validate_story_groups(
            raw_groups,
            articles,
            allow_singleton_fallback=successful_requests > 0,
        )
        artifacts = [story_group_artifact(story) for story in groups]
        return StoryGroupingResult(
            story_groups=groups,
            artifact={
                "status": "ok" if groups else "empty",
                "requests": request_artifacts,
                "split_requests": len(request_artifacts) > 1,
                "story_groups": artifacts,
                "story_threads": artifacts,
            },
        )

    def fit_request(self, articles: list[SelectedArticle]) -> PlannerRequest | None:
        if not articles:
            return None
        budget_tokens = max(0, int(getattr(self.ai_client, "max_input_tokens", 0) or 0))
        excerpt_chars = max(0, int(self.config.enrichment.max_selected_article_excerpt_chars))
        payload = [planner_article_payload(article, excerpt_chars) for article in articles]
        prompt = STORY_GROUPING_USER.format(
            articles=compact_json(payload),
            max_questions_per_story=max(0, int(self.config.enrichment.planner_max_questions_per_story)),
        )
        estimated_tokens = self._estimate_chat_tokens(STORY_GROUPING_SYSTEM, prompt)
        self.debug.log(
            "story_grouping.planner",
            "budget_check",
            articles=len(articles),
            excerpt_chars=excerpt_chars,
            estimated_tokens=estimated_tokens,
            budget_tokens=budget_tokens,
        )
        if budget_tokens > 0 and estimated_tokens <= budget_tokens:
            return PlannerRequest(
                prompt=prompt,
                articles=list(articles),
                excerpt_chars=excerpt_chars,
                estimated_tokens=estimated_tokens,
                budget_tokens=budget_tokens,
                payload=payload,
            )
        return None

    def cache_key(self, request: PlannerRequest) -> str:
        fingerprint = {
            "v": STORY_GROUPING_CACHE_VERSION,
            "stage": "story_grouping",
            "brief_name": self.brief_name,
            "date": self.date,
            "backend": getattr(self.ai_client.config, "backend", ""),
            "model": getattr(self.ai_client.config, "effective_model_label", ""),
            "response_format": getattr(self.ai_client.config, "response_format", ""),
            "schema": "story_grouping",
            "config": {
                "planner_max_questions_per_story": self.config.enrichment.planner_max_questions_per_story,
                "max_selected_article_excerpt_chars": self.config.enrichment.max_selected_article_excerpt_chars,
            },
            "articles": request.payload,
        }
        return JSONCache.make_key(compact_json(fingerprint))

    def _build_requests(self, articles: list[SelectedArticle]) -> list[PlannerRequest]:
        all_articles_request = self.fit_request(articles)
        if all_articles_request is not None:
            return [all_articles_request]

        sorted_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)
        requests: list[PlannerRequest] = []
        current: list[SelectedArticle] = []
        current_request: PlannerRequest | None = None
        for article in sorted_articles:
            candidate = current + [article]
            fitted = self.fit_request(candidate)
            if fitted is not None:
                current = candidate
                current_request = fitted
                continue
            if current_request is not None:
                requests.append(current_request)
            single = self.fit_request([article])
            if single is None:
                self.warning_sink(f"story grouping skipped article {article.candidate.id}: prompt could not fit model budget")
                current = []
                current_request = None
                continue
            current = [article]
            current_request = single
        if current_request is not None:
            requests.append(current_request)
        if requests:
            self.warning_sink(f"story grouping split selected articles into {len(requests)} budget-fit request(s)")
        return requests

    def _run_request(
        self,
        request: PlannerRequest,
        *,
        request_index: int,
        total_requests: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        label = f"story grouping planner {request_index}/{total_requests}"
        if self.brief_name:
            label = f"{label} ({self.brief_name})"
        cache_key = self.cache_key(request)
        if self.cache:
            cached = self.cache.get(cache_key, max_age_seconds=self.config.enrichment.cache_ttl_seconds)
            if cached is not None:
                self.cache_hits += 1
                self.debug.log("story_grouping.planner", "cache_hit", request=f"{request_index}/{total_requests}")
                return cached, True
        try:
            result = self.ai_client.complete_json(
                STORY_GROUPING_SYSTEM,
                request.prompt,
                label=label,
                max_new_tokens=int(getattr(self.ai_client, "max_new_tokens", 0) or 0) or None,
                input_token_limit=request.budget_tokens,
                json_schema=STORY_GROUPING_JSON_SCHEMA,
            )
        except Exception as exc:
            self.warning_sink(f"{label}: skipped after AI error: {type(exc).__name__}: {exc}")
            self.debug.log("story_grouping.planner", "failed", request=f"{request_index}/{total_requests}", error=type(exc).__name__)
            return None, False
        if self.cache:
            self.cache.put(cache_key, result)
        return result, False

    def _raw_story_groups(self, result: dict[str, Any]) -> list[Any]:
        value = result.get("story_groups")
        if not isinstance(value, list):
            value = result.get("story_threads", [])
        return value if isinstance(value, list) else []

    def _validate_story_groups(
        self,
        raw_groups: list[dict[str, Any]],
        articles: list[SelectedArticle],
        *,
        allow_singleton_fallback: bool,
    ) -> list[StoryGroup]:
        result = normalize_story_groups(
            selected=articles,
            raw_groups=raw_groups,
            caller="story grouping",
            allow_singleton_fallback=allow_singleton_fallback,
            fallback_questions=self._fallback_questions,
            question_parser=self._parse_research_questions,
            fallback_when_empty_input=allow_singleton_fallback,
        )
        for warning in result.warnings:
            self.warning_sink(warning)
        return result.groups

    def _parse_research_questions(self, value: Any, story_title: str) -> list[ResearchQuestion]:
        max_questions = max(0, int(self.config.enrichment.planner_max_questions_per_story))
        if max_questions <= 0:
            return []
        questions: list[ResearchQuestion] = []
        if isinstance(value, list):
            for raw in value:
                if not isinstance(raw, dict):
                    continue
                question = clean_text(raw.get("question"), 240)
                queries = string_list(raw.get("queries", []), max_items=3, max_chars=140)
                if not question and not queries:
                    continue
                if not queries and question:
                    queries = [question]
                questions.append(ResearchQuestion(question=question or queries[0], queries=queries))
                if len(questions) >= max_questions:
                    break
        if not questions:
            return self._fallback_questions(story_title)
        return questions

    def _fallback_questions(self, story_title: str) -> list[ResearchQuestion]:
        if int(self.config.enrichment.planner_max_questions_per_story) <= 0:
            return []
        query = clean_text(story_title, 140)
        if not query:
            return []
        return [
            ResearchQuestion(
                question=f"What fresh context helps explain {query}?",
                queries=[query],
            )
        ]

    def _estimate_chat_tokens(self, system: str, user: str) -> int:
        return self.ai_client.estimate_tokens(f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n")


StoryThreadPlanner = StoryGroupingPlanner
