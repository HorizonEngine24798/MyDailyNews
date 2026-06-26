from __future__ import annotations

from typing import Any, Dict, List

from mydailynews.ai.base import AIClient
from mydailynews.ai.prompts import EVIDENCE_DISTILLATION_SYSTEM, EVIDENCE_DISTILLATION_USER
from mydailynews.ai.schemas import EVIDENCE_DISTILLATION_JSON_SCHEMA
from mydailynews.analysis.shared import (
    _append_headline_context,
    _article_ids,
    _dedupe_dicts_by_text,
    _dedupe_strings,
    _ordered_article_groups,
    _short_text,
    article_cache_payload,
    headline_context_cache_payload,
    prior_reports_payload,
    story_thread_payloads,
    topics_payload,
)
from mydailynews.common.cache import JSONCache
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import EvidenceDistillationConfig, PriorReport, SelectedArticle, TopicConfig, UserMemory
from mydailynews.common.utils import compact_json, datetime_to_iso
from mydailynews.story_grouping.models import StoryGroup
from mydailynews.story_grouping.normalization import normalize_story_groups


def _article_sort_key(article: SelectedArticle) -> tuple[float, float, str]:
    return (
        float(article.decision.score),
        float(article.selection_rank_score or article.decision.selection_rank_score or 0.0),
        str(article.candidate.published_at or ""),
    )


def _dedupe_article_ids(values: Any, *, max_items: int) -> List[str]:
    raw_values = values if isinstance(values, list) else [values]
    output: List[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        article_id = str(raw or "").strip()[:80]
        if not article_id or article_id in seen:
            continue
        seen.add(article_id)
        output.append(article_id)
        if len(output) >= max_items:
            break
    return output


def _story_group_boundaries(story_groups: List[StoryGroup] | None) -> List[Dict[str, Any]]:
    if not story_groups:
        return []
    boundaries: List[Dict[str, Any]] = []
    for group in story_groups:
        article_ids = set(_dedupe_article_ids(group.article_ids, max_items=100))
        if not article_ids:
            continue
        boundaries.append(
            {
                "story_id": _short_text(group.story_id, 80),
                "story_title": _short_text(group.story_title, 180),
                "topic": _short_text(getattr(group, "topic", ""), 120),
                "article_ids": article_ids,
            }
        )
    return boundaries

class EvidenceDistiller:
    def __init__(
        self,
        client: AIClient,
        config: EvidenceDistillationConfig,
        include_enrichment_context: bool = True,
        debug: DebugLogger | None = None,
        cache: JSONCache | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.include_enrichment_context = bool(include_enrichment_context)
        self.debug = debug or DebugLogger(False)
        self.cache = cache
        self.cache_ttl_seconds = (
            max(0, int(cache_ttl_seconds))
            if cache_ttl_seconds is not None
            else max(0, int(config.cache_ttl_seconds))
        )
        self.warnings: List[str] = []
        self.group_boundary_warning_count = 0

    def distill(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        brief_name: str = "",
        story_groups: List[StoryGroup] | None = None,
    ) -> Dict[str, Any]:
        self.warnings = []
        self.group_boundary_warning_count = 0
        if not self.config.enabled:
            self.debug.log("analysis.evidence", "skipped_disabled")
            return {}
        if not articles:
            self.warnings.append("evidence distillation skipped: no selected articles")
            return {}

        ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)[: self.config.max_articles]
        effective_story_groups = self._normalize_story_groups_for_articles(story_groups, ordered_articles)
        batches = self._build_article_batches(
            ordered_articles,
            memory=memory,
            topics=topics,
            prior_reports=prior_reports,
            brief_goal=brief_goal,
            date=date,
            story_groups=effective_story_groups,
        )
        self.debug.log(
            "analysis.evidence",
            "starting_batched_distillation",
            articles=len(ordered_articles),
            batches=len(batches),
            batch_size=max(1, int(self.config.max_articles_per_batch)),
        )
        self.debug.set_metric("analysis.evidence.batch_size", max(1, int(self.config.max_articles_per_batch)))
        self.debug.set_metric("analysis.evidence.batches", len(batches))
        if len(batches) > 1:
            self.warnings.append(
                f"evidence distillation split selected articles into {len(batches)} estimated-fit batch(es)."
            )

        results: List[Dict[str, Any]] = []
        for batch_index, batch_articles in enumerate(batches, start=1):
            batch_ids = _article_ids(batch_articles)
            headline_context_articles = [
                article for article in ordered_articles if article.candidate.id not in batch_ids
            ] if len(batches) > 1 else []
            prompt, used_articles, used_reports = self._build_prompt(
                articles=batch_articles,
                memory=memory,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                headline_context_articles=headline_context_articles,
                story_groups=effective_story_groups,
            )
            if not used_articles:
                self.warnings.append(
                    f"evidence distillation batch {batch_index}/{len(batches)} skipped: prompt builder yielded no usable articles"
                )
                continue
            results.append(
                self._run_batch(
                    prompt=prompt,
                    used_articles=used_articles,
                    used_reports=used_reports,
                    memory=memory,
                    topics=topics,
                    brief_goal=brief_goal,
                    date=date,
                    brief_name=brief_name,
                    batch_index=batch_index,
                    total_batches=len(batches),
                    headline_context_articles=headline_context_articles,
                    story_groups=self._groups_for_articles(effective_story_groups, used_articles),
                )
            )

        if not results:
            self.warnings.append("evidence distillation skipped: no batch yielded usable evidence")
            return {}

        result = self._merge_results(results, story_groups=effective_story_groups)
        if not self.config.include_reader_qa:
            result["reader_qa"] = []
        return result

    def _run_batch(
        self,
        *,
        prompt: str,
        used_articles: List[SelectedArticle],
        used_reports: List[PriorReport],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        date: str,
        brief_name: str,
        batch_index: int,
        total_batches: int,
        headline_context_articles: List[SelectedArticle],
        story_groups: List[StoryGroup] | None,
    ) -> Dict[str, Any]:
        label = f"evidence distillation batch {batch_index}/{total_batches}"
        if brief_name:
            label = f"{label} ({brief_name})"
        cache_key = self._cache_key(
            used_articles=used_articles,
            used_reports=used_reports,
            memory=memory,
            topics=topics,
            brief_goal=brief_goal,
            date=date,
            headline_context_articles=headline_context_articles,
            story_groups=story_groups,
        )
        if self.cache:
            cached = self.cache.get(cache_key, max_age_seconds=self.cache_ttl_seconds)
            if cached is not None:
                self.debug.log(
                    "analysis.evidence.batch",
                    "cache_hit",
                    batch=f"{batch_index}/{total_batches}",
                    articles=len(used_articles),
                    prior_reports=len(used_reports),
                )
                return self._normalize_result(cached)

        self.debug.log(
            "analysis.evidence.batch",
            "running",
            batch=f"{batch_index}/{total_batches}",
            articles=len(used_articles),
            prior_reports=len(used_reports),
            prompt_chars=len(prompt),
            max_input_tokens=self.config.max_input_tokens,
            max_new_tokens=self.config.max_new_tokens,
        )
        raw = self.client.complete_json(
            EVIDENCE_DISTILLATION_SYSTEM,
            prompt,
            label=label,
            max_new_tokens=self.config.max_new_tokens,
            input_token_limit=self.config.max_input_tokens,
            json_schema=EVIDENCE_DISTILLATION_JSON_SCHEMA,
        )
        result = self._normalize_result(raw)
        if self.cache:
            self.cache.put(cache_key, result)
        return result

    def _build_prompt(
        self,
        *,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        headline_context_articles: List[SelectedArticle] | None = None,
        story_groups: List[StoryGroup] | None = None,
    ) -> tuple[str, List[SelectedArticle], List[PriorReport]]:
        target_input_tokens = max(1024, min(int(self.config.max_input_tokens), int(self.client.max_input_tokens)))
        prompt_budget_tokens = target_input_tokens
        ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)[: self.config.max_articles]
        active_reports = prior_reports[:3]
        excerpt_options = [
            self.config.max_article_chars,
            min(self.config.max_article_chars, 560),
            min(self.config.max_article_chars, 380),
            240,
            180,
        ]
        dropped_article_ids: list[str] = []
        used_articles = ordered_articles[:]
        prompt = ""

        for excerpt_chars in excerpt_options:
            candidate_articles = used_articles[:]
            candidate_reports = active_reports[:]
            while candidate_articles:
                prompt = self._render_prompt(
                    articles=candidate_articles,
                    excerpt_chars=excerpt_chars,
                    memory=memory,
                    topics=topics,
                    prior_reports=candidate_reports,
                    brief_goal=brief_goal,
                    date=date,
                    headline_context_articles=headline_context_articles or [],
                    story_groups=story_groups,
                )
                estimated_tokens = self.client.estimate_tokens(prompt)
                self.debug.log(
                    "analysis.evidence.prompt",
                    "budget_check",
                    articles=len(candidate_articles),
                    prior_reports=len(candidate_reports),
                    excerpt_chars=excerpt_chars,
                    estimated_tokens=estimated_tokens,
                    budget_tokens=prompt_budget_tokens,
                )
                if estimated_tokens <= prompt_budget_tokens:
                    used_articles = candidate_articles
                    if dropped_article_ids:
                        self.warnings.append(
                            "evidence distillation prompt dropped lower-ranked article(s) to stay within model budget: "
                            + ", ".join(dropped_article_ids)
                        )
                    return prompt, used_articles, candidate_reports
                self.debug.increment("analysis.evidence.prompt_pressure_checks", 1)
                if len(candidate_reports) > 1:
                    candidate_reports = candidate_reports[:-1]
                    self.debug.increment("analysis.evidence.prompt_compaction.drop_prior_report", 1)
                    continue
                if len(candidate_articles) > 2:
                    dropped = candidate_articles.pop()
                    dropped_article_ids.append(dropped.candidate.id)
                    self.debug.increment("analysis.evidence.prompt_compaction.drop_article", 1)
                    continue
                used_articles = candidate_articles
                break

        if dropped_article_ids:
            self.warnings.append(
                "evidence distillation prompt dropped lower-ranked article(s) to stay within model budget: "
                + ", ".join(dropped_article_ids)
            )
        if used_articles and prompt:
            estimated_tokens = self.client.estimate_tokens(prompt)
            if estimated_tokens > prompt_budget_tokens:
                self.warnings.append(
                    f"evidence distillation prompt may exceed budget ({estimated_tokens}>{prompt_budget_tokens}); "
                    "backend limits may truncate context"
                )
            return prompt, used_articles, active_reports[:1]
        return (
            self._render_prompt(
                [],
                0,
                memory,
                topics,
                active_reports[:1],
                brief_goal,
                date,
                headline_context_articles=headline_context_articles or [],
                story_groups=story_groups,
            ),
            [],
            active_reports[:1],
        )

    def _input_token_limit(self) -> int:
        return max(256, min(int(self.config.max_input_tokens), int(self.client.max_input_tokens)))

    def _build_article_batches(
        self,
        articles: List[SelectedArticle],
        *,
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        story_groups: List[StoryGroup] | None = None,
    ) -> List[List[SelectedArticle]]:
        if not articles:
            return []
        batch_size = max(1, int(self.config.max_articles_per_batch))
        single_batch = self._single_batch_after_optional_tail_drop(
            articles,
            memory=memory,
            topics=topics,
            prior_reports=prior_reports,
            brief_goal=brief_goal,
            date=date,
            batch_size=batch_size,
            story_groups=story_groups,
        )
        if single_batch is not None:
            return [single_batch]

        batches: List[List[SelectedArticle]] = []
        current: List[SelectedArticle] = []

        for group in self._ordered_batch_groups(articles, story_groups):
            if current and self._candidate_batch_fits(
                current + group,
                all_articles=articles,
                memory=memory,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                batch_size=batch_size,
                story_groups=story_groups,
            ):
                current.extend(group)
                continue
            if current:
                batches.append(current)
                current = []
            if self._candidate_batch_fits(
                group,
                all_articles=articles,
                memory=memory,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                batch_size=batch_size,
                story_groups=story_groups,
            ):
                current = group[:]
                continue
            if story_groups:
                self._record_group_boundary_warning(
                    "evidence distillation split shared story group because it did not fit a single batch: "
                    + ", ".join(article.candidate.id for article in group)
                )
            for article in group:
                if current and not self._candidate_batch_fits(
                    current + [article],
                    all_articles=articles,
                    memory=memory,
                    topics=topics,
                    prior_reports=prior_reports,
                    brief_goal=brief_goal,
                    date=date,
                    batch_size=batch_size,
                    story_groups=story_groups,
                ):
                    batches.append(current)
                    current = [article]
                    continue
                current.append(article)
        if current:
            batches.append(current)
        return batches

    def _single_batch_after_optional_tail_drop(
        self,
        articles: List[SelectedArticle],
        *,
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        batch_size: int,
        story_groups: List[StoryGroup] | None = None,
    ) -> List[SelectedArticle] | None:
        max_drop = 0 if story_groups is not None else min(
            max(0, int(self.config.max_articles_dropped_to_avoid_split)),
            max(0, len(articles) - 1),
        )
        for drop_count in range(max_drop + 1):
            candidate_articles = articles[: len(articles) - drop_count] if drop_count else articles
            if not self._candidate_batch_fits(
                candidate_articles,
                all_articles=candidate_articles,
                memory=memory,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                batch_size=batch_size,
                include_headline_context=False,
                story_groups=story_groups,
            ):
                continue
            if drop_count:
                dropped = articles[len(candidate_articles) :]
                dropped_ids = [article.candidate.id for article in dropped]
                self.warnings.append(
                    "evidence distillation dropped lower-ranked article(s) instead of splitting into another batch: "
                    + ", ".join(dropped_ids)
                )
                self.debug.log(
                    "analysis.evidence",
                    "dropped_tail_to_avoid_split",
                    dropped=drop_count,
                    kept=len(candidate_articles),
                    max_drop=max_drop,
                )
                self.debug.set_metric("analysis.evidence.avoid_split_dropped_articles", drop_count)
            return candidate_articles
        return None

    def _candidate_batch_fits(
        self,
        candidate_articles: List[SelectedArticle],
        *,
        all_articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        batch_size: int,
        include_headline_context: bool = True,
        story_groups: List[StoryGroup] | None = None,
    ) -> bool:
        if not candidate_articles or len(candidate_articles) > batch_size:
            return False
        candidate_ids = _article_ids(candidate_articles)
        headline_context = (
            [article for article in all_articles if article.candidate.id not in candidate_ids]
            if include_headline_context and len(candidate_articles) < len(all_articles)
            else []
        )
        prompt = self._render_prompt(
            candidate_articles,
            self.config.max_article_chars,
            memory,
            topics,
            prior_reports[:3],
            brief_goal,
            date,
            headline_context_articles=headline_context,
            story_groups=story_groups,
        )
        return self.client.estimate_tokens(prompt) <= self._input_token_limit()

    def _normalize_story_groups_for_articles(
        self,
        story_groups: List[StoryGroup] | None,
        articles: List[SelectedArticle],
    ) -> List[StoryGroup] | None:
        if story_groups is None:
            return None
        result = normalize_story_groups(
            selected=articles,
            raw_groups=story_groups,
            caller="shared evidence grouping",
            allow_singleton_fallback=True,
            fallback_when_empty_input=False,
        )
        for warning in result.warnings:
            self._record_group_boundary_warning(warning)
        return result.groups

    def _ordered_batch_groups(
        self,
        articles: List[SelectedArticle],
        story_groups: List[StoryGroup] | None,
    ) -> List[List[SelectedArticle]]:
        if not story_groups:
            return _ordered_article_groups(articles)
        article_by_id = {article.candidate.id: article for article in articles}
        grouped: List[List[SelectedArticle]] = []
        assigned: set[str] = set()
        for group in story_groups:
            group_articles = [article_by_id[article_id] for article_id in group.article_ids if article_id in article_by_id]
            if not group_articles:
                continue
            assigned.update(article.candidate.id for article in group_articles)
            grouped.append(sorted(group_articles, key=_article_sort_key, reverse=True))
        for article in articles:
            if article.candidate.id not in assigned:
                grouped.append([article])
        grouped.sort(key=lambda group: max(_article_sort_key(article) for article in group), reverse=True)
        return grouped

    def _groups_for_articles(
        self,
        story_groups: List[StoryGroup] | None,
        articles: List[SelectedArticle],
    ) -> List[StoryGroup] | None:
        if story_groups is None:
            return None
        allowed_ids = _article_ids(articles)
        filtered: List[StoryGroup] = []
        for group in story_groups:
            article_ids = [article_id for article_id in group.article_ids if article_id in allowed_ids]
            if not article_ids:
                continue
            filtered.append(
                StoryGroup(
                    story_id=group.story_id,
                    story_title=group.story_title,
                    article_ids=article_ids,
                    research_questions=list(group.research_questions),
                    fallback=bool(group.fallback),
                    topic=getattr(group, "topic", ""),
                )
            )
        return filtered

    def _merge_results(
        self,
        results: List[Dict[str, Any]],
        *,
        story_groups: List[StoryGroup] | None = None,
    ) -> Dict[str, Any]:
        normalized = [self._normalize_result(item) for item in results if isinstance(item, dict)]
        overview = " ".join(_short_text(item.get("overview", ""), 360) for item in normalized if item.get("overview"))
        group_boundaries = _story_group_boundaries(story_groups)
        shared_grouping_mode = story_groups is not None
        selected_ids = set().union(*(boundary["article_ids"] for boundary in group_boundaries)) if group_boundaries else set()

        clusters: List[Dict[str, Any]] = []
        seen_clusters: set[str] = set()
        for result in normalized:
            for raw in result.get("story_clusters", []):
                if not isinstance(raw, dict):
                    continue
                if shared_grouping_mode and not group_boundaries:
                    self._record_group_boundary_warning(
                        "evidence cluster dropped because shared grouping supplied no usable story boundaries"
                    )
                    continue
                article_ids = _dedupe_article_ids(raw.get("article_ids", []), max_items=8)
                boundary = self._boundary_for_cluster(raw, article_ids, group_boundaries)
                if group_boundaries and boundary is None:
                    self._record_group_boundary_warning(
                        f"evidence cluster {raw.get('cluster_id') or raw.get('label') or ''} did not match a supplied story group; dropped"
                    )
                    continue
                if boundary is not None:
                    original_ids = list(article_ids)
                    article_ids = [article_id for article_id in article_ids if article_id in boundary["article_ids"]]
                    if original_ids != article_ids:
                        self._record_group_boundary_warning(
                            f"evidence cluster {raw.get('cluster_id') or raw.get('label') or ''} crossed shared group boundary; trimmed article ids"
                        )
                    if not article_ids:
                        self._record_group_boundary_warning(
                            f"evidence cluster {raw.get('cluster_id') or raw.get('label') or ''} dropped after boundary trim"
                        )
                        continue
                key = (
                    _short_text(raw.get("cluster_id") or raw.get("label"), 120).lower(),
                    tuple(article_ids),
                )
                if key in seen_clusters:
                    continue
                seen_clusters.add(key)
                claims = _dedupe_dicts_by_text(
                    raw.get("key_claims", []),
                    text_key="claim",
                    max_items=max(1, int(self.config.max_claims_per_cluster)),
                )
                if boundary is not None:
                    claims = self._trim_claims_to_boundary(claims, boundary["article_ids"])
                clusters.append(
                    {
                        "cluster_id": boundary["story_id"] if boundary is not None else _short_text(raw.get("cluster_id", ""), 80),
                        "topic": _short_text(raw.get("topic") or (boundary["topic"] if boundary else ""), 80),
                        "label": _short_text(raw.get("label") or (boundary["story_title"] if boundary else ""), 120),
                        "summary": _short_text(raw.get("summary", ""), 280),
                        "article_ids": article_ids,
                        "key_claims": claims,
                        "consensus_points": _dedupe_strings(raw.get("consensus_points", []), max_items=5, max_chars=160),
                        "contested_points": _dedupe_strings(raw.get("contested_points", []), max_items=5, max_chars=160),
                        "known_unknowns": _dedupe_strings(raw.get("known_unknowns", []), max_items=5, max_chars=160),
                        "watch_signals": _dedupe_strings(raw.get("watch_signals", []), max_items=5, max_chars=160),
                    }
                )
                if len(clusters) >= max(1, int(self.config.max_story_clusters)):
                    break
            if len(clusters) >= max(1, int(self.config.max_story_clusters)):
                break

        watch_signals: List[Any] = []
        reader_qa: List[Any] = []
        for result in normalized:
            watch_signals.extend(result.get("global_watch_signals", []))
            reader_qa.extend(result.get("reader_qa", []))
        reader_qa = self._trim_reader_qa(reader_qa, selected_ids, enforce_boundaries=shared_grouping_mode)

        return self._normalize_result(
            {
                "overview": overview[:900],
                "story_clusters": clusters,
                "global_watch_signals": _dedupe_strings(watch_signals, max_items=12, max_chars=160),
                "reader_qa": _dedupe_dicts_by_text(
                    reader_qa,
                    text_key="question",
                    max_items=max(0, int(self.config.max_questions)),
                ),
            }
        )

    def _render_prompt(
        self,
        articles: List[SelectedArticle],
        excerpt_chars: int,
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        headline_context_articles: List[SelectedArticle] | None = None,
        story_groups: List[StoryGroup] | None = None,
    ) -> str:
        payload = [self._article_payload(item, excerpt_chars) for item in articles]
        prompt = EVIDENCE_DISTILLATION_USER.format(
            memory=memory.to_prompt(),
            brief_goal=brief_goal,
            date=date,
            topics=compact_json(topics_payload(topics)),
            prior_reports=compact_json(prior_reports_payload(prior_reports)),
            articles=compact_json(payload),
        )
        prompt = self._append_story_grouping_plan(prompt, articles, story_groups)
        return _append_headline_context(prompt, headline_context_articles or [], stage="evidence")

    def _cache_key(
        self,
        *,
        used_articles: List[SelectedArticle],
        used_reports: List[PriorReport],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        date: str,
        headline_context_articles: List[SelectedArticle],
        story_groups: List[StoryGroup] | None = None,
    ) -> str:
        fingerprint = {
            "v": 7,
            "stage": "evidence_distillation",
            "story_grouping_mode": "shared" if story_groups is not None else "free_clustering",
            "backend": self.client.config.backend,
            "model": self.client.config.effective_model_label,
            "response_format": self.client.config.response_format,
            "date": date,
            "brief_goal": brief_goal,
            "memory": memory.to_prompt(),
            "config": {
                "include_reader_qa": self.config.include_reader_qa,
                "max_input_tokens": self.config.max_input_tokens,
                "max_new_tokens": self.config.max_new_tokens,
                "max_articles": self.config.max_articles,
                "max_articles_per_batch": self.config.max_articles_per_batch,
                "max_article_chars": self.config.max_article_chars,
                "max_context_sources_per_article": self.config.max_context_sources_per_article,
            },
            "topics": topics_payload(topics),
            "articles": [article_cache_payload(item) for item in used_articles],
            "headline_context_articles": [headline_context_cache_payload(item) for item in headline_context_articles],
            "story_groups": self._story_group_payload(story_groups, allowed_ids=_article_ids(used_articles)),
            "prior_reports": prior_reports_payload(used_reports),
        }
        return JSONCache.make_key(compact_json(fingerprint))

    def _append_story_grouping_plan(
        self,
        prompt: str,
        articles: List[SelectedArticle],
        story_groups: List[StoryGroup] | None,
    ) -> str:
        if story_groups is None:
            return prompt
        payload = self._story_group_payload(story_groups, allowed_ids=_article_ids(articles))
        if not payload:
            return (
                prompt
                + "\n\nStory grouping plan:\n[]"
                + "\n\nShared story grouping ran, but no usable story boundaries were supplied for this batch. "
                "Do not invent evidence story_clusters or cross-article story membership. You may still summarize "
                "global watch signals and reader questions when directly supported by the selected articles."
            )
        return (
            prompt
            + "\n\nStory grouping plan:\n"
            + compact_json(payload)
            + "\n\nGroup boundaries are already decided. Enrich each supplied story group into evidence story_clusters; "
            "do not invent cross-group membership. Returned story_clusters[*].article_ids must be subsets of the "
            "matching story_group.article_ids. Prefer story_clusters[*].cluster_id values that match story_group.story_id. "
            "Only drop a supplied group when the batch has no supporting evidence for it."
        )

    def _story_group_payload(
        self,
        story_groups: List[StoryGroup] | None,
        *,
        allowed_ids: set[str],
    ) -> List[Dict[str, Any]]:
        if not story_groups:
            return []
        payload: List[Dict[str, Any]] = []
        for group in story_groups:
            article_ids = [article_id for article_id in group.article_ids if article_id in allowed_ids]
            if not article_ids:
                continue
            item: Dict[str, Any] = {
                "story_id": group.story_id,
                "story_title": group.story_title,
                "article_ids": article_ids,
            }
            if group.topic:
                item["topic"] = group.topic
            if group.fallback:
                item["fallback"] = True
            if group.research_questions:
                item["research_questions"] = [
                    {"question": question.question, "queries": question.queries}
                    for question in group.research_questions[:3]
                ]
            payload.append(item)
        return payload

    def _boundary_for_cluster(
        self,
        raw: Dict[str, Any],
        article_ids: List[str],
        boundaries: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if not boundaries:
            return None
        cluster_id = _short_text(raw.get("cluster_id", ""), 80)
        for boundary in boundaries:
            if cluster_id and cluster_id == boundary["story_id"]:
                return boundary
        for article_id in article_ids:
            for boundary in boundaries:
                if article_id in boundary["article_ids"]:
                    return boundary
        return None

    def _trim_claims_to_boundary(
        self,
        claims: List[Dict[str, Any]],
        allowed_article_ids: set[str],
    ) -> List[Dict[str, Any]]:
        trimmed: List[Dict[str, Any]] = []
        for claim in claims:
            original = _dedupe_article_ids(claim.get("support_article_ids", []), max_items=12)
            support_ids = [article_id for article_id in original if article_id in allowed_article_ids]
            if original != support_ids:
                self._record_group_boundary_warning("evidence claim support ids crossed shared group boundary; trimmed")
            if not support_ids:
                continue
            item = dict(claim)
            item["support_article_ids"] = support_ids
            trimmed.append(item)
        return trimmed

    def _trim_reader_qa(
        self,
        items: List[Any],
        selected_ids: set[str],
        *,
        enforce_boundaries: bool = False,
    ) -> List[Any]:
        if not selected_ids and not enforce_boundaries:
            return items
        output: List[Any] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            article_ids = _dedupe_article_ids(raw.get("article_ids", []), max_items=8)
            trimmed_ids = [article_id for article_id in article_ids if article_id in selected_ids]
            if article_ids != trimmed_ids:
                self._record_group_boundary_warning("evidence reader_qa referenced unknown article ids; trimmed")
            item = dict(raw)
            item["article_ids"] = trimmed_ids
            output.append(item)
        return output

    def _record_group_boundary_warning(self, warning: str) -> None:
        self.group_boundary_warning_count += 1
        self.warnings.append(warning)

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        required = {"overview", "story_clusters", "global_watch_signals", "reader_qa"}
        missing = required.difference(result.keys())
        if missing:
            raise ValueError(f"evidence distillation: missing key(s): {', '.join(sorted(missing))}")
        normalized = dict(result)
        if not isinstance(normalized.get("story_clusters"), list):
            normalized["story_clusters"] = []
        if not isinstance(normalized.get("global_watch_signals"), list):
            normalized["global_watch_signals"] = []
        if not isinstance(normalized.get("reader_qa"), list):
            normalized["reader_qa"] = []
        normalized["overview"] = str(normalized.get("overview", "")).strip()
        return normalized

    def _article_payload(self, article: SelectedArticle, excerpt_chars: int) -> Dict[str, Any]:
        topic = article.decision.topic or article.candidate.metadata.get("topic_name", "")
        payload: Dict[str, Any] = {
            "id": article.candidate.id,
            "topic": topic,
            "headline": article.candidate.title,
            "source": article.candidate.source,
            "url": article.candidate.url,
            "published_at": datetime_to_iso(article.candidate.published_at),
            "score": article.decision.score,
            "article_text": (article.article_text or article.candidate.snippet)[:excerpt_chars],
            "snippet": (article.candidate.snippet or "")[:180],
            "extraction_status": article.extraction_status,
            "story_threads": story_thread_payloads(article, max_items=2),
        }
        if self.include_enrichment_context:
            payload["context_sources"] = [
                {
                    "kind": item.kind,
                    "source": item.source,
                    "title": item.title[:120],
                    "summary": item.summary[:180],
                    "items": item.items[:2],
                }
                for item in article.context_sources[: self.config.max_context_sources_per_article]
            ]
        return payload

