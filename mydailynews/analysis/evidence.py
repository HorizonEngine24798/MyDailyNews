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
    topics_payload,
)
from mydailynews.common.cache import JSONCache
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.domain.event_clusters import candidate_event_cluster_payload
from mydailynews.app.models import EvidenceDistillationConfig, PriorReport, SelectedArticle, TopicConfig, UserMemory
from mydailynews.common.utils import compact_json, datetime_to_iso


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

    def distill(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        brief_name: str = "",
    ) -> Dict[str, Any]:
        self.warnings = []
        if not self.config.enabled:
            self.debug.log("analysis.evidence", "skipped_disabled")
            return {}
        if not articles:
            self.warnings.append("evidence distillation skipped: no selected articles")
            return {}

        ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)[: self.config.max_articles]
        batches = self._build_article_batches(
            ordered_articles,
            memory=memory,
            topics=topics,
            prior_reports=prior_reports,
            brief_goal=brief_goal,
            date=date,
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
                )
            )

        if not results:
            self.warnings.append("evidence distillation skipped: no batch yielded usable evidence")
            return {}

        result = self._merge_results(results)
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
        )
        if single_batch is not None:
            return [single_batch]

        batches: List[List[SelectedArticle]] = []
        current: List[SelectedArticle] = []

        for group in _ordered_article_groups(articles):
            if current and self._candidate_batch_fits(
                current + group,
                all_articles=articles,
                memory=memory,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                batch_size=batch_size,
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
            ):
                current = group[:]
                continue
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
    ) -> List[SelectedArticle] | None:
        max_drop = min(max(0, int(self.config.max_articles_dropped_to_avoid_split)), max(0, len(articles) - 1))
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
        )
        return self.client.estimate_tokens(prompt) <= self._input_token_limit()

    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = [self._normalize_result(item) for item in results if isinstance(item, dict)]
        overview = " ".join(_short_text(item.get("overview", ""), 360) for item in normalized if item.get("overview"))

        clusters: List[Dict[str, Any]] = []
        seen_clusters: set[str] = set()
        for result in normalized:
            for raw in result.get("story_clusters", []):
                if not isinstance(raw, dict):
                    continue
                article_ids = [str(value)[:80] for value in raw.get("article_ids", [])[:8]]
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
                clusters.append(
                    {
                        "cluster_id": _short_text(raw.get("cluster_id", ""), 80),
                        "topic": _short_text(raw.get("topic", ""), 80),
                        "label": _short_text(raw.get("label", ""), 120),
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
    ) -> str:
        fingerprint = {
            "v": 4,
            "stage": "evidence_distillation",
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
            "prior_reports": prior_reports_payload(used_reports),
        }
        return JSONCache.make_key(compact_json(fingerprint))

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
            "event_cluster": candidate_event_cluster_payload(article.candidate),
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

