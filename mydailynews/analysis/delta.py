from __future__ import annotations

from typing import Any, Dict, List

from mydailynews.ai.base import AIClient
from mydailynews.ai.prompts import DELTA_EXTRACTION_SYSTEM, DELTA_EXTRACTION_USER
from mydailynews.ai.schemas import DELTA_EXTRACTION_JSON_SCHEMA
from mydailynews.analysis.shared import (
    _append_headline_context,
    _article_ids,
    _dedupe_dicts_by_text,
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
from mydailynews.app.models import DeltaExtractionConfig, PriorReport, SelectedArticle, TopicConfig, UserMemory
from mydailynews.common.utils import compact_json, datetime_to_iso


class DeltaExtractor:
    def __init__(
        self,
        client: AIClient,
        config: DeltaExtractionConfig,
        debug: DebugLogger | None = None,
        cache: JSONCache | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.cache = cache
        self.cache_ttl_seconds = (
            max(0, int(cache_ttl_seconds))
            if cache_ttl_seconds is not None
            else max(0, int(config.cache_ttl_seconds))
        )
        self.warnings: List[str] = []

    def extract(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        evidence_packet: Dict[str, Any] | None = None,
        brief_name: str = "",
    ) -> Dict[str, Any]:
        self.warnings = []
        evidence_packet = evidence_packet or {}
        if not self.config.enabled:
            self.debug.log("analysis.delta", "skipped_disabled")
            return {}
        if self.config.require_prior_reports and not prior_reports:
            self.warnings.append("delta extraction skipped: require_prior_reports=true but no prior reports were available")
            return {}
        if self.config.input_source == "evidence_only" and not evidence_packet:
            self.warnings.append("delta extraction skipped: input_source=evidence_only but no evidence packet was provided")
            return {}
        if not articles and not evidence_packet:
            self.warnings.append("delta extraction skipped: no selected articles or evidence packet")
            return {}

        if self.config.input_source == "evidence_only":
            ordered_articles: List[SelectedArticle] = []
            batches = [[]]
        else:
            ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)[: self.config.max_articles]
            batches = self._build_article_batches(
                ordered_articles,
                memory=memory,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                evidence_packet=evidence_packet,
            )
            if not batches and evidence_packet:
                batches = [[]]
        self.debug.log(
            "analysis.delta",
            "starting_batched_extraction",
            articles=len(ordered_articles),
            batches=len(batches),
            batch_size=max(1, int(self.config.max_articles_per_batch)),
        )
        self.debug.set_metric("analysis.delta.batch_size", max(1, int(self.config.max_articles_per_batch)))
        self.debug.set_metric("analysis.delta.batches", len(batches))
        if len(batches) > 1:
            self.warnings.append(f"delta extraction split selected articles into {len(batches)} estimated-fit batch(es).")

        results: List[Dict[str, Any]] = []
        for batch_index, batch_articles in enumerate(batches, start=1):
            batch_ids = _article_ids(batch_articles)
            headline_context_articles = [
                article for article in ordered_articles if article.candidate.id not in batch_ids
            ] if len(batches) > 1 else []
            prompt, used_articles, used_reports, reduced_evidence = self._build_prompt(
                articles=batch_articles,
                memory=memory,
                topics=topics,
                prior_reports=prior_reports,
                brief_goal=brief_goal,
                date=date,
                evidence_packet=evidence_packet,
                headline_context_articles=headline_context_articles,
            )
            if not used_articles and not reduced_evidence:
                self.warnings.append(
                    f"delta extraction batch {batch_index}/{len(batches)} skipped: prompt builder yielded no usable evidence"
                )
                continue
            results.append(
                self._run_batch(
                    prompt=prompt,
                    used_articles=used_articles,
                    used_reports=used_reports,
                    reduced_evidence=reduced_evidence,
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
            self.warnings.append("delta extraction skipped: no batch yielded usable evidence")
            return {}
        return self._merge_results(results)

    def _run_batch(
        self,
        *,
        prompt: str,
        used_articles: List[SelectedArticle],
        used_reports: List[PriorReport],
        reduced_evidence: Dict[str, Any],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        date: str,
        brief_name: str,
        batch_index: int,
        total_batches: int,
        headline_context_articles: List[SelectedArticle],
    ) -> Dict[str, Any]:
        label = f"delta extraction batch {batch_index}/{total_batches}"
        if brief_name:
            label = f"{label} ({brief_name})"
        cache_key = self._cache_key(
            used_articles=used_articles,
            used_reports=used_reports,
            reduced_evidence=reduced_evidence,
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
                    "analysis.delta.batch",
                    "cache_hit",
                    batch=f"{batch_index}/{total_batches}",
                    articles=len(used_articles),
                    prior_reports=len(used_reports),
                )
                return self._normalize_result(cached)

        self.debug.log(
            "analysis.delta.batch",
            "running",
            batch=f"{batch_index}/{total_batches}",
            articles=len(used_articles),
            prior_reports=len(used_reports),
            prompt_chars=len(prompt),
            max_input_tokens=self.config.max_input_tokens,
            max_new_tokens=self.config.max_new_tokens,
        )
        raw = self.client.complete_json(
            DELTA_EXTRACTION_SYSTEM,
            prompt,
            label=label,
            max_new_tokens=self.config.max_new_tokens,
            input_token_limit=self.config.max_input_tokens,
            json_schema=DELTA_EXTRACTION_JSON_SCHEMA,
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
        evidence_packet: Dict[str, Any],
        headline_context_articles: List[SelectedArticle] | None = None,
    ) -> tuple[str, List[SelectedArticle], List[PriorReport], Dict[str, Any]]:
        target_input_tokens = max(1024, min(int(self.config.max_input_tokens), int(self.client.max_input_tokens)))
        prompt_budget_tokens = target_input_tokens
        ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)
        used_articles = ordered_articles[: self.config.max_articles]
        used_reports = prior_reports[: self.config.max_prior_reports]
        reduced_evidence = self._compact_evidence_packet(evidence_packet)

        max_article_chars = max(120, int(self.config.max_article_chars))
        excerpt_options = [
            max_article_chars,
            min(max_article_chars, 300),
            min(max_article_chars, 220),
            160,
        ]
        dropped_article_ids: list[str] = []
        prompt = ""
        for excerpt_chars in excerpt_options:
            candidate_articles = used_articles[:]
            candidate_reports = used_reports[:]
            candidate_evidence = dict(reduced_evidence)
            while candidate_articles or candidate_evidence:
                prompt = self._render_prompt(
                    articles=candidate_articles,
                    excerpt_chars=excerpt_chars,
                    memory=memory,
                    topics=topics,
                    prior_reports=candidate_reports,
                    brief_goal=brief_goal,
                    date=date,
                    evidence_packet=candidate_evidence,
                    headline_context_articles=headline_context_articles or [],
                )
                estimated_tokens = self.client.estimate_tokens(prompt)
                self.debug.log(
                    "analysis.delta.prompt",
                    "budget_check",
                    articles=len(candidate_articles),
                    prior_reports=len(candidate_reports),
                    excerpt_chars=excerpt_chars,
                    estimated_tokens=estimated_tokens,
                    budget_tokens=prompt_budget_tokens,
                )
                if estimated_tokens <= prompt_budget_tokens:
                    if dropped_article_ids:
                        self.warnings.append(
                            "delta extraction prompt dropped lower-ranked article(s) to stay within model budget: "
                            + ", ".join(dropped_article_ids)
                        )
                    return prompt, candidate_articles, candidate_reports, candidate_evidence
                self.debug.increment("analysis.delta.prompt_pressure_checks", 1)
                if len(candidate_reports) > 1:
                    candidate_reports = candidate_reports[:-1]
                    self.debug.increment("analysis.delta.prompt_compaction.drop_prior_report", 1)
                    continue
                if candidate_evidence and self.config.input_source in {"evidence_or_articles", "evidence_only"}:
                    candidate_evidence = {}
                    self.debug.increment("analysis.delta.prompt_compaction.drop_evidence_packet", 1)
                    continue
                if len(candidate_articles) > 2:
                    dropped = candidate_articles.pop()
                    dropped_article_ids.append(dropped.candidate.id)
                    self.debug.increment("analysis.delta.prompt_compaction.drop_article", 1)
                    continue
                used_articles = candidate_articles
                used_reports = candidate_reports
                reduced_evidence = candidate_evidence
                break

        if dropped_article_ids:
            self.warnings.append(
                "delta extraction prompt dropped lower-ranked article(s) to stay within model budget: "
                + ", ".join(dropped_article_ids)
            )
        if prompt:
            estimated_tokens = self.client.estimate_tokens(prompt)
            if estimated_tokens > prompt_budget_tokens:
                self.warnings.append(
                    f"delta extraction prompt may exceed budget ({estimated_tokens}>{prompt_budget_tokens}); "
                    "backend limits may truncate context"
                )
        return prompt, used_articles, used_reports[:1], reduced_evidence

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
        evidence_packet: Dict[str, Any],
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
            evidence_packet=evidence_packet,
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
                evidence_packet=evidence_packet,
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
                evidence_packet=evidence_packet,
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
                    evidence_packet=evidence_packet,
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
        evidence_packet: Dict[str, Any],
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
                evidence_packet=evidence_packet,
                batch_size=batch_size,
                include_headline_context=False,
            ):
                continue
            if drop_count:
                dropped = articles[len(candidate_articles) :]
                dropped_ids = [article.candidate.id for article in dropped]
                self.warnings.append(
                    "delta extraction dropped lower-ranked article(s) instead of splitting into another batch: "
                    + ", ".join(dropped_ids)
                )
                self.debug.log(
                    "analysis.delta",
                    "dropped_tail_to_avoid_split",
                    dropped=drop_count,
                    kept=len(candidate_articles),
                    max_drop=max_drop,
                )
                self.debug.set_metric("analysis.delta.avoid_split_dropped_articles", drop_count)
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
        evidence_packet: Dict[str, Any],
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
            articles=candidate_articles,
            excerpt_chars=self.config.max_article_chars,
            memory=memory,
            topics=topics,
            prior_reports=prior_reports[: self.config.max_prior_reports],
            brief_goal=brief_goal,
            date=date,
            evidence_packet=self._compact_evidence_packet(evidence_packet),
            headline_context_articles=headline_context,
        )
        return self.client.estimate_tokens(prompt) <= self._input_token_limit()

    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = [self._normalize_result(item) for item in results if isinstance(item, dict)]
        merged: Dict[str, Any] = {
            "baseline_coverage_note": " ".join(
                _short_text(item.get("baseline_coverage_note", ""), 220)
                for item in normalized
                if item.get("baseline_coverage_note")
            )[:900],
            "new": [],
            "escalated": [],
            "weakened": [],
            "reframed": [],
            "unchanged_but_important": [],
            "evidence_gaps": [],
        }
        for key in ("new", "escalated", "weakened", "reframed", "unchanged_but_important"):
            items: List[Any] = []
            for result in normalized:
                items.extend(result.get(key, []))
            merged[key] = _dedupe_dicts_by_text(items, text_key="item", max_items=16)

        gaps: List[Any] = []
        for result in normalized:
            gaps.extend(result.get("evidence_gaps", []))
        merged["evidence_gaps"] = _dedupe_dicts_by_text(gaps, text_key="gap", max_items=12)
        return self._normalize_result(merged)

    def _render_prompt(
        self,
        *,
        articles: List[SelectedArticle],
        excerpt_chars: int,
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        evidence_packet: Dict[str, Any],
        headline_context_articles: List[SelectedArticle] | None = None,
    ) -> str:
        article_payload = [self._article_payload(item, excerpt_chars) for item in articles]
        reduced_evidence: Dict[str, Any] = {}
        if self.config.input_source in {"evidence_or_articles", "evidence_only"}:
            reduced_evidence = evidence_packet
        fallback_articles = article_payload if self.config.input_source in {"evidence_or_articles", "articles_only"} else []
        prompt = DELTA_EXTRACTION_USER.format(
            memory=memory.to_prompt(),
            brief_goal=brief_goal,
            date=date,
            topics=compact_json(topics_payload(topics)),
            prior_reports=compact_json(prior_reports_payload(prior_reports)),
            evidence_packet=compact_json(reduced_evidence),
            articles=compact_json(fallback_articles),
        )
        return _append_headline_context(prompt, headline_context_articles or [], stage="delta")

    def _cache_key(
        self,
        *,
        used_articles: List[SelectedArticle],
        used_reports: List[PriorReport],
        reduced_evidence: Dict[str, Any],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        date: str,
        headline_context_articles: List[SelectedArticle],
    ) -> str:
        fingerprint = {
            "v": 4,
            "stage": "delta_extraction",
            "backend": self.client.config.backend,
            "model": self.client.config.effective_model_label,
            "response_format": self.client.config.response_format,
            "date": date,
            "brief_goal": brief_goal,
            "memory": memory.to_prompt(),
            "config": {
                "input_source": self.config.input_source,
                "require_prior_reports": self.config.require_prior_reports,
                "max_input_tokens": self.config.max_input_tokens,
                "max_new_tokens": self.config.max_new_tokens,
                "max_articles": self.config.max_articles,
                "max_articles_per_batch": self.config.max_articles_per_batch,
                "max_article_chars": self.config.max_article_chars,
                "max_prior_reports": self.config.max_prior_reports,
            },
            "topics": topics_payload(topics),
            "articles": [article_cache_payload(item) for item in used_articles],
            "headline_context_articles": [
                headline_context_cache_payload(item) for item in headline_context_articles
            ],
            "prior_reports": prior_reports_payload(used_reports),
            "evidence_packet": reduced_evidence,
        }
        return JSONCache.make_key(compact_json(fingerprint))

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        required = {
            "baseline_coverage_note",
            "new",
            "escalated",
            "weakened",
            "reframed",
            "unchanged_but_important",
            "evidence_gaps",
        }
        missing = required.difference(result.keys())
        if missing:
            raise ValueError(f"delta extraction: missing key(s): {', '.join(sorted(missing))}")
        normalized = dict(result)
        for key in ("new", "escalated", "weakened", "reframed", "unchanged_but_important", "evidence_gaps"):
            if not isinstance(normalized.get(key), list):
                normalized[key] = []
        normalized["baseline_coverage_note"] = str(normalized.get("baseline_coverage_note", "")).strip()
        return normalized

    @staticmethod
    def _article_payload(article: SelectedArticle, excerpt_chars: int) -> Dict[str, Any]:
        topic = article.decision.topic or article.candidate.metadata.get("topic_name", "")
        return {
            "id": article.candidate.id,
            "topic": topic,
            "headline": article.candidate.title,
            "source": article.candidate.source,
            "url": article.candidate.url,
            "published_at": datetime_to_iso(article.candidate.published_at),
            "score": article.decision.score,
            "article_text": (article.article_text or article.candidate.snippet)[:excerpt_chars],
            "snippet": (article.candidate.snippet or "")[:160],
            "event_cluster": candidate_event_cluster_payload(article.candidate),
        }

    @staticmethod
    def _compact_evidence_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(packet, dict) or not packet:
            return {}
        clusters = []
        for item in packet.get("story_clusters", [])[:6]:
            if not isinstance(item, dict):
                continue
            clusters.append(
                {
                    "cluster_id": str(item.get("cluster_id", ""))[:60],
                    "topic": str(item.get("topic", ""))[:80],
                    "label": str(item.get("label", ""))[:100],
                    "summary": str(item.get("summary", ""))[:220],
                    "article_ids": [str(value)[:80] for value in item.get("article_ids", [])[:6]],
                    "watch_signals": [str(value)[:140] for value in item.get("watch_signals", [])[:4]],
                }
            )
        reader_qa = []
        for item in packet.get("reader_qa", [])[:4]:
            if not isinstance(item, dict):
                continue
            reader_qa.append(
                {
                    "question": str(item.get("question", ""))[:160],
                    "answer": str(item.get("answer", ""))[:220],
                    "article_ids": [str(value)[:80] for value in item.get("article_ids", [])[:4]],
                }
            )
        return {
            "overview": str(packet.get("overview", ""))[:260],
            "story_clusters": clusters,
            "global_watch_signals": [str(value)[:140] for value in packet.get("global_watch_signals", [])[:6]],
            "reader_qa": reader_qa,
        }
