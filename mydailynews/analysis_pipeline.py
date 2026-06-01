from __future__ import annotations

import json
from typing import Any, Dict, List

from .ai.base import AIClient
from .ai.prompts import (
    DELTA_EXTRACTION_SYSTEM,
    DELTA_EXTRACTION_USER,
    EVIDENCE_DISTILLATION_SYSTEM,
    EVIDENCE_DISTILLATION_USER,
)
from .ai.schemas import DELTA_EXTRACTION_JSON_SCHEMA, EVIDENCE_DISTILLATION_JSON_SCHEMA
from .cache import JSONCache
from .debug import DebugLogger
from .models import (
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
    PriorReport,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from .utils import datetime_to_iso


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _event_cluster_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    cluster_id = str(metadata.get("event_cluster_id", "")).strip()
    if not cluster_id:
        return {}
    return {
        "id": cluster_id,
        "label": str(metadata.get("event_cluster_label", ""))[:180],
        "size": int(metadata.get("event_cluster_size", 1) or 1),
        "source_count": int(metadata.get("event_cluster_source_count", 1) or 1),
        "multi_source": bool(metadata.get("event_cluster_multi_source", False)),
        "latest_published_at": str(metadata.get("event_cluster_latest_published_at", ""))[:64],
    }


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

        prompt, used_articles, used_reports = self._build_prompt(
            articles=articles,
            memory=memory,
            topics=topics,
            prior_reports=prior_reports,
            brief_goal=brief_goal,
            date=date,
        )
        if not used_articles:
            self.warnings.append("evidence distillation skipped: prompt builder yielded no usable articles")
            return {}

        label = "evidence distillation"
        if brief_name:
            label = f"{label} ({brief_name})"
        cache_key = self._cache_key(
            used_articles=used_articles,
            used_reports=used_reports,
            memory=memory,
            topics=topics,
            brief_goal=brief_goal,
            date=date,
        )
        if self.cache:
            cached = self.cache.get(cache_key, max_age_seconds=self.cache_ttl_seconds)
            if cached is not None:
                self.debug.log(
                    "analysis.evidence",
                    "cache_hit",
                    articles=len(used_articles),
                    prior_reports=len(used_reports),
                )
                return self._normalize_result(cached)

        self.debug.log(
            "analysis.evidence",
            "running",
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
        if not self.config.include_reader_qa:
            result["reader_qa"] = []
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
    ) -> tuple[str, List[SelectedArticle], List[PriorReport]]:
        target_input_tokens = max(1024, min(int(self.config.max_input_tokens), int(self.client.max_input_tokens)))
        prompt_budget_tokens = max(900, int(target_input_tokens * 0.9))
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
        return self._render_prompt([], 0, memory, topics, active_reports[:1], brief_goal, date), [], active_reports[:1]

    def _render_prompt(
        self,
        articles: List[SelectedArticle],
        excerpt_chars: int,
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
    ) -> str:
        payload = [self._article_payload(item, excerpt_chars) for item in articles]
        return EVIDENCE_DISTILLATION_USER.format(
            memory=memory.to_prompt(),
            brief_goal=brief_goal,
            date=date,
            topics=_compact_json(self._topics_payload(topics)),
            prior_reports=_compact_json(self._prior_reports_payload(prior_reports)),
            articles=_compact_json(payload),
        )

    def _cache_key(
        self,
        *,
        used_articles: List[SelectedArticle],
        used_reports: List[PriorReport],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        date: str,
    ) -> str:
        fingerprint = {
            "v": 2,
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
                "max_article_chars": self.config.max_article_chars,
                "max_context_sources_per_article": self.config.max_context_sources_per_article,
            },
            "topics": self._topics_payload(topics),
            "articles": [self._article_cache_payload(item) for item in used_articles],
            "prior_reports": self._prior_reports_payload(used_reports),
        }
        return JSONCache.make_key(_compact_json(fingerprint))

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
            "event_cluster": _event_cluster_payload(article.candidate.metadata),
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

    @staticmethod
    def _article_cache_payload(article: SelectedArticle) -> Dict[str, Any]:
        return {
            "id": article.candidate.id,
            "headline": article.candidate.title[:160],
            "source": article.candidate.source,
            "published_at": datetime_to_iso(article.candidate.published_at),
            "score": article.decision.score,
            "article_text": (article.article_text or article.candidate.snippet)[:220],
            "snippet": (article.candidate.snippet or "")[:120],
            "event_cluster": _event_cluster_payload(article.candidate.metadata),
        }

    @staticmethod
    def _topics_payload(topics: List[TopicConfig]) -> List[dict]:
        return [
            {
                "name": topic.name,
                "description": (topic.description or "")[:180],
                "queries": [query[:80] for query in (topic.queries or [topic.name])[:3]],
            }
            for topic in topics
            if topic.enabled
        ]

    @staticmethod
    def _prior_reports_payload(prior_reports: List[PriorReport]) -> List[dict]:
        return [
            {
                "id": report.id,
                "date": report.date,
                "title": report.title,
                "topics": report.topics[:4],
                "summary": report.summary[:320],
                "major_headlines": report.major_headlines[:4],
            }
            for report in prior_reports
        ]


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

        prompt, used_articles, used_reports, reduced_evidence = self._build_prompt(
            articles=articles,
            memory=memory,
            topics=topics,
            prior_reports=prior_reports,
            brief_goal=brief_goal,
            date=date,
            evidence_packet=evidence_packet,
        )
        if not used_articles and not reduced_evidence:
            self.warnings.append("delta extraction skipped: prompt builder yielded no usable evidence")
            return {}

        label = "delta extraction"
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
        )
        if self.cache:
            cached = self.cache.get(cache_key, max_age_seconds=self.cache_ttl_seconds)
            if cached is not None:
                self.debug.log(
                    "analysis.delta",
                    "cache_hit",
                    articles=len(used_articles),
                    prior_reports=len(used_reports),
                )
                return self._normalize_result(cached)

        self.debug.log(
            "analysis.delta",
            "running",
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
    ) -> tuple[str, List[SelectedArticle], List[PriorReport], Dict[str, Any]]:
        target_input_tokens = max(1024, min(int(self.config.max_input_tokens), int(self.client.max_input_tokens)))
        prompt_budget_tokens = max(900, int(target_input_tokens * 0.9))
        ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)
        max_articles = 8
        used_articles = ordered_articles[:max_articles]
        used_reports = prior_reports[: self.config.max_prior_reports]
        reduced_evidence = self._compact_evidence_packet(evidence_packet)

        excerpt_options = [420, 300, 220, 160]
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
    ) -> str:
        article_payload = [self._article_payload(item, excerpt_chars) for item in articles]
        reduced_evidence: Dict[str, Any] = {}
        if self.config.input_source in {"evidence_or_articles", "evidence_only"}:
            reduced_evidence = evidence_packet
        fallback_articles = article_payload if self.config.input_source in {"evidence_or_articles", "articles_only"} else []
        return DELTA_EXTRACTION_USER.format(
            memory=memory.to_prompt(),
            brief_goal=brief_goal,
            date=date,
            topics=_compact_json(EvidenceDistiller._topics_payload(topics)),
            prior_reports=_compact_json(EvidenceDistiller._prior_reports_payload(prior_reports)),
            evidence_packet=_compact_json(reduced_evidence),
            articles=_compact_json(fallback_articles),
        )

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
    ) -> str:
        fingerprint = {
            "v": 2,
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
                "max_prior_reports": self.config.max_prior_reports,
            },
            "topics": EvidenceDistiller._topics_payload(topics),
            "articles": [EvidenceDistiller._article_cache_payload(item) for item in used_articles],
            "prior_reports": EvidenceDistiller._prior_reports_payload(used_reports),
            "evidence_packet": reduced_evidence,
        }
        return JSONCache.make_key(_compact_json(fingerprint))

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
            "event_cluster": _event_cluster_payload(article.candidate.metadata),
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
