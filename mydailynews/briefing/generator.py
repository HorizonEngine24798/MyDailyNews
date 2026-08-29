from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from mydailynews.ai.base import AIClient
from mydailynews.ai.prompts import BRIEF_SYSTEM, BRIEF_USER
from mydailynews.ai.schemas import FINAL_BRIEF_JSON_SCHEMA
from mydailynews.ai.token_budget import TokenBudget, resolve_client_token_budget
from mydailynews.analysis.shared import story_thread_payloads
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import PriorReport, SelectedArticle, TopicConfig, UserMemory
from mydailynews.common.utils import compact_json, datetime_to_iso
from mydailynews.domain.candidate_annotations import candidate_memory_annotation

FINAL_PROMPT_BUDGET_SAFETY_RATIO = 0.95


class BriefGenerator:
    def __init__(
        self,
        client: AIClient,
        max_context_chars: int,
        input_token_limit: int | None = None,
        max_new_tokens: int | None = None,
        include_enrichment_context: bool = True,
        debug: DebugLogger | None = None,
    ) -> None:
        self.client = client
        self.max_context_chars = max(200, max_context_chars)
        self.input_token_limit = input_token_limit
        self.max_new_tokens = max_new_tokens
        self.include_enrichment_context = bool(include_enrichment_context)
        self.debug = debug or DebugLogger(False)
        self.warnings: List[str] = []

    def generate(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        evidence_packet: Dict[str, Any] | None = None,
        delta_packet: Dict[str, Any] | None = None,
        recall_packet: Dict[str, Any] | None = None,
        brief_name: str = "",
    ) -> Dict[str, Any]:
        self.warnings = []
        prompt, used_articles = self._build_prompt(
            articles,
            memory,
            topics,
            prior_reports,
            brief_goal,
            date,
            evidence_packet=evidence_packet or {},
            delta_packet=delta_packet or {},
            recall_packet=recall_packet or {},
        )
        budget = self._request_budget()
        self.debug.log(
            "brief.ai",
            "synthesizing",
            articles=len(used_articles),
            prompt_chars=len(prompt),
            max_input_tokens=budget.input_tokens,
            max_new_tokens=budget.output_tokens,
        )
        label = "final brief generation"
        if brief_name:
            label = f"{label} ({brief_name})"
        result = self.client.complete_json(
            BRIEF_SYSTEM,
            prompt,
            label=label,
            max_new_tokens=budget.output_tokens,
            input_token_limit=budget.input_tokens,
            json_schema=FINAL_BRIEF_JSON_SCHEMA,
        )
        required = {"title", "lead", "topic_reports", "sections"}
        missing = required.difference(result.keys())
        if missing:
            raise ValueError(f"final brief generation: missing key(s): {', '.join(sorted(missing))}")

        result.setdefault("title", f"Daily Brief - {date}")
        result["topic_reports"] = self._normalize_topic_reports(result.get("topic_reports", []))
        result["sections"] = self._normalize_sections(result.get("sections", []))
        result["major_headlines"] = self._major_headlines_payload(used_articles)
        result["selected_articles"] = self._selected_articles_payload(used_articles)
        result["references"] = self._references_payload(used_articles)
        self._ensure_signal_slots(
            result,
            used_articles,
            evidence_packet=evidence_packet or {},
            delta_packet=delta_packet or {},
        )
        self.debug.log("brief.ai", "complete", articles=len(used_articles))
        return result

    def _build_prompt(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        topics: List[TopicConfig],
        prior_reports: List[PriorReport],
        brief_goal: str,
        date: str,
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
        recall_packet: Dict[str, Any] | None = None,
    ) -> tuple[str, List[SelectedArticle]]:
        prompt_budget_tokens = self._prompt_budget_tokens()
        ordered_articles = sorted(articles, key=lambda item: item.decision.score, reverse=True)
        active_reports = prior_reports[:3]
        analysis_options = self._analysis_payload_options(evidence_packet, delta_packet)
        excerpt_options = [
            self.max_context_chars,
            min(self.max_context_chars, 650),
            min(self.max_context_chars, 450),
            280,
        ]
        dropped_article_ids: list[str] = []
        analysis_mode_reduced = False
        used_articles = ordered_articles[:]
        prompt = ""

        for excerpt_chars in excerpt_options:
            candidate_articles = used_articles[:]
            candidate_reports = active_reports[:]
            analysis_index = 0
            while candidate_articles:
                analysis_mode, evidence_payload, delta_payload = analysis_options[analysis_index]
                prompt = self._render_prompt(
                    candidate_articles,
                    excerpt_chars,
                    memory,
                    topics,
                    candidate_reports,
                    brief_goal,
                    date,
                    evidence_packet=evidence_payload,
                    delta_packet=delta_payload,
                    recall_packet=recall_packet or {},
                )
                estimated_tokens = self._estimate_final_input_tokens(prompt)
                self.debug.log(
                    "brief.prompt",
                    "budget_check",
                    articles=len(candidate_articles),
                    prior_reports=len(candidate_reports),
                    excerpt_chars=excerpt_chars,
                    analysis_mode=analysis_mode,
                    estimated_tokens=estimated_tokens,
                    budget_tokens=prompt_budget_tokens,
                )
                if estimated_tokens <= prompt_budget_tokens:
                    used_articles = candidate_articles
                    if analysis_mode != "full":
                        analysis_mode_reduced = True
                    if dropped_article_ids:
                        self._append_article_drop_warning(dropped_article_ids, used_articles, prompt_budget_tokens)
                    if analysis_mode_reduced:
                        self.warnings.append(
                            "final brief prompt used compacted analysis context to stay within the local model budget."
                        )
                    return prompt, used_articles
                if len(candidate_reports) > 1:
                    candidate_reports = candidate_reports[:-1]
                    continue
                if analysis_index < len(analysis_options) - 1:
                    analysis_index += 1
                    analysis_mode_reduced = True
                    continue
                if candidate_articles:
                    dropped = candidate_articles.pop()
                    dropped_article_ids.append(dropped.candidate.id)
                    analysis_index = 0
                    continue

        if dropped_article_ids:
            self._append_article_drop_warning(dropped_article_ids, [], prompt_budget_tokens)
        fallback_mode, fallback_evidence, fallback_delta = analysis_options[-1]
        if fallback_mode != "full":
            self.warnings.append("final brief prompt used compacted analysis context to stay within the local model budget.")
        return (
            self._render_prompt(
                [],
                0,
                memory,
                topics,
                active_reports[:1],
                brief_goal,
                date,
                evidence_packet=fallback_evidence,
                delta_packet=fallback_delta,
                recall_packet=recall_packet or {},
            ),
            [],
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
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
        recall_packet: Dict[str, Any] | None = None,
    ) -> str:
        payload = [self._article_payload(article, excerpt_chars) for article in articles]
        return BRIEF_USER.format(
            memory=memory.to_prompt(),
            date=date,
            brief_goal=brief_goal,
            topics=compact_json(self._topics_payload(topics)),
            prior_reports=compact_json(self._prior_reports_payload(prior_reports)),
            recall_packet=compact_json(recall_packet or {}),
            evidence_packet=compact_json(evidence_packet),
            delta_packet=compact_json(delta_packet),
            articles=compact_json(payload),
        )

    def _estimate_final_input_tokens(self, prompt: str) -> int:
        return self.client.estimate_tokens(f"System:\n{BRIEF_SYSTEM}\n\nUser:\n{prompt}\n\nAssistant:\n")

    def _request_budget(self) -> TokenBudget:
        return resolve_client_token_budget(
            self.client,
            input_tokens=self.input_token_limit,
            output_tokens=self.max_new_tokens,
        )

    def _prompt_budget_tokens(self) -> int:
        return max(64, int(self._request_budget().input_tokens * FINAL_PROMPT_BUDGET_SAFETY_RATIO))

    def _append_article_drop_warning(
        self,
        dropped_article_ids: List[str],
        used_articles: List[SelectedArticle],
        prompt_budget_tokens: int,
    ) -> None:
        unique_ids: List[str] = []
        seen: set[str] = set()
        for article_id in dropped_article_ids:
            if article_id in seen:
                continue
            seen.add(article_id)
            unique_ids.append(article_id)

        if used_articles:
            effective_floor = min(float(item.decision.score) for item in used_articles)
            suffix = f"; effective final score floor is {effective_floor:.2f}"
        else:
            suffix = ""
        self.warnings.append(
            "final brief prompt dropped lower-ranked article(s) to stay within the local model budget "
            f"({prompt_budget_tokens} estimated input tokens): "
            + ", ".join(unique_ids)
            + suffix
        )

    def _article_payload(self, article: SelectedArticle, excerpt_chars: int) -> Dict[str, Any]:
        topic = article.decision.topic or article.candidate.metadata.get("topic_name", "")
        payload: Dict[str, Any] = {
            "id": article.candidate.id,
            "topic": topic,
            "headline": article.candidate.title,
            "source": article.candidate.source,
            "published_at": datetime_to_iso(article.candidate.published_at),
            "score": article.decision.score,
            "article_text": (article.article_text or article.candidate.snippet)[:excerpt_chars],
            "extraction_status": article.extraction_status,
            "story_threads": story_thread_payloads(article, max_items=2),
        }
        memory_annotation = candidate_memory_annotation(article.candidate)
        if memory_annotation is not None and memory_annotation.story_key:
            payload["memory"] = {
                "story_key": memory_annotation.story_key,
                "story_family_key": memory_annotation.story_family_key,
                "today_policy": memory_annotation.today_policy,
                "recent_coverage_count": memory_annotation.recent_coverage_count,
                "score_adjustment": round(float(memory_annotation.score_adjustment), 4),
            }
        if self.include_enrichment_context:
            payload["context_note"] = article.enrichment_reason
            payload["context_sources"] = [
                {
                    "kind": item.kind,
                    "source": item.source,
                    "title": item.title[:120],
                    "summary": item.summary[:180],
                    "items": item.items[:3],
                }
                for item in article.context_sources[:2]
            ]
        return payload

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
                "summary": report.summary[:420],
                "major_headlines": report.major_headlines[:5],
                "story_baselines": [
                    {
                        "story_key": str(item.get("story_key", ""))[:100],
                        "story_family_key": str(item.get("story_family_key", ""))[:100],
                        "title": str(item.get("title", ""))[:140],
                        "change_type": str(item.get("change_type", ""))[:40],
                        "summary": str(item.get("summary", "") or item.get("bullet", ""))[:240],
                        "disposition": str(item.get("disposition", ""))[:40],
                    }
                    for item in (report.story_baselines or [])[:4]
                    if isinstance(item, dict)
                ],
            }
            for report in prior_reports
        ]

    @staticmethod
    def _major_headlines_payload(articles: List[SelectedArticle]) -> List[dict]:
        rows: List[dict] = []
        for article in articles:
            memory_annotation = candidate_memory_annotation(article.candidate)
            rows.append(
                {
                    "headline": article.candidate.title,
                    "source": article.candidate.source,
                    "url": article.candidate.url,
                    "score": article.decision.score,
                    "topic": article.decision.topic or article.candidate.metadata.get("topic_name", ""),
                    "story_threads": story_thread_payloads(article, max_items=None, compact=False),
                    "story_key": memory_annotation.story_key if memory_annotation is not None else "",
                }
            )
        return rows

    @staticmethod
    def _selected_articles_payload(articles: List[SelectedArticle]) -> List[dict]:
        rows: List[dict] = []
        for article in articles:
            memory_annotation = candidate_memory_annotation(article.candidate)
            rows.append(
                {
                    "id": article.candidate.id,
                    "headline": article.candidate.title,
                    "source": article.candidate.source,
                    "url": article.candidate.url,
                    "score": article.decision.score,
                    "topic": article.decision.topic or article.candidate.metadata.get("topic_name", ""),
                    "snippet": article.candidate.snippet or "",
                    "story_threads": story_thread_payloads(article, max_items=None, compact=False),
                    "story_key": memory_annotation.story_key if memory_annotation is not None else "",
                    "story_family_key": memory_annotation.story_family_key if memory_annotation is not None else "",
                }
            )
        return rows

    @staticmethod
    def _references_payload(articles: List[SelectedArticle]) -> List[dict]:
        references: List[dict] = []
        seen: set[str] = set()
        for article in articles:
            title = str(article.candidate.title or "").strip()
            source = str(article.candidate.source or "").strip()
            url = str(article.candidate.url or "").strip()
            key = url or f"{title}|{source}"
            if not key or key in seen:
                continue
            seen.add(key)
            references.append(
                {
                    "title": title,
                    "source": source,
                    "url": url,
                }
            )
        return references

    def _analysis_payload_options(
        self,
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
    ) -> List[tuple[str, Dict[str, Any], Dict[str, Any]]]:
        options_raw = [
            ("full", self._compact_evidence_packet(evidence_packet, mode="full"), self._compact_delta_packet(delta_packet, mode="full")),
            ("compact", self._compact_evidence_packet(evidence_packet, mode="compact"), self._compact_delta_packet(delta_packet, mode="compact")),
            ("minimal", self._compact_evidence_packet(evidence_packet, mode="minimal"), self._compact_delta_packet(delta_packet, mode="minimal")),
            ("none", {}, {}),
        ]
        deduped: List[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        seen: set[str] = set()
        for label, evidence, delta in options_raw:
            signature = compact_json({"evidence": evidence, "delta": delta})
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append((label, evidence, delta))
        return deduped or [("none", {}, {})]

    @staticmethod
    def _compact_evidence_packet(packet: Dict[str, Any], mode: str) -> Dict[str, Any]:
        if not isinstance(packet, dict) or not packet:
            return {}
        overview_limit = {"full": 380, "compact": 240, "minimal": 180}.get(mode, 180)
        cluster_limit = {"full": 6, "compact": 5, "minimal": 3}.get(mode, 3)
        claim_limit = {"full": 4, "compact": 3, "minimal": 2}.get(mode, 2)
        point_limit = {"full": 4, "compact": 3, "minimal": 2}.get(mode, 2)
        question_limit = {"full": 6, "compact": 4, "minimal": 2}.get(mode, 2)

        clusters = []
        for item in packet.get("story_clusters", [])[:cluster_limit]:
            if not isinstance(item, dict):
                continue
            claims = []
            for claim in item.get("key_claims", [])[:claim_limit]:
                if not isinstance(claim, dict):
                    continue
                claims.append(
                    {
                        "claim": str(claim.get("claim", ""))[:140],
                        "support_article_ids": [str(value)[:80] for value in claim.get("support_article_ids", [])[:4]],
                        "confidence": str(claim.get("confidence", ""))[:20],
                    }
                )
            clusters.append(
                {
                    "cluster_id": str(item.get("cluster_id", ""))[:60],
                    "topic": str(item.get("topic", ""))[:80],
                    "label": str(item.get("label", ""))[:100],
                    "summary": str(item.get("summary", ""))[:220],
                    "article_ids": [str(value)[:80] for value in item.get("article_ids", [])[:5]],
                    "key_claims": claims,
                    "consensus_points": [str(value)[:120] for value in item.get("consensus_points", [])[:point_limit]],
                    "contested_points": [str(value)[:120] for value in item.get("contested_points", [])[:point_limit]],
                    "known_unknowns": [str(value)[:120] for value in item.get("known_unknowns", [])[:point_limit]],
                    "watch_signals": [str(value)[:120] for value in item.get("watch_signals", [])[:point_limit]],
                }
            )

        reader_qa = []
        for item in packet.get("reader_qa", [])[:question_limit]:
            if not isinstance(item, dict):
                continue
            reader_qa.append(
                {
                    "question": str(item.get("question", ""))[:140],
                    "answer": str(item.get("answer", ""))[:180],
                    "article_ids": [str(value)[:80] for value in item.get("article_ids", [])[:4]],
                }
            )

        return {
            "overview": str(packet.get("overview", ""))[:overview_limit],
            "story_clusters": clusters,
            "global_watch_signals": [str(value)[:120] for value in packet.get("global_watch_signals", [])[:point_limit + 2]],
            "reader_qa": reader_qa,
        }

    @staticmethod
    def _compact_delta_packet(packet: Dict[str, Any], mode: str) -> Dict[str, Any]:
        if not isinstance(packet, dict) or not packet:
            return {}
        item_limit = {"full": 5, "compact": 3, "minimal": 2}.get(mode, 2)
        summary_limit = {"full": 180, "compact": 140, "minimal": 110}.get(mode, 110)
        note_limit = {"full": 220, "compact": 160, "minimal": 120}.get(mode, 120)

        def _entries(key: str) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for item in packet.get(key, [])[:item_limit]:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "item": str(item.get("item", ""))[:100],
                        "summary": str(item.get("summary", ""))[:summary_limit],
                        "article_ids": [str(value)[:80] for value in item.get("article_ids", [])[:4]],
                    }
                )
            return rows

        def _story_decisions() -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for item in packet.get("story_decisions", [])[:item_limit]:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "story_key": str(item.get("story_key", ""))[:100],
                        "article_ids": [str(value)[:80] for value in item.get("article_ids", [])[:4]],
                        "relationship": str(item.get("relationship", "uncertain"))[:30],
                        "change_type": str(item.get("change_type", "uncertain"))[:30],
                        "disposition": str(item.get("disposition", "uncertain"))[:30],
                        "confidence": item.get("confidence", 0.0),
                        "summary": str(item.get("summary", ""))[:summary_limit],
                        "bullet": str(item.get("bullet", ""))[:summary_limit],
                    }
                )
            return rows

        gaps = []
        for item in packet.get("evidence_gaps", [])[:item_limit]:
            if not isinstance(item, dict):
                continue
            gaps.append(
                {
                    "gap": str(item.get("gap", ""))[:120],
                    "why_it_matters": str(item.get("why_it_matters", ""))[:summary_limit],
                }
            )

        return {
            "baseline_coverage_note": str(packet.get("baseline_coverage_note", ""))[:note_limit],
            "new": _entries("new"),
            "escalated": _entries("escalated"),
            "weakened": _entries("weakened"),
            "reframed": _entries("reframed"),
            "unchanged_but_important": _entries("unchanged_but_important"),
            "story_decisions": _story_decisions(),
            "evidence_gaps": gaps,
        }

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    @classmethod
    def _to_string_list(cls, value: Any) -> List[str]:
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = [value]
        else:
            raw_items = []
        cleaned: List[str] = []
        for raw in raw_items:
            text = cls._normalize_text(raw)
            if text:
                cleaned.append(text)
        return cls._normalized_string_list(cleaned)

    @classmethod
    def _normalize_narrative_changes(cls, value: Any) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        rows: List[Dict[str, str]] = []
        for raw in value:
            if not isinstance(raw, dict):
                continue
            narrative = cls._normalize_text(raw.get("narrative", ""))
            status = cls._normalize_text(raw.get("status", ""))
            summary = cls._normalize_text(raw.get("summary", ""))
            if not (narrative or summary):
                continue
            rows.append(
                {
                    "narrative": narrative,
                    "status": status,
                    "summary": summary,
                }
            )
        return rows

    @classmethod
    def _normalize_topic_reports(cls, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        reports: List[Dict[str, Any]] = []
        for raw in value:
            if not isinstance(raw, dict):
                continue
            topic = cls._normalize_text(raw.get("topic", "")) or "Topic"
            why_it_matters = cls._normalize_text(raw.get("why_it_matters", ""))
            what_changed = cls._normalize_text(raw.get("what_changed", ""))
            narrative_summary = cls._normalize_text(raw.get("narrative_summary", "") or raw.get("summary", ""))
            if not why_it_matters and narrative_summary:
                why_it_matters = narrative_summary

            narrative_changes = cls._normalize_narrative_changes(raw.get("narrative_changes", []))
            if not what_changed and narrative_changes:
                what_changed = cls._normalize_text(narrative_changes[0].get("summary", ""))
            if not what_changed and narrative_summary:
                what_changed = narrative_summary

            who_is_affected = cls._to_string_list(raw.get("who_is_affected", []))
            what_to_watch = cls._to_string_list(raw.get("what_to_watch", raw.get("watch_signals", [])))
            if not narrative_summary:
                parts = [why_it_matters, what_changed]
                narrative_summary = cls._normalize_text(". ".join([part for part in parts if part]))

            reports.append(
                {
                    "topic": topic,
                    "why_it_matters": why_it_matters,
                    "what_changed": what_changed,
                    "who_is_affected": who_is_affected,
                    "narrative_summary": narrative_summary,
                    "narrative_changes": narrative_changes,
                    "what_to_watch": what_to_watch,
                }
            )
        return reports

    @classmethod
    def _normalize_sections(cls, value: Any) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        sections: List[Dict[str, str]] = []
        for raw in value:
            if not isinstance(raw, dict):
                continue
            heading = cls._normalize_text(raw.get("heading", ""))
            summary = cls._normalize_text(raw.get("summary", ""))
            if not (heading or summary):
                continue
            sections.append({"heading": heading or "Section", "summary": summary})
        return sections

    def _ensure_signal_slots(
        self,
        result: Dict[str, Any],
        articles: List[SelectedArticle],
        *,
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
    ) -> None:
        knowns = self._normalized_string_list(result.get("knowns", []))
        unknowns = self._normalized_string_list(result.get("unknowns", []))
        watch_signals = self._normalized_string_list(result.get("watch_signals", []))

        if not knowns:
            knowns = self._fallback_knowns(result, articles, evidence_packet, delta_packet)
        if not unknowns:
            unknowns = self._fallback_unknowns(evidence_packet, delta_packet)
        if not watch_signals:
            watch_signals = self._fallback_watch_signals(result, evidence_packet, delta_packet)

        result["knowns"] = knowns
        result["unknowns"] = unknowns
        result["watch_signals"] = watch_signals

    @staticmethod
    def _normalized_string_list(value: Any, *, limit: int | None = None) -> List[str]:
        items: List[str] = []
        if isinstance(value, list):
            for raw in value:
                text = " ".join(str(raw or "").split()).strip()
                if not text:
                    continue
                items.append(text)
        seen: set[str] = set()
        deduped: List[str] = []
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if limit is not None and len(deduped) >= limit:
                break
        return deduped

    def _fallback_knowns(
        self,
        result: Dict[str, Any],
        articles: List[SelectedArticle],
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
    ) -> List[str]:
        candidates: List[str] = []
        for cluster in evidence_packet.get("story_clusters", []):
            if not isinstance(cluster, dict):
                continue
            for point in cluster.get("consensus_points", []):
                text = str(point).strip()
                if text:
                    candidates.append(text)
        for item in delta_packet.get("unchanged_but_important", []):
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary", "")).strip()
            label = str(item.get("item", "")).strip()
            if summary:
                candidates.append(summary)
            elif label:
                candidates.append(label)
        for report in result.get("topic_reports", []):
            if not isinstance(report, dict):
                continue
            for key in ("why_it_matters", "what_changed", "narrative_summary"):
                text = str(report.get(key, "")).strip()
                if text:
                    candidates.append(text)
        if not candidates:
            for article in articles:
                headline = str(article.candidate.title).strip()
                source = str(article.candidate.source).strip()
                if headline:
                    label = headline
                    if source:
                        label = f"{headline} ({source})"
                    candidates.append(label)
        return self._normalized_string_list(candidates)

    def _fallback_unknowns(
        self,
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
    ) -> List[str]:
        candidates: List[str] = []
        for cluster in evidence_packet.get("story_clusters", []):
            if not isinstance(cluster, dict):
                continue
            for point in cluster.get("known_unknowns", []):
                text = str(point).strip()
                if text:
                    candidates.append(text)
            for point in cluster.get("contested_points", []):
                text = str(point).strip()
                if text:
                    candidates.append(text)
        for item in delta_packet.get("evidence_gaps", []):
            if not isinstance(item, dict):
                continue
            gap = str(item.get("gap", "")).strip()
            why = str(item.get("why_it_matters", "")).strip()
            if gap and why:
                candidates.append(f"{gap} ({why})")
            elif gap:
                candidates.append(gap)
        return self._normalized_string_list(candidates)

    def _fallback_watch_signals(
        self,
        result: Dict[str, Any],
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
    ) -> List[str]:
        candidates: List[str] = []
        for report in result.get("topic_reports", []):
            if not isinstance(report, dict):
                continue
            for item in report.get("what_to_watch", []):
                text = str(item).strip()
                if text:
                    candidates.append(text)
        for item in evidence_packet.get("global_watch_signals", []):
            text = str(item).strip()
            if text:
                candidates.append(text)
        for cluster in evidence_packet.get("story_clusters", []):
            if not isinstance(cluster, dict):
                continue
            for item in cluster.get("watch_signals", []):
                text = str(item).strip()
                if text:
                    candidates.append(text)
        for key in ("new", "escalated"):
            for item in delta_packet.get(key, []):
                if not isinstance(item, dict):
                    continue
                label = str(item.get("item", "")).strip()
                if label:
                    candidates.append(label)
        return self._normalized_string_list(candidates)


def brief_metadata(
    date: str,
    model: str,
    candidate_count: int,
    selected_count: int,
    topics: List[str] | None = None,
    prior_reports_count: int = 0,
    brief_name: str = "",
    warnings: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "2.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "date": date,
        "brief_name": brief_name,
        "model": model,
        "topics": topics or [],
        "prior_reports_count": prior_reports_count,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "warnings": warnings or [],
    }


def no_material_changes_brief(date: str) -> Dict[str, Any]:
    """Return a source-empty brief without giving an LLM room to reconstruct repeats."""

    return {
        "title": f"Daily Brief - {date}",
        "lead": "No material changes met this brief's reporting threshold in the supplied sources.",
        "knowns": [],
        "unknowns": [],
        "watch_signals": [],
        "topic_reports": [],
        "sections": [],
        "major_headlines": [],
        "selected_articles": [],
        "references": [],
    }
