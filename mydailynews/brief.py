from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from .ai.base import AIClient
from .ai.prompts import BRIEF_SYSTEM, BRIEF_USER
from .ai.schemas import FINAL_BRIEF_JSON_SCHEMA
from .debug import DebugLogger
from .models import PriorReport, SelectedArticle, TopicConfig, UserMemory
from .utils import datetime_to_iso

FINAL_PROMPT_BUDGET_SAFETY_RATIO = 0.95


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
        )
        self.debug.log("brief.ai", "synthesizing", articles=len(used_articles), prompt_chars=len(prompt))
        label = "final brief generation"
        if brief_name:
            label = f"{label} ({brief_name})"
        result = self.client.complete_json(
            BRIEF_SYSTEM,
            prompt,
            label=label,
            max_new_tokens=self.max_new_tokens,
            input_token_limit=self.input_token_limit,
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
    ) -> tuple[str, List[SelectedArticle]]:
        target_input_tokens = max(1024, int(self.input_token_limit or self.client.max_input_tokens))
        prompt_budget_tokens = max(512, int(target_input_tokens * FINAL_PROMPT_BUDGET_SAFETY_RATIO))
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
    ) -> str:
        payload = [self._article_payload(article, excerpt_chars) for article in articles]
        return BRIEF_USER.format(
            memory=memory.to_prompt(),
            date=date,
            brief_goal=brief_goal,
            topics=_compact_json(self._topics_payload(topics)),
            prior_reports=_compact_json(self._prior_reports_payload(prior_reports)),
            evidence_packet=_compact_json(evidence_packet),
            delta_packet=_compact_json(delta_packet),
            articles=_compact_json(payload),
        )

    def _estimate_final_input_tokens(self, prompt: str) -> int:
        return self.client.estimate_tokens(f"System:\n{BRIEF_SYSTEM}\n\nUser:\n{prompt}\n\nAssistant:\n")

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
            "event_cluster": self._event_cluster_payload(article.candidate.metadata),
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
            }
            for report in prior_reports
        ]

    @staticmethod
    def _major_headlines_payload(articles: List[SelectedArticle]) -> List[dict]:
        return [
            {
                "headline": article.candidate.title,
                "source": article.candidate.source,
                "url": article.candidate.url,
                "score": article.decision.score,
                "topic": article.decision.topic or article.candidate.metadata.get("topic_name", ""),
                "event_cluster": BriefGenerator._event_cluster_payload(article.candidate.metadata),
            }
            for article in articles
        ]

    @staticmethod
    def _selected_articles_payload(articles: List[SelectedArticle]) -> List[dict]:
        return [
            {
                "id": article.candidate.id,
                "headline": article.candidate.title,
                "source": article.candidate.source,
                "url": article.candidate.url,
                "score": article.decision.score,
                "topic": article.decision.topic or article.candidate.metadata.get("topic_name", ""),
                "snippet": (article.candidate.snippet or "")[:180],
                "event_cluster": BriefGenerator._event_cluster_payload(article.candidate.metadata),
            }
            for article in articles
        ]

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

    @staticmethod
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
            signature = _compact_json({"evidence": evidence, "delta": delta})
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
            "evidence_gaps": gaps,
        }

    @staticmethod
    def _trim_text(value: Any, *, max_chars: int) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        return text[:max_chars]

    @classmethod
    def _to_string_list(cls, value: Any, *, max_items: int, max_chars: int) -> List[str]:
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = [value]
        else:
            raw_items = []
        cleaned: List[str] = []
        for raw in raw_items:
            text = cls._trim_text(raw, max_chars=max_chars)
            if text:
                cleaned.append(text)
        return cls._normalized_string_list(cleaned, limit=max_items)

    @classmethod
    def _normalize_narrative_changes(cls, value: Any) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        rows: List[Dict[str, str]] = []
        for raw in value[:8]:
            if not isinstance(raw, dict):
                continue
            narrative = cls._trim_text(raw.get("narrative", ""), max_chars=90)
            status = cls._trim_text(raw.get("status", ""), max_chars=24)
            summary = cls._trim_text(raw.get("summary", ""), max_chars=220)
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
        for raw in value[:10]:
            if not isinstance(raw, dict):
                continue
            topic = cls._trim_text(raw.get("topic", ""), max_chars=120) or "Topic"
            why_it_matters = cls._trim_text(raw.get("why_it_matters", ""), max_chars=240)
            what_changed = cls._trim_text(raw.get("what_changed", ""), max_chars=240)
            narrative_summary = cls._trim_text(
                raw.get("narrative_summary", "") or raw.get("summary", ""),
                max_chars=280,
            )
            if not why_it_matters and narrative_summary:
                why_it_matters = narrative_summary

            narrative_changes = cls._normalize_narrative_changes(raw.get("narrative_changes", []))
            if not what_changed and narrative_changes:
                what_changed = cls._trim_text(narrative_changes[0].get("summary", ""), max_chars=240)
            if not what_changed and narrative_summary:
                what_changed = narrative_summary

            who_is_affected = cls._to_string_list(raw.get("who_is_affected", []), max_items=5, max_chars=120)
            what_to_watch = cls._to_string_list(
                raw.get("what_to_watch", raw.get("watch_signals", [])),
                max_items=6,
                max_chars=150,
            )
            if not narrative_summary:
                parts = [why_it_matters, what_changed]
                narrative_summary = cls._trim_text(". ".join([part for part in parts if part]), max_chars=280)

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
        for raw in value[:10]:
            if not isinstance(raw, dict):
                continue
            heading = cls._trim_text(raw.get("heading", ""), max_chars=120)
            summary = cls._trim_text(raw.get("summary", ""), max_chars=260)
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
        knowns = self._normalized_string_list(result.get("knowns", []), limit=8)
        unknowns = self._normalized_string_list(result.get("unknowns", []), limit=8)
        watch_signals = self._normalized_string_list(result.get("watch_signals", []), limit=10)

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
    def _normalized_string_list(value: Any, *, limit: int) -> List[str]:
        items: List[str] = []
        if isinstance(value, list):
            for raw in value:
                text = str(raw).strip()
                if not text:
                    continue
                items.append(text[:220])
        seen: set[str] = set()
        deduped: List[str] = []
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
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
            for point in cluster.get("consensus_points", [])[:2]:
                text = str(point).strip()
                if text:
                    candidates.append(text)
        for item in delta_packet.get("unchanged_but_important", [])[:4]:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary", "")).strip()
            label = str(item.get("item", "")).strip()
            if summary:
                candidates.append(summary)
            elif label:
                candidates.append(label)
        for report in result.get("topic_reports", [])[:3]:
            if not isinstance(report, dict):
                continue
            for key in ("why_it_matters", "what_changed", "narrative_summary"):
                text = str(report.get(key, "")).strip()
                if text:
                    candidates.append(text)
        if not candidates:
            for article in articles[:4]:
                headline = str(article.candidate.title).strip()
                source = str(article.candidate.source).strip()
                if headline:
                    label = headline
                    if source:
                        label = f"{headline} ({source})"
                    candidates.append(label)
        return self._normalized_string_list(candidates, limit=8)

    def _fallback_unknowns(
        self,
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
    ) -> List[str]:
        candidates: List[str] = []
        for cluster in evidence_packet.get("story_clusters", []):
            if not isinstance(cluster, dict):
                continue
            for point in cluster.get("known_unknowns", [])[:2]:
                text = str(point).strip()
                if text:
                    candidates.append(text)
            for point in cluster.get("contested_points", [])[:1]:
                text = str(point).strip()
                if text:
                    candidates.append(text)
        for item in delta_packet.get("evidence_gaps", [])[:4]:
            if not isinstance(item, dict):
                continue
            gap = str(item.get("gap", "")).strip()
            why = str(item.get("why_it_matters", "")).strip()
            if gap and why:
                candidates.append(f"{gap} ({why})")
            elif gap:
                candidates.append(gap)
        return self._normalized_string_list(candidates, limit=8)

    def _fallback_watch_signals(
        self,
        result: Dict[str, Any],
        evidence_packet: Dict[str, Any],
        delta_packet: Dict[str, Any],
    ) -> List[str]:
        candidates: List[str] = []
        for report in result.get("topic_reports", [])[:6]:
            if not isinstance(report, dict):
                continue
            for item in report.get("what_to_watch", [])[:2]:
                text = str(item).strip()
                if text:
                    candidates.append(text)
        for item in evidence_packet.get("global_watch_signals", [])[:6]:
            text = str(item).strip()
            if text:
                candidates.append(text)
        for cluster in evidence_packet.get("story_clusters", [])[:4]:
            if not isinstance(cluster, dict):
                continue
            for item in cluster.get("watch_signals", [])[:2]:
                text = str(item).strip()
                if text:
                    candidates.append(text)
        for key in ("new", "escalated"):
            for item in delta_packet.get(key, [])[:3]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("item", "")).strip()
                if label:
                    candidates.append(label)
        return self._normalized_string_list(candidates, limit=10)


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
