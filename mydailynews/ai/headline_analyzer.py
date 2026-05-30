from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .base import AIClient, AIJsonError, write_ai_json_artifact
from .prompts import HEADLINE_ANALYSIS_SYSTEM, HEADLINE_ANALYSIS_USER
from .schemas import HEADLINE_ANALYSIS_JSON_SCHEMA
from ..cache import JSONCache
from ..debug import DebugLogger
from ..models import HeadlineDecision, NewsCandidate, TopicConfig, UserMemory
from ..utils import datetime_to_iso


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class HeadlineAnalyzer:
    def __init__(
        self,
        client: AIClient,
        batch_size: int,
        debug: DebugLogger | None = None,
        cache: JSONCache | None = None,
        cache_ttl_seconds: int = 0,
    ) -> None:
        self.client = client
        self.batch_size = max(1, batch_size)
        self.debug = debug or DebugLogger(False)
        self.cache = cache
        self.cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self.warnings: List[str] = []

    def analyze(
        self,
        candidates: List[NewsCandidate],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        brief_name: str = "",
    ) -> Dict[str, HeadlineDecision]:
        self.warnings = []
        if not candidates:
            return {}

        candidate_payloads = [(item, self._candidate_payload(item)) for item in candidates]
        batches = self._build_batches(candidate_payloads, memory, topics, brief_goal)
        self.debug.log("headline.ai", "starting batched scoring", candidates=len(candidates), batches=len(batches), batch_size=self.batch_size)
        self.debug.set_metric("headline.scoring.batch_size", self.batch_size)

        decisions: Dict[str, HeadlineDecision] = {}
        for batch_index, batch in enumerate(batches, start=1):
            batch_candidates = [item for item, _ in batch]
            batch_payload = [payload for _, payload in batch]
            decisions.update(
                self._analyze_batch(
                    batch_candidates,
                    batch_payload,
                    memory,
                    topics,
                    brief_goal,
                    brief_name,
                    batch_index,
                    len(batches),
                )
            )
        return decisions

    def _analyze_batch(
        self,
        candidates: List[NewsCandidate],
        payload: List[Dict[str, Any]],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        brief_name: str,
        batch_index: int,
        total_batches: int,
    ) -> Dict[str, HeadlineDecision]:
        user_prompt = self._build_user_prompt(memory, topics, brief_goal, payload)
        target_input_tokens = max(1024, int(self.client.max_input_tokens * 0.84))
        # Give multi-headline batches much more room to finish structured output.
        # The scorer is cheap relative to the writer, so a larger generation ceiling
        # is worth it to avoid mid-JSON truncation and omitted decisions.
        dynamic_max_new_tokens = min(self.client.max_new_tokens, max(320, 128 * len(candidates)))
        self.debug.log(
            "headline.ai.batch",
            "scoring",
            batch=f"{batch_index}/{total_batches}",
            items=len(candidates),
            prompt_chars=len(user_prompt),
            max_input_tokens=target_input_tokens,
            max_new_tokens=dynamic_max_new_tokens,
        )
        brief_suffix = f" ({brief_name})" if brief_name else ""
        label = f"headline scoring batch {batch_index}/{total_batches}{brief_suffix}"
        cache_key = self._batch_cache_key(payload, memory, topics, brief_goal)
        if self.cache:
            cached = self.cache.get(cache_key, max_age_seconds=self.cache_ttl_seconds)
            if cached is not None:
                self.debug.log(
                    "headline.ai.batch",
                    "cache_hit",
                    batch=f"{batch_index}/{total_batches}",
                    items=len(candidates),
                )
                self.debug.increment("headline.ai.cache_hits", 1)
                return self._parse_batch_result(cached, candidates, topics, label, batch_index, total_batches)

        try:
            result = self.client.complete_json(
                HEADLINE_ANALYSIS_SYSTEM,
                user_prompt,
                label=label,
                max_new_tokens=dynamic_max_new_tokens,
                input_token_limit=target_input_tokens,
                json_schema=HEADLINE_ANALYSIS_JSON_SCHEMA,
            )
        except AIJsonError as exc:
            warning = f"{label}: skipped {len(candidates)} headline(s) after invalid JSON: {exc}"
            self.debug.increment("headline.ai.invalid_json_batches", 1)
            self.debug.increment("headline.ai.invalid_json_items", len(candidates))
            if self.debug.enabled and len(candidates) > 1:
                replay_path = self._debug_single_item_replay(
                    candidates,
                    payload,
                    memory,
                    topics,
                    brief_goal,
                    brief_name,
                    batch_index,
                    total_batches,
                )
                if replay_path:
                    warning += f"; single-item replay saved to {replay_path}"
            self.warnings.append(warning)
            self.debug.log("headline.ai.batch", "skipped_invalid_json", batch=f"{batch_index}/{total_batches}", items=len(candidates))
            return {}

        if self.cache:
            self.cache.put(cache_key, result)
        return self._parse_batch_result(result, candidates, topics, label, batch_index, total_batches)

    def _parse_batch_result(
        self,
        result: Dict[str, Any],
        candidates: List[NewsCandidate],
        topics: List[TopicConfig],
        label: str,
        batch_index: int,
        total_batches: int,
    ) -> Dict[str, HeadlineDecision]:
        candidate_by_id = {item.id: item for item in candidates}
        decisions: Dict[str, HeadlineDecision] = {}
        for raw in result.get("decisions", []):
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("id", "")).strip()
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                continue
            try:
                score = float(raw.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            decisions[candidate_id] = HeadlineDecision(
                candidate_id=candidate_id,
                score=max(0.0, min(10.0, score)),
                topic=self.best_topic_for_candidate(candidate, topics),
            )

        missing = [item for item in candidates if item.id not in decisions]
        if missing:
            missing_ids = ", ".join(item.id for item in missing[:5])
            warning = (
                f"{label}: skipped {len(missing)} headline(s) because the model omitted decisions; "
                f"first missing id(s): {missing_ids}"
            )
            self.warnings.append(warning)
            self.debug.increment("headline.ai.missing_decisions", len(missing))
            self.debug.log("headline.ai.batch", "incomplete", batch=f"{batch_index}/{total_batches}", missing=len(missing))
        self.debug.log(
            "headline.ai.batch",
            "complete",
            batch=f"{batch_index}/{total_batches}",
            ai_decisions=len(decisions),
        )
        return decisions

    def _batch_cache_key(
        self,
        payload: List[Dict[str, Any]],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
    ) -> str:
        fingerprint = {
            "v": 4,
            "backend": self.client.config.backend,
            "model": self.client.config.effective_model_label,
            "response_format": self.client.config.response_format,
            "brief_goal": brief_goal,
            "memory": memory.to_prompt(),
            "topics": self._topics_payload(topics),
            "items": payload,
        }
        return JSONCache.make_key(_compact_json(fingerprint))

    def _debug_single_item_replay(
        self,
        candidates: List[NewsCandidate],
        payload: List[Dict[str, Any]],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        brief_name: str,
        batch_index: int,
        total_batches: int,
    ) -> str:
        results: List[Dict[str, Any]] = []
        target_input_tokens = max(1024, int(self.client.max_input_tokens * 0.84))
        for item, item_payload in zip(candidates, payload):
            label = f"headline scoring single replay {batch_index}/{total_batches} ({brief_name or 'shared'}) [{item.id}]"
            user_prompt = self._build_user_prompt(memory, topics, brief_goal, [item_payload])
            dynamic_max_new_tokens = min(self.client.max_new_tokens, 192)
            try:
                result = self.client.complete_json(
                    HEADLINE_ANALYSIS_SYSTEM,
                    user_prompt,
                    label=label,
                    max_new_tokens=dynamic_max_new_tokens,
                    input_token_limit=target_input_tokens,
                    json_schema=HEADLINE_ANALYSIS_JSON_SCHEMA,
                )
                results.append(
                    {
                        "candidate_id": item.id,
                        "status": "ok",
                        "result": result,
                    }
                )
            except AIJsonError as exc:
                results.append(
                    {
                        "candidate_id": item.id,
                        "status": "invalid_json",
                        "error": str(exc),
                        "artifact_path": exc.artifact_path,
                        "raw_response_path": exc.raw_response_path,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "candidate_id": item.id,
                        "status": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        try:
            return write_ai_json_artifact(
                "headline_single_replay",
                f"batch_{batch_index}_{brief_name or 'shared'}",
                {
                    "brief_name": brief_name or "shared",
                    "batch": f"{batch_index}/{total_batches}",
                    "candidate_ids": [item.id for item in candidates],
                    "results": results,
                },
            )
        except Exception:
            return ""

    def _build_user_prompt(
        self,
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
        payload: List[Dict[str, Any]],
    ) -> str:
        return HEADLINE_ANALYSIS_USER.format(
            memory=memory.to_prompt(),
            brief_goal=brief_goal,
            topics=_compact_json(self._topics_payload(topics)),
            items=_compact_json(payload),
        )

    def _build_batches(
        self,
        payloads: List[tuple[NewsCandidate, Dict[str, Any]]],
        memory: UserMemory,
        topics: List[TopicConfig],
        brief_goal: str,
    ) -> List[List[tuple[NewsCandidate, Dict[str, Any]]]]:
        if not payloads:
            return []

        base_prompt = self._build_user_prompt(memory, topics, brief_goal, [])
        base_tokens = self.client.estimate_tokens(base_prompt)
        target_input_tokens = max(1024, int(self.client.max_input_tokens * 0.84))

        batches: List[List[tuple[NewsCandidate, Dict[str, Any]]]] = []
        current: List[tuple[NewsCandidate, Dict[str, Any]]] = []
        current_tokens = base_tokens
        for item, payload in payloads:
            payload_tokens = max(1, self.client.estimate_tokens(_compact_json(payload)))
            if current and (len(current) >= self.batch_size or current_tokens + payload_tokens > target_input_tokens):
                batches.append(current)
                current = [(item, payload)]
                current_tokens = base_tokens + payload_tokens
                continue
            current.append((item, payload))
            current_tokens += payload_tokens
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _candidate_payload(item: NewsCandidate) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": item.id,
            "title": item.title,
            "source": item.source,
            "published_at": datetime_to_iso(item.published_at),
            "snippet": (item.snippet or "")[:180],
        }
        topic_hint = str(item.metadata.get("topic_name", "")).strip()
        if topic_hint:
            payload["topic_hint"] = topic_hint
        return payload

    @staticmethod
    def _topics_payload(topics: List[TopicConfig]) -> List[dict]:
        return [
            {
                "name": topic.name,
                "description": (topic.description or "")[:160],
                "queries": [query[:80] for query in (topic.queries or [topic.name])[:3]],
            }
            for topic in topics
            if topic.enabled
        ]

    @classmethod
    def best_topic_for_candidate(cls, item: NewsCandidate, topics: List[TopicConfig]) -> str:
        topic_name = str(item.metadata.get("topic_name", "")).strip()
        if topic_name:
            return topic_name

        best_topic = ""
        best_score = 0.0
        for topic in topics:
            if not topic.enabled:
                continue
            score = cls._topic_match_score(item, topic)
            if score > best_score:
                best_score = score
                best_topic = topic.name
        return best_topic

    @staticmethod
    def _topic_match_score(item: NewsCandidate, topic: TopicConfig) -> float:
        text = f"{item.title or ''} {item.snippet or ''}".lower()
        text_tokens = set(re.findall(r"[a-z0-9]{3,}", text))
        if not text_tokens:
            return 0.0
        topic_tokens = set(re.findall(r"[a-z0-9]{3,}", topic.name.lower()))
        topic_tokens.update(re.findall(r"[a-z0-9]{3,}", topic.description.lower()))
        for query in topic.queries or []:
            topic_tokens.update(re.findall(r"[a-z0-9]{3,}", query.lower()))
        if not topic_tokens:
            return 0.0
        return len(text_tokens.intersection(topic_tokens)) / max(3, len(topic_tokens))
