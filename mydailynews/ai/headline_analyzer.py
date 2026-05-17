from __future__ import annotations

import json
from typing import Dict, Iterable, List

from .client import LocalAIClient
from .prompts import HEADLINE_ANALYSIS_SYSTEM, HEADLINE_ANALYSIS_USER
from ..models import HeadlineDecision, NewsCandidate, UserMemory
from ..utils import datetime_to_iso


class HeadlineAnalyzer:
    def __init__(self, client: LocalAIClient) -> None:
        self.client = client

    def analyze(self, candidates: List[NewsCandidate], memory: UserMemory) -> Dict[str, HeadlineDecision]:
        if not candidates:
            return {}

        payload = [
            {
                "id": item.id,
                "title": item.title,
                "source": item.source,
                "category": item.category,
                "url": item.url,
                "published_at": datetime_to_iso(item.published_at),
                "snippet": item.snippet[:500],
                "tags": item.tags[:8],
            }
            for item in candidates
        ]
        result = self.client.complete_json(
            HEADLINE_ANALYSIS_SYSTEM,
            HEADLINE_ANALYSIS_USER.format(
                memory=memory.to_prompt(),
                items=json.dumps(payload, ensure_ascii=False, indent=2),
            ),
        )
        if not result:
            return self._fallback(candidates, memory)

        decisions: Dict[str, HeadlineDecision] = {}
        for raw in result.get("decisions", []):
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("id", ""))
            if not candidate_id:
                continue
            try:
                score = float(raw.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            decisions[candidate_id] = HeadlineDecision(
                candidate_id=candidate_id,
                score=max(0.0, min(10.0, score)),
                reason=str(raw.get("reason", "")),
                summary=str(raw.get("summary", "")),
                tags=[str(tag) for tag in raw.get("tags", []) if tag],
                duplicate_of=raw.get("duplicate_of"),
            )

        missing = [item for item in candidates if item.id not in decisions]
        decisions.update(self._fallback(missing, memory))
        return decisions

    @staticmethod
    def _fallback(candidates: Iterable[NewsCandidate], memory: UserMemory) -> Dict[str, HeadlineDecision]:
        preferred_topics = " ".join(memory.preferred_topics).lower()
        preferred_sources = {source.lower() for source in memory.preferred_sources}
        decisions: Dict[str, HeadlineDecision] = {}
        for item in candidates:
            haystack = f"{item.title} {item.snippet} {' '.join(item.tags)}".lower()
            score = 5.0
            if item.source.lower() in preferred_sources:
                score += 1.0
            for topic in preferred_topics.split():
                if topic and topic in haystack:
                    score += 0.4
            decisions[item.id] = HeadlineDecision(
                candidate_id=item.id,
                score=max(0.0, min(10.0, score)),
                reason="Fallback heuristic used because AI scoring did not return a decision.",
                summary=item.snippet[:220] or item.title,
                tags=item.tags[:5],
            )
        return decisions
