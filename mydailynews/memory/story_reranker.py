from __future__ import annotations

"""Bounded second-stage validation for retrieved story candidates.

The retriever establishes a small, source-backed candidate set.  A reranker
may only reorder or reject that set; it must never search the full ledger or
create a story key.  This keeps identity decisions inspectable and makes a
model failure degrade to the existing heuristic behaviour.
"""

from dataclasses import replace
from typing import Protocol, Sequence

from mydailynews.app.models import NewsCandidate
from mydailynews.memory.story_retrieval import StoryCandidateMatch


DEFAULT_RERANK_ACCEPTANCE_THRESHOLD = 0.5


class StoryCandidateReranker(Protocol):
    """Score already-retrieved candidates as same-story probabilities."""

    def score(
        self,
        candidate: NewsCandidate,
        matches: Sequence[StoryCandidateMatch],
        *,
        source_text: str = "",
    ) -> Sequence[float]: ...


def rerank_story_candidates(
    candidate: NewsCandidate,
    matches: Sequence[StoryCandidateMatch],
    reranker: StoryCandidateReranker | None,
    *,
    source_text: str = "",
    acceptance_threshold: float = DEFAULT_RERANK_ACCEPTANCE_THRESHOLD,
    reject_below_threshold: bool = False,
) -> list[StoryCandidateMatch]:
    """Apply an optional pairwise validator without widening candidate scope.

    On an unavailable or malformed model score, retain the heuristic match.
    Production uses score/rank-only mode by default: an uncertain local model
    cannot silently discard a retrieved continuation.  Hard rejection is an
    explicit experiment because the corpus shows this small reranker has false
    negatives. Callers can expose every recorded score for auditability.
    """

    source_matches = list(matches)
    if not source_matches or reranker is None:
        return source_matches
    try:
        scores = list(reranker.score(candidate, source_matches, source_text=source_text))
    except Exception:
        return source_matches
    if len(scores) != len(source_matches):
        return source_matches
    threshold = max(0.0, min(1.0, float(acceptance_threshold)))
    scored: list[StoryCandidateMatch] = []
    for match, value in zip(source_matches, scores):
        try:
            score = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return source_matches
        if not reject_below_threshold or score >= threshold:
            scored.append(replace(match, reranker_score=score))
    return sorted(
        scored,
        key=lambda match: (float(match.reranker_score or 0.0), match.score, match.record.last_seen),
        reverse=True,
    )
