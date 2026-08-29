from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, SelectedArticle
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.evaluation.schema import EvalCorpus
from mydailynews.memory.story_store import DEFAULT_CANDIDATE_THRESHOLD, StoryStore


@dataclass
class StoryRetrievalDiagnostics:
    """Gold-assisted diagnostic that isolates candidate retrieval quality.

    Gold canonical identity is used only after each day to seed the historical
    store. It is never supplied to retrieval for the current day.
    """

    historical_continuations: int = 0
    hits_at_1: int = 0
    hits_at_3: int = 0
    reciprocal_rank_sum: float = 0.0
    same_day_only_continuations: int = 0
    unseen_stories: int = 0
    unseen_without_candidates: int = 0
    related_theme_items: int = 0
    related_theme_with_candidates: int = 0
    candidates_returned: int = 0
    documents: int = 0
    continuation_misses: List[str] = field(default_factory=list)
    continuation_miss_details: List[Dict[str, Any]] = field(default_factory=list)
    unseen_with_candidates: List[str] = field(default_factory=list)
    unseen_candidate_details: List[Dict[str, Any]] = field(default_factory=list)

    def payload(self) -> Dict[str, Any]:
        continuation_denominator = max(1, self.historical_continuations)
        unseen_denominator = max(1, self.unseen_stories)
        document_denominator = max(1, self.documents)
        return {
            "uses_private_gold_for_historical_writeback": True,
            "documents": self.documents,
            "historical_continuations": self.historical_continuations,
            "recall_at_1": round(self.hits_at_1 / continuation_denominator, 4),
            "recall_at_3": round(self.hits_at_3 / continuation_denominator, 4),
            "mean_reciprocal_rank": round(self.reciprocal_rank_sum / continuation_denominator, 4),
            "same_day_only_continuations_excluded": self.same_day_only_continuations,
            "unseen_stories": self.unseen_stories,
            "new_story_without_candidate_rate": round(
                self.unseen_without_candidates / unseen_denominator,
                4,
            ),
            "related_theme_items": self.related_theme_items,
            "related_theme_with_candidate_rate": round(
                self.related_theme_with_candidates / max(1, self.related_theme_items),
                4,
            ),
            "mean_candidates_per_document": round(
                self.candidates_returned / document_denominator,
                4,
            ),
            "continuation_misses": list(self.continuation_misses),
            "continuation_miss_details": list(self.continuation_miss_details),
            "unseen_with_candidates": list(self.unseen_with_candidates),
            "unseen_candidate_details": list(self.unseen_candidate_details),
        }


def evaluate_story_store_retrieval(
    corpus: EvalCorpus,
    *,
    threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
    limit: int = 3,
) -> StoryRetrievalDiagnostics:
    """Measure the source-backed retriever against prior-day story history.

    This is an intervention diagnostic, not an end-to-end production score:
    private canonical IDs are used to make historical store writeback perfect.
    Current documents are retrieved from title/snippet/body only.
    """

    result = StoryRetrievalDiagnostics()
    with TemporaryDirectory(prefix="mydailynews-story-store-eval-") as raw_root:
        root = Path(raw_root)
        for arc in corpus.arcs:
            store = StoryStore(root / f"{arc.id}.json")
            seen_before_day: set[str] = set()
            for day in arc.days:
                expected_by_id = {item.document_id: item for item in day.expectations}
                day_articles: List[SelectedArticle] = []
                day_story_ids: set[str] = set()
                for document in day.documents:
                    expected = expected_by_id[document.id]
                    candidate = document.to_candidate()
                    canonical_key = f"oracle:{arc.id}:{expected.canonical_story_id}"
                    matches = store.candidate_stories(
                        candidate,
                        source_text=document.body,
                        limit=limit,
                        min_score=threshold,
                    )
                    keys = [match.record.story_key for match in matches]
                    result.documents += 1
                    result.candidates_returned += len(matches)

                    if expected.relationship == "same_story" and canonical_key in seen_before_day:
                        result.historical_continuations += 1
                        if canonical_key in keys:
                            rank = keys.index(canonical_key) + 1
                            result.reciprocal_rank_sum += 1.0 / rank
                            result.hits_at_3 += 1
                            if rank == 1:
                                result.hits_at_1 += 1
                        else:
                            result.continuation_misses.append(document.id)
                            result.continuation_miss_details.append(
                                {
                                    "document_id": document.id,
                                    "expected_story_key": canonical_key,
                                    "permissive_top3": [
                                        match.metadata()
                                        for match in store.candidate_stories(
                                            candidate,
                                            source_text=document.body,
                                            limit=3,
                                            min_score=0.0,
                                        )
                                    ],
                                }
                            )
                    elif expected.relationship == "same_story":
                        # A duplicate first appearing in the same batch is a
                        # current-story grouping problem, not historical recall.
                        result.same_day_only_continuations += 1
                    elif expected.relationship == "related_theme":
                        result.related_theme_items += 1
                        if matches:
                            result.related_theme_with_candidates += 1
                    elif canonical_key not in seen_before_day:
                        result.unseen_stories += 1
                        if not matches:
                            result.unseen_without_candidates += 1
                        else:
                            result.unseen_with_candidates.append(document.id)
                            result.unseen_candidate_details.append(
                                {
                                    "document_id": document.id,
                                    "candidates": [match.metadata() for match in matches],
                                }
                            )

                    set_memory_annotation(
                        candidate,
                        MemoryAnnotation(
                            story_key=canonical_key,
                            story_family_key=f"oracle:{arc.id}",
                            story_title=candidate.title,
                            match_confidence=1.0,
                        ),
                    )
                    day_articles.append(
                        SelectedArticle(
                            candidate=candidate,
                            decision=HeadlineDecision(candidate.id, score=8.0),
                            article_text=document.body,
                            extraction_status="fixture",
                        )
                    )
                    day_story_ids.add(canonical_key)

                store.update_selected(selected=day_articles, date=day.date)
                seen_before_day.update(day_story_ids)
    return result
