from __future__ import annotations

import unittest

from mydailynews.app.models import UserMemory
from mydailynews.evaluation.schema import (
    EvalArc,
    EvalCorpus,
    EvalDay,
    EvalDocument,
    EvalExpectation,
)
from tools.run_semantic_thread_evaluation import (
    chronological_batches,
    chronological_documents,
    pairwise_clustering,
    production_candidate,
)


class SemanticThreadEvaluatorTests(unittest.TestCase):
    def test_production_candidate_strips_evaluation_only_metadata(self):
        document = EvalDocument(
            id="doc-1",
            source="Example Wire",
            title="A source-backed headline",
            url="https://example.test/doc-1",
            snippet="A source-backed snippet.",
            body="A source-backed article body.",
            published_at="2026-08-30T08:00:00Z",
            category="blind-genre-label",
            tags=["hard-negative", "transition-archetype"],
        )

        candidate = production_candidate(document)

        self.assertEqual(candidate.category, "")
        self.assertEqual(candidate.tags, [])
        self.assertEqual(candidate.metadata, {})
        self.assertEqual(candidate.title, document.title)
        self.assertEqual(candidate.snippet, document.snippet)
        self.assertEqual(document.tags, ["hard-negative", "transition-archetype"])

    def test_chronological_documents_interleaves_event_families(self):
        late = self._arc("late-family", "2026-08-30", "late-doc")
        early = self._arc("early-family", "2026-08-29", "early-doc")
        corpus = EvalCorpus("change_monitor.eval.v1", "test", "test", [late, early])

        timeline = chronological_documents(corpus)

        self.assertEqual([entry[0].id for entry in timeline], ["early-family", "late-family"])

    def test_chronological_batches_groups_same_day_across_families(self):
        first = self._arc("first-family", "2026-08-29", "first-doc")
        second = self._arc("second-family", "2026-08-29", "second-doc")
        later = self._arc("later-family", "2026-08-30", "later-doc")
        corpus = EvalCorpus(
            "change_monitor.eval.v1",
            "test",
            "test",
            [later, first, second],
        )

        batches = chronological_batches(corpus)

        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual({entry[0].id for entry in batches[0]}, {"first-family", "second-family"})

    def test_pairwise_clustering_counts_cross_family_false_merge(self):
        rows = [
            {"canonical_cluster_id": "a:one", "predicted_cluster_id": "merged"},
            {"canonical_cluster_id": "a:one", "predicted_cluster_id": "merged"},
            {"canonical_cluster_id": "b:two", "predicted_cluster_id": "merged"},
        ]

        result = pairwise_clustering(rows)

        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_merge"], 2)
        self.assertEqual(result["false_split"], 0)

    @staticmethod
    def _arc(arc_id: str, date: str, document_id: str) -> EvalArc:
        document = EvalDocument(
            id=document_id,
            source="Example Wire",
            title=f"Headline for {document_id}",
            url=f"https://example.test/{document_id}",
            snippet="Snippet.",
            body="Body.",
            published_at=f"{date}T08:00:00Z",
        )
        expected = EvalExpectation(
            document_id=document_id,
            canonical_story_id=f"story-{document_id}",
            relationship="new_story",
            delta_type="new",
            material=True,
            display="full_report",
            profile_relevance="eligible",
            should_select=True,
        )
        return EvalArc(
            id=arc_id,
            split="holdout",
            tags=["test-only"],
            profile=UserMemory(),
            fact_catalog={},
            days=[EvalDay(date=date, documents=[document], expectations=[expected])],
        )


if __name__ == "__main__":
    unittest.main()
