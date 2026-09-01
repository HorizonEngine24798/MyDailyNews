from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest
from urllib.parse import urlparse

from mydailynews.evaluation.schema import load_corpus


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "evals" / "cases" / "change_monitoring.real_blind.v1.json"
PROVENANCE_PATH = (
    REPO_ROOT / "evals" / "cases" / "change_monitoring.real_blind.v1.provenance.json"
)
METHODOLOGY_PATH = REPO_ROOT / "docs" / "real-blind-corpus-methodology-2026-08-30.md"
SEAL_PATH = REPO_ROOT / "evals" / "cases" / "change_monitoring.real_blind.v1.seal.json"


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RealBlindCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(CORPUS_PATH)
        cls.raw = _read_json(CORPUS_PATH)
        cls.provenance = _read_json(PROVENANCE_PATH)

    def test_strict_schema_and_frozen_scope(self) -> None:
        self.assertEqual(self.corpus.schema_version, "change_monitor.eval.v1")
        self.assertEqual(len(self.corpus.arcs), 18)

        documents = [
            document
            for arc in self.corpus.arcs
            for day in arc.days
            for document in day.documents
        ]
        expectations = [
            expected
            for arc in self.corpus.arcs
            for day in arc.days
            for expected in day.expectations
        ]
        self.assertEqual(len(documents), 120)
        self.assertEqual(len(expectations), 120)
        self.assertEqual(sum(len(arc.days) for arc in self.corpus.arcs), 107)
        self.assertTrue(all(arc.split == "holdout" for arc in self.corpus.arcs))
        self.assertTrue(all(document.category == "real" for document in documents))

        document_ids = [document.id for document in documents]
        expectation_ids = [expected.document_id for expected in expectations]
        self.assertEqual(len(document_ids), len(set(document_ids)))
        self.assertEqual(len(expectation_ids), len(set(expectation_ids)))
        self.assertEqual(set(document_ids), set(expectation_ids))

    def test_public_inputs_hide_gold_and_trap_metadata(self) -> None:
        for arc in self.corpus.arcs:
            public = arc.public_input()
            self.assertFalse(hasattr(public, "tags"))
            self.assertFalse(hasattr(public, "fact_catalog"))
            self.assertFalse(hasattr(public, "split"))
            for day in public.days:
                for document in day.documents:
                    self.assertFalse(hasattr(document, "canonical_story_id"))
                    candidate = document.to_candidate()
                    self.assertNotIn("expected", candidate.metadata)
                    self.assertNotIn("canonical_story_id", candidate.metadata)

    def test_sources_are_public_provenance_not_copied_articles(self) -> None:
        documents = [
            document
            for arc in self.corpus.arcs
            for day in arc.days
            for document in day.documents
        ]
        blocked_hosts = {
            "theguardian.com",
            "www.theguardian.com",
            "bbc.com",
            "www.bbc.com",
            "bbc.co.uk",
            "www.bbc.co.uk",
            "reuters.com",
            "www.reuters.com",
            "apnews.com",
            "www.apnews.com",
        }
        for document in documents:
            parsed = urlparse(document.url)
            self.assertIn(parsed.scheme, {"http", "https"}, document.id)
            self.assertTrue(parsed.netloc, document.id)
            self.assertNotIn(parsed.hostname, blocked_hosts, document.id)
            self.assertNotIn(parsed.hostname, {"localhost", "fixture.test", "example.com"})
            self.assertIn("official_source", document.tags, document.id)
            self.assertIn("paraphrased", document.tags, document.id)
            self.assertLessEqual(len(document.snippet.split()), 25, document.id)
            self.assertLessEqual(len(document.body.split()), 100, document.id)

    def test_every_family_contains_continuation_and_hard_negative(self) -> None:
        for arc in self.corpus.arcs:
            relationships = {
                expected.relationship
                for day in arc.days
                for expected in day.expectations
            }
            self.assertIn("same_story", relationships, arc.id)
            self.assertIn("related_theme", relationships, arc.id)

        story_families: dict[str, set[str]] = defaultdict(set)
        for arc in self.corpus.arcs:
            for day in arc.days:
                for expected in day.expectations:
                    story_families[expected.canonical_story_id].add(arc.id)
        leaked = {
            story_id: sorted(families)
            for story_id, families in story_families.items()
            if len(families) != 1
        }
        self.assertEqual(leaked, {}, "canonical event threads may not cross families")

    def test_requested_genres_have_redundant_family_coverage(self) -> None:
        required = {
            "genre_sports": 2,
            "genre_celebrity_culture": 2,
            "genre_economy_business": 2,
            "genre_war_conflict": 2,
            "genre_climate": 2,
            "genre_science": 2,
            "genre_ai_technology": 2,
            "genre_space": 2,
            "genre_technology": 2,
        }
        counts = Counter(tag for arc in self.corpus.arcs for tag in set(arc.tags))
        for tag, minimum in required.items():
            self.assertGreaterEqual(counts[tag], minimum, tag)

    def test_label_and_adversarial_slice_diversity(self) -> None:
        expectations = [
            expected
            for arc in self.corpus.arcs
            for day in arc.days
            for expected in day.expectations
        ]
        self.assertTrue(
            {"new_story", "same_story", "related_theme"}.issubset(
                {expected.relationship for expected in expectations}
            )
        )
        self.assertTrue(
            {
                "new",
                "material_update",
                "status_change",
                "correction",
                "resolved",
                "reframed",
                "incremental",
                "unchanged",
            }.issubset({expected.delta_type for expected in expectations})
        )
        self.assertEqual({expected.material for expected in expectations}, {False, True})

        document_tags = {
            tag
            for arc in self.corpus.arcs
            for day in arc.days
            for document in day.documents
            for tag in document.tags
        }
        self.assertTrue(
            {
                "long_source",
                "translated_release",
                "non_english_source",
                "explicit_correction",
                "provisional_status",
                "retrospective",
                "post_cutoff_2026",
                "same_event_hard_negative",
            }.issubset(document_tags)
        )

    def test_provenance_is_one_to_one_and_complete(self) -> None:
        self.assertEqual(
            self.provenance.get("schema_version"), "real_blind.provenance.v1"
        )
        families = self.provenance.get("families")
        self.assertIsInstance(families, list)
        self.assertEqual(len(families), 18)

        corpus_arc_ids = {arc.id for arc in self.corpus.arcs}
        provenance_arc_ids = {family["arc_id"] for family in families}
        self.assertEqual(provenance_arc_ids, corpus_arc_ids)

        corpus_document_ids = {
            document.id
            for arc in self.corpus.arcs
            for day in arc.days
            for document in day.documents
        }
        provenance_document_ids = [
            document_id
            for family in families
            for document_id in family["document_ids"]
        ]
        self.assertEqual(len(provenance_document_ids), len(set(provenance_document_ids)))
        self.assertEqual(set(provenance_document_ids), corpus_document_ids)
        self.assertTrue(all(family["genre_tags"] for family in families))
        self.assertTrue(all(family["structural_categories"] for family in families))

    def test_cryptographic_seal_matches_artifacts(self) -> None:
        seal = _read_json(SEAL_PATH)
        self.assertEqual(seal.get("schema_version"), "real_blind.seal.v1")
        self.assertEqual(seal.get("status"), "sealed_unscored")
        self.assertEqual(seal.get("hash_algorithm"), "sha256")
        self.assertIsNone(seal.get("first_blind_run"))

        expected_paths = {
            "corpus": CORPUS_PATH,
            "provenance": PROVENANCE_PATH,
            "methodology": METHODOLOGY_PATH,
        }
        artifacts = seal.get("artifacts", {})
        self.assertEqual(set(artifacts), set(expected_paths))
        for name, path in expected_paths.items():
            self.assertEqual(artifacts[name]["sha256"], _sha256(path), name)
            self.assertEqual(
                (REPO_ROOT / artifacts[name]["path"]).resolve(), path.resolve(), name
            )

        self.assertEqual(
            seal.get("counts"),
            {
                "event_families": 18,
                "documents": 120,
                "expectations": 120,
                "chronological_days": 107,
                "holdout_families": 18,
            },
        )


if __name__ == "__main__":
    unittest.main()
