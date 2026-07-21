from __future__ import annotations

import unittest
from types import SimpleNamespace

from mydailynews.briefing.narrative import NarrativeBriefGenerator, _compact_source_brief


class NarrativePromptCompactionTests(unittest.TestCase):
    def test_target_words_is_reachable_within_output_budget(self) -> None:
        generator = NarrativeBriefGenerator(SimpleNamespace(max_new_tokens=512), target_words=1800)

        self.assertEqual(generator.requested_target_words, 1800)
        self.assertEqual(generator.target_words, 192)

    def test_keeps_reader_facing_brief_and_drops_duplicate_pipeline_payloads(self) -> None:
        compact = _compact_source_brief(
            {
                "title": "Daily brief",
                "lead": "Material lead",
                "sections": [{"heading": "Update", "body": "What changed"}],
                "selected_articles": [{"article_text": "large duplicate source text"}],
                "references": [{"url": "https://example.com"}],
                "analysis": {"evidence_packet": {"raw": "large duplicate analysis"}},
                "metadata": {"diagnostics": "internal"},
            }
        )

        self.assertEqual(compact["lead"], "Material lead")
        self.assertEqual(compact["sections"][0]["body"], "What changed")
        self.assertNotIn("selected_articles", compact)
        self.assertNotIn("references", compact)
        self.assertNotIn("analysis", compact)
        self.assertNotIn("metadata", compact)


if __name__ == "__main__":
    unittest.main()
