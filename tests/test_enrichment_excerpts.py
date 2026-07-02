from __future__ import annotations

import unittest

from mydailynews.enrichment.payloads import excerpt_text


class EnrichmentExcerptTests(unittest.TestCase):
    def test_relevant_windows_include_body_terms_past_boilerplate(self) -> None:
        text = (
            "Subscribe now. Cookie choices. Newsletter signup. " * 20
            + "Regulators expanded chip export controls after supply-chain warnings. "
            + "The new licensing rule affects advanced AI accelerators."
        )

        excerpt = excerpt_text(
            text,
            strategy="relevant_windows",
            terms_text="chip export controls AI accelerators",
            max_chars=240,
            lead_chars=0,
            window_chars=180,
            max_windows=1,
        )

        self.assertIn("chip export controls", excerpt)
        self.assertNotIn("Subscribe now", excerpt)

    def test_prefix_strategy_preserves_old_truncation(self) -> None:
        text = "First words only. Later chip export controls matter."

        self.assertEqual(
            excerpt_text(
                text,
                strategy="prefix",
                terms_text="chip export",
                max_chars=16,
                lead_chars=0,
                window_chars=100,
                max_windows=1,
            ),
            "First words only",
        )

    def test_empty_and_short_text_are_stable(self) -> None:
        self.assertEqual(
            excerpt_text(
                "",
                strategy="relevant_windows",
                terms_text="anything",
                max_chars=100,
                lead_chars=10,
                window_chars=20,
                max_windows=1,
            ),
            "",
        )
        self.assertEqual(
            excerpt_text(
                "Short body.",
                strategy="relevant_windows",
                terms_text="body",
                max_chars=100,
                lead_chars=20,
                window_chars=20,
                max_windows=1,
            ),
            "Short body.",
        )


if __name__ == "__main__":
    unittest.main()
