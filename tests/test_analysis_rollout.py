from __future__ import annotations

import unittest

from mydailynews.analysis.rollout import resolve_analysis_stage_configs
from mydailynews.app.models import (
    AnalysisConfig,
    AnalysisRolloutConfig,
    AnalysisRolloutModeConfig,
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
)


class AnalysisRolloutResolutionTests(unittest.TestCase):
    def test_rollout_disabled_preserves_direct_stage_flags(self) -> None:
        disabled = AnalysisConfig(
            evidence_distillation=EvidenceDistillationConfig(enabled=False),
            delta_extraction=DeltaExtractionConfig(enabled=False),
            rollout=AnalysisRolloutConfig(enabled=False, profile="quality_focused"),
        )
        evidence, delta, metadata = resolve_analysis_stage_configs(disabled, "detailed")

        self.assertFalse(metadata["rollout_enabled"])
        self.assertFalse(evidence.enabled)
        self.assertFalse(delta.enabled)
        self.assertEqual(metadata["evidence_skip_reason"], "disabled")
        self.assertEqual(metadata["delta_skip_reason"], "disabled")

        direct_enabled = AnalysisConfig(
            evidence_distillation=EvidenceDistillationConfig(enabled=True),
            delta_extraction=DeltaExtractionConfig(enabled=True),
            rollout=AnalysisRolloutConfig(enabled=False, profile="safe_local"),
        )
        evidence, delta, metadata = resolve_analysis_stage_configs(direct_enabled, "general")

        self.assertFalse(metadata["rollout_enabled"])
        self.assertTrue(evidence.enabled)
        self.assertTrue(delta.enabled)
        self.assertEqual(metadata["evidence_skip_reason"], "enabled")
        self.assertEqual(metadata["delta_skip_reason"], "enabled")

    def test_profile_defaults_resolve_by_brief(self) -> None:
        analysis = AnalysisConfig(
            rollout=AnalysisRolloutConfig(enabled=True, profile="balanced_local"),
        )

        general_evidence, general_delta, general_meta = resolve_analysis_stage_configs(analysis, "general")
        detailed_evidence, detailed_delta, detailed_meta = resolve_analysis_stage_configs(analysis, "detailed")

        self.assertEqual(general_meta["rollout_profile"], "balanced_local")
        self.assertFalse(general_evidence.enabled)
        self.assertFalse(general_delta.enabled)
        self.assertTrue(detailed_evidence.enabled)
        self.assertTrue(detailed_delta.enabled)
        self.assertEqual(detailed_meta["rollout_mode"], "detailed")

    def test_brief_specific_override_wins(self) -> None:
        analysis = AnalysisConfig(
            rollout=AnalysisRolloutConfig(
                enabled=True,
                profile="quality_focused",
                general=AnalysisRolloutModeConfig(
                    evidence_enabled=False,
                    delta_enabled=True,
                ),
            ),
        )

        evidence, delta, metadata = resolve_analysis_stage_configs(analysis, "general")

        self.assertEqual(metadata["rollout_profile"], "quality_focused")
        self.assertFalse(evidence.enabled)
        self.assertTrue(delta.enabled)

    def test_rollout_caps_keep_stricter_limit(self) -> None:
        analysis = AnalysisConfig(
            evidence_distillation=EvidenceDistillationConfig(
                enabled=True,
                max_input_tokens=8000,
                max_new_tokens=1200,
                max_articles=10,
                max_articles_per_batch=9,
                max_articles_dropped_to_avoid_split=6,
                max_article_chars=900,
            ),
            delta_extraction=DeltaExtractionConfig(
                enabled=True,
                max_input_tokens=7000,
                max_new_tokens=1100,
                max_articles=8,
                max_articles_per_batch=7,
                max_articles_dropped_to_avoid_split=5,
                max_article_chars=500,
                max_prior_reports=6,
            ),
            rollout=AnalysisRolloutConfig(
                enabled=True,
                profile="balanced_local",
                detailed=AnalysisRolloutModeConfig(
                    evidence_max_input_tokens=9000,
                    evidence_max_new_tokens=900,
                    evidence_max_articles=4,
                    evidence_max_articles_per_batch=3,
                    evidence_max_articles_dropped_to_avoid_split=2,
                    evidence_max_article_chars=600,
                    delta_max_input_tokens=6000,
                    delta_max_new_tokens=1000,
                    delta_max_articles=3,
                    delta_max_articles_per_batch=2,
                    delta_max_articles_dropped_to_avoid_split=1,
                    delta_max_article_chars=360,
                    delta_max_prior_reports=4,
                ),
            ),
        )

        evidence, delta, _metadata = resolve_analysis_stage_configs(analysis, "detailed")

        self.assertEqual(evidence.max_input_tokens, 8000)
        self.assertEqual(evidence.max_new_tokens, 900)
        self.assertEqual(evidence.max_articles, 4)
        self.assertEqual(evidence.max_articles_per_batch, 3)
        self.assertEqual(evidence.max_articles_dropped_to_avoid_split, 2)
        self.assertEqual(evidence.max_article_chars, 600)
        self.assertEqual(delta.max_input_tokens, 6000)
        self.assertEqual(delta.max_new_tokens, 1000)
        self.assertEqual(delta.max_articles, 3)
        self.assertEqual(delta.max_articles_per_batch, 2)
        self.assertEqual(delta.max_articles_dropped_to_avoid_split, 1)
        self.assertEqual(delta.max_article_chars, 360)
        self.assertEqual(delta.max_prior_reports, 4)


if __name__ == "__main__":
    unittest.main()
