from __future__ import annotations

import inspect
import unittest

from mydailynews.pipeline.brief_execution import run_brief


class BriefExecutionOrderTests(unittest.TestCase):
    def test_analysis_runs_after_fetch_without_inline_enrichment(self) -> None:
        source = inspect.getsource(run_brief)

        article_fetch_stage = source.index("article_fetch_result = _fetch_articles_stage")
        article_fetch_checkpoint = source.index('stage="article_fetch"')
        analysis_resolution = source.index(
            "evidence_config, delta_config, analysis_rollout_meta = resolve_analysis_stage_configs"
        )
        story_grouping_stage = source.index("story_grouping_result = _story_grouping_stage")
        story_grouping_checkpoint = source.index('stage="story_grouping"')
        evidence_stage = source.index("evidence_result = _run_evidence_stage")
        write_handoff_stage = source.index('stage="write_handoff"')

        self.assertLess(article_fetch_stage, analysis_resolution)
        self.assertLess(analysis_resolution, article_fetch_checkpoint)
        self.assertLess(analysis_resolution, story_grouping_stage)
        self.assertLess(story_grouping_stage, story_grouping_checkpoint)
        self.assertLess(story_grouping_checkpoint, evidence_stage)
        self.assertNotIn("enrichment_result = _enrich_articles_stage", source)
        self.assertLess(source.index('stage="write_output"'), write_handoff_stage)
        write_output_stage = source.index('stage="write_output"')
        self.assertIn("if _checkpoint_stage", source[write_output_stage - 120:write_output_stage])


if __name__ == "__main__":
    unittest.main()
