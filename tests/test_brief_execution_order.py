from __future__ import annotations

import inspect
import unittest

from mydailynews.pipeline.brief_execution import run_brief


class BriefExecutionOrderTests(unittest.TestCase):
    def test_analysis_config_resolves_after_fetch_before_story_grouping_and_enrichment(self) -> None:
        source = inspect.getsource(run_brief)

        article_fetch_stage = source.index("article_fetch_result = _fetch_articles_stage")
        article_fetch_checkpoint = source.index('stage="article_fetch"')
        analysis_resolution = source.index(
            "evidence_config, delta_config, analysis_rollout_meta = resolve_analysis_stage_configs"
        )
        story_grouping_stage = source.index("story_grouping_result = _story_grouping_stage")
        story_grouping_checkpoint = source.index('stage="story_grouping"')
        enrichment_stage = source.index("enrichment_result = _enrich_articles_stage")

        self.assertLess(article_fetch_stage, analysis_resolution)
        self.assertLess(analysis_resolution, article_fetch_checkpoint)
        self.assertLess(analysis_resolution, story_grouping_stage)
        self.assertLess(story_grouping_stage, story_grouping_checkpoint)
        self.assertLess(story_grouping_checkpoint, enrichment_stage)


if __name__ == "__main__":
    unittest.main()
