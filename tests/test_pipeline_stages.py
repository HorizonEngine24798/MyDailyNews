from __future__ import annotations

import unittest

from mydailynews.pipeline_stages import (
    PipelineRunOptions,
    normalize_brief_selection,
    normalize_stage_name,
)


class PipelineStageOptionTests(unittest.TestCase):
    def test_normalize_stage_name_accepts_hyphenated_input(self) -> None:
        self.assertEqual(normalize_stage_name("headline-select"), "headline_select")
        self.assertEqual(normalize_stage_name("snapshot"), "snapshot")

    def test_normalize_stage_name_rejects_unknown_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported stage"):
            normalize_stage_name("not_a_stage")

    def test_normalize_brief_selection(self) -> None:
        self.assertEqual(normalize_brief_selection("both"), ("general", "detailed"))
        self.assertEqual(normalize_brief_selection("general"), ("general",))
        self.assertEqual(normalize_brief_selection("detailed"), ("detailed",))

    def test_run_options_from_cli(self) -> None:
        options = PipelineRunOptions.from_cli(
            brief="general",
            stop_after_stage="headline-select",
            save_intermediate=False,
            no_save_intermediate=False,
            dump_stage_artifacts=True,
            stage_artifact_dir="output/tmp/stages",
        )
        self.assertEqual(options.briefs, ("general",))
        self.assertEqual(options.stop_after_stage, "headline_select")
        self.assertTrue(options.save_intermediate)
        self.assertTrue(options.dump_stage_artifacts)
        self.assertEqual(options.stage_artifact_dir, "output/tmp/stages")

    def test_no_save_intermediate_overrides_stage_default(self) -> None:
        options = PipelineRunOptions.from_cli(
            brief="detailed",
            stop_after_stage="final_brief",
            save_intermediate=False,
            no_save_intermediate=True,
            dump_stage_artifacts=False,
            stage_artifact_dir="",
        )
        self.assertFalse(options.save_intermediate)


if __name__ == "__main__":
    unittest.main()
