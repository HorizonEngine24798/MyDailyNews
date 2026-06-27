from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

from mydailynews.app.models import BriefOutput, EnrichmentOutput, NarrativeBriefOutput, PipelineResult
from mydailynews.pipeline.orchestrator import NewsOrchestrator
from mydailynews.pipeline.stages import PipelineRunOptions


def _brief_output() -> BriefOutput:
    return BriefOutput(
        name="general",
        markdown_path="output/current_general.md",
        json_path="output/current_general.json",
        candidate_count=1,
        selected_count=1,
        handoff_path="output/handoff/current_general_handoff.json",
    )


def _orchestrator(*, skip_modules: tuple[str, ...] = ()) -> NewsOrchestrator:
    orchestrator = NewsOrchestrator.__new__(NewsOrchestrator)
    orchestrator.config = SimpleNamespace(
        pipeline=SimpleNamespace(default_series=["briefs", "enrichment", "narrative_brief"]),
        enrichment=SimpleNamespace(enabled=True, mode="story_llm"),
        narrative_briefing=SimpleNamespace(enabled=True),
    )
    orchestrator.run_options = PipelineRunOptions(skip_modules=skip_modules)
    orchestrator.warnings = []
    orchestrator.stopped_after_stage = ""
    return orchestrator


class PipelineModuleContractTests(unittest.TestCase):
    def test_series_passes_current_run_artifacts_without_disk_fallback(self) -> None:
        orchestrator = _orchestrator()
        brief = _brief_output()
        enrichment = EnrichmentOutput(
            name="enrichment",
            json_path="output/current_enrichment.json",
            markdown_path="output/current_enrichment.md",
        )
        narrative = NarrativeBriefOutput(
            name="narrative",
            markdown_path="output/current_narrative.md",
            json_path="output/current_narrative.json",
        )
        calls: dict[str, dict] = {}

        def run_briefs(self, *, date: str) -> PipelineResult:
            calls["briefs"] = {"date": date}
            return PipelineResult(outputs=[brief], warnings=self.warnings)

        def run_enrichment(
            self,
            *,
            date: str,
            source_outputs=None,
            allow_disk_fallback: bool = True,
        ) -> PipelineResult:
            calls["enrichment"] = {
                "date": date,
                "source_outputs": list(source_outputs or []),
                "allow_disk_fallback": allow_disk_fallback,
            }
            return PipelineResult(enrichment_outputs=[enrichment], warnings=self.warnings)

        def run_narrative_brief(
            self,
            *,
            date: str,
            outputs=None,
            enrichment_json_path: str = "",
            allow_disk_fallback: bool = True,
            use_enrichment=None,
        ) -> PipelineResult:
            calls["narrative"] = {
                "date": date,
                "outputs": list(outputs or []),
                "enrichment_json_path": enrichment_json_path,
                "allow_disk_fallback": allow_disk_fallback,
                "use_enrichment": use_enrichment,
            }
            return PipelineResult(outputs=list(outputs or []), narrative_outputs=[narrative], warnings=self.warnings)

        orchestrator.run_briefs = MethodType(run_briefs, orchestrator)
        orchestrator.run_enrichment = MethodType(run_enrichment, orchestrator)
        orchestrator.run_narrative_brief = MethodType(run_narrative_brief, orchestrator)

        result = orchestrator.run_series(date="2026-06-14")

        self.assertEqual(result.outputs, [brief])
        self.assertEqual(result.enrichment_outputs, [enrichment])
        self.assertEqual(result.narrative_outputs, [narrative])
        self.assertEqual(calls["enrichment"]["source_outputs"], [brief])
        self.assertFalse(calls["enrichment"]["allow_disk_fallback"])
        self.assertEqual(calls["narrative"]["outputs"], [brief])
        self.assertEqual(calls["narrative"]["enrichment_json_path"], enrichment.json_path)
        self.assertFalse(calls["narrative"]["allow_disk_fallback"])
        self.assertTrue(calls["narrative"]["use_enrichment"])

    def test_series_skip_enrichment_does_not_offer_stale_enrichment_to_narrative(self) -> None:
        orchestrator = _orchestrator(skip_modules=("enrichment",))
        brief = _brief_output()
        calls: dict[str, dict] = {}

        def run_briefs(self, *, date: str) -> PipelineResult:
            return PipelineResult(outputs=[brief], warnings=self.warnings)

        def run_enrichment(self, **kwargs) -> PipelineResult:
            raise AssertionError("enrichment should be skipped")

        def run_narrative_brief(
            self,
            *,
            date: str,
            outputs=None,
            enrichment_json_path: str = "",
            allow_disk_fallback: bool = True,
            use_enrichment=None,
        ) -> PipelineResult:
            calls["narrative"] = {
                "enrichment_json_path": enrichment_json_path,
                "allow_disk_fallback": allow_disk_fallback,
                "use_enrichment": use_enrichment,
            }
            return PipelineResult(warnings=self.warnings)

        orchestrator.run_briefs = MethodType(run_briefs, orchestrator)
        orchestrator.run_enrichment = MethodType(run_enrichment, orchestrator)
        orchestrator.run_narrative_brief = MethodType(run_narrative_brief, orchestrator)

        orchestrator.run_series(date="2026-06-14")

        self.assertEqual(calls["narrative"]["enrichment_json_path"], "")
        self.assertFalse(calls["narrative"]["allow_disk_fallback"])
        self.assertFalse(calls["narrative"]["use_enrichment"])
        self.assertTrue(any("enrichment: module skipped by run option" in warning for warning in orchestrator.warnings))


if __name__ == "__main__":
    unittest.main()

