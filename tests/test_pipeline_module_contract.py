from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

from mydailynews.app.models import (
    BriefOutput,
    NarrativeBriefOutput,
    PerspectivesReportOutput,
    PipelineResult,
    TTSOutput,
)
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
        pipeline=SimpleNamespace(default_series=["briefs", "perspectives_report", "narrative_brief"]),
        narrative_briefing=SimpleNamespace(enabled=True),
        perspectives_report=SimpleNamespace(enabled=True),
    )
    orchestrator.run_options = PipelineRunOptions(skip_modules=skip_modules)
    orchestrator.warnings = []
    orchestrator.stopped_after_stage = ""
    return orchestrator


class PipelineModuleContractTests(unittest.TestCase):
    def test_series_passes_current_run_artifacts_without_disk_fallback(self) -> None:
        orchestrator = _orchestrator()
        brief = _brief_output()
        perspectives = PerspectivesReportOutput(
            name="perspectives_report",
            markdown_path="output/current_perspectives.md",
            json_path="output/current_perspectives.json",
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

        def run_perspectives_report(
            self,
            *,
            date: str,
            outputs=None,
            allow_disk_fallback: bool = True,
        ) -> PipelineResult:
            calls["perspectives_report"] = {
                "date": date,
                "outputs": list(outputs or []),
                "allow_disk_fallback": allow_disk_fallback,
            }
            return PipelineResult(perspectives_report_outputs=[perspectives], warnings=self.warnings)

        def run_narrative_brief(
            self,
            *,
            date: str,
            outputs=None,
            perspectives_json_path: str = "",
            allow_disk_fallback: bool = True,
        ) -> PipelineResult:
            calls["narrative"] = {
                "date": date,
                "outputs": list(outputs or []),
                "perspectives_json_path": perspectives_json_path,
                "allow_disk_fallback": allow_disk_fallback,
            }
            return PipelineResult(outputs=list(outputs or []), narrative_outputs=[narrative], warnings=self.warnings)

        orchestrator.run_briefs = MethodType(run_briefs, orchestrator)
        orchestrator.run_perspectives_report = MethodType(run_perspectives_report, orchestrator)
        orchestrator.run_narrative_brief = MethodType(run_narrative_brief, orchestrator)

        result = orchestrator.run_series(date="2026-06-14")

        self.assertEqual(result.outputs, [brief])
        self.assertEqual(result.perspectives_report_outputs, [perspectives])
        self.assertEqual(result.narrative_outputs, [narrative])
        self.assertEqual(calls["perspectives_report"]["outputs"], [brief])
        self.assertFalse(calls["perspectives_report"]["allow_disk_fallback"])
        self.assertEqual(calls["narrative"]["outputs"], [brief])
        self.assertEqual(calls["narrative"]["perspectives_json_path"], perspectives.json_path)
        self.assertFalse(calls["narrative"]["allow_disk_fallback"])

    def test_series_tts_uses_current_run_narrative_markdown(self) -> None:
        orchestrator = _orchestrator()
        orchestrator.config.pipeline.default_series = ["briefs", "narrative_brief", "tts"]
        orchestrator.config.tts = SimpleNamespace(enabled=True, backend="kokoro")
        brief = _brief_output()
        narrative = NarrativeBriefOutput(
            name="narrative",
            markdown_path="output/current_narrative.md",
            json_path="output/current_narrative.json",
        )
        tts = TTSOutput(
            name="tts",
            wav_path="output/current_narrative.wav",
            json_path="output/current_narrative_audio.json",
            markdown_path=narrative.markdown_path,
            chunk_count=1,
        )
        calls: dict[str, dict] = {}

        def close(self) -> None:
            calls["closed_before_tts"] = {"tts_called": "tts" in calls}

        def run_briefs(self, *, date: str) -> PipelineResult:
            return PipelineResult(outputs=[brief], warnings=self.warnings)

        def run_narrative_brief(
            self,
            *,
            date: str,
            outputs=None,
            perspectives_json_path: str = "",
            allow_disk_fallback: bool = True,
        ) -> PipelineResult:
            return PipelineResult(outputs=list(outputs or []), narrative_outputs=[narrative], warnings=self.warnings)

        def run_tts(
            self,
            *,
            date: str,
            markdown_path: str = "",
            allow_disk_fallback: bool = True,
        ) -> PipelineResult:
            calls["tts"] = {
                "date": date,
                "markdown_path": markdown_path,
                "allow_disk_fallback": allow_disk_fallback,
            }
            return PipelineResult(tts_outputs=[tts], warnings=self.warnings)

        orchestrator.run_briefs = MethodType(run_briefs, orchestrator)
        orchestrator.run_narrative_brief = MethodType(run_narrative_brief, orchestrator)
        orchestrator.run_tts = MethodType(run_tts, orchestrator)
        orchestrator.close = MethodType(close, orchestrator)

        result = orchestrator.run_series(date="2026-06-14")

        self.assertEqual(result.tts_outputs, [tts])
        self.assertEqual(calls["closed_before_tts"], {"tts_called": False})
        self.assertEqual(calls["tts"]["markdown_path"], narrative.markdown_path)
        self.assertFalse(calls["tts"]["allow_disk_fallback"])

    def test_series_tts_runs_after_all_selected_modules_and_speaks_each_markdown(self) -> None:
        orchestrator = _orchestrator()
        orchestrator.config.pipeline.default_series = [
            "tts",
            "briefs",
            "narrative_brief",
            "perspectives_report",
        ]
        orchestrator.config.perspectives_report.enabled = True
        orchestrator.config.tts = SimpleNamespace(
            enabled=True,
            backend="kokoro",
            modules=["briefs", "narrative_brief", "perspectives_report"],
        )
        brief = _brief_output()
        narrative = NarrativeBriefOutput(
            name="narrative",
            markdown_path="output/current_narrative.md",
            json_path="output/current_narrative.json",
        )
        perspectives = PerspectivesReportOutput(
            name="perspectives_report",
            markdown_path="output/current_perspectives.md",
            json_path="output/current_perspectives.json",
        )
        calls: list[str] = []

        def close(self) -> None:
            calls.append("closed")

        def run_briefs(self, *, date: str) -> PipelineResult:
            return PipelineResult(outputs=[brief], warnings=self.warnings)

        def run_narrative_brief(self, **kwargs) -> PipelineResult:
            return PipelineResult(narrative_outputs=[narrative], warnings=self.warnings)

        def run_perspectives_report(self, **kwargs) -> PipelineResult:
            return PipelineResult(perspectives_report_outputs=[perspectives], warnings=self.warnings)

        def run_tts(self, *, date: str, markdown_path: str = "", allow_disk_fallback: bool = True) -> PipelineResult:
            calls.append(markdown_path)
            return PipelineResult(
                tts_outputs=[
                    TTSOutput(
                        name="tts",
                        wav_path=f"{markdown_path}.wav",
                        json_path=f"{markdown_path}.json",
                        markdown_path=markdown_path,
                        chunk_count=1,
                    )
                ],
                warnings=self.warnings,
            )

        orchestrator.run_briefs = MethodType(run_briefs, orchestrator)
        orchestrator.run_narrative_brief = MethodType(run_narrative_brief, orchestrator)
        orchestrator.run_perspectives_report = MethodType(run_perspectives_report, orchestrator)
        orchestrator.run_tts = MethodType(run_tts, orchestrator)
        orchestrator.close = MethodType(close, orchestrator)

        result = orchestrator.run_series(date="2026-06-14")

        self.assertEqual(
            calls,
            [
                "closed",
                brief.markdown_path,
                narrative.markdown_path,
                perspectives.markdown_path,
            ],
        )
        self.assertEqual(len(result.tts_outputs), 3)


if __name__ == "__main__":
    unittest.main()
