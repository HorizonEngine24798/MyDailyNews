from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, TextIO, TYPE_CHECKING

from mydailynews.app.models import PipelineResult

if TYPE_CHECKING:
    from mydailynews.diagnostics.debug import DebugLogger
    from mydailynews.app.models import AIConfig, AppConfig
    from mydailynews.pipeline.stages import PipelineRunOptions


class CliReporter:
    def __init__(self, *, enabled: bool = True, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stdout

    def run_start(
        self,
        *,
        config_path: Path | str,
        config: AppConfig,
        run_options: PipelineRunOptions,
    ) -> None:
        if not self.enabled:
            return
        self._print("MyDailyNews run starting")
        self._print(f"Config: {config_path}")
        self._print(f"Module: {run_options.module}")
        if run_options.date:
            self._print(f"Date: {run_options.date}")
        self._print(f"Briefs: {','.join(run_options.briefs)}")
        if run_options.skip_modules:
            self._print(f"Skipped modules: {','.join(run_options.skip_modules)}")
        self._print(f"AI: {self._ai_label(config.ai_summary)} -> {self._ai_label(config.ai_final)}")
        enrichment = "enabled" if bool(getattr(config.enrichment, "enabled", False)) else "disabled"
        self._print(f"Enrichment: {enrichment}")
        self._print("")

    def phase(self, message: str) -> None:
        text = str(message or "").strip()
        if not self.enabled or not text:
            return
        self._print(text)

    def outputs(self, result: PipelineResult) -> None:
        if not self.enabled or not (result.outputs or result.enrichment_outputs or result.narrative_outputs):
            return
        self._print("")
        for output in result.outputs:
            label = output.name.title()
            self._print(f"{label} markdown brief: {output.markdown_path}")
            self._print(f"{label} JSON brief:     {output.json_path}")
            self._print(
                f"{label} selected {output.selected_count} articles "
                f"from {output.candidate_count} candidates."
            )
            if output.handoff_path:
                self._print(f"{label} handoff:        {output.handoff_path}")
        for output in result.enrichment_outputs:
            label = output.name.title()
            source_label = ",".join(output.source_briefs) if output.source_briefs else "none"
            if output.markdown_path:
                self._print(f"{label} markdown:      {output.markdown_path}")
            self._print(f"{label} JSON:          {output.json_path}")
            self._print(f"{label} source briefs: {source_label}")
            self._print(f"{label} story threads: {output.story_thread_count}")
        for output in result.narrative_outputs:
            label = output.name.title()
            source_label = ",".join(output.source_briefs) if output.source_briefs else "none"
            self._print(f"{label} markdown brief: {output.markdown_path}")
            self._print(f"{label} JSON brief:     {output.json_path}")
            self._print(f"{label} source briefs:  {source_label}")

    def stopped(self, stage: str, artifact_paths: Iterable[str] | None = None) -> None:
        if not self.enabled:
            return
        self._print(f"Run stopped after stage: {stage}")
        self.stage_artifacts(artifact_paths)

    def stage_artifacts(self, artifact_paths: Iterable[str] | None = None) -> None:
        paths = self._path_list(artifact_paths)
        if not self.enabled or not paths:
            return
        self._print("Stage artifacts:")
        for path in paths:
            self._print(f"- {path}")

    def warnings(self, warnings: Iterable[str]) -> None:
        warning_lines = [str(warning) for warning in warnings if str(warning).strip()]
        if not self.enabled or not warning_lines:
            return
        self._print("")
        self._print("Warnings:")
        for warning in warning_lines:
            self._print(f"- {warning}")

    def debug_summary(
        self,
        *,
        debug: DebugLogger,
        output_dir: str | Path,
        artifact_paths: Iterable[str] | None = None,
    ) -> None:
        if not self.enabled or not bool(getattr(debug, "enabled", False)):
            return
        analytics_path = debug.write_analytics_artifact(output_dir)
        lines = debug.analytics_summary_lines()
        paths = self._path_list(artifact_paths)
        if not lines and not analytics_path and not paths:
            return

        self._print("")
        self._print("Debug summary")
        for line in lines:
            self._print(f"- {line}")

        self._print("")
        self._print("Debug artifacts")
        if analytics_path:
            self._print(f"- Analytics JSON: {analytics_path}")
        if paths:
            self._print("- Stage artifacts:")
            for path in paths:
                self._print(f"  {path}")
        self._print(f"- Llama server logs: {Path(output_dir) / 'diagnostics' / 'llama_server'}")

    def _print(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    @staticmethod
    def _ai_label(config: AIConfig) -> str:
        backend = str(getattr(config, "backend", "") or "unknown")
        model = str(getattr(config, "effective_model_label", "") or getattr(config, "model_id", "") or "unknown")
        return f"{backend}:{model}"

    @staticmethod
    def _path_list(artifact_paths: Iterable[str] | None) -> list[str]:
        return [str(path) for path in artifact_paths or [] if str(path).strip()]
