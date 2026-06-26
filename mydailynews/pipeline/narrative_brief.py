from __future__ import annotations

from pathlib import Path
from typing import List

from mydailynews.app.models import BriefOutput, NarrativeBriefOutput
from mydailynews.briefing.narrative import (
    NarrativeBriefGenerator,
    NarrativeSourceBrief,
    load_source_brief,
    write_narrative_outputs,
)
from mydailynews.common.warnings import extend_warnings


NARRATIVE_SOURCE_BRIEF_NAMES = ("general", "detailed")


def run_narrative_brief(orchestrator, *, outputs: List[BriefOutput], date: str) -> NarrativeBriefOutput | None:
    config = orchestrator.config.narrative_briefing
    if not config.enabled:
        return None

    run_warnings: List[str] = []
    try:
        source_briefs = _collect_source_briefs(outputs, output_dir=Path(orchestrator.config.output_dir), date=date)
    except Exception as exc:
        warning = (
            f"narrative: source brief loading failed ({type(exc).__name__}): {exc}; "
            "continuing with already written structured briefs."
        )
        orchestrator.warnings.append(warning)
        orchestrator.debug.set_metric("brief.narrative.status", "failed")
        orchestrator.debug.set_metric("brief.narrative.error", f"{type(exc).__name__}: {exc}")
        orchestrator.debug.log("narrative_brief.run", "failed_source_load", error=type(exc).__name__)
        return None
    if not source_briefs:
        orchestrator.warnings.append("narrative: No source brief JSON files were available for the narrative briefing pass.")
        return None

    source_names = [source.name for source in source_briefs]
    orchestrator.reporter.phase("Writing narrative Markdown brief...")
    orchestrator.debug.set_metric("brief.narrative.status", "running")
    orchestrator.debug.log("narrative_brief.run", "starting", source_briefs=",".join(source_names))

    output: NarrativeBriefOutput | None = None
    generator: NarrativeBriefGenerator | None = None
    try:
        generator = NarrativeBriefGenerator(
            orchestrator.final_ai_client,
            input_token_limit=config.max_input_tokens or orchestrator.config.ai_final.max_input_tokens,
            max_new_tokens=config.max_new_tokens or orchestrator.config.ai_final.max_new_tokens,
            target_words=config.target_words,
            editorial_style=config.editorial_style,
            debug=orchestrator.debug,
        )
        with orchestrator.debug.span("brief.narrative.generation"):
            narrative_brief = generator.generate(
                source_briefs,
                orchestrator.config.user_memory,
                date=date,
            )
        extend_warnings(run_warnings, generator.warnings)

        output_dir = Path(orchestrator.config.output_dir)
        with orchestrator.debug.span("brief.narrative.write_output"):
            markdown_path, json_path = write_narrative_outputs(output_dir, date, narrative_brief)

        orchestrator._record_stage_artifact(
            stage="narrative_brief",
            brief_name="pipeline",
            payload=orchestrator._stage_payload(
                stage="narrative_brief",
                brief_name="pipeline",
                summary={
                    "source_briefs": source_names,
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                    "segments": len(narrative_brief.get("segments", [])),
                    "markdown_chars": len(markdown_path.read_text(encoding="utf-8")),
                    "warnings": len(run_warnings),
                },
                next_stage_input={
                    "narrative_brief": narrative_brief,
                    "source_briefs": [
                        {
                            "name": source.name,
                            "json_path": source.json_path,
                        }
                        for source in source_briefs
                    ],
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                },
            ),
        )
        orchestrator.debug.set_metric("brief.narrative.status", "completed")
        orchestrator.debug.log(
            "narrative_brief.run",
            "complete",
            markdown=markdown_path,
            json=json_path,
            warnings=len(run_warnings),
        )
        output = NarrativeBriefOutput(
            name="narrative",
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            source_briefs=source_names,
            warnings=run_warnings,
        )
    except Exception as exc:
        if generator is not None:
            extend_warnings(run_warnings, generator.warnings)
        warning = (
            f"narrative: generation failed ({type(exc).__name__}): {exc}; "
            "continuing with already written structured briefs."
        )
        run_warnings.append(warning)
        orchestrator.debug.set_metric("brief.narrative.status", "failed")
        orchestrator.debug.set_metric("brief.narrative.error", f"{type(exc).__name__}: {exc}")
        orchestrator.debug.log("narrative_brief.run", "failed", error=type(exc).__name__)
    finally:
        try:
            orchestrator.final_ai_client.unload()
        except Exception as exc:
            run_warnings.append(f"narrative: final model unload failed ({type(exc).__name__}): {exc}")
            orchestrator.debug.set_metric("brief.narrative.unload_error", f"{type(exc).__name__}: {exc}")
        extend_warnings(orchestrator.warnings, run_warnings)
    return output


def _collect_source_briefs(
    outputs: List[BriefOutput],
    *,
    output_dir: Path,
    date: str,
) -> List[NarrativeSourceBrief]:
    by_name: dict[str, Path] = {}
    for output in outputs:
        name = str(output.name or "").strip().lower()
        if name in NARRATIVE_SOURCE_BRIEF_NAMES:
            by_name[name] = Path(output.json_path)

    for name in NARRATIVE_SOURCE_BRIEF_NAMES:
        if name in by_name:
            continue
        candidate_path = output_dir / f"{date}_{name}_brief.json"
        if candidate_path.exists():
            by_name[name] = candidate_path

    source_briefs: List[NarrativeSourceBrief] = []
    for name in NARRATIVE_SOURCE_BRIEF_NAMES:
        path = by_name.get(name)
        if path is None or not path.exists():
            continue
        source_briefs.append(load_source_brief(name, path))
    return source_briefs
