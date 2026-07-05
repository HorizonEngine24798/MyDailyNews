from __future__ import annotations

from pathlib import Path
from typing import List

from mydailynews.app.models import TTSOutput
from mydailynews.common.warnings import extend_warnings
from mydailynews.tts.kokoro_backend import synthesize_narrative_markdown


def run_tts(
    orchestrator,
    *,
    date: str = "",
    markdown_path: str = "",
    narrative_markdown_path: str = "",
    allow_disk_fallback: bool = True,
) -> TTSOutput | None:
    config = getattr(orchestrator.config, "tts", None)
    if not bool(getattr(config, "enabled", False)):
        orchestrator.warnings.append("tts: module is disabled by config; skipped.")
        orchestrator.debug.set_metric("module.tts.status", "skipped_disabled")
        return None
    if str(getattr(config, "backend", "kokoro") or "kokoro").strip().lower() != "kokoro":
        orchestrator.warnings.append("tts: unsupported backend; skipped.")
        orchestrator.debug.set_metric("module.tts.status", "skipped_backend")
        return None

    output_dir = Path(orchestrator.config.output_dir)
    input_path = _markdown_path(
        output_dir=output_dir,
        date=date,
        markdown_path=markdown_path or narrative_markdown_path,
        allow_disk_fallback=allow_disk_fallback,
    )
    if input_path is None:
        suffix = f" for {date}" if date else ""
        warning = f"tts: no Markdown input was available{suffix}."
        orchestrator.warnings.append(warning)
        orchestrator.debug.set_metric("module.tts.status", "skipped_no_input")
        return None

    run_warnings: List[str] = []
    wav_path = output_dir / f"{input_path.stem}.wav"
    orchestrator.reporter.phase("Synthesizing audio...")
    orchestrator.debug.set_metric("module.tts.status", "running")
    orchestrator.debug.log("tts.module", "starting", markdown=str(input_path), wav=str(wav_path))
    try:
        with orchestrator.debug.span("module.tts"):
            output = synthesize_narrative_markdown(input_path, wav_path, config)
    except Exception as exc:
        run_warnings.append(f"tts: synthesis failed ({type(exc).__name__}: {exc}).")
        extend_warnings(orchestrator.warnings, run_warnings)
        orchestrator.debug.set_metric("module.tts.status", "failed")
        orchestrator.debug.set_metric("module.tts.error", f"{type(exc).__name__}: {exc}")
        orchestrator.debug.log("tts.module", "failed", error=type(exc).__name__)
        return None

    _record_tts_artifact(orchestrator, output=output)
    orchestrator.debug.set_metric("module.tts.status", "completed")
    orchestrator.debug.log(
        "tts.module",
        "complete",
        wav=output.wav_path,
        json=output.json_path,
        chunks=output.chunk_count,
    )
    extend_warnings(orchestrator.warnings, run_warnings)
    return output


def _markdown_path(
    *,
    output_dir: Path,
    date: str,
    markdown_path: str = "",
    allow_disk_fallback: bool = True,
) -> Path | None:
    explicit = str(markdown_path or "").strip()
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        if not allow_disk_fallback:
            return None
    if allow_disk_fallback:
        path = output_dir / f"{date}_narrative_brief.md"
        if path.exists():
            return path
    return None


def _record_tts_artifact(orchestrator, *, output: TTSOutput) -> None:
    stage_payload_builder = getattr(orchestrator, "_stage_payload", None)
    record_stage_artifact = getattr(orchestrator, "_record_stage_artifact", None)
    if not callable(stage_payload_builder) or not callable(record_stage_artifact):
        return
    record_stage_artifact(
        stage="tts",
        brief_name="pipeline",
        payload=stage_payload_builder(
            stage="tts",
            brief_name="pipeline",
            summary={
                "markdown_path": output.markdown_path,
                "wav_path": output.wav_path,
                "json_path": output.json_path,
                "backend": output.backend,
                "voice": output.voice,
                "chunks": output.chunk_count,
                "warnings": len(output.warnings),
            },
            next_stage_input={
                "markdown_path": output.markdown_path,
                "wav_path": output.wav_path,
                "json_path": output.json_path,
            },
        ),
    )
