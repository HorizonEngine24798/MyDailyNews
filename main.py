import argparse
from pathlib import Path

from mydailynews.config import get_ai_model_presets, load_config
from mydailynews.pipeline_stages import ALL_STAGE_ORDER, PipelineRunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local-first topic-focused news brief.")
    parser.add_argument("--config", default="config.json", help="Path to the JSON config file.")
    parser.add_argument("--no-enrichment", action="store_true", help="Skip Wikipedia and past-news enrichment.")
    parser.add_argument("--debug", action="store_true", help="Print safe progress diagnostics while the pipeline runs.")
    parser.add_argument(
        "--brief",
        default="both",
        choices=("general", "detailed", "both"),
        help="Run only one brief mode or both.",
    )
    parser.add_argument(
        "--stop-after-stage",
        default="",
        help=f"Stop the run after a stage checkpoint. Supported: {', '.join(ALL_STAGE_ORDER)}",
    )
    parser.add_argument(
        "--dump-stage-artifacts",
        action="store_true",
        help="Write JSON checkpoints for each executed stage.",
    )
    parser.add_argument(
        "--save-intermediate",
        "--save_intermediate",
        action="store_true",
        help="Save full intermediate stage payloads for debugging and replay tooling.",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Disable intermediate payload saving even in stage-by-stage runs.",
    )
    parser.add_argument(
        "--stage-artifact-dir",
        default="",
        help="Optional directory for stage checkpoint JSON files.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available --stop-after-stage values and exit.",
    )
    parser.add_argument(
        "--list-model-presets",
        action="store_true",
        help="List built-in ai.preset options and exit.",
    )
    return parser


def _print_model_presets() -> None:
    presets = get_ai_model_presets()
    print("Available ai.preset options:")
    for preset_name, meta in presets.items():
        generation_ref = str(meta.get("max_generation_tokens_note") or meta["max_generation_tokens"])
        print(f"- {preset_name}")
        print(f"  model_id: {meta['model_id']}")
        print(f"  parameters: {meta['parameter_count']}")
        print(
            f"  context_window_tokens: {meta['context_window_tokens']} "
            f"(max_generation_tokens: {generation_ref})"
        )
        print(
            f"  default_runtime_tokens: "
            f"max_input_tokens={meta['max_input_tokens']}, max_new_tokens={meta['max_new_tokens']}"
        )
        print(f"  notes: {meta['notes']}")
        print(f"  source: {meta['source']}")


def _print_debug_analytics(orchestrator, output_dir: str) -> None:
    analytics_path = orchestrator.debug.write_analytics_artifact(output_dir)
    lines = orchestrator.debug.analytics_summary_lines()
    if not lines and not analytics_path:
        return
    print("")
    print("Debug Analytics:")
    for line in lines:
        print(f"- {line}")
    if analytics_path:
        print(f"Analytics JSON:       {analytics_path}")


def main() -> int:
    args = build_parser().parse_args()
    if args.list_stages:
        print("Available pipeline stages:")
        for stage in ALL_STAGE_ORDER:
            print(f"- {stage}")
        return 0
    if args.list_model_presets:
        _print_model_presets()
        return 0

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Create or restore config.json before running the pipeline.")
        return 1

    config = load_config(config_path)
    if args.no_enrichment:
        config.enrichment.enabled = False
    try:
        run_options = PipelineRunOptions.from_cli(
            brief=args.brief,
            stop_after_stage=args.stop_after_stage,
            save_intermediate=args.save_intermediate,
            no_save_intermediate=args.no_save_intermediate,
            dump_stage_artifacts=args.dump_stage_artifacts,
            stage_artifact_dir=args.stage_artifact_dir,
        )
    except ValueError as exc:
        print(f"Invalid run option: {exc}")
        return 1

    from mydailynews.orchestrator import NewsOrchestrator

    orchestrator = NewsOrchestrator(config, debug=args.debug)
    result = None
    run_error: Exception | None = None
    try:
        result = orchestrator.run(run_options=run_options)
    except Exception as exc:
        run_error = exc
    finally:
        orchestrator.close()
        if args.debug:
            _print_debug_analytics(orchestrator, config.output_dir)

    if run_error is not None:
        if isinstance(run_error, RuntimeError):
            print(f"Run failed: {run_error}")
            return 1
        raise run_error

    if orchestrator.stopped_after_stage:
        print(f"Run stopped after stage: {orchestrator.stopped_after_stage}")
    if orchestrator.stage_artifact_paths:
        print("Stage artifacts:")
        for path in orchestrator.stage_artifact_paths:
            print(f"- {path}")

    for output in result.outputs:
        print(f"{output.name.title()} markdown brief: {output.markdown_path}")
        print(f"{output.name.title()} JSON brief:     {output.json_path}")
        print(f"{output.name.title()} selected {output.selected_count} articles from {output.candidate_count} candidates.")
    if result.warnings:
        print("")
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
