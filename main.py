import argparse
from pathlib import Path

from mydailynews.app.config import load_config
from mydailynews.app.runtime_config import find_runtime_config_issues, format_runtime_config_issues
from mydailynews.pipeline.stages import ALL_STAGE_ORDER, PipelineRunOptions
from mydailynews.diagnostics.reporting import CliReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local-first topic-focused news brief.")
    parser.add_argument("--config", default="config.local.json", help="Path to the JSON config file.")
    parser.add_argument("--no-enrichment", action="store_true", help="Force-skip story-thread/context enrichment.")
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
        help="Write replay-oriented JSON artifacts for each executed stage.",
    )
    parser.add_argument(
        "--save-intermediate",
        "--save_intermediate",
        action="store_true",
        help="Write stage artifacts even when the run is not stopped at a checkpoint.",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Disable the --save-intermediate artifact-writing trigger.",
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_stages:
        print("Available pipeline stages:")
        for stage in ALL_STAGE_ORDER:
            print(f"- {stage}")
        return 0

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Create a local config before running the pipeline:")
        print("  copy config.example.json config.local.json")
        print("  python tools/autoconfig.py --config config.local.json --write config.recommended.json")
        print("  python main.py --config config.recommended.json")
        return 1

    config = load_config(config_path)
    runtime_issues = find_runtime_config_issues(config)
    if runtime_issues:
        print(f"Config is not ready to run: {config_path}")
        print(format_runtime_config_issues(runtime_issues))
        print("Run tools/autoconfig.py or edit the local config with your llama.cpp paths and token limits.")
        return 1
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

    reporter = CliReporter(enabled=True)
    reporter.run_start(config_path=config_path, config=config, run_options=run_options)

    from mydailynews.pipeline.orchestrator import NewsOrchestrator

    orchestrator = NewsOrchestrator(config, debug=args.debug, reporter=reporter)
    result = None
    run_error: Exception | None = None
    try:
        result = orchestrator.run(run_options=run_options)
    except Exception as exc:
        run_error = exc
    finally:
        orchestrator.close()

    if run_error is not None:
        reporter.warnings(orchestrator.warnings)
        if args.debug:
            reporter.debug_summary(
                debug=orchestrator.debug,
                output_dir=config.output_dir,
                artifact_paths=orchestrator.stage_artifact_paths,
            )
        if isinstance(run_error, RuntimeError):
            print(f"Run failed: {run_error}")
            return 1
        raise run_error

    if orchestrator.stopped_after_stage:
        reporter.stopped(orchestrator.stopped_after_stage, orchestrator.stage_artifact_paths)
    else:
        reporter.stage_artifacts(orchestrator.stage_artifact_paths)

    if result is None:
        print("Run failed: no result returned.")
        return 1
    reporter.outputs(result)
    reporter.warnings(result.warnings)
    if args.debug:
        reporter.debug_summary(
            debug=orchestrator.debug,
            output_dir=config.output_dir,
            artifact_paths=orchestrator.stage_artifact_paths,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
