import argparse
from pathlib import Path

from mydailynews.config import get_ai_model_presets, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local-first topic-focused news brief.")
    parser.add_argument("--config", default="config.json", help="Path to the JSON config file.")
    parser.add_argument("--no-enrichment", action="store_true", help="Skip Wikipedia and past-news enrichment.")
    parser.add_argument("--debug", action="store_true", help="Print safe progress diagnostics while the pipeline runs.")
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
    if args.list_model_presets:
        _print_model_presets()
        return 0

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Use config.example.json as the starter config.")
        return 1

    config = load_config(config_path)
    if args.no_enrichment:
        config.enrichment.enabled = False

    from mydailynews.orchestrator import NewsOrchestrator

    orchestrator = NewsOrchestrator(config, debug=args.debug)
    result = None
    run_error: Exception | None = None
    try:
        result = orchestrator.run()
    except Exception as exc:
        run_error = exc
    finally:
        if args.debug:
            _print_debug_analytics(orchestrator, config.output_dir)

    if run_error is not None:
        if isinstance(run_error, RuntimeError):
            print(f"Run failed: {run_error}")
            return 1
        raise run_error

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
