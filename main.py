import argparse
from pathlib import Path

from mydailynews.config import load_config
from mydailynews.orchestrator import NewsOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local-first daily news brief.")
    parser.add_argument("--config", default="config.json", help="Path to the JSON config file.")
    parser.add_argument("--no-enrichment", action="store_true", help="Skip Wikipedia and past-news enrichment.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Use config.example.json as the starter config.")
        return 1

    config = load_config(config_path)
    if args.no_enrichment:
        config.enrichment.enabled = False

    result = NewsOrchestrator(config).run()
    print(f"Markdown brief: {result.markdown_path}")
    print(f"JSON brief:     {result.json_path}")
    print(f"Selected {result.selected_count} articles from {result.candidate_count} candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
