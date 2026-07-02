from __future__ import annotations

import argparse
from pathlib import Path

from mydailynews.gui.server import serve_gui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local MyDailyNews web GUI.")
    parser.add_argument("--config", default="config.local.json", help="Path to the JSON config file.")
    parser.add_argument("--host", "--gui-host", default="127.0.0.1", help="Host interface for the GUI.")
    parser.add_argument("--port", "--gui-port", type=int, default=8765, help="Port for the GUI.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Create a local config before launching the GUI:")
        print("  cp config.example.json config.local.json")
        print("  python tools/autoconfig.py --config config.local.json --write config.recommended.json")
        return 1
    return serve_gui(
        root=Path.cwd(),
        config_path=config_path,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
