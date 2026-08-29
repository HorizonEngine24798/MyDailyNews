from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mydailynews.evaluation.retrieval_diagnostics import evaluate_story_store_retrieval  # noqa: E402
from mydailynews.evaluation.schema import load_corpus  # noqa: E402
from mydailynews.memory.story_store import DEFAULT_CANDIDATE_THRESHOLD  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure source-backed prior-story retrieval. Private canonical IDs are used only "
            "for historical store writeback, so this is a retrieval intervention diagnostic."
        )
    )
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "evals" / "cases" / "change_monitoring.v1.json"),
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_CANDIDATE_THRESHOLD)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_story_store_retrieval(
        load_corpus(args.corpus),
        threshold=args.threshold,
        limit=args.limit,
    ).payload()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
