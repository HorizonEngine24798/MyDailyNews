from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.evaluation.adapters import (  # noqa: E402
    FaultInjectionAdapter,
    PredictionFileAdapter,
    ProductionHeuristicAdapter,
    ScriptedOracleAdapter,
    LocalDeltaModelAdapter,
)
from mydailynews.evaluation.runner import evaluate_adapter, write_evaluation_report  # noqa: E402
from mydailynews.evaluation.schema import load_corpus, load_predictions  # noqa: E402


DEFAULT_CORPUS = REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline personalized change-monitoring evaluations.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--arc", action="append", default=[], help="Evaluate only this arc ID; repeatable.")
    parser.add_argument("--split", choices=("development", "holdout"), help="Evaluate only one corpus split.")
    parser.add_argument("--adapter", choices=("heuristic", "oracle", "predictions", "delta-model"), default="heuristic")
    parser.add_argument("--predictions", type=Path, help="Standardized prediction JSON for --adapter predictions")
    parser.add_argument("--config", type=Path, help="Application config for --adapter delta-model")
    parser.add_argument("--model-role", choices=("summary", "final"), default="summary")
    parser.add_argument(
        "--model-max-new-tokens",
        type=int,
        help="Override the configured model output ceiling for this evaluation run.",
    )
    parser.add_argument(
        "--delta-output-mode",
        choices=("full", "decision_only"),
        help="Override the delta contract; decision_only is intended for constrained local models.",
    )
    parser.add_argument(
        "--delta-max-articles-per-batch",
        type=int,
        help="Override delta batch width; use a small value when output tokens are constrained.",
    )
    parser.add_argument(
        "--fixture-mode",
        choices=("direct", "rss"),
        default="direct",
        help="Use fast direct fixtures or route them through the production RSS parser.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--fault",
        choices=(
            "drop_every_other",
            "merge_all_stories",
            "omit_everything",
            "unsupported_claims",
            "call_everything_new",
            "hallucinate_quiet_days",
        ),
    )
    parser.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Optional CI gate against an overall metric; repeatable.",
    )
    parser.add_argument(
        "--fail-over",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Optional CI ceiling for error/rate metrics; repeatable.",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    requested_arcs = {str(value or "").strip() for value in args.arc if str(value or "").strip()}
    selected_arcs = [
        arc
        for arc in corpus.arcs
        if (not requested_arcs or arc.id in requested_arcs)
        and (not args.split or arc.split == args.split)
    ]
    if requested_arcs:
        unknown_arcs = requested_arcs.difference(arc.id for arc in corpus.arcs)
        if unknown_arcs:
            parser.error(f"unknown --arc value(s): {sorted(unknown_arcs)}")
    if not selected_arcs:
        parser.error("corpus filters selected no arcs")
    corpus = replace(corpus, arcs=selected_arcs)
    client = None
    if args.adapter == "oracle":
        adapter = ScriptedOracleAdapter(corpus)
    elif args.adapter == "predictions":
        if args.predictions is None:
            parser.error("--predictions is required with --adapter predictions")
        adapter = PredictionFileAdapter(load_predictions(args.predictions), name=args.predictions.stem)
    elif args.adapter == "delta-model":
        if args.config is None:
            parser.error("--config is required with --adapter delta-model")
        from mydailynews.ai.factory import create_ai_client
        from mydailynews.app.config import load_config
        from mydailynews.diagnostics.debug import DebugLogger

        app_config = load_config(args.config)
        debug = DebugLogger(False)
        model_config = app_config.ai_summary if args.model_role == "summary" else app_config.ai_final
        if args.model_max_new_tokens is not None:
            if args.model_max_new_tokens <= 0:
                parser.error("--model-max-new-tokens must be positive")
            model_config = replace(model_config, max_new_tokens=args.model_max_new_tokens)
        delta_output_mode = args.delta_output_mode or app_config.analysis.delta_extraction.output_mode
        delta_batch_size = (
            args.delta_max_articles_per_batch
            if args.delta_max_articles_per_batch is not None
            else app_config.analysis.delta_extraction.max_articles_per_batch
        )
        if delta_batch_size <= 0:
            parser.error("--delta-max-articles-per-batch must be positive")
        client = create_ai_client(model_config, debug)
        delta_config = replace(
            app_config.analysis.delta_extraction,
            enabled=True,
            model_role=args.model_role,
            input_source="evidence_or_articles",
            output_mode=delta_output_mode,
            require_prior_reports=False,
            max_articles_per_batch=delta_batch_size,
            max_articles_dropped_to_avoid_split=(
                0
                if delta_output_mode == "decision_only"
                else app_config.analysis.delta_extraction.max_articles_dropped_to_avoid_split
            ),
        )
        adapter = LocalDeltaModelAdapter(
            client,
            delta_config,
            debug=debug,
            name=f"delta_model:{model_config.effective_model_label}",
            fixture_mode=args.fixture_mode,
        )
    else:
        adapter = ProductionHeuristicAdapter(fixture_mode=args.fixture_mode)
    if args.fault:
        adapter = FaultInjectionAdapter(adapter, args.fault)

    try:
        result = evaluate_adapter(corpus, adapter)
    finally:
        if client is not None:
            client.close()
    json_path, markdown_path = write_evaluation_report(result, args.output)
    print(f"Evaluation JSON: {json_path}")
    print(f"Evaluation Markdown: {markdown_path}")
    print(f"Cases: {result['overall']['cases']}")
    print(f"Identity F1: {result['overall']['story_identity_pairwise_f1']:.4f}")
    print(f"Delta accuracy: {result['overall']['delta_type_accuracy']:.4f}")
    print(f"Display accuracy: {result['overall']['display_policy_accuracy']:.4f}")

    validation_errors = result.get("validation", {}).get("errors", [])
    if validation_errors:
        print("Prediction validation failed:")
        for error in validation_errors:
            print(f"- {error}")
        return 2

    failures = []
    for gate in args.fail_under:
        if "=" not in gate:
            parser.error(f"invalid --fail-under {gate!r}; expected METRIC=VALUE")
        metric, raw_threshold = gate.split("=", 1)
        try:
            threshold = float(raw_threshold)
        except ValueError:
            parser.error(f"invalid threshold in --fail-under {gate!r}")
        value = result.get("overall", {}).get(metric)
        if not isinstance(value, (int, float)):
            parser.error(f"unknown numeric overall metric {metric!r}")
        if float(value) < threshold:
            failures.append(f"{metric}={value} < {threshold}")
    for gate in args.fail_over:
        if "=" not in gate:
            parser.error(f"invalid --fail-over {gate!r}; expected METRIC=VALUE")
        metric, raw_threshold = gate.split("=", 1)
        try:
            threshold = float(raw_threshold)
        except ValueError:
            parser.error(f"invalid threshold in --fail-over {gate!r}")
        value = result.get("overall", {}).get(metric)
        if not isinstance(value, (int, float)):
            parser.error(f"unknown numeric overall metric {metric!r}")
        if float(value) > threshold:
            failures.append(f"{metric}={value} > {threshold}")
    if failures:
        print("Quality gate failed: " + "; ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
