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
    CandidateMetadataReplayAdapter,
    FaultInjectionAdapter,
    PredictionFileAdapter,
    ProductionHeuristicAdapter,
    ScriptedOracleAdapter,
    LocalDeltaModelAdapter,
)
from mydailynews.evaluation.runner import evaluate_adapter, write_evaluation_report  # noqa: E402
from mydailynews.evaluation.investigations import (  # noqa: E402
    INVESTIGATION_MODES,
    build_investigation,
)
from mydailynews.evaluation.schema import load_corpus, load_predictions  # noqa: E402


DEFAULT_CORPUS = REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline personalized change-monitoring evaluations.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--arc", action="append", default=[], help="Evaluate only this arc ID; repeatable.")
    parser.add_argument("--split", choices=("development", "holdout"), help="Evaluate only one corpus split.")
    parser.add_argument("--adapter", choices=("heuristic", "oracle", "predictions", "delta-model"), default="heuristic")
    parser.add_argument("--predictions", type=Path, help="Standardized prediction JSON for --adapter predictions")
    parser.add_argument(
        "--replay-candidate-metadata",
        action="store_true",
        help="Reconstruct broad-baseline candidate metadata for a historical local-delta prediction file.",
    )
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
        "--investigation-mode",
        choices=tuple(sorted(INVESTIGATION_MODES)),
        default="baseline",
        help=(
            "Model capability condition: baseline, gold-blind retrieved_top3, or the "
            "private-gold oracle_candidate/oracle_ledger ceilings."
        ),
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=3,
        help="Maximum prior stories supplied in retrieved_top3 mode, from 1 to 3 (default: 3).",
    )
    parser.add_argument(
        "--candidate-min-score",
        type=float,
        default=0.34,
        help="Minimum lexical candidate score in retrieved_top3 mode (default: 0.34).",
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
    if args.investigation_mode != "baseline" and args.adapter != "delta-model":
        parser.error("--investigation-mode other than baseline requires --adapter delta-model")
    if args.replay_candidate_metadata and args.adapter != "predictions":
        parser.error("--replay-candidate-metadata requires --adapter predictions")
    if not 1 <= args.candidate_limit <= 3:
        parser.error("--candidate-limit must be between 1 and 3")
    if not 0.0 <= args.candidate_min_score <= 1.0:
        parser.error("--candidate-min-score must be between 0 and 1")
    client = None
    if args.adapter == "oracle":
        adapter = ScriptedOracleAdapter(corpus)
    elif args.adapter == "predictions":
        if args.predictions is None:
            parser.error("--predictions is required with --adapter predictions")
        adapter = PredictionFileAdapter(load_predictions(args.predictions), name=args.predictions.stem)
        if args.replay_candidate_metadata:
            adapter = CandidateMetadataReplayAdapter(adapter)
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
            name=f"delta_model:{model_config.effective_model_label}:{args.investigation_mode}",
            fixture_mode=args.fixture_mode,
            investigation=build_investigation(corpus, args.investigation_mode),
            candidate_limit=args.candidate_limit,
            candidate_min_score=args.candidate_min_score,
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
    print(f"Candidate recall@3: {result['overall'].get('candidate_recall_at_3')}")
    print(
        "Relationship given correct candidate: "
        f"{result['overall'].get('relationship_accuracy_given_candidate')}"
    )
    print(
        "Correct candidate actually linked: "
        f"{result['overall'].get('same_story_link_accuracy_given_candidate')}"
    )
    print(
        "Continuation delta given correct identity: "
        f"{result['overall'].get('continuation_delta_accuracy_given_correct_identity')}"
    )

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
