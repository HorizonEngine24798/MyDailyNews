from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

from mydailynews.evaluation.adapters import EvaluationAdapter
from mydailynews.evaluation.schema import EvalCorpus
from mydailynews.evaluation.scoring import score_predictions


def evaluate_adapter(corpus: EvalCorpus, adapter: EvaluationAdapter) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    predictions = []
    for arc in corpus.arcs:
        predictions.extend(adapter.predict(arc.public_input()))
    duration_ms = round((perf_counter() - started) * 1000.0, 4)
    score = score_predictions(corpus, predictions)
    diagnostics = adapter.diagnostics() if callable(getattr(adapter, "diagnostics", None)) else {}
    disclosure = adapter.disclosure() if callable(getattr(adapter, "disclosure", None)) else {}
    if disclosure.get("uses_private_gold") is True:
        score["validation"]["warnings"].append(
            "This run used private gold diagnostic context. Treat it only as a conditional "
            "capability ceiling; it is not a production-comparable quality score."
        )
    ai_totals = diagnostics.get("ai", {}).get("totals", {}) if isinstance(diagnostics, dict) else {}
    ai_duration_ms = float(ai_totals.get("duration_ms", 0.0) or 0.0)
    output_tokens = int(ai_totals.get("output_tokens", 0) or 0)
    return {
        **score,
        "run": {
            "adapter": adapter.name,
            "started_at": started_at.isoformat(),
            "duration_ms": duration_ms,
            "ai_output_tokens_per_second": (
                round(output_tokens / (ai_duration_ms / 1000.0), 4)
                if output_tokens and ai_duration_ms > 0
                else None
            ),
            "diagnostics": diagnostics,
            "investigation": disclosure,
        },
        "predictions": [item.to_dict() for item in predictions],
    }


def write_evaluation_report(
    result: Dict[str, Any],
    output: Path | str,
) -> Tuple[Path, Path]:
    target = Path(output)
    if target.suffix.lower() == ".json":
        json_path = target
        markdown_path = target.with_suffix(".md")
    else:
        json_path = target / "evaluation_report.json"
        markdown_path = target / "evaluation_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    return json_path, markdown_path


def _markdown_report(result: Dict[str, Any]) -> str:
    corpus = result.get("corpus", {})
    run = result.get("run", {})
    counts = result.get("prediction_counts", {})
    metrics = result.get("overall", {})
    investigation = run.get("investigation", {})
    lines = [
        f"# Evaluation: {corpus.get('name', 'unnamed')}",
        "",
        f"Adapter: `{run.get('adapter', '')}`  ",
        f"Cases: {metrics.get('cases', 0)}  ",
        f"Runtime: {run.get('duration_ms', 0)} ms  ",
        f"AI output throughput: {_format_metric(run.get('ai_output_tokens_per_second'))} tokens/s  ",
        f"Missing predictions: {counts.get('missing', 0)}",
        "",
    ]
    if investigation:
        lines.extend(
            [
                "## Investigation disclosure",
                "",
                f"Mode: `{investigation.get('mode', '')}`  ",
                f"Uses private gold: `{bool(investigation.get('uses_private_gold', False))}`  ",
                f"Production-comparable: `{bool(investigation.get('production_comparable', False))}`  ",
                str(investigation.get("purpose", "")),
                "",
            ]
        )
    lines.extend(["## Quality dashboard", "", "| Metric | Score |", "|---|---:|"])
    ordered_metrics = [
        "story_identity_pairwise_f1",
        "relationship_accuracy",
        "delta_type_accuracy",
        "continuation_delta_type_accuracy",
        "material_update_f1",
        "novelty_detection_f1",
        "material_continuation_recall",
        "linked_material_continuation_recall",
        "display_policy_accuracy",
        "selection_f1",
        "profile_relevance_accuracy",
        "candidate_recall_at_1",
        "candidate_recall_at_3",
        "relationship_accuracy_given_candidate",
        "same_story_link_accuracy_given_candidate",
        "continuation_delta_accuracy_given_correct_identity",
        "display_accuracy_given_correct_semantics",
        "false_suppression_rate",
        "unchanged_full_report_rate",
        "quiet_day_abstention_rate",
        "claim_evaluation_coverage",
        "required_fact_recall",
        "faithfulness_pass_rate",
        "latency_ms_p95",
    ]
    for name in ordered_metrics:
        lines.append(f"| {name} | {_format_metric(metrics.get(name))} |")
    stage = metrics.get("stage_diagnostics", {})
    if isinstance(stage, dict) and stage:
        candidate = stage.get("candidate_retrieval", {})
        identity = stage.get("identity", {})
        delta = stage.get("delta", {})
        policy = stage.get("policy", {})
        lines.extend(
            [
                "",
                "## Conditional stage funnel",
                "",
                "| Stage | Opportunity cases | Conditional result |",
                "|---|---:|---:|",
                f"| Candidate retrieval @3 | {candidate.get('continuation_cases', 0)} | {_format_metric(candidate.get('recall_at_3'))} |",
                f"| Relationship with correct candidate | {identity.get('cases_with_correct_candidate', 0)} | {_format_metric(identity.get('relationship_accuracy_given_correct_candidate'))} |",
                f"| Correct candidate actually linked | {identity.get('cases_with_correct_candidate', 0)} | {_format_metric(identity.get('same_story_link_accuracy_given_correct_candidate'))} |",
                f"| Continuation delta with correct identity | {delta.get('continuation_cases_with_correct_identity', 0)} | {_format_metric(delta.get('continuation_accuracy_given_correct_identity'))} |",
                f"| Display with correct semantics | {policy.get('cases_with_correct_semantics', 0)} | {_format_metric(policy.get('display_accuracy_given_correct_semantics'))} |",
            ]
        )
    lines.extend(["", "## Trap slices", "", "| Trap | Cases | Identity F1 | Delta | Display |", "|---|---:|---:|---:|---:|"])
    for tag, slice_metrics in sorted(result.get("by_tag", {}).items()):
        lines.append(
            f"| {tag} | {slice_metrics.get('cases', 0)} | "
            f"{_format_metric(slice_metrics.get('story_identity_pairwise_f1'))} | "
            f"{_format_metric(slice_metrics.get('delta_type_accuracy'))} | "
            f"{_format_metric(slice_metrics.get('display_policy_accuracy'))} |"
        )
    validation = result.get("validation", {})
    warnings = validation.get("warnings", [])
    errors = validation.get("errors", [])
    if warnings or errors:
        lines.extend(["", "## Validation", ""])
        for error in errors:
            lines.append(f"- Error: {error}")
        for warning in warnings:
            lines.append(f"- Warning: {warning}")
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value if value is not None else "n/a")
