from __future__ import annotations

"""Probe whether a constrained local model can judge one story comparison.

This is intentionally an oracle diagnostic: private canonical IDs select the
true prior document and a hard negative.  It must never be reported as a
production evaluation.  Its purpose is to distinguish a bad broad pipeline
contract from a hard small-model capability limit.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.ai.base import JSONSchemaSpec  # noqa: E402
from mydailynews.ai.factory import create_ai_client  # noqa: E402
from mydailynews.app.config import load_config  # noqa: E402
from mydailynews.evaluation.schema import EvalDocument, EvalExpectation, load_corpus  # noqa: E402
from mydailynews.memory.story_retrieval import source_fact_texts  # noqa: E402


DELTA_TYPES = [
    "new",
    "material_update",
    "status_change",
    "correction",
    "resolved",
    "reframed",
    "incremental",
    "unchanged",
    "uncertain",
    "not_applicable",
]
SPIKE_SCHEMA = JSONSchemaSpec(
    name="story_continuation_spike",
    schema={
        "type": "object",
        "properties": {
            "same_story": {"type": "boolean"},
            "delta_type": {"type": "string", "enum": DELTA_TYPES},
            "old_evidence": {"type": "array", "items": {"type": "string"}},
            "new_evidence": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["same_story", "delta_type", "old_evidence", "new_evidence", "reason"],
    },
)


@dataclass(frozen=True)
class Probe:
    case_id: str
    current: EvalDocument
    prior: EvalDocument
    expected_same_story: bool
    expected_delta: str | None
    prior_facts: list[str]
    current_facts: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an oracle-controlled Qwen story-continuation capability spike.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.cpu-qwen1.7b-full.json")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json")
    parser.add_argument("--max-continuations", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "evaluations" / f"qwen_continuation_spike_{datetime.now():%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    if args.max_continuations <= 0:
        parser.error("--max-continuations must be positive")

    probes = build_probes(load_corpus(args.corpus), args.max_continuations)
    if not probes:
        raise RuntimeError("No continuation probes with cited prior and current facts were available.")
    app_config = load_config(args.config)
    client = create_ai_client(replace(app_config.ai_summary, max_new_tokens=args.max_new_tokens))
    variants = ("raw_pair", "compact_source", "oracle_fact_ledger", "staged_decision")
    rows: list[dict[str, Any]] = []
    try:
        for variant in variants:
            for probe in probes:
                for negative in (False, True):
                    tested = replace_probe_with_negative(probe, probes, negative)
                    user = render_prompt(tested, variant)
                    try:
                        prediction = client.complete_json(
                            SYSTEM_PROMPT,
                            user,
                            label=f"continuation_spike.{variant}.{tested.case_id}",
                            max_new_tokens=args.max_new_tokens,
                            json_schema=SPIKE_SCHEMA,
                        )
                        error = ""
                    except Exception as exc:  # Diagnostics must record bad calls, not hide them.
                        prediction = {}
                        error = f"{type(exc).__name__}: {exc}"
                    rows.append(result_row(variant, tested, negative, prediction, error))
    finally:
        client.close()

    report = {
        "kind": "qwen_continuation_capability_spike.v1",
        "contaminated": True,
        "warning": "Private canonical IDs and fact labels selected these comparison pairs. This is a capability probe, not a production score.",
        "model": app_config.ai_summary.effective_model_label,
        "probes": len(probes),
        "variants": list(variants),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "report.json"
    markdown_path = args.output / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    for variant, values in report["summary"].items():
        print(f"{variant}: relationship={values['relationship_accuracy']:.3f}; delta_on_true_pairs={values['delta_accuracy_on_true_pairs']:.3f}; errors={values['errors']}")
    return 0


SYSTEM_PROMPT = """You are comparing two source-grounded news records. Return only the requested JSON. Do not invent facts. `same_story` means both records concern the same continuing real-world story, not merely a similar topic. If false, use delta_type `not_applicable`."""


def build_probes(corpus, maximum: int) -> list[Probe]:
    candidates: list[Probe] = []
    for arc in corpus.arcs:
        history: dict[str, list[tuple[EvalDocument, EvalExpectation]]] = {}
        for day in arc.days:
            expected = {item.document_id: item for item in day.expectations}
            for current in day.documents:
                current_expected = expected[current.id]
                prior_rows = history.get(current_expected.canonical_story_id, [])
                if current_expected.relationship == "same_story" and prior_rows:
                    prior, prior_expected = prior_rows[-1]
                    prior_facts = facts_for(arc.fact_catalog, prior_expected)
                    current_facts = facts_for(arc.fact_catalog, current_expected)
                    if prior_facts and current_facts:
                        candidates.append(Probe(
                            case_id=f"{arc.id}:{day.date}:{current.id}", current=current, prior=prior,
                            expected_same_story=True, expected_delta=current_expected.delta_type,
                            prior_facts=prior_facts, current_facts=current_facts,
                        ))
                history.setdefault(current_expected.canonical_story_id, []).append((current, current_expected))

    # Preserve differing delta types before filling remaining slots chronologically.
    selected: list[Probe] = []
    seen_delta: set[str] = set()
    for probe in candidates:
        if probe.expected_delta not in seen_delta and len(selected) < maximum:
            selected.append(probe)
            seen_delta.add(str(probe.expected_delta))
    for probe in candidates:
        if len(selected) >= maximum:
            break
        if probe not in selected:
            selected.append(probe)
    return selected


def facts_for(catalog: dict[str, str], expected: EvalExpectation) -> list[str]:
    return [catalog[item] for item in expected.required_fact_ids if catalog.get(item)][:4]


def replace_probe_with_negative(probe: Probe, probes: list[Probe], negative: bool) -> Probe:
    if not negative:
        return probe
    # A different continuation's prior source is a realistic hard-ish negative:
    # it is news evidence, not a fabricated counterexample.
    alternatives = [item for item in probes if item.case_id != probe.case_id]
    prior_probe = max(alternatives, key=lambda item: token_overlap(probe.current.title, item.prior.title))
    return Probe(
        case_id=f"{probe.case_id}:negative:{prior_probe.case_id}", current=probe.current,
        prior=prior_probe.prior, expected_same_story=False, expected_delta=None,
        prior_facts=prior_probe.prior_facts, current_facts=probe.current_facts,
    )


def token_overlap(left: str, right: str) -> int:
    return len(set(left.casefold().split()).intersection(right.casefold().split()))


def source_packet(document: EvalDocument) -> str:
    facts = source_fact_texts(document.to_candidate(), source_text=document.body)[:4]
    return "\n".join(f"- {text}" for _, text in facts)


def render_prompt(probe: Probe, variant: str) -> str:
    if variant == "raw_pair":
        evidence = f"PRIOR ARTICLE\nTitle: {probe.prior.title}\nBody: {probe.prior.body[:1200]}\n\nCURRENT ARTICLE\nTitle: {probe.current.title}\nBody: {probe.current.body[:1200]}"
        instruction = "Compare the articles directly."
    elif variant == "compact_source":
        evidence = f"PRIOR SOURCE EVIDENCE\n{source_packet(probe.prior)}\n\nCURRENT SOURCE EVIDENCE\n{source_packet(probe.current)}"
        instruction = "Compare only these compact source-evidence bullets."
    elif variant == "oracle_fact_ledger":
        evidence = f"PRIOR CITED FACT LEDGER\n" + "\n".join(f"- {fact}" for fact in probe.prior_facts) + f"\n\nCURRENT CITED FACT LEDGER\n" + "\n".join(f"- {fact}" for fact in probe.current_facts)
        instruction = "Compare only these cited fact-ledger entries."
    else:
        evidence = f"PRIOR SOURCE EVIDENCE\n{source_packet(probe.prior)}\n\nCURRENT SOURCE EVIDENCE\n{source_packet(probe.current)}"
        instruction = "First decide whether the records are one continuing story. Only if they are, identify the single most specific supported difference; otherwise use not_applicable."
    return f"{instruction}\n\n{evidence}\n\nReturn same_story, delta_type, short old_evidence and new_evidence lists copied or closely paraphrased from the supplied evidence, and a short reason."


def result_row(variant: str, probe: Probe, negative: bool, prediction: dict[str, Any], error: str) -> dict[str, Any]:
    same_story = prediction.get("same_story") if isinstance(prediction.get("same_story"), bool) else None
    delta = str(prediction.get("delta_type", "") or "")
    return {
        "variant": variant,
        "case_id": probe.case_id,
        "negative_pair": negative,
        "expected_same_story": probe.expected_same_story,
        "expected_delta": probe.expected_delta,
        "predicted_same_story": same_story,
        "predicted_delta": delta,
        "relationship_correct": same_story == probe.expected_same_story,
        "delta_correct": bool(probe.expected_delta and delta == probe.expected_delta),
        "error": error,
        "prediction": prediction,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        true_pairs = [row for row in group if not row["negative_pair"]]
        summary[variant] = {
            "calls": len(group),
            "errors": sum(bool(row["error"]) for row in group),
            "relationship_accuracy": round(sum(row["relationship_correct"] for row in group) / max(1, len(group)), 4),
            "delta_accuracy_on_true_pairs": round(sum(row["delta_correct"] for row in true_pairs) / max(1, len(true_pairs)), 4),
            "same_story_recall": round(sum(row["predicted_same_story"] is True for row in true_pairs) / max(1, len(true_pairs)), 4),
            "different_story_recall": round(sum(row["predicted_same_story"] is False for row in group if row["negative_pair"]) / max(1, len(group) - len(true_pairs)), 4),
        }
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Qwen continuation capability spike", "", "**Contaminated oracle diagnostic:** private labels selected comparison pairs and fact packets. This is not a production score.", "", f"Probes: {report['probes']} true continuations plus the same number of negative pairs.", "", "| Variant | Relationship accuracy | Delta accuracy on true pairs | Same-story recall | Different-story recall | Errors |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in report["summary"].items():
        lines.append(f"| {name} | {values['relationship_accuracy']:.4f} | {values['delta_accuracy_on_true_pairs']:.4f} | {values['same_story_recall']:.4f} | {values['different_story_recall']:.4f} | {values['errors']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
