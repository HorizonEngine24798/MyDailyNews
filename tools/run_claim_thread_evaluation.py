from __future__ import annotations

"""Replay the production heuristic-retrieval and claim-thread path end to end."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.analysis.deterministic_delta import build_deterministic_delta_scaffold  # noqa: E402
from mydailynews.analysis.identity_gate import enforce_candidate_identity_gate  # noqa: E402
from mydailynews.app.models import HeadlineDecision, MemoryConfig, SelectedArticle  # noqa: E402
from mydailynews.domain.candidate_annotations import candidate_memory_annotation  # noqa: E402
from mydailynews.evaluation.schema import load_corpus  # noqa: E402
from mydailynews.memory.context import build_story_memory_context  # noqa: E402
from mydailynews.memory.ranking import annotate_candidates_with_memory  # noqa: E402
from mydailynews.memory.recall import apply_delta_signals_to_selected  # noqa: E402
from mydailynews.memory.story_store import StoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate production retrieval plus claim-thread semantics.")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


def evaluate(corpus_path: Path) -> dict:
    corpus = load_corpus(corpus_path)
    rows = []
    with TemporaryDirectory(prefix="mydailynews-claim-thread-eval-") as raw_root:
        for arc in corpus.arcs:
            store = StoryStore(Path(raw_root) / f"{arc.id}.json")
            canonical_to_predicted: dict[str, str] = {}
            for day in arc.days:
                expected_by_id = {item.document_id: item for item in day.expectations}
                for document in day.documents:
                    expected = expected_by_id[document.id]
                    candidate = document.to_candidate()
                    headline = HeadlineDecision(candidate.id, score=8.0)
                    annotate_candidates_with_memory(
                        candidates=[candidate], decisions={candidate.id: headline}, memory_config=MemoryConfig(),
                        coverage_store=None, story_store=store, date=day.date,
                    )
                    article = SelectedArticle(
                        candidate=candidate, decision=headline, article_text=document.body, extraction_status="fixture",
                    )
                    memory = build_story_memory_context(
                        selected=[article], story_groups=[], story_store=store, coverage_store=None,
                        prior_reports=[], date=day.date,
                    )
                    packet = build_deterministic_delta_scaffold(
                        [article], [], story_memory=memory, story_store=store,
                    )
                    packet = enforce_candidate_identity_gate(packet, memory)
                    apply_delta_signals_to_selected(selected=[article], delta_packet=packet)
                    decision = packet["story_decisions"][0]
                    predicted_key = candidate_memory_annotation(candidate).story_key
                    expected_existing_key = canonical_to_predicted.get(expected.canonical_story_id, "")
                    identity_correct = (
                        predicted_key == expected_existing_key
                        if expected.relationship == "same_story"
                        else predicted_key not in set(canonical_to_predicted.values())
                    )
                    if expected.canonical_story_id not in canonical_to_predicted:
                        canonical_to_predicted[expected.canonical_story_id] = predicted_key
                    predicted_material = float(decision.get("materiality", 0.0) or 0.0) >= 0.7
                    rows.append({
                        "arc": arc.id, "document_id": document.id,
                        "slice": "absurd_magic_elf" if "invented_domain" in arc.tags else "standard_news",
                        "is_continuation": expected.relationship == "same_story",
                        "should_select": expected.should_select,
                        "expected_relationship": expected.relationship,
                        "predicted_relationship": _relationship(decision.get("relationship")),
                        "identity_correct": identity_correct,
                        "expected_delta": expected.delta_type, "predicted_delta": decision.get("change_type"),
                        "correct_delta": decision.get("change_type") == expected.delta_type,
                        "correct_material": predicted_material == expected.material,
                        "correct_display": decision.get("disposition") == expected.display,
                        "expected_material": expected.material, "predicted_material": predicted_material,
                        "expected_display": expected.display, "predicted_display": decision.get("disposition"),
                        "summary": decision.get("summary", ""),
                    })
                    store.update_selected(selected=[article], date=day.date, delta_packet=packet)
    return {
        "kind": "claim_thread_end_to_end_evaluation.v1",
        "note": "Production heuristic retrieval, candidate-bounded identity gate, claim-state delta classification, annotation linking, and StoryStore writeback are replayed without gold inputs. Gold labels are used only by this scorer.",
        "summary": summarize(rows),
        "mismatches": [row for row in rows if row["is_continuation"] and not (row["identity_correct"] and row["correct_delta"] and row["correct_material"] and row["correct_display"])],
        "rows": rows,
    }


def _relationship(value) -> str:
    text = str(value or "")
    return "new_story" if text == "distinct_story" else text


def summarize(rows: list[dict]) -> dict:
    groups = {
        "overall": rows,
        "continuations": [row for row in rows if row["is_continuation"]],
        "pipeline_eligible": [row for row in rows if row["should_select"]],
        "standard_news": [row for row in rows if row["slice"] == "standard_news"],
        "absurd_magic_elf": [row for row in rows if row["slice"] == "absurd_magic_elf"],
    }
    return {name: metrics(group) for name, group in groups.items() if group}


def metrics(rows: list[dict]) -> dict:
    return {
        "documents": len(rows),
        "identity_accuracy": ratio([row["identity_correct"] for row in rows]),
        "delta_accuracy": ratio([row["correct_delta"] for row in rows]),
        "materiality_accuracy": ratio([row["correct_material"] for row in rows]),
        "display_accuracy": ratio([row["correct_display"] for row in rows]),
        "joint_identity_semantic_policy_accuracy": ratio([
            row["identity_correct"] and row["correct_delta"] and row["correct_material"] and row["correct_display"]
            for row in rows
        ]),
    }


def ratio(values: list[bool]) -> float:
    return round(sum(values) / max(1, len(values)), 4)


def render(report: dict) -> str:
    lines = ["# Claim-thread end-to-end evaluation", "", report["note"], "", "| Slice | Docs | Identity | Delta | Materiality | Display | Joint |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in report["summary"].items():
        lines.append(f"| {name} | {item['documents']} | {item['identity_accuracy']:.4f} | {item['delta_accuracy']:.4f} | {item['materiality_accuracy']:.4f} | {item['display_accuracy']:.4f} | {item['joint_identity_semantic_policy_accuracy']:.4f} |")
    lines.extend(["", f"Continuation mismatches: {len(report['mismatches'])}", ""])
    for row in report["mismatches"]:
        lines.append(f"- `{row['document_id']}`: identity={row['identity_correct']}; expected {row['expected_delta']}/{row['expected_material']}/{row['expected_display']}; predicted {row['predicted_delta']}/{row['predicted_material']}/{row['predicted_display']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
