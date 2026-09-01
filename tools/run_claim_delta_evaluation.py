from __future__ import annotations

"""Evaluate source-claim delta classification with oracle story identity."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.analysis.deterministic_delta import build_deterministic_delta_scaffold  # noqa: E402
from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, SelectedArticle  # noqa: E402
from mydailynews.domain.candidate_annotations import set_memory_annotation  # noqa: E402
from mydailynews.evaluation.schema import load_corpus  # noqa: E402
from mydailynews.memory.story_retrieval import StoryCandidateMatch  # noqa: E402
from mydailynews.memory.story_store import StoryStore, story_baseline_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic claim/state delta classification.")
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
    with TemporaryDirectory(prefix="mydailynews-claim-delta-eval-") as raw_root:
        for arc in corpus.arcs:
            store = StoryStore(Path(raw_root) / f"{arc.id}.json")
            for day in arc.days:
                expected_by_id = {item.document_id: item for item in day.expectations}
                for document in day.documents:
                    expected = expected_by_id[document.id]
                    story_key = f"oracle:{arc.id}:{expected.canonical_story_id}"
                    prior = next((record for record in store.records() if record.story_key == story_key), None)
                    candidate = document.to_candidate()
                    set_memory_annotation(candidate, MemoryAnnotation(
                        story_key=story_key, story_family_key=f"oracle:{arc.id}",
                        story_title=candidate.title, match_confidence=1.0,
                    ))
                    article = SelectedArticle(
                        candidate=candidate, decision=HeadlineDecision(candidate.id, score=8.0),
                        article_text=document.body, extraction_status="fixture",
                    )
                    baselines = []
                    if prior is not None:
                        match = StoryCandidateMatch(
                            score=1.0, record=prior, lexical_score=1.0, alias_score=1.0,
                            entity_score=1.0, event_score=1.0, fact_score=1.0, numeric_conflict=False,
                        )
                        baselines = [story_baseline_payload(match, max_facts=4)]
                    memory = {"stories": [{
                        "story_key": story_key,
                        "current_title": candidate.title,
                        "current_article_ids": [candidate.id],
                        "prior_baselines": baselines,
                    }]}
                    packet = build_deterministic_delta_scaffold(
                        [article], [], story_memory=memory, story_store=store,
                    )
                    decision = packet["story_decisions"][0]
                    predicted_material = float(decision["materiality"]) >= 0.7
                    rows.append({
                        "arc": arc.id,
                        "document_id": document.id,
                        "slice": "absurd_magic_elf" if "invented_domain" in arc.tags else "standard_news",
                        "is_continuation": expected.relationship == "same_story",
                        "should_select": expected.should_select,
                        "expected_delta": expected.delta_type,
                        "predicted_delta": decision["change_type"],
                        "expected_material": expected.material,
                        "predicted_material": predicted_material,
                        "expected_display": expected.display,
                        "predicted_display": decision["disposition"],
                        "correct_delta": decision["change_type"] == expected.delta_type,
                        "correct_material": predicted_material == expected.material,
                        "correct_display": decision["disposition"] == expected.display,
                        "summary": decision["summary"],
                        "claim_delta": decision.get("claim_delta", {}),
                    })
                    store.update_selected(selected=[article], date=day.date, delta_packet=packet)
    return {
        "kind": "claim_delta_evaluation.v1",
        "note": "Oracle canonical identity supplies the correct prior StoryStore thread so this isolates source-claim delta classification. No expected delta, materiality, display, or fact label is supplied to the classifier.",
        "summary": summarize(rows),
        "mismatches": [row for row in rows if not (row["correct_delta"] and row["correct_material"] and row["correct_display"])],
        "rows": rows,
    }


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
        "delta_accuracy": ratio([row["correct_delta"] for row in rows]),
        "materiality_accuracy": ratio([row["correct_material"] for row in rows]),
        "display_accuracy": ratio([row["correct_display"] for row in rows]),
        "joint_semantic_policy_accuracy": ratio([
            row["correct_delta"] and row["correct_material"] and row["correct_display"] for row in rows
        ]),
    }


def ratio(values: list[bool]) -> float:
    return round(sum(values) / max(1, len(values)), 4)


def render(report: dict) -> str:
    lines = ["# Claim/state delta evaluation", "", report["note"], "", "| Slice | Documents | Delta accuracy | Materiality accuracy | Display accuracy | Joint accuracy |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in report["summary"].items():
        lines.append(f"| {name} | {item['documents']} | {item['delta_accuracy']:.4f} | {item['materiality_accuracy']:.4f} | {item['display_accuracy']:.4f} | {item['joint_semantic_policy_accuracy']:.4f} |")
    lines.extend(["", f"Mismatches: {len(report['mismatches'])}", ""])
    for row in report["mismatches"]:
        lines.append(f"- `{row['document_id']}`: expected {row['expected_delta']}/{row['expected_material']}/{row['expected_display']}; predicted {row['predicted_delta']}/{row['predicted_material']}/{row['predicted_display']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
