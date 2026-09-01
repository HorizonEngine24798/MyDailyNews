from __future__ import annotations

"""Evaluate heuristic retrieval followed by bounded Qwen pairwise validation."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.ai.qwen_story_reranker import QwenStoryReranker  # noqa: E402
from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, SelectedArticle  # noqa: E402
from mydailynews.domain.candidate_annotations import set_memory_annotation  # noqa: E402
from mydailynews.evaluation.schema import load_corpus  # noqa: E402
from mydailynews.memory.story_reranker import rerank_story_candidates  # noqa: E402
from mydailynews.memory.story_store import StoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate bounded Qwen reranking over StoryStore retrieval.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hard-rejection", action="store_true", help="Reject candidates below the threshold (experimental).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    payload = evaluate(
        args.corpus, QwenStoryReranker(args.model_path), threshold=args.threshold, hard_rejection=args.hard_rejection,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def evaluate(corpus_path: Path, reranker, *, threshold: float, hard_rejection: bool = False) -> dict:
    corpus = load_corpus(corpus_path)
    rows = []
    with TemporaryDirectory(prefix="mydailynews-reranked-story-eval-") as raw_root:
        for arc in corpus.arcs:
            store = StoryStore(Path(raw_root) / f"{arc.id}.json")
            seen_before_day: set[str] = set()
            for day in arc.days:
                expected_by_id = {item.document_id: item for item in day.expectations}
                writeback = []
                for document in day.documents:
                    expected = expected_by_id[document.id]
                    canonical_key = f"oracle:{arc.id}:{expected.canonical_story_id}"
                    candidate = document.to_candidate()
                    heuristic = store.candidate_stories(candidate, source_text=document.body)
                    reranked = rerank_story_candidates(
                        candidate, heuristic, reranker, source_text=document.body, acceptance_threshold=threshold,
                        reject_below_threshold=hard_rejection,
                    )
                    historical_same = expected.relationship == "same_story" and canonical_key in seen_before_day
                    slice_name = "absurd_magic_elf" if "invented_domain" in arc.tags else "standard_news"
                    rows.append({
                        "document_id": document.id,
                        "slice": slice_name,
                        "historical_same_story": historical_same,
                        "heuristic_keys": [item.record.story_key for item in heuristic],
                        "reranked_keys": [item.record.story_key for item in reranked],
                        "reranker_scores": [round(float(item.reranker_score or 0.0), 6) for item in reranked],
                        "expected_key": canonical_key if historical_same else "",
                    })
                    set_memory_annotation(candidate, MemoryAnnotation(
                        story_key=canonical_key,
                        story_family_key=f"oracle:{arc.id}", story_title=candidate.title, match_confidence=1.0,
                    ))
                    writeback.append(SelectedArticle(
                        candidate=candidate, decision=HeadlineDecision(candidate.id, score=8.0),
                        article_text=document.body, extraction_status="fixture",
                    ))
                store.update_selected(selected=writeback, date=day.date)
                seen_before_day.update(
                    f"oracle:{arc.id}:{expected_by_id[item.id].canonical_story_id}" for item in day.documents
                )
    return {
        "kind": "reranked_story_retrieval_evaluation.v1",
        "note": "Gold canonical IDs are used only to write prior-day test history and score outputs. Qwen sees only current source evidence plus heuristic-retrieved StoryStore facts. This evaluates identity routing, not factual delta classification.",
        "threshold": threshold,
        "hard_rejection": hard_rejection,
        "summary": summarize(rows),
        "rows": rows,
    }


def summarize(rows: list[dict]) -> dict:
    groups = {"overall": rows}
    groups.update({name: [row for row in rows if row["slice"] == name] for name in ("standard_news", "absurd_magic_elf")})
    return {name: metrics(group) for name, group in groups.items() if group}


def metrics(rows: list[dict]) -> dict:
    historical = [row for row in rows if row["historical_same_story"]]
    new = [row for row in rows if not row["historical_same_story"]]
    return {
        "documents": len(rows),
        "historical_continuations": len(historical),
        "heuristic_candidate_recall": ratio([row["expected_key"] in row["heuristic_keys"] for row in historical]),
        "reranked_candidate_recall": ratio([row["expected_key"] in row["reranked_keys"] for row in historical]),
        "reranked_correct_top1": ratio([bool(row["reranked_keys"]) and row["reranked_keys"][0] == row["expected_key"] for row in historical]),
        "new_or_unrelated_without_reranked_candidate": ratio([not row["reranked_keys"] for row in new]),
        "mean_heuristic_candidates": round(sum(len(row["heuristic_keys"]) for row in rows) / max(1, len(rows)), 4),
        "mean_reranked_candidates": round(sum(len(row["reranked_keys"]) for row in rows) / max(1, len(rows)), 4),
    }


def ratio(values: list[bool]) -> float:
    return round(sum(values) / max(1, len(values)), 4)


def render(payload: dict) -> str:
    lines = ["# Reranked StoryStore retrieval evaluation", "", payload["note"], "", "| Slice | Docs | Historical continuations | Heuristic recall | Reranked recall | Reranked correct top-1 | New/unrelated rejected | Mean heuristic candidates | Mean reranked candidates |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in payload["summary"].items():
        lines.append(f"| {name} | {item['documents']} | {item['historical_continuations']} | {item['heuristic_candidate_recall']:.4f} | {item['reranked_candidate_recall']:.4f} | {item['reranked_correct_top1']:.4f} | {item['new_or_unrelated_without_reranked_candidate']:.4f} | {item['mean_heuristic_candidates']:.4f} | {item['mean_reranked_candidates']:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
