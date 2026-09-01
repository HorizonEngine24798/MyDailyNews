from __future__ import annotations

"""Evaluate local pair models on continuation identity, including absurd cases.

Canonical IDs are used only by the scorer to construct historical positive and
hard-negative pairs.  Models receive source text and an explicit task
instruction, never the labels or IDs.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.evaluation.schema import EvalDocument, EvalExpectation, load_corpus  # noqa: E402
from mydailynews.memory.story_retrieval import source_fact_texts  # noqa: E402


@dataclass(frozen=True)
class Pair:
    pair_id: str
    slice_name: str
    prior: EvalDocument
    current: EvalDocument
    same_story: bool
    story_id: str


TASK = "Decide whether the current news report is a continuation of the same real-world story as the prior report. Answer yes only for the same continuing event or development; shared topic, words, people, or location alone are not enough."


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local pair models for story continuation identity.")
    parser.add_argument("--model", choices=("qwen-reranker", "modernbert-nli"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    pairs = build_pairs(load_corpus(args.corpus))
    scorer = qwen_reranker if args.model == "qwen-reranker" else modernbert_nli
    rows = []
    for pair in pairs:
        score = scorer(args.model_path, pair, args.max_length)
        rows.append({
            "pair_id": pair.pair_id,
            "slice": pair.slice_name,
            "same_story": pair.same_story,
            "score": round(float(score), 6),
            "predicted_same_story": bool(score >= 0.5),
        })
    report = {
        "kind": "pair_model_continuation_evaluation.v1",
        "model": args.model,
        "model_path": str(args.model_path),
        "pairs": len(rows),
        "note": "Pairs are scorer-constructed from private canonical labels; models see only source text. Scores measure pairwise continuation identity, not delta classification.",
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


def build_pairs(corpus) -> list[Pair]:
    positives: list[Pair] = []
    all_prior_documents: list[tuple[EvalDocument, str]] = []
    for arc in corpus.arcs:
        by_story: dict[str, list[tuple[EvalDocument, EvalExpectation]]] = {}
        slice_name = "absurd_magic_elf" if "invented_domain" in arc.tags else "standard_news"
        for day in arc.days:
            expectations = {row.document_id: row for row in day.expectations}
            for current in day.documents:
                expected = expectations[current.id]
                previous = by_story.get(expected.canonical_story_id, [])
                if expected.relationship == "same_story" and previous:
                    prior, _ = previous[-1]
                    base_id = f"{arc.id}:{day.date}:{current.id}"
                    positives.append(Pair(base_id + ":positive", slice_name, prior, current, True, expected.canonical_story_id))
                all_prior_documents.append((current, expected.canonical_story_id))
                by_story.setdefault(expected.canonical_story_id, []).append((current, expected))
    pairs: list[Pair] = []
    for positive in positives:
        pairs.append(positive)
        negative_prior, _ = max(
            (item for item in all_prior_documents if item[1] != positive.story_id),
            key=lambda item: lexical_overlap(
                positive.current.title + " " + positive.current.body,
                item[0].title + " " + item[0].body,
            ),
        )
        pairs.append(Pair(
            positive.pair_id.replace(":positive", ":negative"),
            positive.slice_name,
            negative_prior,
            positive.current,
            False,
            positive.story_id,
        ))
    return pairs


def lexical_overlap(left: str, right: str) -> int:
    return len(set(left.casefold().split()).intersection(right.casefold().split()))


def evidence_text(document: EvalDocument) -> str:
    facts = source_fact_texts(document.to_candidate(), source_text=document.body)[:4]
    return "\n".join(f"- {text}" for _, text in facts)


def qwen_reranker(model_path: Path, pair: Pair, max_length: int) -> float:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, tokenizer = _qwen_model(model_path)
    query = f"{TASK}\n\nCURRENT REPORT:\n{evidence_text(pair.current)}"
    document = f"PRIOR REPORT:\n{evidence_text(pair.prior)}"
    instruction = "Classify whether the prior report is the same continuing real-world news story as the current report."
    formatted = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tokens = tokenizer.encode(prefix, add_special_tokens=False) + tokenizer.encode(formatted, add_special_tokens=False)[:max_length] + tokenizer.encode(suffix, add_special_tokens=False)
    inputs = {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long)}
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        yes = logits[0, tokenizer.convert_tokens_to_ids("yes")]
        no = logits[0, tokenizer.convert_tokens_to_ids("no")]
        return float(torch.softmax(torch.stack([no, yes]), dim=0)[1])


_QWEN_CACHE = None


def _qwen_model(model_path: Path):
    global _QWEN_CACHE
    if _QWEN_CACHE is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").eval()
        _QWEN_CACHE = (model, tokenizer)
    return _QWEN_CACHE


_MODERNBERT_CACHE = None


def modernbert_nli(model_path: Path, pair: Pair, max_length: int) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    global _MODERNBERT_CACHE
    if _MODERNBERT_CACHE is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype="auto").eval()
        _MODERNBERT_CACHE = (model, tokenizer)
    model, tokenizer = _MODERNBERT_CACHE
    premise = f"Prior report:\n{evidence_text(pair.prior)}"
    hypothesis = f"The current report continues the same real-world news story. Current report:\n{evidence_text(pair.current)}"
    inputs = tokenizer(premise, hypothesis, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=0)
        return float(probabilities[model.config.label2id["entailment"]])


def summarize(rows: list[dict]) -> dict[str, dict]:
    groups = {"overall": rows, "standard_news": [row for row in rows if row["slice"] == "standard_news"], "absurd_magic_elf": [row for row in rows if row["slice"] == "absurd_magic_elf"]}
    return {name: metrics(group) for name, group in groups.items() if group}


def metrics(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["same_story"]]
    negatives = [row for row in rows if not row["same_story"]]
    return {
        "pairs": len(rows),
        "accuracy_at_0_5": round(sum(row["same_story"] == row["predicted_same_story"] for row in rows) / len(rows), 4),
        "positive_recall_at_0_5": round(sum(row["predicted_same_story"] for row in positives) / max(1, len(positives)), 4),
        "negative_recall_at_0_5": round(sum(not row["predicted_same_story"] for row in negatives) / max(1, len(negatives)), 4),
        "mean_positive_score": round(sum(row["score"] for row in positives) / max(1, len(positives)), 4),
        "mean_negative_score": round(sum(row["score"] for row in negatives) / max(1, len(negatives)), 4),
        "auc": round(auc(positives, negatives), 4),
    }


def auc(positives: list[dict], negatives: list[dict]) -> float:
    if not positives or not negatives:
        return 0.0
    wins = sum(1.0 if positive["score"] > negative["score"] else 0.5 if positive["score"] == negative["score"] else 0.0 for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def render_markdown(report: dict) -> str:
    lines = ["# Pair-model continuation evaluation", "", report["note"], "", "| Slice | Pairs | Accuracy @ 0.5 | Same-story recall | Different-story recall | Mean positive | Mean negative | AUC |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in report["summary"].items():
        lines.append(f"| {name} | {value['pairs']} | {value['accuracy_at_0_5']:.4f} | {value['positive_recall_at_0_5']:.4f} | {value['negative_recall_at_0_5']:.4f} | {value['mean_positive_score']:.4f} | {value['mean_negative_score']:.4f} | {value['auc']:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
