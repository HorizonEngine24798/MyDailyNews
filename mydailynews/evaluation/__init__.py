"""Offline evaluation contracts for personalized change monitoring."""

from mydailynews.evaluation.runner import evaluate_adapter, write_evaluation_report
from mydailynews.evaluation.investigations import build_investigation
from mydailynews.evaluation.retrieval_diagnostics import evaluate_story_ledger_retrieval
from mydailynews.evaluation.schema import EvalCorpus, EvalPrediction, load_corpus

__all__ = [
    "EvalCorpus",
    "EvalPrediction",
    "build_investigation",
    "evaluate_adapter",
    "evaluate_story_ledger_retrieval",
    "load_corpus",
    "write_evaluation_report",
]
