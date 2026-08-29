"""Offline evaluation contracts for personalized change monitoring."""

from mydailynews.evaluation.runner import evaluate_adapter, write_evaluation_report
from mydailynews.evaluation.schema import EvalCorpus, EvalPrediction, load_corpus

__all__ = [
    "EvalCorpus",
    "EvalPrediction",
    "evaluate_adapter",
    "load_corpus",
    "write_evaluation_report",
]
