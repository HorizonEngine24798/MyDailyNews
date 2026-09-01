from __future__ import annotations

"""Optional local Qwen3 reranker adapter.

Transformers and torch are intentionally imported only at scoring time, so a
normal application install does not acquire a heavyweight inference dependency.
"""

from pathlib import Path
from typing import Sequence

from mydailynews.app.models import NewsCandidate
from mydailynews.memory.story_retrieval import StoryCandidateMatch, source_fact_texts


_TASK = (
    "Decide whether the current news report is a continuation of the same real-world "
    "story as the prior report. Answer yes only for the same continuing event or "
    "development; shared topic, words, people, or location alone are not enough."
)


class QwenStoryReranker:
    """Local, lazy Qwen/Qwen3-Reranker adapter for bounded story candidates."""

    def __init__(self, model_path: Path | str, *, max_length: int = 512) -> None:
        self.model_path = Path(model_path)
        self.max_length = max(128, int(max_length))
        self._model = None
        self._tokenizer = None

    def score(
        self,
        candidate: NewsCandidate,
        matches: Sequence[StoryCandidateMatch],
        *,
        source_text: str = "",
    ) -> Sequence[float]:
        import torch

        model, tokenizer = self._load()
        current = _current_evidence(candidate, source_text=source_text)
        results: list[float] = []
        for match in matches:
            prior = _prior_evidence(match)
            tokens = _tokens_for_pair(tokenizer, current=current, prior=prior, max_length=self.max_length)
            inputs = {
                "input_ids": torch.tensor([tokens]),
                "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
            }
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :]
                yes = logits[0, tokenizer.convert_tokens_to_ids("yes")]
                no = logits[0, tokenizer.convert_tokens_to_ids("no")]
                results.append(float(torch.softmax(torch.stack([no, yes]), dim=0)[1]))
        return results

    def _load(self):
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto").eval()
        return self._model, self._tokenizer


def _current_evidence(candidate: NewsCandidate, *, source_text: str) -> str:
    return "\n".join(
        f"- {text}" for _, text in source_fact_texts(candidate, source_text=source_text)[:4]
    )


def _prior_evidence(match: StoryCandidateMatch) -> str:
    facts = list(match.record.facts[-4:])
    rows = [f"- {match.record.title}"]
    rows.extend(f"- {fact.text}" for fact in facts)
    return "\n".join(rows)


def _tokens_for_pair(tokenizer, *, current: str, prior: str, max_length: int) -> list[int]:
    query = f"{_TASK}\n\nCURRENT REPORT:\n{current}"
    document = f"PRIOR REPORT:\n{prior}"
    formatted = (
        "<Instruct>: Classify whether the prior report is the same continuing real-world "
        "news story as the current report.\n<Query>: "
        f"{query}\n<Document>: {document}"
    )
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return (
        tokenizer.encode(prefix, add_special_tokens=False)
        + tokenizer.encode(formatted, add_special_tokens=False)[:max_length]
        + tokenizer.encode(suffix, add_special_tokens=False)
    )
