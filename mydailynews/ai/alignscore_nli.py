from __future__ import annotations

"""Small, dependency-optional adapter for the official AlignScore NLI head.

The upstream AlignScore package targets an old PyTorch Lightning stack.  This
adapter loads only the RoBERTa encoder and three-way NLI head, so MyDailyNews
can use the published weights with its current CPU environment.  It deliberately
does not turn probabilities into story or transition labels; policy remains a
separate, testable layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mydailynews.analysis.claim_delta import (
    BidirectionalEntailmentScore,
    ClaimEvidence,
)


@dataclass(frozen=True)
class NliProbabilities:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def predicted_label(self) -> str:
        values = {
            "entailment": self.entailment,
            "neutral": self.neutral,
            "contradiction": self.contradiction,
        }
        return max(values, key=values.get)


@dataclass(frozen=True)
class BidirectionalNliScore:
    current_to_prior: NliProbabilities
    prior_to_current: NliProbabilities

    def entailment_score(self) -> BidirectionalEntailmentScore:
        return BidirectionalEntailmentScore(
            current_entails_prior=self.current_to_prior.entailment,
            prior_entails_current=self.prior_to_current.entailment,
        )


class AlignScoreNliScorer:
    """Run the official AlignScore three-way head without Lightning.

    ``runtime_checkpoint`` is the compact safetensors artifact produced by
    ``tools/prepare_alignscore_runtime.py``.  ``tokenizer_dir`` must contain a
    local RoBERTa-base config/tokenizer snapshot.  No network calls occur at
    runtime.
    """

    def __init__(
        self,
        runtime_checkpoint: str | Path,
        tokenizer_dir: str | Path,
        *,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        checkpoint = Path(runtime_checkpoint)
        tokenizer_root = Path(tokenizer_dir)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AlignScore runtime checkpoint not found: {checkpoint}")
        if not tokenizer_root.is_dir():
            raise FileNotFoundError(f"AlignScore tokenizer directory not found: {tokenizer_root}")

        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoTokenizer, RobertaModel
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "AlignScore requires the optional CPU evaluation environment "
                "with torch, transformers, and safetensors."
            ) from exc

        config = AutoConfig.from_pretrained(tokenizer_root, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        state = load_file(str(checkpoint), device="cpu")

        encoder = RobertaModel(config)
        encoder_state = {
            key.removeprefix("base_model."): value
            for key, value in state.items()
            if key.startswith("base_model.")
        }
        encoder.load_state_dict(encoder_state, strict=True)

        tri_head = torch.nn.Linear(config.hidden_size, 3)
        tri_state = {
            key.removeprefix("tri_layer."): value
            for key, value in state.items()
            if key.startswith("tri_layer.")
        }
        tri_head.load_state_dict(tri_state, strict=True)
        encoder.eval()
        tri_head.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._encoder = encoder
        self._tri_head = tri_head
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, min(int(max_length), int(tokenizer.model_max_length)))
        self.checkpoint_path = checkpoint
        self.tokenizer_dir = tokenizer_root

    def score(self, current: ClaimEvidence, prior: ClaimEvidence) -> BidirectionalEntailmentScore:
        """Implement the model-agnostic claim scorer protocol."""

        return self.score_bidirectional([current.text], [prior.text])[0].entailment_score()

    def score_bidirectional(
        self,
        current_texts: Sequence[str],
        prior_texts: Sequence[str],
    ) -> list[BidirectionalNliScore]:
        if len(current_texts) != len(prior_texts):
            raise ValueError("current_texts and prior_texts must have equal length")
        if not current_texts:
            return []

        current = [str(value or "") for value in current_texts]
        prior = [str(value or "") for value in prior_texts]
        directed = self.score_pairs(current + prior, prior + current)
        midpoint = len(current)
        return [
            BidirectionalNliScore(
                current_to_prior=directed[index],
                prior_to_current=directed[midpoint + index],
            )
            for index in range(midpoint)
        ]

    def score_pairs(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> list[NliProbabilities]:
        """Return probabilities in the checkpoint's MNLI label order.

        AlignScore uses label 0 for entailment, 1 for neutral, and 2 for
        contradiction.  The premise is the possible supporting text and the
        hypothesis is the proposition being tested.
        """

        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have equal length")
        output: list[NliProbabilities] = []
        for start in range(0, len(premises), self.batch_size):
            premise_batch = [str(value or "") for value in premises[start : start + self.batch_size]]
            hypothesis_batch = [str(value or "") for value in hypotheses[start : start + self.batch_size]]
            encoded = self._tokenizer(
                premise_batch,
                hypothesis_batch,
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self._torch.inference_mode():
                pooled = self._encoder(**encoded).pooler_output
                probabilities = self._torch.softmax(self._tri_head(pooled), dim=-1).cpu()
            output.extend(
                NliProbabilities(
                    entailment=round(float(row[0]), 6),
                    neutral=round(float(row[1]), 6),
                    contradiction=round(float(row[2]), 6),
                )
                for row in probabilities
            )
        return output


__all__ = [
    "AlignScoreNliScorer",
    "BidirectionalNliScore",
    "NliProbabilities",
]
