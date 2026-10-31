from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Collection, Iterable, List


# ``[^\W_]`` means a Unicode letter or digit without treating underscore as a
# word character.  It keeps the matching layer useful for invented names and
# non-English profiles, while remaining deterministic and dependency-free.
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class TokenSimilarity:
    overlap_count: int = 0
    containment: float = 0.0
    jaccard: float = 0.0
    left_novelty: float = 1.0
    right_novelty: float = 1.0
    numeric_conflict: bool = False

    @property
    def confidence(self) -> float:
        if self.numeric_conflict:
            # Distinct numbers are often product versions, quantities, dates,
            # or legal counts.  They are identity evidence, not stopwords.
            return round(min(0.45, max(self.containment * 0.78, self.jaccard)), 4)
        return round(max(self.containment * 0.78, self.jaccard), 4)


def word_tokens(
    text: Any,
    *,
    stopwords: Collection[str] | None = None,
    min_alpha_chars: int = 2,
    keep_numbers: bool = True,
) -> List[str]:
    """Return normalized Unicode word tokens without embedding domain meaning."""

    normalized_stopwords = {str(item).casefold() for item in (stopwords or ())}
    output: List[str] = []
    for match in _WORD_RE.finditer(str(text or "").casefold()):
        token = match.group(0)
        if token in normalized_stopwords:
            continue
        if token.isdecimal():
            if keep_numbers:
                output.append(token)
            continue
        if len(token) < max(1, int(min_alpha_chars)):
            continue
        output.append(token)
    return output


def unique_tokens(tokens: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for raw in tokens:
        token = str(raw or "").strip().casefold()
        if not token or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def compare_token_sets(left: Iterable[str], right: Iterable[str]) -> TokenSimilarity:
    left_set = {str(item).casefold() for item in left if str(item).strip()}
    right_set = {str(item).casefold() for item in right if str(item).strip()}
    if not left_set or not right_set:
        return TokenSimilarity()

    overlap = left_set.intersection(right_set)
    left_numbers = {item for item in left_set if item.isdecimal()}
    right_numbers = {item for item in right_set if item.isdecimal()}
    numeric_conflict = bool(left_numbers and right_numbers and left_numbers.isdisjoint(right_numbers))
    return TokenSimilarity(
        overlap_count=len(overlap),
        containment=round(len(overlap) / max(1, min(len(left_set), len(right_set))), 4),
        jaccard=round(len(overlap) / max(1, len(left_set.union(right_set))), 4),
        left_novelty=round(len(left_set.difference(right_set)) / max(1, len(left_set)), 4),
        right_novelty=round(len(right_set.difference(left_set)) / max(1, len(right_set)), 4),
        numeric_conflict=numeric_conflict,
    )


def normalized_word_text(text: Any) -> str:
    """A stable lexical fingerprint used only for duplicate detection."""

    return " ".join(word_tokens(text, min_alpha_chars=1, keep_numbers=True))
