from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Iterable, List, Sequence

from mydailynews.app.models import NewsCandidate
from mydailynews.domain.text_similarity import compare_token_sets, normalized_word_text, word_tokens
from mydailynews.memory.story_keys import STOPWORDS, story_identity_for_candidate, story_tokens_for_candidate


DEFAULT_CANDIDATE_THRESHOLD = 0.25
MAX_CANDIDATES = 3


@dataclass(frozen=True)
class StoryCandidateMatch:
    score: float
    record: Any
    lexical_score: float
    alias_score: float
    entity_score: float
    event_score: float
    fact_score: float
    numeric_conflict: bool
    reasons: List[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "story_key": self.record.story_key,
            "title": self.record.title,
            "score": round(float(self.score), 4),
            "last_seen": self.record.last_seen,
            "source_document_ids": list(self.record.source_document_ids[-4:]),
            "signals": {
                "lexical": self.lexical_score,
                "alias": self.alias_score,
                "entity": self.entity_score,
                "event": self.event_score,
                "fact": self.fact_score,
                "numeric_conflict": self.numeric_conflict,
            },
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SourceSignals:
    normalized_title: str
    all_tokens: List[str]
    entity_tokens: List[str]
    event_tokens: List[str]
    number_tokens: List[str]
    fact_token_sets: List[List[str]]


def retrieve_story_candidates(
    candidate: NewsCandidate,
    records: Iterable[Any],
    *,
    source_text: str = "",
    limit: int = MAX_CANDIDATES,
    min_score: float = DEFAULT_CANDIDATE_THRESHOLD,
) -> List[StoryCandidateMatch]:
    """Return candidates from the current heuristic retriever.

    Older notes call this "hybrid", but it combines hand-weighted lexical
    signals rather than sparse and dense retrieval. Leave it alone while
    evaluation remains healthy; if it starts failing, compare BM25,
    embedding/cosine retrieval, rank fusion, and pairwise reranking.
    """

    signals = source_signals(candidate, source_text=source_text)
    matches = [_candidate_match(candidate, signals, record) for record in records]
    threshold = max(0.0, min(1.0, float(min_score)))
    matches = [match for match in matches if match.score >= threshold]
    matches.sort(
        key=lambda match: (
            match.score,
            match.record.last_seen,
            match.record.story_key,
        ),
        reverse=True,
    )
    return matches[: max(0, min(MAX_CANDIDATES, int(limit)))]


def source_signals(candidate: NewsCandidate, *, source_text: str = "") -> SourceSignals:
    title_tokens = _tokens(candidate.title)
    context_text = " ".join(
        value for value in [candidate.snippet, source_text] if str(value or "").strip()
    )
    context_tokens = _tokens(context_text)
    metadata_tokens = _tokens(
        " ".join(
            [
                str(candidate.category or ""),
                str(candidate.metadata.get("topic_name", "") or ""),
                " ".join(str(tag) for tag in (candidate.tags or [])[:12]),
            ]
        )
    )
    numbers = _dedupe(
        token
        for token in [*title_tokens, *context_tokens]
        if any(character.isdigit() for character in token)
    )
    capitalized = _capitalized_title_tokens(candidate.title)
    recurring = [token for token in title_tokens if token in set(context_tokens)]
    # Topic/category labels are broad recall context, not entity evidence.
    entities = _dedupe([*capitalized, *recurring])[:24]
    entity_set = set(entities)
    number_set = set(numbers)
    events = _dedupe(
        token
        for token in [*title_tokens, *context_tokens[:18]]
        if token not in entity_set and token not in number_set
    )[:28]
    facts = source_fact_texts(candidate, source_text=source_text)
    fact_token_sets = [_tokens(text)[:36] for _, text in facts]
    all_tokens = _dedupe(
        [
            *story_tokens_for_candidate(candidate),
            *context_tokens[:24],
            *metadata_tokens,
        ]
    )[:48]
    return SourceSignals(
        normalized_title=normalized_word_text(candidate.title),
        all_tokens=all_tokens,
        entity_tokens=entities,
        event_tokens=events,
        number_tokens=numbers,
        fact_token_sets=fact_token_sets,
    )


def source_fact_texts(candidate: NewsCandidate, *, source_text: str = "") -> List[tuple[str, str]]:
    rows: List[tuple[str, str]] = []
    title = _clean_source_text(candidate.title, max_chars=280)
    if title:
        rows.append(("headline", title))
    body = _clean_source_text(source_text or candidate.snippet, max_chars=2400)
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", body):
        text = _clean_source_text(sentence, max_chars=420)
        if len(text) < 24:
            continue
        rows.append(("source_sentence", text))
        if len(rows) >= 7:
            break
    output: List[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, text in rows:
        normalized = normalized_word_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append((kind, text))
    return output


def provisional_story_key(
    candidate: NewsCandidate,
    *,
    occupied_story_keys: Iterable[str],
) -> str:
    base = story_identity_for_candidate(candidate).story_key or "story"
    occupied = {str(value or "").strip() for value in occupied_story_keys}
    if base not in occupied:
        return base
    seed = str(candidate.id or candidate.url or candidate.title or base)
    suffix = sha256(seed.encode("utf-8")).hexdigest()[:8]
    candidate_key = f"{base}--{suffix}"
    counter = 2
    while candidate_key in occupied:
        candidate_key = f"{base}--{suffix}-{counter}"
        counter += 1
    return candidate_key


def _candidate_match(
    candidate: NewsCandidate,
    current: SourceSignals,
    record: Any,
) -> StoryCandidateMatch:
    aliases = _dedupe([record.title, *record.aliases])
    record_tokens = _dedupe(
        [
            *record.tokens,
            *record.entity_tokens,
            *record.event_tokens,
            *record.number_tokens,
            *[token for alias in aliases for token in _tokens(alias)],
            *[token for fact in record.facts[-12:] for token in fact.tokens],
        ]
    )[:100]
    lexical = _similarity(current.all_tokens, record_tokens)
    alias = max(
        [_similarity(_tokens(candidate.title), _tokens(value)) for value in aliases]
        or [0.0]
    )
    entity = _similarity(current.entity_tokens, record.entity_tokens)
    event = _similarity(current.event_tokens, record.event_tokens)
    fact = max(
        [
            _similarity(current_fact, stored_fact.tokens)
            for current_fact in current.fact_token_sets
            for stored_fact in record.facts[-16:]
        ]
        or [0.0]
    )
    current_numbers = set(current.number_tokens)
    prior_numbers = set(record.number_tokens)
    numeric_conflict = bool(current_numbers and prior_numbers and current_numbers.isdisjoint(prior_numbers))
    score = (0.25 * lexical) + (0.18 * alias) + (0.24 * entity) + (0.13 * event) + (0.20 * fact)
    reasons: List[str] = []
    base_key = story_identity_for_candidate(candidate).story_key
    if current.normalized_title and current.normalized_title in {
        normalized_word_text(value) for value in aliases
    }:
        score = max(score, 0.98)
        reasons.append("exact_title_alias")
    elif base_key and base_key == record.story_key:
        score = max(score, 0.9)
        reasons.append("exact_generated_story_key")
    if entity >= 0.55 and (event >= 0.25 or fact >= 0.25):
        score += 0.1
        reasons.append("entity_plus_event_or_fact")
    if fact >= 0.48:
        score += 0.08
        reasons.append("source_fact_overlap")
    if alias >= 0.58:
        reasons.append("title_alias_overlap")
    if numeric_conflict:
        score -= 0.15 if entity < 0.8 or fact < 0.5 else 0.05
        reasons.append("numeric_conflict_penalty")
    if max(alias, entity, fact) < 0.28:
        score *= 0.7
    score = round(max(0.0, min(1.0, score)), 4)
    return StoryCandidateMatch(
        score=score,
        record=record,
        lexical_score=round(lexical, 4),
        alias_score=round(alias, 4),
        entity_score=round(entity, 4),
        event_score=round(event, 4),
        fact_score=round(fact, 4),
        numeric_conflict=numeric_conflict,
        reasons=reasons,
    )


def _capitalized_title_tokens(title: str) -> List[str]:
    chunks = re.findall(r"[^\W_][\w'’-]*", str(title or ""), flags=re.UNICODE)
    output: List[str] = []
    for index, chunk in enumerate(chunks):
        normalized = _tokens(chunk)
        if not normalized:
            continue
        has_case_signal = chunk[:1].isupper() and (index > 0 or chunk.isupper())
        has_identifier_signal = any(character.isdigit() for character in chunk)
        if has_case_signal or has_identifier_signal:
            output.extend(normalized)
    return _dedupe(output)


def _similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    return float(compare_token_sets(left, right).confidence)


def _tokens(value: Any) -> List[str]:
    return word_tokens(value, stopwords=STOPWORDS, min_alpha_chars=2, keep_numbers=True)


def _dedupe(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _clean_source_text(value: Any, *, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]
