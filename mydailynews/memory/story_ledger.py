from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

from mydailynews.app.models import NewsCandidate, SelectedArticle
from mydailynews.common.utils import datetime_to_iso
from mydailynews.domain.candidate_annotations import candidate_memory_annotation
from mydailynews.domain.text_similarity import compare_token_sets, normalized_word_text, word_tokens
from mydailynews.memory.story_keys import STOPWORDS, story_identity_for_candidate, story_tokens_for_candidate


STORY_LEDGER_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_THRESHOLD = 0.25
MAX_CANDIDATES = 3
MAX_FACTS_PER_STORY = 80


@dataclass(frozen=True)
class SourceFact:
    fact_id: str
    text: str
    kind: str
    source_id: str
    source_name: str
    source_url: str
    published_at: str
    observed_at: str
    tokens: List[str] = field(default_factory=list)
    user_visible: bool = False


@dataclass(frozen=True)
class StoryLedgerRecord:
    story_key: str
    story_family_key: str
    title: str
    aliases: List[str]
    entity_tokens: List[str]
    event_tokens: List[str]
    number_tokens: List[str]
    first_seen: str
    last_seen: str
    last_shown: str = ""
    source_document_ids: List[str] = field(default_factory=list)
    facts: List[SourceFact] = field(default_factory=list)
    last_user_visible_fact_ids: List[str] = field(default_factory=list)
    last_change_type: str = ""
    last_materiality: float = 0.0
    last_disposition: str = ""


@dataclass(frozen=True)
class StoryCandidateMatch:
    score: float
    record: StoryLedgerRecord
    lexical_score: float
    alias_score: float
    entity_score: float
    event_score: float
    fact_score: float
    numeric_conflict: bool
    reasons: List[str] = field(default_factory=list)

    def metadata(self) -> Dict[str, Any]:
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


class StoryLedgerStore:
    """File-backed source evidence and retrieval signals for durable stories."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records: List[StoryLedgerRecord] | None = None

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "StoryLedgerStore":
        return cls(Path(state_dir) / "story_ledger.json")

    def records(self) -> List[StoryLedgerRecord]:
        if self._records is None:
            self._records = self._read_records()
        return list(self._records)

    def candidate_stories(
        self,
        candidate: NewsCandidate,
        *,
        source_text: str = "",
        limit: int = MAX_CANDIDATES,
        min_score: float = DEFAULT_CANDIDATE_THRESHOLD,
        fallback_records: Iterable[Any] | None = None,
    ) -> List[StoryCandidateMatch]:
        """Return candidates from the current heuristic retriever.

        Older notes call this "hybrid", but it combines hand-weighted lexical
        signals rather than sparse and dense retrieval. Leave it alone while
        evaluation remains healthy; if it starts failing, compare BM25,
        embedding/cosine retrieval, rank fusion, and pairwise reranking.
        """
        signals = source_signals(candidate, source_text=source_text)
        records = {record.story_key: record for record in self.records()}
        for fallback in fallback_records or []:
            converted = _record_from_story_index(fallback)
            if converted is not None and converted.story_key not in records:
                records[converted.story_key] = converted

        matches = [
            _candidate_match(candidate, signals, record)
            for record in records.values()
        ]
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

    def update_selected(
        self,
        *,
        selected: List[SelectedArticle],
        date: str,
        visible_article_ids: Iterable[str] = (),
        delta_packet: Dict[str, Any] | None = None,
    ) -> List[StoryLedgerRecord]:
        visible_ids = {str(value or "").strip() for value in visible_article_ids if str(value or "").strip()}
        decisions = _decisions_by_article(delta_packet)
        updated = {record.story_key: record for record in self.records()}
        for article in selected:
            annotation = candidate_memory_annotation(article.candidate)
            if annotation is None or not annotation.story_key:
                continue
            story_key = annotation.story_key
            previous = updated.get(story_key)
            current_signals = source_signals(
                article.candidate,
                source_text=article.article_text or article.candidate.snippet,
            )
            is_visible = str(article.candidate.id) in visible_ids
            current_facts = source_facts_for_article(article, observed_at=date, user_visible=is_visible)
            facts_by_id = {fact.fact_id: fact for fact in (previous.facts if previous else [])}
            for fact in current_facts:
                existing = facts_by_id.get(fact.fact_id)
                if existing is not None:
                    # Visibility is monotonic evidence: processing the same
                    # source again in a brief where it is omitted must not erase
                    # the fact that the user already saw it elsewhere.
                    fact = replace(fact, user_visible=existing.user_visible or fact.user_visible)
                facts_by_id[fact.fact_id] = fact
            decision = decisions.get(str(article.candidate.id), {})
            if is_visible:
                same_day_visible = (
                    previous.last_user_visible_fact_ids
                    if previous is not None and previous.last_shown == str(date or "")
                    else []
                )
                visible_fact_ids = _merge_strings(
                    same_day_visible,
                    [fact.fact_id for fact in current_facts],
                    max_items=12,
                    max_chars=80,
                )
            else:
                visible_fact_ids = list(previous.last_user_visible_fact_ids) if previous else []
            facts = _bounded_fact_history(
                facts_by_id.values(),
                protected_fact_ids=visible_fact_ids,
            )
            retained_fact_ids = {fact.fact_id for fact in facts}
            visible_fact_ids = [
                fact_id for fact_id in visible_fact_ids if fact_id in retained_fact_ids
            ]
            updated[story_key] = StoryLedgerRecord(
                story_key=story_key,
                story_family_key=(
                    annotation.story_family_key
                    or (previous.story_family_key if previous else "")
                ),
                title=str(article.candidate.title or annotation.story_title or (previous.title if previous else ""))[:180],
                aliases=_merge_strings(
                    previous.aliases if previous else [],
                    [article.candidate.title, annotation.story_title],
                    max_items=16,
                    max_chars=180,
                ),
                entity_tokens=_merge_tokens(
                    previous.entity_tokens if previous else [],
                    current_signals.entity_tokens,
                    max_items=32,
                ),
                event_tokens=_merge_tokens(
                    previous.event_tokens if previous else [],
                    current_signals.event_tokens,
                    max_items=40,
                ),
                number_tokens=_merge_tokens(
                    previous.number_tokens if previous else [],
                    current_signals.number_tokens,
                    max_items=20,
                ),
                first_seen=previous.first_seen if previous else str(date or ""),
                last_seen=str(date or ""),
                last_shown=str(date or "") if is_visible else (previous.last_shown if previous else ""),
                source_document_ids=_merge_strings(
                    previous.source_document_ids if previous else [],
                    [article.candidate.id],
                    max_items=40,
                    max_chars=120,
                ),
                facts=facts,
                last_user_visible_fact_ids=visible_fact_ids[-12:],
                last_change_type=str(decision.get("change_type", "") or (previous.last_change_type if previous else "")),
                last_materiality=_bounded_float(
                    decision.get("materiality"),
                    previous.last_materiality if previous else 0.0,
                ),
                last_disposition=str(decision.get("disposition", "") or (previous.last_disposition if previous else "")),
            )

        records = sorted(updated.values(), key=lambda record: record.story_key)
        self._records = records
        self._write_records(records)
        return records

    def _read_records(self) -> List[StoryLedgerRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            return []
        rows = payload.get("stories", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []
        output: List[StoryLedgerRecord] = []
        for row in rows:
            record = _ledger_record_from_payload(row)
            if record is not None:
                output.append(record)
        return output

    def _write_records(self, records: List[StoryLedgerRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STORY_LEDGER_SCHEMA_VERSION,
            "stories": [asdict(record) for record in records],
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


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
    # Topic/category labels are useful broad recall context, but they are not
    # entities. Treating labels such as "world" or "synthetic" as entity
    # evidence causes unrelated stories in the same feed to overmatch.
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


def source_facts_for_article(
    article: SelectedArticle,
    *,
    observed_at: str,
    user_visible: bool,
) -> List[SourceFact]:
    candidate = article.candidate
    published_at = datetime_to_iso(candidate.published_at)
    output: List[SourceFact] = []
    for kind, text in source_fact_texts(
        candidate,
        source_text=article.article_text or candidate.snippet,
    ):
        normalized = normalized_word_text(text)
        digest = sha256(
            f"{candidate.id}\n{kind}\n{normalized}".encode("utf-8")
        ).hexdigest()[:20]
        output.append(
            SourceFact(
                fact_id=f"fact:{digest}",
                text=text,
                kind=kind,
                source_id=str(candidate.id or ""),
                source_name=str(candidate.source or "")[:120],
                source_url=str(candidate.url or "")[:500],
                published_at=published_at,
                observed_at=str(observed_at or ""),
                tokens=_tokens(text)[:36],
                user_visible=bool(user_visible),
            )
        )
    return output


def ledger_baseline_payload(match: StoryCandidateMatch, *, max_facts: int = 6) -> Dict[str, Any]:
    record = match.record
    visible_ids = set(record.last_user_visible_fact_ids)
    ordered_facts = sorted(
        record.facts,
        key=lambda fact: (
            fact.fact_id in visible_ids,
            fact.user_visible,
            fact.observed_at,
        ),
        reverse=True,
    )[: max(0, int(max_facts))]
    return {
        "story_key": record.story_key,
        "story_family_key": record.story_family_key,
        "title": record.title,
        "aliases": list(record.aliases[-6:]),
        "first_seen": record.first_seen,
        "last_seen": record.last_seen,
        "last_shown": record.last_shown,
        "last_change_type": record.last_change_type,
        "knowns": [fact.text for fact in ordered_facts],
        "source_facts": [
            {
                "fact_id": fact.fact_id,
                "text": fact.text,
                "kind": fact.kind,
                "source_id": fact.source_id,
                "source": fact.source_name,
                "url": fact.source_url,
                "published_at": fact.published_at,
                "observed_at": fact.observed_at,
                "user_visible": fact.user_visible,
            }
            for fact in ordered_facts
        ],
        "candidate_score": round(float(match.score), 4),
        "candidate_signals": match.metadata()["signals"],
        "candidate_reasons": list(match.reasons),
    }


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
    record: StoryLedgerRecord,
) -> StoryCandidateMatch:
    record_tokens = _dedupe(
        [
            *record.entity_tokens,
            *record.event_tokens,
            *record.number_tokens,
            *[token for alias in record.aliases for token in _tokens(alias)],
            *[token for fact in record.facts[-12:] for token in fact.tokens],
        ]
    )[:100]
    lexical = _similarity(current.all_tokens, record_tokens)
    alias = max(
        [_similarity(_tokens(candidate.title), _tokens(value)) for value in record.aliases]
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
        normalized_word_text(value) for value in record.aliases
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


def _ledger_record_from_payload(raw: Any) -> StoryLedgerRecord | None:
    if not isinstance(raw, dict):
        return None
    story_key = str(raw.get("story_key", "") or "").strip()
    if not story_key:
        return None
    raw_facts = raw.get("facts", [])
    facts = (
        [fact for item in raw_facts for fact in [_fact_from_payload(item)] if fact is not None]
        if isinstance(raw_facts, list)
        else []
    )
    return StoryLedgerRecord(
        story_key=story_key,
        story_family_key=str(raw.get("story_family_key", "") or "").strip(),
        title=str(raw.get("title", "") or "").strip()[:180],
        aliases=_string_list(raw.get("aliases", []), max_items=16, max_chars=180),
        entity_tokens=_token_list(raw.get("entity_tokens", []), 32),
        event_tokens=_token_list(raw.get("event_tokens", []), 40),
        number_tokens=_token_list(raw.get("number_tokens", []), 20),
        first_seen=str(raw.get("first_seen", "") or "").strip(),
        last_seen=str(raw.get("last_seen", "") or "").strip(),
        last_shown=str(raw.get("last_shown", "") or "").strip(),
        source_document_ids=_string_list(raw.get("source_document_ids", []), max_items=40, max_chars=120),
        facts=facts[-MAX_FACTS_PER_STORY:],
        last_user_visible_fact_ids=_string_list(raw.get("last_user_visible_fact_ids", []), max_items=12, max_chars=80),
        last_change_type=str(raw.get("last_change_type", "") or "").strip(),
        last_materiality=_bounded_float(raw.get("last_materiality"), 0.0),
        last_disposition=str(raw.get("last_disposition", "") or "").strip(),
    )


def _fact_from_payload(raw: Any) -> SourceFact | None:
    if not isinstance(raw, dict):
        return None
    fact_id = str(raw.get("fact_id", "") or "").strip()
    text = str(raw.get("text", "") or "").strip()
    source_id = str(raw.get("source_id", "") or "").strip()
    if not fact_id or not text or not source_id:
        return None
    return SourceFact(
        fact_id=fact_id,
        text=text[:420],
        kind=str(raw.get("kind", "source_sentence") or "source_sentence").strip(),
        source_id=source_id,
        source_name=str(raw.get("source_name", "") or "").strip()[:120],
        source_url=str(raw.get("source_url", "") or "").strip()[:500],
        published_at=str(raw.get("published_at", "") or "").strip(),
        observed_at=str(raw.get("observed_at", "") or "").strip(),
        tokens=_token_list(raw.get("tokens", []), 36),
        user_visible=bool(raw.get("user_visible", False)),
    )


def _record_from_story_index(raw: Any) -> StoryLedgerRecord | None:
    story_key = str(getattr(raw, "story_key", "") or "").strip()
    if not story_key:
        return None
    title = str(getattr(raw, "title", "") or "").strip()
    tokens = _token_list(getattr(raw, "tokens", []), 40)
    return StoryLedgerRecord(
        story_key=story_key,
        story_family_key=str(getattr(raw, "story_family_key", "") or "").strip(),
        title=title,
        aliases=[title] if title else [],
        entity_tokens=[],
        event_tokens=tokens,
        number_tokens=[token for token in tokens if any(char.isdigit() for char in token)],
        first_seen=str(getattr(raw, "first_seen", "") or "").strip(),
        last_seen=str(getattr(raw, "last_seen", "") or "").strip(),
        last_shown="",
        source_document_ids=[],
        facts=[],
        last_user_visible_fact_ids=[],
        last_change_type=str(getattr(raw, "last_change_type", "") or "").strip(),
        last_materiality=0.0,
        last_disposition=str(getattr(raw, "last_disposition", "") or "").strip(),
    )


def _decisions_by_article(delta_packet: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    rows = delta_packet.get("story_decisions", []) if isinstance(delta_packet, dict) else []
    if not isinstance(rows, list):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        article_ids = row.get("article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for value in article_ids:
            article_id = str(value or "").strip()
            if article_id:
                output[article_id] = row
    return output


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


def _merge_tokens(*groups: Iterable[str], max_items: int) -> List[str]:
    return _dedupe(value for group in groups for value in group)[:max_items]


def _bounded_fact_history(
    facts: Iterable[SourceFact],
    *,
    protected_fact_ids: Iterable[str],
) -> List[SourceFact]:
    ordered = sorted(
        facts,
        key=lambda fact: (fact.observed_at, fact.source_id, fact.fact_id),
    )
    if len(ordered) <= MAX_FACTS_PER_STORY:
        return ordered
    protected = {str(value or "").strip() for value in protected_fact_ids}
    by_id = {fact.fact_id: fact for fact in ordered}
    selected = ordered[-MAX_FACTS_PER_STORY:]
    selected_ids = {fact.fact_id for fact in selected}
    missing_protected = [
        by_id[fact_id]
        for fact_id in protected
        if fact_id in by_id and fact_id not in selected_ids
    ]
    if not missing_protected:
        return selected
    removable = [fact for fact in selected if fact.fact_id not in protected]
    for protected_fact in missing_protected:
        if not removable:
            break
        removed = removable.pop(0)
        selected.remove(removed)
        selected.append(protected_fact)
    return sorted(
        selected,
        key=lambda fact: (fact.observed_at, fact.source_id, fact.fact_id),
    )


def _merge_strings(
    *groups: Iterable[Any],
    max_items: int,
    max_chars: int,
) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            text = " ".join(str(value or "").split()).strip()[:max_chars]
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            output.append(text)
    return output[-max_items:]


def _string_list(value: Any, *, max_items: int, max_chars: int) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return _merge_strings(value, max_items=max_items, max_chars=max_chars)


def _token_list(value: Any, max_items: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return _dedupe(str(item) for item in value)[:max_items]


def _clean_source_text(value: Any, *, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def _bounded_float(value: Any, default: float) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return round(max(0.0, min(1.0, float(default))), 4)
