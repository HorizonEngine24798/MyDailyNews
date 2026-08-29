from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date as date_type, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from mydailynews.app.models import MemoryAnnotation, NewsCandidate, SelectedArticle
from mydailynews.common.utils import datetime_to_iso
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.domain.text_similarity import compare_token_sets, normalized_word_text, word_tokens
from mydailynews.memory.story_keys import STOPWORDS, StoryIdentity, slugify_text, story_identity_for_candidate
from mydailynews.memory.story_retrieval import (
    DEFAULT_CANDIDATE_THRESHOLD,
    MAX_CANDIDATES,
    StoryCandidateMatch,
    provisional_story_key,
    retrieve_story_candidates,
    source_fact_texts,
    source_signals,
)


STORY_STORE_SCHEMA_VERSION = 1
MATCH_CONFIDENCE_THRESHOLD = 0.58
MAX_FACTS_PER_STORY = 80
STORY_STATUSES = {"active", "stale"}
LEGACY_STORY_FILES = ("story_index.json", "story_ledger.json")


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
class StoryRecord:
    """Canonical identity, lifecycle, semantic state, and source evidence."""

    story_key: str
    story_family_key: str = ""
    title: str = ""
    topic: str = ""
    tokens: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    entity_tokens: List[str] = field(default_factory=list)
    event_tokens: List[str] = field(default_factory=list)
    number_tokens: List[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    status: str = "active"
    last_shown: str = ""
    source_document_ids: List[str] = field(default_factory=list)
    facts: List[SourceFact] = field(default_factory=list)
    last_user_visible_fact_ids: List[str] = field(default_factory=list)
    last_material_change_date: str = ""
    last_change_type: str = ""
    last_materiality: float = 0.0
    last_delta_summary: str = ""
    last_knowns: List[str] = field(default_factory=list)
    last_unknowns: List[str] = field(default_factory=list)
    last_watch_signals: List[str] = field(default_factory=list)
    last_disposition: str = ""
    last_report_id: str = ""


class StoryStore:
    """Single file-backed source of truth for durable story memory.

    `story_store.json` supersedes both legacy story files. If the canonical
    file does not exist, legacy index and ledger rows are merged in memory and
    the next normal write persists the unified representation. Legacy files
    remain untouched as migration backups and are ignored after that write.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        legacy_index_path: Path | str | None = None,
        legacy_ledger_path: Path | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.legacy_index_path = Path(legacy_index_path) if legacy_index_path is not None else None
        self.legacy_ledger_path = Path(legacy_ledger_path) if legacy_ledger_path is not None else None
        self._records: List[StoryRecord] | None = None

    @classmethod
    def from_state_dir(cls, state_dir: Path | str) -> "StoryStore":
        root = Path(state_dir)
        return cls(
            root / "story_store.json",
            legacy_index_path=root / "story_index.json",
            legacy_ledger_path=root / "story_ledger.json",
        )

    @property
    def using_legacy_migration(self) -> bool:
        return not self.path.exists() and any(
            path is not None and path.exists()
            for path in (self.legacy_index_path, self.legacy_ledger_path)
        )

    def records(self) -> List[StoryRecord]:
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
    ) -> List[StoryCandidateMatch]:
        return retrieve_story_candidates(
            candidate,
            self.records(),
            source_text=source_text,
            limit=limit,
            min_score=min_score,
        )

    def match_candidate(self, candidate: NewsCandidate) -> StoryIdentity:
        """Compatibility matcher for callers that only need compact identity."""

        base = story_identity_for_candidate(candidate)
        best: tuple[float, StoryRecord] | None = None
        for record in self.records():
            confidence = _candidate_record_confidence(base, record)
            if confidence < MATCH_CONFIDENCE_THRESHOLD:
                continue
            if best is None or (confidence, record.last_seen) > (best[0], best[1].last_seen):
                best = (confidence, record)
        if best is None:
            return base
        confidence, record = best
        return StoryIdentity(
            story_key=record.story_key,
            story_family_key=record.story_family_key,
            story_title=record.title,
            tokens=base.tokens,
            match_confidence=confidence,
        )

    def update_selected(
        self,
        *,
        selected: List[SelectedArticle],
        date: str,
        visible_article_ids: Iterable[str] = (),
        story_groups: Iterable[Any] | None = None,
        delta_packet: Dict[str, Any] | None = None,
        stale_after_days: int = 7,
        retention_days: int = 30,
    ) -> List[StoryRecord]:
        visible_ids = {
            str(value or "").strip()
            for value in visible_article_ids
            if str(value or "").strip()
        }
        decisions = _decisions_by_article(delta_packet)
        groups = _story_group_by_article_id(story_groups or [])
        updated = {record.story_key: record for record in self.records()}

        for article in selected:
            annotation = candidate_memory_annotation(article.candidate)
            if annotation is None or not annotation.story_key:
                continue
            group = groups.get(str(article.candidate.id))
            if group is not None:
                annotation = _annotation_with_group(annotation, group)
                set_memory_annotation(article.candidate, annotation)

            story_key = annotation.story_key
            previous = updated.get(story_key)
            candidate_identity = story_identity_for_candidate(article.candidate)
            current_signals = source_signals(
                article.candidate,
                source_text=article.article_text or article.candidate.snippet,
            )
            is_visible = str(article.candidate.id) in visible_ids
            current_facts = source_facts_for_article(
                article,
                observed_at=date,
                user_visible=is_visible,
            )
            facts_by_id = {fact.fact_id: fact for fact in (previous.facts if previous else [])}
            for fact in current_facts:
                existing = facts_by_id.get(fact.fact_id)
                if existing is not None:
                    fact = replace(fact, user_visible=existing.user_visible or fact.user_visible)
                facts_by_id[fact.fact_id] = fact

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
            facts = _bounded_fact_history(facts_by_id.values(), protected_fact_ids=visible_fact_ids)
            retained_fact_ids = {fact.fact_id for fact in facts}
            visible_fact_ids = [fact_id for fact_id in visible_fact_ids if fact_id in retained_fact_ids]

            decision = decisions.get(str(article.candidate.id), {})
            semantic = _semantic_baseline_fields(previous, decision, date)
            title = (
                annotation.story_title
                or (previous.title if previous else "")
                or article.candidate.title
            )
            topic = str(article.decision.topic or article.candidate.metadata.get("topic_name", "") or "")
            updated[story_key] = StoryRecord(
                story_key=story_key,
                story_family_key=annotation.story_family_key or (previous.story_family_key if previous else ""),
                title=str(title or "")[:180],
                topic=(topic or (previous.topic if previous else ""))[:120],
                tokens=_merge_tokens(
                    previous.tokens if previous else [],
                    annotation.story_key.split("-"),
                    candidate_identity.tokens,
                    max_items=24,
                ),
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
                status="active",
                last_shown=str(date or "") if is_visible else (previous.last_shown if previous else ""),
                source_document_ids=_merge_strings(
                    previous.source_document_ids if previous else [],
                    [article.candidate.id],
                    max_items=40,
                    max_chars=120,
                ),
                facts=facts,
                last_user_visible_fact_ids=visible_fact_ids[-12:],
                last_materiality=_bounded_float(
                    decision.get("materiality"),
                    previous.last_materiality if previous else 0.0,
                ),
                **semantic,
            )

        records = _refresh_lifecycle_records(
            sorted(updated.values(), key=lambda record: record.story_key),
            as_of_date=date,
            stale_after_days=stale_after_days,
            retention_days=retention_days,
            prune=True,
        )
        return self.replace_records(records)

    def refresh_lifecycle(
        self,
        *,
        as_of_date: str,
        stale_after_days: int = 7,
        retention_days: int = 30,
        prune: bool = False,
    ) -> List[StoryRecord]:
        records = _refresh_lifecycle_records(
            self.records(),
            as_of_date=as_of_date,
            stale_after_days=stale_after_days,
            retention_days=retention_days,
            prune=prune,
        )
        return self.replace_records(records)

    def replace_records(self, records: Iterable[StoryRecord]) -> List[StoryRecord]:
        output = sorted(list(records), key=lambda record: record.story_key)
        keys = [record.story_key for record in output]
        if any(not key for key in keys):
            raise ValueError("Story records require story_key.")
        if len(keys) != len(set(keys)):
            raise ValueError("Story records require unique story keys.")
        self._records = output
        self._write_records(output)
        return list(output)

    def _read_records(self) -> List[StoryRecord]:
        if self.path.exists():
            return _read_story_file(self.path)

        index_records = (
            _read_story_file(self.legacy_index_path)
            if self.legacy_index_path is not None and self.legacy_index_path.exists()
            else []
        )
        ledger_records = (
            _read_story_file(self.legacy_ledger_path)
            if self.legacy_ledger_path is not None and self.legacy_ledger_path.exists()
            else []
        )
        return _merge_legacy_records(index_records, ledger_records)

    def _write_records(self, records: Sequence[StoryRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STORY_STORE_SCHEMA_VERSION,
            "stories": [asdict(record) for record in records],
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


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
        digest = sha256(f"{candidate.id}\n{kind}\n{normalized}".encode("utf-8")).hexdigest()[:20]
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
                tokens=word_tokens(
                    text,
                    stopwords=STOPWORDS,
                    min_alpha_chars=2,
                    keep_numbers=True,
                )[:36],
                user_visible=bool(user_visible),
            )
        )
    return output


def story_baseline_payload(match: StoryCandidateMatch, *, max_facts: int = 6) -> Dict[str, Any]:
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
    knowns = [fact.text for fact in ordered_facts] or list(record.last_knowns)
    return {
        "story_key": record.story_key,
        "story_family_key": record.story_family_key,
        "title": record.title,
        "topic": record.topic,
        "aliases": list(record.aliases[-6:]),
        "first_seen": record.first_seen,
        "last_seen": record.last_seen,
        "status": record.status,
        "last_shown": record.last_shown,
        "last_material_change_date": record.last_material_change_date,
        "last_change_type": record.last_change_type,
        "last_delta_summary": record.last_delta_summary,
        "knowns": knowns,
        "unknowns": list(record.last_unknowns),
        "watch_signals": list(record.last_watch_signals),
        "last_disposition": record.last_disposition,
        "last_report_id": record.last_report_id,
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


def story_record_from_payload(raw: Any) -> StoryRecord | None:
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
    title = str(raw.get("title", "") or "").strip()[:180]
    tokens = _token_list(raw.get("tokens", []), 24)
    aliases = _string_list(raw.get("aliases", []), max_items=16, max_chars=180)
    if not aliases and title:
        aliases = [title]
    event_tokens = _token_list(raw.get("event_tokens", []), 40)
    if not event_tokens:
        event_tokens = list(tokens)
    if not tokens:
        tokens = _merge_tokens(
            _token_list(raw.get("entity_tokens", []), 32),
            event_tokens,
            _token_list(raw.get("number_tokens", []), 20),
            max_items=24,
        )
    status = str(raw.get("status", "active") or "active").strip().lower() or "active"
    if status not in STORY_STATUSES:
        status = "active"
    return StoryRecord(
        story_key=story_key,
        story_family_key=str(raw.get("story_family_key", "") or "").strip(),
        title=title,
        topic=str(raw.get("topic", "") or "").strip()[:120],
        tokens=tokens,
        aliases=aliases,
        entity_tokens=_token_list(raw.get("entity_tokens", []), 32),
        event_tokens=event_tokens,
        number_tokens=_token_list(raw.get("number_tokens", []), 20),
        first_seen=str(raw.get("first_seen", "") or "").strip(),
        last_seen=str(raw.get("last_seen", "") or "").strip(),
        status=status,
        last_shown=str(raw.get("last_shown", "") or "").strip(),
        source_document_ids=_string_list(raw.get("source_document_ids", []), max_items=40, max_chars=120),
        facts=facts[-MAX_FACTS_PER_STORY:],
        last_user_visible_fact_ids=_string_list(
            raw.get("last_user_visible_fact_ids", []),
            max_items=12,
            max_chars=80,
        ),
        last_material_change_date=str(raw.get("last_material_change_date", "") or "").strip(),
        last_change_type=str(raw.get("last_change_type", "") or "").strip(),
        last_materiality=_bounded_float(raw.get("last_materiality"), 0.0),
        last_delta_summary=str(raw.get("last_delta_summary", "") or "").strip()[:400],
        last_knowns=_string_list(raw.get("last_knowns", []), max_items=6, max_chars=180),
        last_unknowns=_string_list(raw.get("last_unknowns", []), max_items=6, max_chars=180),
        last_watch_signals=_string_list(raw.get("last_watch_signals", []), max_items=6, max_chars=180),
        last_disposition=str(raw.get("last_disposition", "") or "").strip(),
        last_report_id=str(raw.get("last_report_id", "") or "").strip(),
    )


def merge_story_records(
    records: Sequence[StoryRecord],
    overrides: Dict[str, Any] | None = None,
) -> StoryRecord:
    if not records:
        raise ValueError("At least one story record is required.")
    latest = max(records, key=lambda record: (record.last_seen, record.story_key))
    raw = dict(overrides or {})
    facts_by_id: Dict[str, SourceFact] = {}
    for record in records:
        for fact in record.facts:
            existing = facts_by_id.get(fact.fact_id)
            facts_by_id[fact.fact_id] = (
                replace(fact, user_visible=True)
                if existing is not None and existing.user_visible and not fact.user_visible
                else fact
            )
    visible_ids = _merge_strings(
        *(record.last_user_visible_fact_ids for record in records),
        max_items=12,
        max_chars=80,
    )
    facts = _bounded_fact_history(facts_by_id.values(), protected_fact_ids=visible_ids)
    active = any(record.status == "active" for record in records)
    override_tokens = _token_list(raw.get("tokens", []), 24)
    requested_status = str(raw.get("status", "active" if active else "stale") or "active").strip().lower()
    status = requested_status if requested_status in STORY_STATUSES else ("active" if active else "stale")
    return replace(
        latest,
        story_key=str(raw.get("story_key", latest.story_key) or latest.story_key).strip(),
        story_family_key=str(raw.get("story_family_key") or latest.story_family_key).strip(),
        title=str(raw.get("title") or latest.title).strip()[:180],
        topic=str(raw.get("topic") or latest.topic).strip()[:120],
        tokens=(
            override_tokens
            if override_tokens
            else _merge_tokens(*(record.tokens for record in records), max_items=24)
        ),
        aliases=_merge_strings(*(record.aliases for record in records), max_items=16, max_chars=180),
        entity_tokens=_merge_tokens(*(record.entity_tokens for record in records), max_items=32),
        event_tokens=_merge_tokens(*(record.event_tokens for record in records), max_items=40),
        number_tokens=_merge_tokens(*(record.number_tokens for record in records), max_items=20),
        first_seen=(
            str(raw.get("first_seen", "") or "").strip()
            or _min_nonempty(record.first_seen for record in records)
        ),
        last_seen=(
            str(raw.get("last_seen", "") or "").strip()
            or _max_nonempty(record.last_seen for record in records)
        ),
        status=status,
        last_shown=_max_nonempty(record.last_shown for record in records),
        source_document_ids=_merge_strings(
            *(record.source_document_ids for record in records),
            max_items=40,
            max_chars=120,
        ),
        facts=facts,
        last_user_visible_fact_ids=[fact_id for fact_id in visible_ids if fact_id in {fact.fact_id for fact in facts}],
    )


def _read_story_file(path: Path) -> List[StoryRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return []
    rows = payload.get("stories", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [record for row in rows for record in [story_record_from_payload(row)] if record is not None]


def _merge_legacy_records(
    index_records: Sequence[StoryRecord],
    ledger_records: Sequence[StoryRecord],
) -> List[StoryRecord]:
    by_key: Dict[str, StoryRecord] = {record.story_key: record for record in index_records}
    for ledger in ledger_records:
        index = by_key.get(ledger.story_key)
        if index is None:
            by_key[ledger.story_key] = ledger
            continue
        merged = merge_story_records([index, ledger])
        # The compact index owns semantic summaries/lifecycle; the ledger owns
        # exact evidence and retrieval signals. Preserve each side's authority.
        by_key[ledger.story_key] = replace(
            merged,
            title=index.title or ledger.title,
            topic=index.topic,
            tokens=index.tokens or ledger.tokens,
            status=index.status,
            last_material_change_date=index.last_material_change_date,
            last_change_type=index.last_change_type or ledger.last_change_type,
            last_delta_summary=index.last_delta_summary,
            last_knowns=index.last_knowns,
            last_unknowns=index.last_unknowns,
            last_watch_signals=index.last_watch_signals,
            last_disposition=index.last_disposition or ledger.last_disposition,
            last_report_id=index.last_report_id,
            last_materiality=ledger.last_materiality,
        )
    return sorted(by_key.values(), key=lambda record: record.story_key)


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
        source_name=str(raw.get("source_name", raw.get("source", "")) or "").strip()[:120],
        source_url=str(raw.get("source_url", raw.get("url", "")) or "").strip()[:500],
        published_at=str(raw.get("published_at", "") or "").strip(),
        observed_at=str(raw.get("observed_at", "") or "").strip(),
        tokens=_token_list(raw.get("tokens", []), 36),
        user_visible=bool(raw.get("user_visible", False)),
    )


def _candidate_record_confidence(base: StoryIdentity, record: StoryRecord) -> float:
    if record.story_key == base.story_key:
        return 1.0
    similarity = compare_token_sets(base.tokens, record.tokens)
    confidence = similarity.confidence
    if similarity.numeric_conflict:
        return confidence
    left = set(base.story_key.split("-"))
    right = set(record.story_key.split("-"))
    overlap = left.intersection(right)
    if len(overlap) >= 2 and len(overlap) / max(1, min(len(left), len(right))) >= 0.4:
        confidence = max(confidence, 0.62)
    return confidence


def _refresh_lifecycle_records(
    records: Sequence[StoryRecord],
    *,
    as_of_date: str,
    stale_after_days: int,
    retention_days: int,
    prune: bool,
) -> List[StoryRecord]:
    as_of = _parse_date(as_of_date)
    if as_of is None:
        return sorted(records, key=lambda record: record.story_key)
    stale_cutoff = as_of - timedelta(days=max(0, int(stale_after_days)))
    prune_cutoff = as_of - timedelta(days=max(0, int(retention_days)))
    refreshed: List[StoryRecord] = []
    for record in records:
        last_seen = _parse_date(record.last_seen)
        if prune and last_seen is not None and last_seen < prune_cutoff:
            continue
        status = "stale" if last_seen is not None and last_seen < stale_cutoff else "active"
        refreshed.append(replace(record, status=status))
    return sorted(refreshed, key=lambda record: record.story_key)


def _semantic_baseline_fields(
    previous: StoryRecord | None,
    decision: Dict[str, Any] | None,
    date: str,
) -> Dict[str, Any]:
    if not decision:
        return {
            "last_material_change_date": previous.last_material_change_date if previous else "",
            "last_change_type": previous.last_change_type if previous else "",
            "last_delta_summary": previous.last_delta_summary if previous else "",
            "last_knowns": list(previous.last_knowns) if previous else [],
            "last_unknowns": list(previous.last_unknowns) if previous else [],
            "last_watch_signals": list(previous.last_watch_signals) if previous else [],
            "last_disposition": previous.last_disposition if previous else "",
            "last_report_id": previous.last_report_id if previous else "",
        }
    change_type = str(decision.get("change_type", "") or "").strip()
    try:
        material = float(decision.get("materiality", 0.0) or 0.0) >= 0.7
    except (TypeError, ValueError):
        material = False
    return {
        "last_material_change_date": (
            str(date or "")
            if material
            else (previous.last_material_change_date if previous else "")
        ),
        "last_change_type": change_type or (previous.last_change_type if previous else ""),
        "last_delta_summary": (
            str(decision.get("summary", "") or decision.get("bullet", "") or "").strip()[:400]
            or (previous.last_delta_summary if previous else "")
        ),
        "last_knowns": _string_list(
            decision.get("knowns", previous.last_knowns if previous else []),
            max_items=6,
            max_chars=180,
        ),
        "last_unknowns": _string_list(
            decision.get("unknowns", previous.last_unknowns if previous else []),
            max_items=6,
            max_chars=180,
        ),
        "last_watch_signals": _string_list(
            decision.get("watch_signals", previous.last_watch_signals if previous else []),
            max_items=6,
            max_chars=180,
        ),
        "last_disposition": str(
            decision.get("disposition", previous.last_disposition if previous else "") or ""
        ).strip(),
        "last_report_id": str(
            decision.get("prior_report_id", previous.last_report_id if previous else "") or ""
        ).strip(),
    }


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


def _story_group_by_article_id(story_groups: Iterable[Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for group in story_groups:
        article_ids = getattr(group, "article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for article_id in article_ids:
            key = str(article_id or "").strip()
            if key and key not in output:
                output[key] = group
    return output


def _annotation_with_group(annotation: MemoryAnnotation, group: Any) -> MemoryAnnotation:
    title = str(getattr(group, "story_title", "") or annotation.story_title).strip()
    topic = str(getattr(group, "topic", "") or "").strip()
    family = annotation.story_family_key
    if topic:
        family = slugify_text(topic, max_tokens=4) or family
    return MemoryAnnotation(
        story_key=annotation.story_key,
        story_family_key=family,
        story_title=title or annotation.story_title,
        match_confidence=annotation.match_confidence,
        recent_coverage_count=annotation.recent_coverage_count,
        recent_lead_count=annotation.recent_lead_count,
        covered_yesterday=annotation.covered_yesterday,
        change_type=annotation.change_type,
        materiality=annotation.materiality,
        score_adjustment=annotation.score_adjustment,
        today_policy=annotation.today_policy,
        reason=annotation.reason,
    )


def _bounded_fact_history(
    facts: Iterable[SourceFact],
    *,
    protected_fact_ids: Iterable[str],
) -> List[SourceFact]:
    ordered = sorted(facts, key=lambda fact: (fact.observed_at, fact.source_id, fact.fact_id))
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
    removable = [fact for fact in selected if fact.fact_id not in protected]
    for protected_fact in missing_protected:
        if not removable:
            break
        selected.remove(removable.pop(0))
        selected.append(protected_fact)
    return sorted(selected, key=lambda fact: (fact.observed_at, fact.source_id, fact.fact_id))


def _merge_tokens(*groups: Iterable[str], max_items: int) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            normalized = str(value or "").strip().casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
    return output[:max_items]


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
    return _merge_tokens((str(item) for item in value), max_items=max_items)


def _parse_date(value: str) -> date_type | None:
    try:
        return date_type.fromisoformat(str(value or "").strip())
    except ValueError:
        return None


def _min_nonempty(values: Iterable[str]) -> str:
    output = sorted(str(value or "").strip() for value in values if str(value or "").strip())
    return output[0] if output else ""


def _max_nonempty(values: Iterable[str]) -> str:
    output = sorted(str(value or "").strip() for value in values if str(value or "").strip())
    return output[-1] if output else ""


def _bounded_float(value: Any, default: float) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return round(max(0.0, min(1.0, float(default))), 4)


__all__ = [
    "DEFAULT_CANDIDATE_THRESHOLD",
    "LEGACY_STORY_FILES",
    "MATCH_CONFIDENCE_THRESHOLD",
    "MAX_FACTS_PER_STORY",
    "SourceFact",
    "StoryCandidateMatch",
    "StoryRecord",
    "StoryStore",
    "merge_story_records",
    "provisional_story_key",
    "story_baseline_payload",
    "story_record_from_payload",
]
