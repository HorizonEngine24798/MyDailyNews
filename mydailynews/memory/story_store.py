from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date as date_type, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from mydailynews.analysis.claim_delta import FACT_OPERATIONS
from mydailynews.app.models import MemoryAnnotation, NewsCandidate, SelectedArticle
from mydailynews.common.utils import datetime_to_iso
from mydailynews.common.storage import (
    MEMORY_DATABASE_NAME,
    claim_migration,
    metadata_value,
    open_database,
    set_metadata,
)
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


STORY_STORE_SCHEMA_VERSION = 6
MATCH_CONFIDENCE_THRESHOLD = 0.58
# Source facts are an evidence cache, not a transcript archive.  The last
# user-visible facts are always protected; the remaining slots retain the most
# recent distinct evidence for retrieval and a bounded delta comparison.
MAX_FACTS_PER_STORY = 32
MAX_THREAD_EVENTS_PER_STORY = 24
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
class StoryThreadEvent:
    event_id: str
    observed_at: str
    article_ids: List[str] = field(default_factory=list)
    relationship: str = ""
    change_type: str = ""
    materiality: float = 0.0
    disposition: str = ""
    summary: str = ""
    current_evidence_ids: List[str] = field(default_factory=list)
    prior_evidence_ids: List[str] = field(default_factory=list)
    superseded_prior_evidence_ids: List[str] = field(default_factory=list)
    operations: List[Dict[str, Any]] = field(default_factory=list)


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
    active_fact_ids: List[str] = field(default_factory=list)
    thread_events: List[StoryThreadEvent] = field(default_factory=list)
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
    """SQLite-backed source of truth for durable story memory.

    Existing JSON stores are imported once and left untouched as migration
    backups. ``path`` remains the legacy path for caller compatibility;
    ``database_path`` is the live store.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        legacy_index_path: Path | str | None = None,
        legacy_ledger_path: Path | str | None = None,
        database_path: Path | str | None = None,
    ) -> None:
        self.path = Path(path)
        is_database_path = self.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
        self.database_path = Path(database_path) if database_path is not None else (
            self.path if is_database_path else self.path.with_suffix(".sqlite3")
        )
        self._legacy_store_path = None if self.database_path == self.path else self.path
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
            database_path=root / MEMORY_DATABASE_NAME,
        )

    @property
    def using_legacy_migration(self) -> bool:
        legacy_exists = any(
            path is not None and path.exists()
            for path in (self._legacy_store_path, self.legacy_index_path, self.legacy_ledger_path)
        )
        if not legacy_exists:
            return False
        if not self.database_path.exists():
            return True
        with open_database(self.database_path) as database:
            return metadata_value(database, "migration.story_store.v1") is None

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
        database: Any | None = None,
    ) -> List[StoryRecord]:
        visible_ids = {
            str(value or "").strip()
            for value in visible_article_ids
            if str(value or "").strip()
        }
        decisions = _decisions_by_article(delta_packet)
        validated_operation_packet = bool(
            isinstance(delta_packet, dict)
            and delta_packet.get("story_delta_version") in {"story-cards.v1", "story-cards.v2"}
        )
        groups = _story_group_by_article_id(story_groups or [])
        current_facts_by_article_id = {
            str(article.candidate.id): source_facts_for_article(
                article,
                observed_at=date,
                user_visible=str(article.candidate.id) in visible_ids,
            )
            for article in selected
        }
        existing_records = (
            self._read_records_from_database(database)
            if database is not None
            else self.records()
        )
        updated = {record.story_key: record for record in existing_records}

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
            current_facts = current_facts_by_article_id.get(str(article.candidate.id), [])
            decision = decisions.get(str(article.candidate.id), {})
            decision_article_ids = _string_list(
                decision.get("article_ids", [article.candidate.id]),
                max_items=max(1, len(current_facts_by_article_id)),
                max_chars=120,
            )
            operation_current_facts = [
                fact
                for article_id in decision_article_ids
                for fact in current_facts_by_article_id.get(article_id, [])
            ] or current_facts
            facts_by_id = {fact.fact_id: fact for fact in (previous.facts if previous else [])}
            for fact in operation_current_facts:
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
            claim_delta = (
                decision.get("claim_delta", {})
                if isinstance(decision.get("claim_delta"), dict)
                else {}
            )
            semantic_fact_ids = _merge_strings(
                decision.get("current_evidence_ids", []),
                decision.get("prior_evidence_ids", []),
                decision.get("superseded_prior_evidence_ids", []),
                claim_delta.get("current_evidence_ids", []),
                claim_delta.get("prior_evidence_ids", []),
                claim_delta.get("superseded_prior_evidence_ids", []),
                max_items=20,
                max_chars=80,
            )
            active_fact_ids = _next_active_fact_ids(
                previous,
                operation_current_facts,
                decision,
                operations_validated=(
                    validated_operation_packet
                    and decision.get("editor_safe_to_defer") is True
                ),
            )
            protected_fact_ids = _merge_strings(
                visible_fact_ids,
                semantic_fact_ids,
                max_items=MAX_FACTS_PER_STORY,
                max_chars=80,
            )
            facts = _bounded_fact_history(
                facts_by_id.values(),
                protected_fact_ids=protected_fact_ids,
            )
            retained_fact_ids = {fact.fact_id for fact in facts}
            visible_fact_ids = [fact_id for fact_id in visible_fact_ids if fact_id in retained_fact_ids]
            active_fact_id_set = set(active_fact_ids)
            active_fact_ids = [
                fact.fact_id for fact in facts if fact.fact_id in active_fact_id_set
            ]

            semantic = _semantic_baseline_fields(previous, decision, date)
            thread_events = _updated_thread_events(
                previous.thread_events if previous else [],
                decision,
                article_id=str(article.candidate.id or ""),
                observed_at=str(date or ""),
            )
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
                active_fact_ids=active_fact_ids,
                thread_events=thread_events,
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
        return self.replace_records(records, database=database)

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

    def replace_records(
        self,
        records: Iterable[StoryRecord],
        *,
        database: Any | None = None,
    ) -> List[StoryRecord]:
        output = sorted(list(records), key=lambda record: record.story_key)
        keys = [record.story_key for record in output]
        if any(not key for key in keys):
            raise ValueError("Story records require story_key.")
        if len(keys) != len(set(keys)):
            raise ValueError("Story records require unique story keys.")
        if database is None:
            self._records = output
        self._write_records(output, database=database)
        return list(output)

    def _read_records(self) -> List[StoryRecord]:
        self._import_legacy_once()
        with open_database(self.database_path) as database:
            return self._read_records_from_database(database)

    @staticmethod
    def _read_records_from_database(database: Any) -> List[StoryRecord]:
        output: List[StoryRecord] = []
        rows = database.execute("SELECT story_key, payload FROM stories ORDER BY story_key").fetchall()
        for row in rows:
            record = story_record_from_payload(_json_object(row["payload"]))
            if record is None or record.story_key != str(row["story_key"]):
                raise ValueError(f"SQLite stories row has an invalid payload: {row['story_key']}")
            output.append(record)
        return output

    def _write_records(
        self,
        records: Sequence[StoryRecord],
        *,
        database: Any | None = None,
    ) -> None:
        if database is None:
            with open_database(self.database_path) as opened:
                self._write_records(records, database=opened)
            return
        database.execute("DELETE FROM stories")
        database.executemany(
            "INSERT INTO stories(story_key, payload) VALUES (?, ?)",
            [
                (
                    record.story_key,
                    json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")),
                )
                for record in records
            ],
        )
        set_metadata(database, "migration.story_store.v1")
        set_metadata(database, "schema.story_store", str(STORY_STORE_SCHEMA_VERSION))

    def _import_legacy_once(self) -> None:
        with open_database(self.database_path) as database:
            if not claim_migration(database, "migration.story_store.v1"):
                return
            existing = database.execute("SELECT 1 FROM stories LIMIT 1").fetchone()
            records: Sequence[StoryRecord] = []
            if existing is None:
                if self._legacy_store_path is not None and self._legacy_store_path.exists():
                    records = _read_story_file(self._legacy_store_path)
                else:
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
                    records = _merge_legacy_records(index_records, ledger_records)
                database.executemany(
                    "INSERT OR IGNORE INTO stories(story_key, payload) VALUES (?, ?)",
                    [
                        (
                            record.story_key,
                            json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")),
                        )
                        for record in records
                    ],
                )
            set_metadata(database, "migration.story_store.v1")
            set_metadata(database, "schema.story_store", str(STORY_STORE_SCHEMA_VERSION))


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
        # Use the normalized source claim rather than the article ID.  Wire
        # rewrites and syndications otherwise consume a new durable slot each
        # time despite adding no evidence to the story state.
        digest = sha256(f"{kind}\n{normalized}".encode("utf-8")).hexdigest()[:20]
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
    active_ids = set(record.active_fact_ids)
    ordered_facts = sorted(
        (fact for fact in record.facts if fact.fact_id in active_ids),
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
        "thread_events": [
            {
                "event_id": event.event_id,
                "observed_at": event.observed_at,
                "article_ids": list(event.article_ids),
                "relationship": event.relationship,
                "change_type": event.change_type,
                "materiality": event.materiality,
                "disposition": event.disposition,
                "summary": event.summary,
                "current_evidence_ids": list(event.current_evidence_ids),
                "prior_evidence_ids": list(event.prior_evidence_ids),
                "superseded_prior_evidence_ids": list(event.superseded_prior_evidence_ids),
                "operations": [dict(item) for item in event.operations],
            }
            for event in record.thread_events[-4:]
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
    raw_events = raw.get("thread_events", [])
    thread_events = (
        [event for item in raw_events for event in [_thread_event_from_payload(item)] if event is not None]
        if isinstance(raw_events, list)
        else []
    )[-MAX_THREAD_EVENTS_PER_STORY:]
    visible_fact_ids = _string_list(
        raw.get("last_user_visible_fact_ids", []),
        max_items=12,
        max_chars=80,
    )
    active_fact_ids = (
        _string_list(
            raw.get("active_fact_ids", []),
            max_items=MAX_FACTS_PER_STORY,
            max_chars=80,
        )
        if "active_fact_ids" in raw
        else [fact.fact_id for fact in facts]
    )
    facts = _bounded_fact_history(
        facts,
        protected_fact_ids=visible_fact_ids,
    )
    retained_fact_ids = {fact.fact_id for fact in facts}
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
        active_fact_ids=[
            fact_id for fact_id in active_fact_ids if fact_id in retained_fact_ids
        ],
        thread_events=thread_events,
        last_user_visible_fact_ids=[
            fact_id for fact_id in visible_fact_ids if fact_id in retained_fact_ids
        ],
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
    requested_active_ids = (
        _string_list(
            raw.get("active_fact_ids", []),
            max_items=MAX_FACTS_PER_STORY,
            max_chars=80,
        )
        if "active_fact_ids" in raw
        else _merge_strings(
            *(record.active_fact_ids for record in records),
            max_items=MAX_FACTS_PER_STORY,
            max_chars=80,
        )
    )
    facts = _bounded_fact_history(
        facts_by_id.values(),
        protected_fact_ids=visible_ids,
    )
    retained_fact_ids = {fact.fact_id for fact in facts}
    thread_events = _merge_thread_events(*(record.thread_events for record in records))
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
        active_fact_ids=[
            fact_id for fact_id in requested_active_ids if fact_id in retained_fact_ids
        ],
        thread_events=thread_events,
        last_user_visible_fact_ids=[fact_id for fact_id in visible_ids if fact_id in retained_fact_ids],
    )


def _read_story_file(path: Path) -> List[StoryRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise RuntimeError(f"Could not read legacy story memory file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Legacy story memory file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy story memory file must contain an object: {path}")
    rows = payload.get("stories", [])
    if not isinstance(rows, list):
        raise ValueError(f"Legacy story memory file must contain a stories list: {path}")
    records: List[StoryRecord] = []
    for index, row in enumerate(rows, start=1):
        record = story_record_from_payload(row) if isinstance(row, dict) else None
        if record is None:
            raise ValueError(f"Legacy story memory file has an invalid story at row {index}: {path}")
        records.append(record)
    return records


def _json_object(value: Any) -> Dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _thread_event_from_payload(raw: Any) -> StoryThreadEvent | None:
    if not isinstance(raw, dict):
        return None
    event_id = str(raw.get("event_id", "") or "").strip()
    observed_at = str(raw.get("observed_at", "") or "").strip()
    if not event_id or not observed_at:
        return None
    return StoryThreadEvent(
        event_id=event_id,
        observed_at=observed_at,
        article_ids=_string_list(raw.get("article_ids", []), max_items=8, max_chars=120),
        relationship=str(raw.get("relationship", "") or "").strip(),
        change_type=str(raw.get("change_type", "") or "").strip(),
        materiality=_bounded_float(raw.get("materiality"), 0.0),
        disposition=str(raw.get("disposition", "") or "").strip(),
        summary=str(raw.get("summary", "") or "").strip()[:400],
        current_evidence_ids=_string_list(raw.get("current_evidence_ids", []), max_items=12, max_chars=80),
        prior_evidence_ids=_string_list(raw.get("prior_evidence_ids", []), max_items=12, max_chars=80),
        superseded_prior_evidence_ids=_string_list(
            raw.get("superseded_prior_evidence_ids", []),
            max_items=12,
            max_chars=80,
        ),
        operations=_operation_list(raw.get("operations", [])),
    )


def _updated_thread_events(
    previous: Sequence[StoryThreadEvent],
    decision: Dict[str, Any] | None,
    *,
    article_id: str,
    observed_at: str,
) -> List[StoryThreadEvent]:
    if not decision:
        return list(previous[-MAX_THREAD_EVENTS_PER_STORY:])
    relationship = str(decision.get("relationship", "") or "").strip()
    change_type = str(decision.get("change_type", "") or "").strip()
    summary = str(decision.get("summary", "") or decision.get("reason", "") or "").strip()[:400]
    event_article_ids = _string_list(
        decision.get("article_ids", [article_id] if article_id else []),
        max_items=8,
        max_chars=120,
    )
    if not event_article_ids and article_id:
        event_article_ids = [article_id]
    digest = sha256(
        (
            f"{observed_at}\n{','.join(event_article_ids)}\n{relationship}\n"
            f"{change_type}\n{normalized_word_text(summary)}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    event = StoryThreadEvent(
        event_id=f"event:{digest}",
        observed_at=observed_at,
        article_ids=event_article_ids,
        relationship=relationship,
        change_type=change_type,
        materiality=_bounded_float(decision.get("materiality"), 0.0),
        disposition=str(decision.get("disposition", "") or "").strip(),
        summary=summary,
        current_evidence_ids=_string_list(
            decision.get("current_evidence_ids", []),
            max_items=12,
            max_chars=80,
        ),
        prior_evidence_ids=_string_list(
            decision.get("prior_evidence_ids", []),
            max_items=12,
            max_chars=80,
        ),
        superseded_prior_evidence_ids=_string_list(
            decision.get("superseded_prior_evidence_ids", []),
            max_items=12,
            max_chars=80,
        ),
        operations=_operation_list(decision.get("operations", [])),
    )
    return _merge_thread_events(previous, [event])


def _merge_thread_events(*groups: Iterable[StoryThreadEvent]) -> List[StoryThreadEvent]:
    by_id: Dict[str, StoryThreadEvent] = {}
    for group in groups:
        for event in group:
            by_id[event.event_id] = event
    return sorted(
        by_id.values(),
        key=lambda event: (event.observed_at, event.event_id),
    )[-MAX_THREAD_EVENTS_PER_STORY:]


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


def _next_active_fact_ids(
    previous: StoryRecord | None,
    current_facts: Sequence[SourceFact],
    decision: Dict[str, Any] | None,
    *,
    operations_validated: bool,
) -> List[str]:
    previous_fact_ids = {fact.fact_id for fact in previous.facts} if previous else set()
    active = set(previous.active_fact_ids) if previous else set()
    current_fact_ids = {fact.fact_id for fact in current_facts}
    fail_open = active.union(current_fact_ids)
    if not operations_validated or not isinstance(decision, dict):
        return sorted(fail_open)

    operations = _operation_list(decision.get("operations", []))
    operation_current_ids = {
        str(operation.get("current_evidence_id", "") or "")
        for operation in operations
    }
    expected_current_ids = set(
        _string_list(
            decision.get("current_evidence_ids", []),
            max_items=16,
            max_chars=80,
        )
    )
    references_are_valid = bool(operations) and all(
        operation.get("operation") != "uncertain"
        and str(operation.get("current_evidence_id", "") or "") in current_fact_ids
        and all(
            prior_id in previous_fact_ids
            for prior_id in operation.get("prior_fact_ids", [])
        )
        and (
            bool(operation.get("prior_fact_ids", []))
            if operation.get("operation") in {"repeat", "replace", "resolve"}
            else not operation.get("prior_fact_ids", [])
        )
        for operation in operations
    )
    if expected_current_ids and operation_current_ids != expected_current_ids:
        references_are_valid = False
    if not references_are_valid:
        return sorted(fail_open)

    for operation in operations:
        if operation["operation"] != "repeat":
            active.add(str(operation["current_evidence_id"]))
    return sorted(active)


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
    # Collapse exact repeated source claims even when they originated in old
    # schema-v1 records whose IDs included the article ID.  Prefer a fact the
    # user has already seen; otherwise retain the most recent provenance.
    distinct: Dict[str, SourceFact] = {}
    for fact in facts:
        key = f"{fact.kind}\n{normalized_word_text(fact.text)}"
        if not key.strip():
            continue
        existing = distinct.get(key)
        if existing is None or (
            (fact.user_visible, fact.observed_at, fact.source_id, fact.fact_id)
            > (existing.user_visible, existing.observed_at, existing.source_id, existing.fact_id)
        ):
            distinct[key] = fact
    ordered = sorted(distinct.values(), key=lambda fact: (fact.observed_at, fact.source_id, fact.fact_id))
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


def _operation_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output: List[Dict[str, Any]] = []
    for raw in value[:16]:
        if not isinstance(raw, dict):
            continue
        operation = str(raw.get("operation", "uncertain") or "uncertain").strip()
        current_id = str(raw.get("current_evidence_id", "") or "").strip()[:80]
        if operation not in FACT_OPERATIONS or not current_id:
            continue
        output.append(
            {
                "operation": operation,
                "current_evidence_id": current_id,
                "prior_fact_ids": _string_list(
                    raw.get("prior_fact_ids", []),
                    max_items=8,
                    max_chars=80,
                ),
            }
        )
    return output


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
    "MAX_THREAD_EVENTS_PER_STORY",
    "SourceFact",
    "StoryThreadEvent",
    "StoryCandidateMatch",
    "StoryRecord",
    "StoryStore",
    "merge_story_records",
    "provisional_story_key",
    "story_baseline_payload",
    "story_record_from_payload",
]
