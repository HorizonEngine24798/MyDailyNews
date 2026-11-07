from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from mydailynews.app.models import NewsCandidate, SelectedArticle
from mydailynews.common.storage import (
    MEMORY_DATABASE_NAME,
    claim_migration,
    metadata_value,
    open_database,
    set_metadata,
)
from mydailynews.domain.candidate_annotations import candidate_memory_annotation, set_memory_annotation
from mydailynews.memory.story_keys import StoryIdentity, story_identity_for_candidate
from mydailynews.memory.story_records import (
    LEGACY_STORY_FILES,
    MATCH_CONFIDENCE_THRESHOLD,
    MAX_FACTS_PER_STORY,
    MAX_THREAD_EVENTS_PER_STORY,
    STORY_STATUSES,
    STORY_STORE_SCHEMA_VERSION,
    SourceFact,
    StoryRecord,
    StoryThreadEvent,
    _annotation_with_group,
    _bounded_fact_history,
    _bounded_float,
    _candidate_record_confidence,
    _decisions_by_article,
    _fact_from_payload,
    _json_object,
    _max_nonempty,
    _merge_legacy_records,
    _merge_strings,
    _merge_thread_events,
    _merge_tokens,
    _min_nonempty,
    _next_active_fact_ids,
    _operation_list,
    _parse_date,
    _read_story_file,
    _refresh_lifecycle_records,
    _semantic_baseline_fields,
    _story_group_by_article_id,
    _string_list,
    _thread_event_from_payload,
    _token_list,
    _updated_thread_events,
    merge_story_records,
    source_facts_for_article,
    story_baseline_payload,
    story_record_from_payload,
)
from mydailynews.memory.story_retrieval import (
    DEFAULT_CANDIDATE_THRESHOLD,
    MAX_CANDIDATES,
    StoryCandidateMatch,
    provisional_story_key,
    retrieve_story_candidates,
    source_signals,
)


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
            decision = decisions.get(str(article.candidate.id), {})
            group = groups.get(str(article.candidate.id))
            if group is not None:
                annotation = _annotation_with_group(
                    annotation,
                    group,
                    decision=decision,
                )
                set_memory_annotation(article.candidate, annotation)

            story_key = annotation.story_key
            previous = updated.get(story_key)
            if (
                previous is None
                and not decision.get("referenced_prior_story_keys")
                and article.candidate.metadata.get("memory_identity_state") == "provisional"
            ):
                prior_candidates = article.candidate.metadata.get("memory_prior_story_candidates", [])
                if isinstance(prior_candidates, list):
                    for item in prior_candidates:
                        if not isinstance(item, dict):
                            continue
                        candidate_key = str(item.get("story_key", "") or "").strip()
                        signals = item.get("signals", {}) if isinstance(item.get("signals"), dict) else {}
                        if bool(signals.get("numeric_conflict")):
                            continue
                        try:
                            candidate_score = float(item.get("score", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            candidate_score = 0.0
                        if candidate_score >= MATCH_CONFIDENCE_THRESHOLD and candidate_key in updated:
                            story_key = candidate_key
                            previous = updated[candidate_key]
                            annotation = replace(
                                annotation,
                                story_key=story_key,
                                story_family_key=previous.story_family_key or annotation.story_family_key,
                                story_title=previous.title or annotation.story_title,
                            )
                            set_memory_annotation(article.candidate, annotation)
                            article.candidate.metadata["memory_identity_state"] = "fallback_linked"
                            break
            candidate_identity = story_identity_for_candidate(article.candidate)
            current_signals = source_signals(
                article.candidate,
                source_text=article.article_text or article.candidate.snippet,
            )
            is_visible = str(article.candidate.id) in visible_ids
            current_facts = current_facts_by_article_id.get(str(article.candidate.id), [])
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
            prune=False,
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
    "source_facts_for_article",
    "story_baseline_payload",
    "story_record_from_payload",
]
