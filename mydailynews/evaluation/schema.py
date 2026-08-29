from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as date_type, datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

from mydailynews.app.models import NewsCandidate, UserMemory


EVAL_SCHEMA_VERSION = "change_monitor.eval.v1"
RELATIONSHIPS = {"new_story", "same_story", "related_theme", "uncertain"}
DELTA_TYPES = {
    "new",
    "material_update",
    "status_change",
    "correction",
    "resolved",
    "reframed",
    "incremental",
    "unchanged",
    "uncertain",
}
DISPLAYS = {"full_report", "continuing_bullet", "omit"}
RELEVANCE_LABELS = {"must_select", "eligible", "irrelevant"}
SPLITS = {"development", "holdout"}


@dataclass(frozen=True)
class EvalDocument:
    id: str
    source: str
    title: str
    url: str
    snippet: str
    body: str
    published_at: str
    category: str = "synthetic"
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalDocument":
        return cls(
            id=_text(raw.get("id")),
            source=_text(raw.get("source")),
            title=_text(raw.get("title")),
            url=_text(raw.get("url")),
            snippet=_text(raw.get("snippet")),
            body=_text(raw.get("body")),
            published_at=_text(raw.get("published_at")),
            category=_text(raw.get("category")) or "synthetic",
            tags=_strings(raw.get("tags", [])),
        )

    def to_candidate(self) -> NewsCandidate:
        published_at: datetime | None = None
        if self.published_at:
            try:
                published_at = datetime.fromisoformat(self.published_at.replace("Z", "+00:00"))
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except ValueError:
                published_at = None
        return NewsCandidate(
            id=self.id,
            source=self.source,
            category=self.category,
            title=self.title,
            url=self.url,
            snippet=self.snippet,
            published_at=published_at,
            tags=list(self.tags),
            metadata={"fixture_source": True},
        )


@dataclass(frozen=True)
class EvalExpectation:
    document_id: str
    canonical_story_id: str
    relationship: str
    delta_type: str
    material: bool
    display: str
    profile_relevance: str
    should_select: bool
    required_fact_ids: List[str] = field(default_factory=list)
    forbidden_fact_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalExpectation":
        return cls(
            document_id=_text(raw.get("document_id")),
            canonical_story_id=_text(raw.get("canonical_story_id")),
            relationship=_text(raw.get("relationship")),
            delta_type=_text(raw.get("delta_type")),
            material=_required_bool(raw, "material"),
            display=_text(raw.get("display")),
            profile_relevance=_text(raw.get("profile_relevance")),
            should_select=_required_bool(raw, "should_select"),
            required_fact_ids=_strings(raw.get("required_fact_ids", [])),
            forbidden_fact_ids=_strings(raw.get("forbidden_fact_ids", [])),
        )


@dataclass(frozen=True)
class EvalDay:
    date: str
    documents: List[EvalDocument]
    expectations: List[EvalExpectation]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalDay":
        documents = raw.get("documents", [])
        expectations = raw.get("expectations", [])
        return cls(
            date=_text(raw.get("date")),
            documents=[EvalDocument.from_dict(item) for item in documents if isinstance(item, dict)],
            expectations=[EvalExpectation.from_dict(item) for item in expectations if isinstance(item, dict)],
        )


@dataclass(frozen=True)
class EvalArc:
    id: str
    split: str
    tags: List[str]
    profile: UserMemory
    fact_catalog: Dict[str, str]
    days: List[EvalDay]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalArc":
        profile_raw = raw.get("profile", {}) if isinstance(raw.get("profile"), dict) else {}
        allowed_profile_fields = set(UserMemory.__dataclass_fields__)
        profile_values = {key: value for key, value in profile_raw.items() if key in allowed_profile_fields}
        days = raw.get("days", [])
        facts = raw.get("fact_catalog", {})
        return cls(
            id=_text(raw.get("id")),
            split=_text(raw.get("split")) or "development",
            tags=_strings(raw.get("tags", [])),
            profile=UserMemory(**profile_values),
            fact_catalog={str(key): _text(value) for key, value in facts.items()} if isinstance(facts, dict) else {},
            days=[EvalDay.from_dict(item) for item in days if isinstance(item, dict)],
        )

    def public_input(self) -> "EvalArcInput":
        # Deliberately excludes split, trap tags, fact IDs, canonical story IDs,
        # and every expected label. Adapters cannot win by reading the answer key.
        return EvalArcInput(
            id=self.id,
            profile=self.profile,
            days=[EvalDayInput(date=day.date, documents=day.documents) for day in self.days],
        )


@dataclass(frozen=True)
class EvalCorpus:
    schema_version: str
    name: str
    description: str
    arcs: List[EvalArc]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalCorpus":
        arcs = raw.get("arcs", [])
        corpus = cls(
            schema_version=_text(raw.get("schema_version")),
            name=_text(raw.get("name")),
            description=_text(raw.get("description")),
            arcs=[EvalArc.from_dict(item) for item in arcs if isinstance(item, dict)],
        )
        corpus.validate()
        return corpus

    def validate(self) -> None:
        errors: List[str] = []
        if self.schema_version != EVAL_SCHEMA_VERSION:
            errors.append(f"schema_version must be {EVAL_SCHEMA_VERSION!r}")
        if not self.name:
            errors.append("name is required")
        if not self.arcs:
            errors.append("at least one arc is required")

        arc_ids: set[str] = set()
        global_document_ids: set[str] = set()
        for arc in self.arcs:
            prefix = f"arc {arc.id or '<missing>'}"
            if not arc.id:
                errors.append("arc id is required")
            elif arc.id in arc_ids:
                errors.append(f"duplicate arc id: {arc.id}")
            arc_ids.add(arc.id)
            if arc.split not in SPLITS:
                errors.append(f"{prefix}: invalid split {arc.split!r}")
            if not arc.tags:
                errors.append(f"{prefix}: at least one trap tag is required")
            if not arc.days:
                errors.append(f"{prefix}: at least one day is required")
            previous_date: date_type | None = None
            arc_document_ids: set[str] = set()
            seen_canonical_stories: set[str] = set()
            for day in arc.days:
                try:
                    parsed_date = date_type.fromisoformat(day.date)
                except ValueError:
                    parsed_date = None
                    errors.append(f"{prefix}: invalid date {day.date!r}")
                if parsed_date is not None and previous_date is not None and parsed_date <= previous_date:
                    errors.append(f"{prefix}: days must be strictly chronological")
                if parsed_date is not None:
                    previous_date = parsed_date
                day_ids: set[str] = set()
                for document in day.documents:
                    document_label = f"{prefix}/{day.date}/{document.id or '<missing>'}"
                    if not document.id or not document.source or not document.title or not document.url or not document.body:
                        errors.append(
                            f"{document_label}: each document needs id, source, title, url, and body"
                        )
                    parsed_url = urlparse(document.url)
                    if document.url and (parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc):
                        errors.append(f"{document_label}: url must be an absolute HTTP(S) URL")
                    published_at: datetime | None = None
                    if not document.published_at:
                        errors.append(f"{document_label}: published_at is required")
                    else:
                        try:
                            published_at = datetime.fromisoformat(
                                document.published_at.replace("Z", "+00:00")
                            )
                        except ValueError:
                            errors.append(f"{document_label}: invalid published_at {document.published_at!r}")
                        if published_at is not None and published_at.tzinfo is None:
                            errors.append(f"{document_label}: published_at must include a timezone")
                        if (
                            published_at is not None
                            and parsed_date is not None
                            and published_at.date() != parsed_date
                        ):
                            errors.append(f"{document_label}: published_at must fall on its evaluation day")
                    if document.id in global_document_ids:
                        errors.append(f"duplicate document id: {document.id}")
                    global_document_ids.add(document.id)
                    arc_document_ids.add(document.id)
                    day_ids.add(document.id)
                expected_ids = {item.document_id for item in day.expectations}
                if len(expected_ids) != len(day.expectations):
                    errors.append(f"{prefix}/{day.date}: duplicate expectation document_id")
                if day_ids != expected_ids:
                    errors.append(
                        f"{prefix}/{day.date}: document/expectation mismatch "
                        f"(missing={sorted(day_ids - expected_ids)}, extra={sorted(expected_ids - day_ids)})"
                    )
                for expected in day.expectations:
                    label = f"{prefix}/{day.date}/{expected.document_id}"
                    if not expected.canonical_story_id:
                        errors.append(f"{label}: canonical_story_id is required")
                    if expected.relationship not in RELATIONSHIPS:
                        errors.append(f"{label}: invalid relationship {expected.relationship!r}")
                    if expected.delta_type not in DELTA_TYPES:
                        errors.append(f"{label}: invalid delta_type {expected.delta_type!r}")
                    if expected.display not in DISPLAYS:
                        errors.append(f"{label}: invalid display {expected.display!r}")
                    if expected.profile_relevance not in RELEVANCE_LABELS:
                        errors.append(f"{label}: invalid profile_relevance {expected.profile_relevance!r}")
                    story_was_seen = expected.canonical_story_id in seen_canonical_stories
                    if expected.relationship == "same_story" and not story_was_seen:
                        errors.append(f"{label}: same_story requires an earlier canonical-story occurrence")
                    if expected.relationship == "new_story" and story_was_seen:
                        errors.append(f"{label}: new_story cannot reuse an earlier canonical story")
                    if expected.canonical_story_id:
                        seen_canonical_stories.add(expected.canonical_story_id)
                    required = set(expected.required_fact_ids)
                    forbidden = set(expected.forbidden_fact_ids)
                    if required.intersection(forbidden):
                        errors.append(f"{label}: required and forbidden fact IDs overlap")
                    unknown_facts = required.union(forbidden).difference(arc.fact_catalog)
                    if unknown_facts:
                        errors.append(f"{label}: unknown fact IDs {sorted(unknown_facts)}")
                    if expected.display == "omit" and expected.required_fact_ids:
                        errors.append(f"{label}: omitted items cannot require reported facts")
                    if expected.display == "omit" and expected.should_select:
                        errors.append(f"{label}: omitted items cannot be marked should_select")
                    if expected.profile_relevance == "irrelevant" and expected.should_select:
                        errors.append(f"{label}: irrelevant items cannot be marked should_select")
                    if expected.required_fact_ids and not expected.should_select:
                        errors.append(f"{label}: required reported facts require should_select")
            if not arc_document_ids:
                errors.append(f"{prefix}: no documents")
        if errors:
            raise ValueError("Invalid evaluation corpus:\n- " + "\n- ".join(errors))

    def expectations_by_key(self) -> Dict[tuple[str, str, str], EvalExpectation]:
        return {
            (arc.id, day.date, expected.document_id): expected
            for arc in self.arcs
            for day in arc.days
            for expected in day.expectations
        }

    def quiet_days(self) -> set[tuple[str, str]]:
        return {
            (arc.id, day.date)
            for arc in self.arcs
            for day in arc.days
            if not day.documents
        }


@dataclass(frozen=True)
class EvalDayInput:
    date: str
    documents: List[EvalDocument]


@dataclass(frozen=True)
class EvalArcInput:
    id: str
    profile: UserMemory
    days: List[EvalDayInput]


@dataclass
class EvalPrediction:
    arc_id: str
    date: str
    document_id: str
    predicted_story_id: str
    relationship: str
    delta_type: str
    material: bool
    display: str
    profile_relevance: str
    selected: bool
    reported_fact_ids: List[str] | None = None
    unsupported_claims: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalPrediction":
        reported = raw.get("reported_fact_ids")
        if reported is not None and not isinstance(reported, list):
            raise ValueError("reported_fact_ids must be an array or null")
        return cls(
            arc_id=_text(raw.get("arc_id")),
            date=_text(raw.get("date")),
            document_id=_text(raw.get("document_id")),
            predicted_story_id=_text(raw.get("predicted_story_id")),
            relationship=_text(raw.get("relationship")) or "uncertain",
            delta_type=_text(raw.get("delta_type")) or "uncertain",
            material=_required_bool(raw, "material"),
            display=_text(raw.get("display")) or "full_report",
            profile_relevance=_text(raw.get("profile_relevance")) or "eligible",
            selected=_required_bool(raw, "selected"),
            reported_fact_ids=_strings(reported) if isinstance(reported, list) else None,
            unsupported_claims=_strings(raw.get("unsupported_claims", [])),
            latency_ms=_float(raw.get("latency_ms")),
            metadata=dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_corpus(path: Path | str) -> EvalCorpus:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation corpus root must be an object")
    return EvalCorpus.from_dict(payload)


def load_predictions(path: Path | str) -> List[EvalPrediction]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = payload.get("predictions", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Prediction file must be a list or contain a predictions list")
    predictions: List[EvalPrediction] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"prediction[{index}] must be an object")
        try:
            predictions.append(EvalPrediction.from_dict(item))
        except ValueError as exc:
            raise ValueError(f"prediction[{index}]: {exc}") from exc
    return predictions


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _strings(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _required_bool(raw: Dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a JSON boolean")
    return value


def prediction_keys(predictions: Iterable[EvalPrediction]) -> List[tuple[str, str, str]]:
    return [(item.arc_id, item.date, item.document_id) for item in predictions]
