from __future__ import annotations

"""Evidence contracts for story-thread and claim-delta inference.

This module deliberately does not infer semantic transitions from words or
phrases. Deterministic code is limited to source provenance and exact claim
repetition. Everything requiring event understanding is exposed as a small,
injectable inference contract and validated before it can affect story memory.
"""

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from mydailynews.domain.text_similarity import normalized_word_text


CLAIM_DELTA_POLICY_VERSION = "claim-evidence-contract.v3"

RELATIONSHIPS = {"same_story", "related_theme", "distinct_story", "uncertain"}
CHANGE_TYPES = {
    "new", "material_update", "status_change", "correction", "resolved",
    "incremental", "escalated", "weakened", "reframed", "unchanged", "uncertain",
}
DISPOSITIONS = {"full_report", "continuing_bullet", "omit", "uncertain"}
CLAIM_RELATIONS = {
    "equivalent",
    "supports",
    "adds_detail",
    "weaker_restatement",
    "new_fact_in_story",
    "non_substantive",
    "contradicts",
    "supersedes",
    "temporal_successor",
    "context_only",
    "uncertain",
}
ENTAILMENT_VALUES = {"yes", "no", "uncertain"}


@dataclass(frozen=True)
class ClaimEvidence:
    """One bounded source claim with enough provenance to audit it."""

    claim_id: str
    text: str
    side: str
    kind: str = "source_sentence"
    story_key: str = ""
    source_id: str = ""
    source_name: str = ""
    source_url: str = ""
    published_at: str = ""
    observed_at: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "side": self.side,
            "kind": self.kind,
            "story_key": self.story_key,
            "source_id": self.source_id,
            "source": self.source_name,
            "url": self.source_url,
            "published_at": self.published_at,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class ClaimAlignment:
    current_claim_id: str
    prior_claim_id: str
    relation: str = "exact_repetition"

    def payload(self) -> dict[str, str]:
        return {
            "current_claim_id": self.current_claim_id,
            "prior_claim_id": self.prior_claim_id,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class ClaimComparisonRequest:
    """Model-agnostic input for comparing a current report with candidates."""

    current_claims: tuple[ClaimEvidence, ...]
    prior_claims: tuple[ClaimEvidence, ...]
    exact_alignments: tuple[ClaimAlignment, ...] = ()
    exact_repetition_story_key: str = ""

    @property
    def current_ids(self) -> set[str]:
        return {claim.claim_id for claim in self.current_claims}

    @property
    def prior_ids(self) -> set[str]:
        return {claim.claim_id for claim in self.prior_claims}

    def prior_ids_for_story(self, story_key: str) -> set[str]:
        return {
            claim.claim_id
            for claim in self.prior_claims
            if claim.story_key == str(story_key or "").strip()
        }

    def payload(self) -> dict[str, Any]:
        return {
            "policy_version": CLAIM_DELTA_POLICY_VERSION,
            "current_claims": [claim.payload() for claim in self.current_claims],
            "prior_claims": [claim.payload() for claim in self.prior_claims],
            "exact_alignments": [item.payload() for item in self.exact_alignments],
            "exact_repetition_story_key": self.exact_repetition_story_key,
        }


@dataclass(frozen=True)
class SemanticClaimRelation:
    """Directed semantic edge between one current and one prior claim."""

    current_claim_id: str
    prior_claim_id: str
    relation: str
    current_entails_prior: str = "uncertain"
    prior_entails_current: str = "uncertain"

    def payload(self) -> dict[str, str]:
        return {
            "current_claim_id": self.current_claim_id,
            "prior_claim_id": self.prior_claim_id,
            "relation": self.relation,
            "current_entails_prior": self.current_entails_prior,
            "prior_entails_current": self.prior_entails_current,
        }


@dataclass(frozen=True)
class SemanticDeltaDecision:
    """Typed output accepted from any semantic inference implementation."""

    relationship: str
    change_type: str
    materiality: float
    confidence: float
    disposition: str
    summary: str
    prior_story_key: str = ""
    current_evidence_ids: tuple[str, ...] = ()
    prior_evidence_ids: tuple[str, ...] = ()
    superseded_prior_evidence_ids: tuple[str, ...] = ()
    claim_relations: tuple[SemanticClaimRelation, ...] = ()


@runtime_checkable
class SemanticDeltaInferencer(Protocol):
    """Interchangeable semantic backend: local model, service, or ensemble."""

    def compare(self, request: ClaimComparisonRequest) -> SemanticDeltaDecision | None:
        ...


@dataclass(frozen=True)
class BidirectionalEntailmentScore:
    """Backend-neutral pair score suitable for NLI or alignment models."""

    current_entails_prior: float
    prior_entails_current: float


@runtime_checkable
class BidirectionalEntailmentScorer(Protocol):
    """Optional lower-level plug-in used to verify individual claim edges."""

    def score(
        self,
        current: ClaimEvidence,
        prior: ClaimEvidence,
    ) -> BidirectionalEntailmentScore:
        ...


@dataclass(frozen=True)
class ClaimDeltaAssessment:
    relationship: str
    change_type: str
    materiality: float
    confidence: float
    disposition: str
    summary: str
    prior_story_key: str = ""
    added_claims: list[str] = field(default_factory=list)
    repeated_claims: list[str] = field(default_factory=list)
    superseded_claims: list[str] = field(default_factory=list)
    current_claims: list[dict[str, Any]] = field(default_factory=list)
    prior_claims: list[dict[str, Any]] = field(default_factory=list)
    exact_alignments: list[dict[str, str]] = field(default_factory=list)
    current_evidence_ids: list[str] = field(default_factory=list)
    prior_evidence_ids: list[str] = field(default_factory=list)
    superseded_prior_evidence_ids: list[str] = field(default_factory=list)
    claim_relations: list[dict[str, str]] = field(default_factory=list)
    decision_basis: str = ""
    requires_semantic_inference: bool = False


def current_claim_evidence(
    *,
    article_id: str,
    title: str,
    text: str,
    source_name: str = "",
    source_url: str = "",
    published_at: str = "",
    observed_at: str = "",
    max_claims: int = 7,
) -> list[ClaimEvidence]:
    """Create stable, bounded current-side claims without interpreting them."""

    namespace = f"current:{str(article_id or '').strip() or 'unknown'}"
    rows = _claim_rows(title, text, max_claims=max_claims)
    return [
        _make_evidence(
            namespace=namespace,
            side="current",
            kind=kind,
            text=claim_text,
            source_id=str(article_id or "").strip(),
            source_name=source_name,
            source_url=source_url,
            published_at=published_at,
            observed_at=observed_at,
        )
        for kind, claim_text in rows
    ]


def prior_claim_evidence(
    baseline: Mapping[str, Any],
    *,
    max_claims: int = 8,
) -> list[ClaimEvidence]:
    """Read source-backed prior claims from a bounded baseline payload."""

    story_key = _clean_text(baseline.get("story_key"), 160)
    if not story_key:
        return []
    namespace = f"prior:{story_key}"
    rows: list[ClaimEvidence] = []
    raw_facts = baseline.get("source_facts", [])
    if isinstance(raw_facts, list):
        for raw in raw_facts:
            if not isinstance(raw, Mapping):
                continue
            claim_text = _clean_text(raw.get("text"), 420)
            if not claim_text:
                continue
            rows.append(
                _make_evidence(
                    namespace=namespace,
                    side="prior",
                    kind=_clean_text(raw.get("kind"), 40) or "source_sentence",
                    text=claim_text,
                    story_key=story_key,
                    source_id=_clean_text(raw.get("source_id"), 120),
                    source_name=_clean_text(raw.get("source"), 120),
                    source_url=_clean_text(raw.get("url"), 500),
                    published_at=_clean_text(raw.get("published_at"), 48),
                    observed_at=_clean_text(raw.get("observed_at"), 48),
                    stable_hint=_clean_text(raw.get("fact_id"), 120),
                )
            )
            if len(rows) >= max(1, int(max_claims)):
                break

    # Memory-disabled prior-report fallbacks may have no source-fact row. Their
    # report ID is explicit provenance and the weaker evidence kind is retained.
    if not rows:
        knowns = baseline.get("knowns", [])
        if isinstance(knowns, str):
            knowns = [knowns]
        if isinstance(knowns, list):
            for index, value in enumerate(knowns[: max(1, int(max_claims))]):
                claim_text = _clean_text(value, 420)
                if claim_text:
                    rows.append(
                        _make_evidence(
                            namespace=namespace,
                            side="prior",
                            kind="prior_report_claim",
                            text=claim_text,
                            story_key=story_key,
                            source_id=_clean_text(baseline.get("last_report_id"), 120),
                            observed_at=_clean_text(baseline.get("last_seen"), 48),
                            stable_hint=f"known:{index}",
                        )
                    )

    if not rows:
        title = _clean_text(baseline.get("title"), 280)
        if title:
            rows.append(
                _make_evidence(
                    namespace=namespace,
                    side="prior",
                    kind="headline",
                    text=title,
                    story_key=story_key,
                    source_id=_clean_text(baseline.get("last_report_id"), 120),
                    observed_at=_clean_text(baseline.get("last_seen"), 48),
                )
            )
    return _dedupe_evidence(rows)[: max(1, int(max_claims))]


def build_claim_comparison(
    current_claims: Sequence[ClaimEvidence],
    prior_claims: Sequence[ClaimEvidence],
) -> ClaimComparisonRequest:
    """Align only normalized-exact propositions; leave paraphrases semantic."""

    current = tuple(_dedupe_evidence(current_claims))
    prior = tuple(_dedupe_evidence(prior_claims))
    prior_by_text: dict[str, list[ClaimEvidence]] = {}
    for claim in prior:
        key = normalized_word_text(claim.text)
        if key:
            prior_by_text.setdefault(key, []).append(claim)

    alignments: list[ClaimAlignment] = []
    comparable_current = [claim for claim in current if claim.kind != "headline"] or list(current)
    matched_current_ids: set[str] = set()
    for claim in comparable_current:
        candidates = prior_by_text.get(normalized_word_text(claim.text), [])
        if not candidates:
            continue
        for candidate in candidates:
            alignments.append(
                ClaimAlignment(
                    current_claim_id=claim.claim_id,
                    prior_claim_id=candidate.claim_id,
                )
            )
        matched_current_ids.add(claim.claim_id)

    exact_story_key = ""
    if comparable_current and len(matched_current_ids) == len(comparable_current):
        story_sets = [
            {
                old.story_key
                for old in prior_by_text.get(normalized_word_text(claim.text), [])
                if old.story_key
            }
            for claim in comparable_current
        ]
        common_stories = set.intersection(*story_sets) if story_sets else set()
        headline_stories = {
            old.story_key
            for claim in current
            if claim.kind == "headline"
            for old in prior_by_text.get(normalized_word_text(claim.text), [])
            if old.story_key
        }
        enough_exact_evidence = (
            len(comparable_current) >= 2
            or bool(common_stories.intersection(headline_stories))
        )
        if len(common_stories) == 1 and enough_exact_evidence:
            exact_story_key = next(iter(common_stories))

    return ClaimComparisonRequest(
        current_claims=current,
        prior_claims=prior,
        exact_alignments=tuple(alignments),
        exact_repetition_story_key=exact_story_key,
    )


def assess_claim_comparison(
    request: ClaimComparisonRequest,
    *,
    inferencer: SemanticDeltaInferencer | None = None,
) -> ClaimDeltaAssessment:
    """Apply structural facts, then optionally ask an injected semantic engine."""

    current_text = {claim.claim_id: claim.text for claim in request.current_claims}
    prior_text = {claim.claim_id: claim.text for claim in request.prior_claims}
    repeated_ids = {item.current_claim_id for item in request.exact_alignments}
    repeated = [claim.text for claim in request.current_claims if claim.claim_id in repeated_ids]
    added = [
        claim.text for claim in request.current_claims
        if claim.kind != "headline" and claim.claim_id not in repeated_ids
    ]

    if not request.prior_claims:
        return _assessment(
            request,
            relationship="distinct_story", change_type="new", materiality=1.0,
            confidence=0.95, disposition="full_report",
            summary="No candidate prior evidence exists; this is a first observation.",
            added=added or [claim.text for claim in request.current_claims],
            current_evidence_ids=[claim.claim_id for claim in request.current_claims],
            decision_basis="first_observation",
        )

    if request.exact_repetition_story_key:
        return _assessment(
            request,
            relationship="same_story", change_type="unchanged", materiality=0.0,
            confidence=0.98, disposition="omit",
            summary="All current source propositions exactly repeat one candidate story.",
            prior_story_key=request.exact_repetition_story_key,
            repeated=repeated,
            current_evidence_ids=[
                item.current_claim_id for item in request.exact_alignments
                if item.current_claim_id in current_text
            ],
            prior_evidence_ids=[
                item.prior_claim_id for item in request.exact_alignments
                if item.prior_claim_id in prior_text
            ],
            decision_basis="exact_repetition",
        )

    if inferencer is not None:
        try:
            proposed = inferencer.compare(request)
        except Exception:
            proposed = None
        validated, _ = validate_semantic_decision(proposed, request)
        if validated is not None:
            return _assessment(
                request,
                relationship=validated.relationship,
                change_type=validated.change_type,
                materiality=validated.materiality,
                confidence=validated.confidence,
                disposition=validated.disposition,
                summary=validated.summary,
                prior_story_key=validated.prior_story_key,
                added=added,
                repeated=repeated,
                superseded=[
                    prior_text[claim_id]
                    for claim_id in validated.superseded_prior_evidence_ids
                    if claim_id in prior_text
                ],
                current_evidence_ids=validated.current_evidence_ids,
                prior_evidence_ids=validated.prior_evidence_ids,
                superseded_prior_evidence_ids=validated.superseded_prior_evidence_ids,
                claim_relations=validated.claim_relations,
                decision_basis="semantic_inference",
            )

    return _assessment(
        request,
        relationship="uncertain", change_type="uncertain", materiality=0.5,
        confidence=0.0, disposition="full_report",
        summary="Source claims differ; semantic comparison is required.",
        added=added, repeated=repeated,
        decision_basis="semantic_inference_required",
        requires_semantic_inference=True,
    )


def assess_claim_delta(
    current_title: str,
    current_text: str,
    prior_facts: Sequence[str],
    *,
    inferencer: SemanticDeltaInferencer | None = None,
) -> ClaimDeltaAssessment:
    """Compatibility wrapper around the evidence-based comparison contract."""

    current = current_claim_evidence(article_id="current", title=current_title, text=current_text)
    baseline = {
        "story_key": "prior",
        "source_facts": [
            {
                "fact_id": f"legacy:{index}", "text": value,
                "kind": "source_sentence", "source_id": "prior",
            }
            for index, value in enumerate(prior_facts)
        ],
    }
    prior = prior_claim_evidence(baseline, max_claims=max(1, len(prior_facts)))
    return assess_claim_comparison(build_claim_comparison(current, prior), inferencer=inferencer)


def semantic_decision_from_mapping(value: Any) -> SemanticDeltaDecision | None:
    if not isinstance(value, Mapping):
        return None
    return SemanticDeltaDecision(
        relationship=_clean_text(value.get("relationship"), 40) or "uncertain",
        change_type=_clean_text(value.get("change_type"), 40) or "uncertain",
        materiality=_bounded_float(value.get("materiality"), 0.5),
        confidence=_bounded_float(value.get("confidence"), 0.0),
        disposition=_clean_text(value.get("disposition"), 40) or "uncertain",
        summary=_clean_text(value.get("summary"), 400),
        prior_story_key=_clean_text(value.get("prior_story_key"), 160),
        current_evidence_ids=tuple(_string_list(value.get("current_evidence_ids"), 8, 160)),
        prior_evidence_ids=tuple(_string_list(value.get("prior_evidence_ids"), 8, 160)),
        superseded_prior_evidence_ids=tuple(
            _string_list(value.get("superseded_prior_evidence_ids"), 8, 160)
        ),
        claim_relations=tuple(_semantic_relations(value.get("claim_relations"))),
    )


def validate_semantic_decision(
    value: SemanticDeltaDecision | Mapping[str, Any] | None,
    request: ClaimComparisonRequest,
) -> tuple[SemanticDeltaDecision | None, list[str]]:
    """Reject ungrounded or internally inconsistent semantic decisions."""

    proposed = value if isinstance(value, SemanticDeltaDecision) else semantic_decision_from_mapping(value)
    if proposed is None:
        return None, ["missing semantic decision"]

    errors: list[str] = []
    if proposed.relationship not in RELATIONSHIPS:
        errors.append("invalid relationship")
    if proposed.change_type not in CHANGE_TYPES:
        errors.append("invalid change type")
    if proposed.disposition not in DISPOSITIONS:
        errors.append("invalid disposition")

    current_ids = set(proposed.current_evidence_ids)
    prior_ids = set(proposed.prior_evidence_ids)
    superseded_ids = set(proposed.superseded_prior_evidence_ids)
    if not current_ids or not current_ids.issubset(request.current_ids):
        errors.append("current evidence references are missing or unknown")
    if request.prior_claims and (not prior_ids or not prior_ids.issubset(request.prior_ids)):
        errors.append("prior evidence references are missing or unknown")
    if not superseded_ids.issubset(prior_ids):
        errors.append("superseded evidence must be cited prior evidence")
    if request.prior_claims and not proposed.claim_relations:
        errors.append("claim relations are required for a candidate comparison")
    related_current_ids: set[str] = set()
    related_prior_ids: set[str] = set()
    for relation in proposed.claim_relations:
        related_current_ids.add(relation.current_claim_id)
        related_prior_ids.add(relation.prior_claim_id)
        if relation.current_claim_id not in current_ids:
            errors.append("claim relation cites uncited current evidence")
        if relation.prior_claim_id not in prior_ids:
            errors.append("claim relation cites uncited prior evidence")
        if relation.relation not in CLAIM_RELATIONS:
            errors.append("invalid claim relation")
        if relation.current_entails_prior not in ENTAILMENT_VALUES:
            errors.append("invalid current-to-prior entailment")
        if relation.prior_entails_current not in ENTAILMENT_VALUES:
            errors.append("invalid prior-to-current entailment")
        if relation.relation == "equivalent" and not (
            relation.current_entails_prior == "yes"
            and relation.prior_entails_current == "yes"
        ):
            errors.append("equivalent relation requires bidirectional entailment")
        if relation.relation in {"supports", "adds_detail"} and (
            relation.current_entails_prior != "yes"
        ):
            errors.append("support/detail relation requires current-to-prior entailment")
        if relation.relation == "weaker_restatement" and not (
            relation.prior_entails_current == "yes"
            and relation.current_entails_prior != "yes"
        ):
            errors.append(
                "weaker restatement requires prior-to-current entailment only"
            )

    if request.prior_claims and current_ids.difference(related_current_ids):
        errors.append("every cited current claim must participate in a relation")
    if request.prior_claims and prior_ids.difference(related_prior_ids):
        errors.append("every cited prior claim must participate in a relation")
    superseding_prior_ids = {
        relation.prior_claim_id
        for relation in proposed.claim_relations
        if relation.relation in {"supersedes", "contradicts"}
    }
    if superseded_ids.difference(superseding_prior_ids):
        errors.append("superseded evidence requires a superseding or contradictory relation")

    if proposed.relationship == "same_story":
        allowed = request.prior_ids_for_story(proposed.prior_story_key)
        if not proposed.prior_story_key or not allowed:
            errors.append("same-story decision has no supplied candidate story key")
        elif not prior_ids.issubset(allowed):
            errors.append("prior evidence does not belong to the selected story")
        if proposed.change_type == "new":
            errors.append("same-story decision cannot be new")
        substantive_relations = {
            relation.relation
            for relation in proposed.claim_relations
            if relation.relation not in {
                "context_only", "non_substantive", "uncertain",
            }
        }
        if proposed.change_type != "uncertain" and not substantive_relations:
            errors.append("same-story change requires a substantive claim relation")
        if proposed.change_type == "unchanged":
            comparable_current_ids = {
                claim.claim_id
                for claim in request.current_claims
                if claim.kind != "headline"
            } or request.current_ids
            if comparable_current_ids.difference(current_ids):
                errors.append("unchanged must cite every current source claim")
            allowed_unchanged_relations = {
                "equivalent", "weaker_restatement", "non_substantive",
            }
            if any(
                relation.relation not in allowed_unchanged_relations
                for relation in proposed.claim_relations
            ):
                errors.append("unchanged contains a relation that denotes change")
            if not substantive_relations.intersection(
                {"equivalent", "weaker_restatement"}
            ):
                errors.append(
                    "unchanged requires an equivalent or weaker-restatement relation"
                )
        if proposed.change_type == "correction" and not substantive_relations.intersection(
            {"contradicts", "supersedes"}
        ):
            errors.append("correction requires contradiction or supersession evidence")
    elif proposed.relationship in {"related_theme", "distinct_story"}:
        if proposed.prior_story_key:
            errors.append("non-linking decision cannot select a prior story key")
        if proposed.change_type not in {"new", "uncertain"}:
            errors.append("non-linking decision cannot assert a continuation change")
        if any(
            relation.relation not in {"context_only", "uncertain"}
            for relation in proposed.claim_relations
        ):
            errors.append("non-linking decision cannot assert a substantive claim relation")
    elif proposed.relationship == "uncertain" and proposed.change_type != "uncertain":
        errors.append("uncertain relationship requires uncertain change type")

    if proposed.disposition == "omit" and not (
        proposed.relationship == "same_story" and proposed.change_type == "unchanged"
    ):
        errors.append("only a confirmed unchanged continuation may be omitted")
    if not proposed.summary:
        errors.append("summary is required")
    return (None, errors) if errors else (proposed, [])


def request_from_claim_delta_payload(value: Any) -> ClaimComparisonRequest | None:
    """Reconstruct the validation request stored in a scaffold decision."""

    if not isinstance(value, Mapping):
        return None
    current = _evidence_from_payloads(value.get("current_claims"), expected_side="current")
    prior = _evidence_from_payloads(value.get("prior_claims"), expected_side="prior")
    if not current:
        return None
    return build_claim_comparison(current, prior)


def _assessment(
    request: ClaimComparisonRequest,
    *,
    relationship: str,
    change_type: str,
    materiality: float,
    confidence: float,
    disposition: str,
    summary: str,
    prior_story_key: str = "",
    added: Iterable[str] = (),
    repeated: Iterable[str] = (),
    superseded: Iterable[str] = (),
    current_evidence_ids: Iterable[str] = (),
    prior_evidence_ids: Iterable[str] = (),
    superseded_prior_evidence_ids: Iterable[str] = (),
    claim_relations: Iterable[SemanticClaimRelation] = (),
    decision_basis: str,
    requires_semantic_inference: bool = False,
) -> ClaimDeltaAssessment:
    return ClaimDeltaAssessment(
        relationship=relationship,
        change_type=change_type,
        materiality=_bounded_float(materiality, 0.5),
        confidence=_bounded_float(confidence, 0.0),
        disposition=disposition,
        summary=summary,
        prior_story_key=prior_story_key,
        added_claims=_dedupe_text(added)[:6],
        repeated_claims=_dedupe_text(repeated)[:6],
        superseded_claims=_dedupe_text(superseded)[:4],
        current_claims=[claim.payload() for claim in request.current_claims[:8]],
        prior_claims=[claim.payload() for claim in request.prior_claims[:24]],
        exact_alignments=[item.payload() for item in request.exact_alignments[:12]],
        current_evidence_ids=_string_list(list(current_evidence_ids), 8, 160),
        prior_evidence_ids=_string_list(list(prior_evidence_ids), 8, 160),
        superseded_prior_evidence_ids=_string_list(list(superseded_prior_evidence_ids), 8, 160),
        claim_relations=[relation.payload() for relation in list(claim_relations)[:12]],
        decision_basis=decision_basis,
        requires_semantic_inference=requires_semantic_inference,
    )


def _claim_rows(title: str, text: str, *, max_claims: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    clean_title = _clean_text(title, 280)
    if clean_title:
        rows.append(("headline", clean_title))
    body = _clean_text(text, 2400)
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", body):
        clean = _clean_text(sentence, 420)
        if len(clean) < 24:
            continue
        rows.append(("source_sentence", clean))
        if len(rows) >= max(1, int(max_claims)):
            break
    return _dedupe_claim_rows(rows)[: max(1, int(max_claims))]


def _make_evidence(
    *,
    namespace: str,
    side: str,
    kind: str,
    text: str,
    story_key: str = "",
    source_id: str = "",
    source_name: str = "",
    source_url: str = "",
    published_at: str = "",
    observed_at: str = "",
    stable_hint: str = "",
) -> ClaimEvidence:
    # Match StoryStore's content-addressed fact identity so semantic edges can
    # refer to bounded durable facts without duplicating claim text in events.
    normalized = normalized_word_text(text)
    digest = sha256(f"{kind}\n{normalized}".encode("utf-8")).hexdigest()[:20]
    claim_id = stable_hint if stable_hint.startswith("fact:") else f"fact:{digest}"
    return ClaimEvidence(
        claim_id=claim_id, text=_clean_text(text, 420), side=side,
        kind=_clean_text(kind, 40) or "source_sentence",
        story_key=_clean_text(story_key, 160), source_id=_clean_text(source_id, 120),
        source_name=_clean_text(source_name, 120), source_url=_clean_text(source_url, 500),
        published_at=_clean_text(published_at, 48), observed_at=_clean_text(observed_at, 48),
    )


def _evidence_from_payloads(value: Any, *, expected_side: str) -> list[ClaimEvidence]:
    if not isinstance(value, list):
        return []
    output: list[ClaimEvidence] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        claim_id = _clean_text(raw.get("claim_id"), 160)
        text = _clean_text(raw.get("text"), 420)
        side = _clean_text(raw.get("side"), 20)
        if not claim_id or not text or side != expected_side:
            continue
        output.append(
            ClaimEvidence(
                claim_id=claim_id, text=text, side=side,
                kind=_clean_text(raw.get("kind"), 40) or "source_sentence",
                story_key=_clean_text(raw.get("story_key"), 160),
                source_id=_clean_text(raw.get("source_id"), 120),
                source_name=_clean_text(raw.get("source"), 120),
                source_url=_clean_text(raw.get("url"), 500),
                published_at=_clean_text(raw.get("published_at"), 48),
                observed_at=_clean_text(raw.get("observed_at"), 48),
            )
        )
    return _dedupe_evidence(output)


def _semantic_relations(value: Any) -> list[SemanticClaimRelation]:
    if not isinstance(value, list):
        return []
    output: list[SemanticClaimRelation] = []
    for raw in value[:12]:
        if not isinstance(raw, Mapping):
            continue
        output.append(
            SemanticClaimRelation(
                current_claim_id=_clean_text(raw.get("current_claim_id"), 160),
                prior_claim_id=_clean_text(raw.get("prior_claim_id"), 160),
                relation=_clean_text(raw.get("relation"), 40),
                current_entails_prior=(
                    _clean_text(raw.get("current_entails_prior"), 20) or "uncertain"
                ),
                prior_entails_current=(
                    _clean_text(raw.get("prior_entails_current"), 20) or "uncertain"
                ),
            )
        )
    return output


def _dedupe_claim_rows(rows: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, text in rows:
        key = normalized_word_text(text)
        if key and key not in seen:
            seen.add(key)
            output.append((kind, text))
    return output


def _dedupe_evidence(values: Iterable[ClaimEvidence]) -> list[ClaimEvidence]:
    output: list[ClaimEvidence] = []
    seen: set[str] = set()
    for value in values:
        if value.claim_id and value.claim_id not in seen:
            seen.add(value.claim_id)
            output.append(value)
    return output


def _dedupe_text(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value, 420)
        key = normalized_word_text(text)
        if text and key and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _string_list(value: Any, max_items: int, max_chars: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    output: list[str] = []
    for item in value:
        text = _clean_text(item, max_chars)
        if text and text not in output:
            output.append(text)
        if len(output) >= max_items:
            break
    return output


def _clean_text(value: Any, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def _bounded_float(value: Any, default: float) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return default
