from __future__ import annotations

"""Source-evidence and fact-operation contracts for story-delta inference."""

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Iterable, Mapping

from mydailynews.domain.text_similarity import normalized_word_text


# One canonical vocabulary for the production operation prompt, schema, parser,
# validator, and StoryStore integration.
FACT_OPERATIONS = ("add", "repeat", "replace", "resolve", "uncertain")


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
class FactOperation:
    """One model-proposed operation over supplied evidence IDs."""

    operation: str
    current_evidence_id: str
    prior_fact_ids: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "current_evidence_id": self.current_evidence_id,
            "prior_fact_ids": list(self.prior_fact_ids),
        }


@dataclass(frozen=True)
class FactOperationRequest:
    """One grouped story-day compared with at most one prior story."""

    story_key: str
    current_claims: tuple[ClaimEvidence, ...]
    prior_claims: tuple[ClaimEvidence, ...] = ()
    article_ids: tuple[str, ...] = ()

    @property
    def current_ids(self) -> set[str]:
        return {claim.claim_id for claim in self.current_claims}

    @property
    def prior_ids(self) -> set[str]:
        return {claim.claim_id for claim in self.prior_claims}

    def payload(self) -> dict[str, Any]:
        return {
            "story_key": self.story_key,
            "article_ids": list(self.article_ids),
            "current_evidence": [claim.payload() for claim in self.current_claims],
            "prior_facts": [claim.payload() for claim in self.prior_claims],
        }


@dataclass(frozen=True)
class FactOperationValidation:
    story_key: str
    operations: tuple[FactOperation, ...] = ()
    diagnostics: tuple[str, ...] = ()
    structurally_valid_references: bool = False
    safe_to_apply: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "story_key": self.story_key,
            "operations": [operation.payload() for operation in self.operations],
            "diagnostics": list(self.diagnostics),
            "structurally_valid_references": self.structurally_valid_references,
            "safe_to_apply": self.safe_to_apply,
        }


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


def validate_fact_operations(
    value: Mapping[str, Any] | None,
    request: FactOperationRequest,
) -> FactOperationValidation:
    """Validate a narrow model response without making editorial judgments.

    Any malformed row, unknown ID, conflict, or unsafe mutation makes the
    whole response fail open. Valid ``uncertain`` responses are also visible
    and non-applicable, but retain their cited source evidence for diagnostics.
    """

    diagnostics: list[str] = []
    if not isinstance(value, Mapping):
        return FactOperationValidation(
            story_key=request.story_key,
            diagnostics=("missing or malformed operation response",),
        )

    reference_errors = False
    story_key = _clean_text(value.get("story_key"), 160)
    if not story_key:
        diagnostics.append("story_key is required")
        reference_errors = True
    elif story_key != request.story_key:
        diagnostics.append("story_key was not the supplied story key")
        reference_errors = True

    raw_operations = value.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        diagnostics.append("operations must be a non-empty list")
        raw_operations = []

    parsed: list[FactOperation] = []
    seen_exact: set[tuple[str, str, tuple[str, ...]]] = set()
    by_current: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    current_by_id = {claim.claim_id: claim for claim in request.current_claims}
    prior_by_id = {claim.claim_id: claim for claim in request.prior_claims}

    for index, raw in enumerate(raw_operations[:16]):
        if not isinstance(raw, Mapping):
            diagnostics.append(f"operation {index} is malformed")
            continue
        operation = _clean_text(raw.get("operation"), 20)
        current_id = _clean_text(raw.get("current_evidence_id"), 160)
        raw_prior_ids = raw.get("prior_fact_ids")
        if not operation or not current_id or not isinstance(raw_prior_ids, list):
            diagnostics.append(f"operation {index} is incomplete")
            continue
        prior_ids = tuple(_string_list(raw_prior_ids, 8, 160))
        if operation not in FACT_OPERATIONS:
            diagnostics.append(f"operation {index} has an unknown operation")
            continue
        if current_id not in current_by_id:
            diagnostics.append(f"operation {index} cites an unknown current evidence ID")
            reference_errors = True
            continue
        if any(
            prior_id not in prior_by_id
            or prior_by_id[prior_id].story_key != request.story_key
            for prior_id in prior_ids
        ):
            diagnostics.append(f"operation {index} cites an unknown or cross-story prior fact ID")
            reference_errors = True
            continue
        if operation in {"repeat", "replace", "resolve"} and not prior_ids:
            diagnostics.append(f"operation {index} requires a cited prior fact")
            continue
        if operation in {"add", "uncertain"} and prior_ids:
            diagnostics.append(f"operation {index} must not cite a prior fact")
            continue
        if operation == "replace" and not all(
            _explicitly_replaces(current_by_id[current_id].text, prior_by_id[prior_id].text)
            for prior_id in prior_ids
        ):
            diagnostics.append(f"operation {index} lacks explicit correction or supersession evidence")
            continue
        if operation == "resolve" and not all(
            _explicitly_resolves(current_by_id[current_id].text, prior_by_id[prior_id].text)
            for prior_id in prior_ids
        ):
            diagnostics.append(f"operation {index} does not cite an explicitly open prior state")
            continue
        key = (operation, current_id, prior_ids)
        if key in seen_exact:
            continue
        seen_exact.add(key)
        parsed.append(FactOperation(operation, current_id, prior_ids))
        by_current.setdefault(current_id, set()).add((operation, prior_ids))

    conflicted = {
        current_id for current_id, operations in by_current.items()
        if len(operations) > 1
    }
    if conflicted:
        diagnostics.append(
            "conflicting operations for current evidence: " + ", ".join(sorted(conflicted))
        )
        parsed = [item for item in parsed if item.current_evidence_id not in conflicted]

    covered_current_ids = {item.current_evidence_id for item in parsed}
    missing_current_ids = request.current_ids.difference(covered_current_ids)
    if missing_current_ids:
        diagnostics.append(
            "missing operations for current evidence: " + ", ".join(sorted(missing_current_ids))
        )

    references_valid = not reference_errors
    safe_to_apply = not diagnostics and bool(parsed) and all(
        item.operation != "uncertain" for item in parsed
    )
    return FactOperationValidation(
        story_key=request.story_key,
        operations=tuple(parsed),
        diagnostics=tuple(diagnostics),
        structurally_valid_references=references_valid,
        safe_to_apply=safe_to_apply,
    )


def publication_policy_for_operations(
    validation: FactOperationValidation,
    request: FactOperationRequest,
) -> str:
    """Derive an initial publication candidate after operation validation."""

    if not validation.safe_to_apply:
        return "visible_fail_open"
    operations = list(validation.operations)
    if operations and all(item.operation == "repeat" for item in operations):
        body_ids = {
            claim.claim_id for claim in request.current_claims if claim.kind != "headline"
        } or request.current_ids
        covered = {item.current_evidence_id for item in operations}
        return "no_change" if body_ids.issubset(covered) else "visible_fail_open"
    if any(item.operation in {"replace", "resolve"} for item in operations):
        return "materiality_candidate"
    if any(item.operation == "add" for item in operations):
        return "incremental"
    return "visible_fail_open"


_REPLACEMENT_MARKERS = re.compile(
    r"\b(amend(?:ed|ment)|correct(?:ed|ion)|den(?:y|ied|ies)|false|no longer|"
    r"retract(?:ed|ion)|revis(?:ed|ion)|instead|rather than|supersed(?:e|ed|es)|"
    r"replac(?:e|ed|es)|binding change)\b",
    re.IGNORECASE,
)
_NOT_A_REPLACEMENT = re.compile(
    r"\b(?:not|isn't|wasn't) (?:a |an )?(?:correction|retraction|replacement)\b",
    re.IGNORECASE,
)
_RESOLUTION_MARKERS = re.compile(
    r"\b(resolv(?:e|ed|es)|fix(?:ed|es)?|repair(?:ed|s)?|ended|closed|cleared|"
    r"restored|prevent(?:ed|s)?|completed|returned|released)\b",
    re.IGNORECASE,
)
_OPEN_STATE_MARKERS = re.compile(
    r"\b(open|unresolved|uncertain|uncertainty|pending|await(?:ing|ed)?|remains?|"
    r"could|can|risk|fault|problem|issue|investigat(?:e|ed|ing|ion)|not yet|"
    r"planned?|proposed|expected|scheduled)\b",
    re.IGNORECASE,
)


def _explicitly_replaces(current_text: str, prior_text: str) -> bool:
    current = _clean_text(current_text, 420)
    prior = _clean_text(prior_text, 420)
    return bool(
        current
        and prior
        and normalized_word_text(current) != normalized_word_text(prior)
        and _REPLACEMENT_MARKERS.search(current)
        and not _NOT_A_REPLACEMENT.search(current)
    )


def _explicitly_resolves(current_text: str, prior_text: str) -> bool:
    return bool(
        _RESOLUTION_MARKERS.search(_clean_text(current_text, 420))
        and _OPEN_STATE_MARKERS.search(_clean_text(prior_text, 420))
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
