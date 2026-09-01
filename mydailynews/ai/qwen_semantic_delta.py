from __future__ import annotations

"""Compact evidence-contract adapter for small JSON-capable local models."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from mydailynews.ai.base import AIClient, JSONSchemaSpec
from mydailynews.analysis.claim_delta import (
    ClaimComparisonRequest,
    SemanticClaimRelation,
    SemanticDeltaDecision,
)
from mydailynews.common.utils import compact_json


COMPACT_IDENTITY_SCHEMA = JSONSchemaSpec(
    name="compact_story_identity",
    schema={
        "type": "object",
        "properties": {
            "candidate": {"type": "string"},
            "relation": {
                "type": "string",
                "enum": [
                    "same_event",
                    "direct_successor",
                    "related_context",
                    "different_event",
                    "uncertain",
                ],
            },
            "current_evidence": {"type": "string"},
            "prior_evidence": {"type": "string"},
        },
        "required": ["candidate", "relation", "current_evidence", "prior_evidence"],
        "additionalProperties": False,
    },
)


PAIR_RELATIONS = (
    "equivalent",
    "adds_specificity",
    "weaker_restatement",
    "temporal_successor",
    "completed_successor",
    "intensified_successor",
    "weakened_successor",
    "explicit_replacement",
    "conflict",
    "new_unaligned",
    "non_substantive",
    "uncertain",
)


IDENTITY_SYSTEM_PROMPT = """Decide event identity from bounded source claims.
Return only schema-valid JSON. same_event means the same concrete occurrence or
state. direct_successor means a source-evidenced later step or consequence in
the same bounded process. Shared actors, place, or topic are insufficient, and
a disconnected recurring event is different_event. related_context is useful
background without event identity. Cite only supplied opaque IDs; choose
uncertain when the propositions do not establish the relation."""


PAIR_SYSTEM_PROMPT = """Classify each fixed source-claim pair independently.
Return only schema-valid JSON. equivalent is the same proposition;
adds_specificity means the current proposition implies the prior and adds
information; weaker_restatement means the prior implies the current and the
current adds nothing; successor labels require a later linked occurrence or
state; explicit_replacement requires an explicit correction, withdrawal, or
replacement; disagreement without replacement is conflict; new_unaligned is a
substantive current proposition not addressed by its closest prior claim;
non_substantive is document commentary rather than an external-world claim.
Preserve attribution, negation, quantity, modality, and time. Use uncertain
when the pair does not establish a label."""


@dataclass(frozen=True)
class _PairSlot:
    slot: str
    current: Any
    prior: Any
    alignment_score: float


class QwenSemanticDeltaInferencer:
    """Produce one grounded semantic decision with a deliberately small output."""

    def __init__(
        self,
        client: AIClient,
        *,
        max_current_claims: int = 5,
        max_prior_claims: int = 8,
        pair_scorer: Any = None,
        max_pairs_per_current: int = 2,
        max_pairs_per_request: int = 6,
        max_new_tokens: int = 128,
        input_token_limit: int | None = None,
        label_prefix: str = "semantic_delta",
    ) -> None:
        self.client = client
        self.max_current_claims = max(1, int(max_current_claims))
        self.max_prior_claims = max(1, int(max_prior_claims))
        self.pair_scorer = pair_scorer
        self.max_pairs_per_current = max(1, int(max_pairs_per_current))
        self.max_pairs_per_request = max(1, int(max_pairs_per_request))
        self.max_new_tokens = max(128, int(max_new_tokens))
        self.input_token_limit = input_token_limit
        self.label_prefix = str(label_prefix or "semantic_delta")
        self.calls = 0
        self.requests = 0
        self.last_request_count = 0
        self.last_raw: dict[str, Any] = {}

    def compare(self, request: ClaimComparisonRequest) -> SemanticDeltaDecision | None:
        current = list(request.current_claims[: self.max_current_claims])
        prior = _round_robin_prior_claims(request, self.max_prior_claims)
        if not current or not prior:
            return None
        current_aliases = {f"c{index}": item for index, item in enumerate(current)}
        prior_aliases = {f"p{index}": item for index, item in enumerate(prior)}
        story_keys = list(dict.fromkeys(item.story_key for item in prior if item.story_key))
        story_aliases = {f"s{index}": key for index, key in enumerate(story_keys)}
        story_alias_by_key = {key: alias for alias, key in story_aliases.items()}
        payload = {
            "n": [
                _claim_payload(item, alias)
                for alias, item in current_aliases.items()
            ],
            "s": _candidate_payload(
                prior_aliases,
                story_alias_by_key,
            ),
        }
        identity_prompt = f"""`n` is the current report; `s` contains candidate stories.
Choose one candidate and classify its event relation. Cite the strongest
current and prior evidence. The cited prior evidence must belong to the chosen
candidate.

Evidence:
{compact_json(payload)}"""
        self.calls += 1
        self.last_request_count = 1
        self.requests += 1
        identity_raw = self.client.complete_json(
            IDENTITY_SYSTEM_PROMPT,
            identity_prompt,
            label=f"{self.label_prefix}.identity.{self.calls}",
            max_new_tokens=min(96, self.max_new_tokens),
            input_token_limit=self.input_token_limit,
            json_schema=_bounded_identity_schema(
                current_aliases=current_aliases,
                prior_aliases=prior_aliases,
                story_aliases=story_aliases,
            ),
        )
        identity_relation = str(identity_raw.get("relation", "uncertain") or "uncertain")
        candidate_alias = str(identity_raw.get("candidate", "") or "")
        selected_story_key = story_aliases.get(candidate_alias, "")
        identity_prior = prior_aliases.get(
            str(identity_raw.get("prior_evidence", "") or "")
        )
        pair_slots: list[_PairSlot] = []
        pair_labels: dict[str, str] = {}
        pair_outputs: list[dict[str, Any]] = []
        identity_links = identity_relation in {"same_event", "direct_successor"}
        identity_owner_matches = bool(
            selected_story_key
            and identity_prior is not None
            and identity_prior.story_key == selected_story_key
        )
        if identity_links and identity_owner_matches:
            selected_prior = [
                claim
                for claim in request.prior_claims
                if claim.story_key == selected_story_key
            ][: self.max_prior_claims]
            pair_slots = _aligned_pair_slots(
                current,
                selected_prior,
                scorer=self.pair_scorer,
                top_k=self.max_pairs_per_current,
            )
            for batch_index, start in enumerate(
                range(0, len(pair_slots), self.max_pairs_per_request)
            ):
                batch = pair_slots[start : start + self.max_pairs_per_request]
                pair_prompt = f"""The reports are already linked to one story. Label
each fixed claim pair independently. Keys such as r0 are fixed pair slots, not
evidence IDs.

Pairs:
{compact_json([_pair_payload(item) for item in batch])}"""
                self.last_request_count += 1
                self.requests += 1
                raw = self.client.complete_json(
                    PAIR_SYSTEM_PROMPT,
                    pair_prompt,
                    label=f"{self.label_prefix}.pairs.{self.calls}.{batch_index}",
                    max_new_tokens=self.max_new_tokens,
                    input_token_limit=self.input_token_limit,
                    json_schema=_pair_relation_schema(batch),
                )
                output = dict(raw) if isinstance(raw, dict) else {}
                pair_outputs.append(output)
                pair_labels.update(
                    (slot, str(label or "uncertain"))
                    for slot, label in output.items()
                )
        self.last_raw = {
            "identity": dict(identity_raw) if isinstance(identity_raw, dict) else {},
            "pair_relations": pair_outputs,
            "resolved_identity": {
                "story_key": selected_story_key,
                "prior_claim_id": identity_prior.claim_id if identity_prior else "",
            },
            "alignments": [
                {
                    "slot": item.slot,
                    "current_claim_id": item.current.claim_id,
                    "prior_claim_id": item.prior.claim_id,
                    "score": round(item.alignment_score, 6),
                }
                for item in pair_slots
            ],
        }
        return _decision_from_pair_labels(
            identity_raw,
            pair_slots,
            pair_labels,
            current_aliases=current_aliases,
            identity_prior_aliases=prior_aliases,
            story_aliases=story_aliases,
        )


def _round_robin_prior_claims(
    request: ClaimComparisonRequest,
    maximum: int,
):
    groups: dict[str, list[Any]] = {}
    for claim in request.prior_claims:
        groups.setdefault(claim.story_key, []).append(claim)
    output = []
    depth = 0
    while len(output) < maximum:
        added = False
        for claims in groups.values():
            if depth < len(claims):
                output.append(claims[depth])
                added = True
                if len(output) >= maximum:
                    break
        if not added:
            break
        depth += 1
    return output


def _candidate_payload(prior_aliases, story_alias_by_key):
    groups: dict[str, list[tuple[str, Any]]] = {}
    for alias, claim in prior_aliases.items():
        groups.setdefault(claim.story_key, []).append((alias, claim))
    return [
        {
            "i": story_alias_by_key[story_key],
            "c": [_claim_payload(item, alias) for alias, item in values],
        }
        for story_key, values in groups.items()
    ]


def _claim_payload(claim, alias):
    return {
        "i": alias,
        "t": claim.text,
    }


def _delta_claims(claims):
    claims = list(claims)
    source_claims = [claim for claim in claims if claim.kind != "headline"]
    return source_claims or claims


def _aligned_pair_slots(current_claims, prior_claims, *, scorer, top_k):
    current = _delta_claims(current_claims)
    prior = _delta_claims(prior_claims)
    if not current or not prior:
        return []
    cartesian = [(left, right) for left in current for right in prior]
    if scorer is None:
        scores = [0.0 for _ in cartesian]
    else:
        scored = scorer.score_bidirectional(
            [left.text for left, _ in cartesian],
            [right.text for _, right in cartesian],
        )
        scores = [_alignment_value(item) for item in scored]
    ranked_by_current: dict[str, list[tuple[float, Any, Any]]] = {}
    for (left, right), score in zip(cartesian, scores):
        ranked_by_current.setdefault(left.claim_id, []).append((score, left, right))
    selected = []
    for left in current:
        ranked = sorted(
            ranked_by_current.get(left.claim_id, []),
            key=lambda item: (item[0], item[2].claim_id),
            reverse=True,
        )
        selected.extend(ranked[: max(1, int(top_k))])
    return [
        _PairSlot(
            slot=f"r{index}",
            current=left,
            prior=right,
            alignment_score=float(score),
        )
        for index, (score, left, right) in enumerate(selected)
    ]


def _alignment_value(score):
    forward = score.current_to_prior
    backward = score.prior_to_current
    return max(
        float(forward.entailment),
        float(backward.entailment),
        float(forward.contradiction),
        float(backward.contradiction),
    )


def _pair_payload(item: _PairSlot):
    return {
        "slot": item.slot,
        "current": item.current.text,
        "prior": item.prior.text,
    }


def _decision_from_pair_labels(
    identity_raw: Any,
    pair_slots,
    pair_labels,
    *,
    current_aliases,
    identity_prior_aliases,
    story_aliases,
) -> SemanticDeltaDecision | None:
    if not isinstance(identity_raw, dict):
        return None
    identity_relation = str(identity_raw.get("relation", "uncertain") or "uncertain")
    selected_story_key = story_aliases.get(
        str(identity_raw.get("candidate", "") or ""),
        "",
    )
    identity_current = current_aliases.get(
        str(identity_raw.get("current_evidence", "") or "")
    )
    identity_prior = identity_prior_aliases.get(
        str(identity_raw.get("prior_evidence", "") or "")
    )
    if (
        identity_current is None
        or identity_prior is None
        or not selected_story_key
        or identity_prior.story_key != selected_story_key
    ):
        return None

    relationship = {
        "same_event": "same_story",
        "direct_successor": "same_story",
        "related_context": "related_theme",
        "different_event": "distinct_story",
    }.get(identity_relation, "uncertain")

    if relationship == "same_story" and pair_slots:
        selected_pairs = _select_pair_labels(pair_slots, pair_labels)
        change_type = _aggregate_change_type(selected_pairs)
        materiality = _materiality(change_type)
        confidence = 0.0 if change_type == "uncertain" else 0.75
        disposition = _disposition(change_type, materiality)
        prior_story_key = selected_story_key
        edges = tuple(
            _claim_relation(item, label)
            for item, label in selected_pairs
        )
        current_ids = _unique(item.current.claim_id for item, _ in selected_pairs)
        prior_ids = _unique(item.prior.claim_id for item, _ in selected_pairs)
        superseded_ids = _unique(
            item.prior.claim_id
            for item, label in selected_pairs
            if label == "explicit_replacement"
        )
        summary_claim = _summary_claim(selected_pairs, change_type) or identity_current
    elif relationship in {"related_theme", "distinct_story"}:
        change_type = "new"
        materiality = 1.0
        confidence = 0.75
        disposition = "full_report"
        prior_story_key = ""
        edges = (
            SemanticClaimRelation(
                current_claim_id=identity_current.claim_id,
                prior_claim_id=identity_prior.claim_id,
                relation="context_only",
                current_entails_prior="uncertain",
                prior_entails_current="uncertain",
            ),
        )
        current_ids = (identity_current.claim_id,)
        prior_ids = (identity_prior.claim_id,)
        superseded_ids = ()
        summary_claim = identity_current
    else:
        relationship = "uncertain"
        change_type = "uncertain"
        materiality = 0.5
        confidence = 0.0
        disposition = "full_report"
        prior_story_key = ""
        edges = (
            SemanticClaimRelation(
                current_claim_id=identity_current.claim_id,
                prior_claim_id=identity_prior.claim_id,
                relation="uncertain",
                current_entails_prior="uncertain",
                prior_entails_current="uncertain",
            ),
        )
        current_ids = (identity_current.claim_id,)
        prior_ids = (identity_prior.claim_id,)
        superseded_ids = ()
        summary_claim = identity_current

    return SemanticDeltaDecision(
        relationship=relationship,
        change_type=change_type,
        materiality=materiality,
        confidence=confidence,
        disposition=disposition,
        summary=summary_claim.text[:240],
        prior_story_key=prior_story_key,
        current_evidence_ids=current_ids,
        prior_evidence_ids=prior_ids,
        superseded_prior_evidence_ids=superseded_ids,
        claim_relations=edges,
    )


def _select_pair_labels(pair_slots, pair_labels):
    grouped: dict[str, list[tuple[_PairSlot, str]]] = {}
    for item in pair_slots:
        label = str(pair_labels.get(item.slot, "uncertain") or "uncertain")
        if label not in PAIR_RELATIONS:
            label = "uncertain"
        grouped.setdefault(item.current.claim_id, []).append((item, label))
    return [
        _resolve_claim_pairs(values)
        for values in grouped.values()
    ]


def _resolve_claim_pairs(values):
    labels = {label for _, label in values}
    precedence = (
        "explicit_replacement",
        "conflict",
        "completed_successor",
        "intensified_successor",
        "weakened_successor",
        "temporal_successor",
        "adds_specificity",
        "equivalent",
        "weaker_restatement",
        "new_unaligned",
        "non_substantive",
        "uncertain",
    )
    for wanted in precedence:
        if wanted in labels:
            return next((item, label) for item, label in values if label == wanted)
    return values[0][0], "uncertain"


def _aggregate_change_type(selected_pairs):
    labels = [label for _, label in selected_pairs if label != "non_substantive"]
    if not labels or any(label in {"conflict", "uncertain"} for label in labels):
        return "uncertain"
    if "explicit_replacement" in labels:
        return "correction"
    if "completed_successor" in labels:
        return "resolved"
    if "intensified_successor" in labels:
        return "escalated"
    if "weakened_successor" in labels:
        return "weakened"
    if "temporal_successor" in labels:
        return "status_change"
    if any(label in {"adds_specificity", "new_unaligned"} for label in labels):
        return "incremental"
    if all(label in {"equivalent", "weaker_restatement"} for label in labels):
        return "unchanged"
    return "uncertain"


def _claim_relation(item, label):
    relation, forward, backward = {
        "equivalent": ("equivalent", "yes", "yes"),
        "adds_specificity": ("adds_detail", "yes", "no"),
        "weaker_restatement": ("weaker_restatement", "no", "yes"),
        "temporal_successor": ("temporal_successor", "no", "no"),
        "completed_successor": ("temporal_successor", "no", "no"),
        "intensified_successor": ("temporal_successor", "no", "no"),
        "weakened_successor": ("temporal_successor", "no", "no"),
        "explicit_replacement": ("supersedes", "no", "no"),
        "conflict": ("contradicts", "no", "no"),
        "new_unaligned": ("new_fact_in_story", "no", "no"),
        "non_substantive": ("non_substantive", "no", "no"),
    }.get(label, ("uncertain", "uncertain", "uncertain"))
    return SemanticClaimRelation(
        current_claim_id=item.current.claim_id,
        prior_claim_id=item.prior.claim_id,
        relation=relation,
        current_entails_prior=forward,
        prior_entails_current=backward,
    )


def _summary_claim(selected_pairs, change_type):
    priorities = {
        "correction": {"explicit_replacement"},
        "resolved": {"completed_successor"},
        "escalated": {"intensified_successor"},
        "weakened": {"weakened_successor"},
        "status_change": {"temporal_successor"},
        "incremental": {"adds_specificity", "new_unaligned"},
        "unchanged": {"equivalent", "weaker_restatement"},
    }
    wanted = priorities.get(change_type, set())
    for item, label in selected_pairs:
        if label in wanted:
            return item.current
    return selected_pairs[0][0].current if selected_pairs else None


def _materiality(change_type):
    if change_type == "unchanged":
        return 0.0
    if change_type == "incremental":
        return 0.6
    if change_type == "uncertain":
        return 0.5
    return 0.9


def _unique(values):
    return tuple(dict.fromkeys(str(value) for value in values if str(value or "")))


def _bounded_identity_schema(*, current_aliases, prior_aliases, story_aliases):
    """Constrain opaque references to IDs actually present in this request."""

    schema = deepcopy(COMPACT_IDENTITY_SCHEMA.schema)
    properties = schema["properties"]
    properties["candidate"]["enum"] = list(story_aliases)
    properties["current_evidence"]["enum"] = list(current_aliases)
    properties["prior_evidence"]["enum"] = list(prior_aliases)
    return JSONSchemaSpec(name=COMPACT_IDENTITY_SCHEMA.name, schema=schema)


def _pair_relation_schema(batch):
    slots = [item.slot for item in batch]
    return JSONSchemaSpec(
        name=f"compact_pair_relations_{len(slots)}",
        schema={
            "type": "object",
            "properties": {
                slot: {"type": "string", "enum": list(PAIR_RELATIONS)}
                for slot in slots
            },
            "required": slots,
            "additionalProperties": False,
        },
    )


def _disposition(change_type: str, materiality: float) -> str:
    if change_type == "unchanged":
        return "omit"
    if change_type == "uncertain" or materiality >= 0.7:
        return "full_report"
    return "continuing_bullet"


__all__ = [
    "COMPACT_IDENTITY_SCHEMA",
    "PAIR_RELATIONS",
    "QwenSemanticDeltaInferencer",
]
