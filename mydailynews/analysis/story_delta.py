from __future__ import annotations

from typing import Any, Dict, List

from mydailynews.ai.base import AIClient
from mydailynews.ai.prompts import (
    FACT_OPERATION_SYSTEM,
    FACT_OPERATION_USER,
    STORY_EDITOR_SYSTEM,
    STORY_EDITOR_USER,
    STORY_IDENTITY_SYSTEM,
    STORY_IDENTITY_USER,
)
from mydailynews.ai.schemas import (
    FACT_OPERATION_JSON_SCHEMA,
    STORY_EDITOR_JSON_SCHEMA,
    STORY_IDENTITY_JSON_SCHEMA,
)
from mydailynews.ai.token_budget import resolve_client_token_budget
from mydailynews.analysis.shared import _short_text
from mydailynews.analysis.claim_delta import (
    FactOperation,
    FactOperationRequest,
    FactOperationValidation,
    current_claim_evidence,
    prior_claim_evidence,
    publication_policy_for_operations,
    validate_fact_operations,
)
from mydailynews.common.cache import JSONCache
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import DeltaExtractionConfig, SelectedArticle, UserMemory
from mydailynews.common.utils import compact_json, datetime_to_iso


MAX_EDITOR_FULL_REPORTS = 5
EDITOR_PROMPT_SAFETY_RATIO = 0.95


def _current_story_claims(
    article_ids: tuple[str, ...],
    by_id: Dict[str, SelectedArticle],
    *,
    max_current_claims: int = 8,
    max_article_chars: int | None = None,
) -> List[Any]:
    per_article_claims: List[List[Any]] = []
    for article_id in article_ids:
        article = by_id[article_id]
        claims = current_claim_evidence(
            article_id=article_id,
            title=str(article.candidate.title or ""),
            text=_short_text(
                article.article_text or article.candidate.snippet,
                max_article_chars,
            ),
            source_name=str(article.candidate.source or ""),
            source_url=str(article.candidate.url or ""),
            published_at=datetime_to_iso(article.candidate.published_at),
            max_claims=4,
        )
        body = [claim for claim in claims if claim.kind != "headline"]
        per_article_claims.append(body or claims[:1])

    current_claims: List[Any] = []
    seen: set[str] = set()
    for claim_index in range(4):
        for claims in per_article_claims:
            if claim_index < len(claims) and claims[claim_index].claim_id not in seen:
                current_claims.append(claims[claim_index])
                seen.add(claims[claim_index].claim_id)
            if len(current_claims) >= max(1, int(max_current_claims)):
                return current_claims
    return current_claims


class StoryDeltaAnalyzer:
    """Production story comparison: identity, one prior, operations, then editor."""

    def __init__(
        self,
        client: AIClient,
        config: DeltaExtractionConfig,
        debug: DebugLogger | None = None,
        cache: JSONCache | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.cache = cache
        self.cache_ttl_seconds = max(
            0,
            int(config.cache_ttl_seconds if cache_ttl_seconds is None else cache_ttl_seconds),
        )
        self.warnings: List[str] = []

    def extract(
        self,
        articles: List[SelectedArticle],
        memory: UserMemory,
        brief_goal: str,
        story_memory: Dict[str, Any],
        *,
        brief_name: str = "",
    ) -> Dict[str, Any]:
        self.warnings = []
        by_id = {str(article.candidate.id): article for article in articles}
        stories = story_memory.get("stories", []) if isinstance(story_memory, dict) else []
        cards: List[Dict[str, Any]] = []
        for index, story in enumerate(stories if isinstance(stories, list) else [], start=1):
            if not isinstance(story, dict):
                continue
            article_ids = tuple(
                article_id for article_id in (
                    str(value or "").strip() for value in story.get("current_article_ids", [])
                ) if article_id in by_id
            )
            if not article_ids:
                continue
            cards.append(self._build_story_card(index, story, article_ids, by_id, brief_name))

        if not cards:
            return {}
        editor_decisions = self._edit_story_cards(cards, memory, brief_goal, brief_name)
        return _story_delta_packet(cards, editor_decisions)

    def _build_story_card(
        self,
        index: int,
        story: Dict[str, Any],
        article_ids: tuple[str, ...],
        by_id: Dict[str, SelectedArticle],
        brief_name: str,
    ) -> Dict[str, Any]:
        current_key = str(story.get("story_key", "") or "").strip() or f"article:{article_ids[0]}"
        current_claims = tuple(
            _current_story_claims(
                article_ids,
                by_id,
                max_article_chars=max(120, int(self.config.max_article_chars)),
            )
        )
        baselines = [
            item for item in story.get("prior_baselines", [])
            if isinstance(item, dict) and str(item.get("story_key", "") or "").strip()
        ]
        identity = self._select_identity(
            current_key=current_key,
            story=story,
            current_claims=current_claims,
            baselines=baselines,
            label=f"story identity {index} ({brief_name or 'brief'})",
        )
        prior_key = identity["prior_story_key"] if identity["relationship"] == "same_story" else ""
        baseline = next(
            (item for item in baselines if str(item.get("story_key", "") or "") == prior_key),
            None,
        )
        prior_claims = tuple(prior_claim_evidence(baseline, max_claims=8)) if baseline else ()
        resolved_key = prior_key or current_key
        request = FactOperationRequest(
            story_key=resolved_key,
            current_claims=current_claims,
            prior_claims=prior_claims,
            article_ids=article_ids,
        )

        if identity["relationship"] == "same_story" and prior_claims:
            try:
                raw_operations = self._complete_json(
                    FACT_OPERATION_SYSTEM,
                    FACT_OPERATION_USER.format(comparison=compact_json(request.payload())),
                    label=f"story delta {index} ({brief_name or 'brief'})",
                    schema=FACT_OPERATION_JSON_SCHEMA,
                    max_new_tokens=min(512, max(192, int(self.config.max_new_tokens))),
                )
                validation = validate_fact_operations(raw_operations, request)
            except Exception as exc:
                self.warnings.append(
                    f"story delta {index} failed ({type(exc).__name__}: {exc}); kept visible for editor."
                )
                validation = FactOperationValidation(
                    story_key=resolved_key,
                    diagnostics=("comparison request failed",),
                )
        elif identity["relationship"] == "distinct_story":
            raw_operations = {
                "story_key": resolved_key,
                "operations": [
                    FactOperation("add", claim.claim_id).payload() for claim in current_claims
                ],
            }
            validation = validate_fact_operations(raw_operations, request)
        else:
            raw_operations = {
                "story_key": resolved_key,
                "operations": [
                    FactOperation("uncertain", claim.claim_id).payload() for claim in current_claims
                ],
            }
            validation = validate_fact_operations(raw_operations, request)

        member_articles = [by_id[article_id] for article_id in article_ids]
        return {
            "card_id": f"story-card-{index:03d}",
            "story_key": resolved_key,
            "current_story_key": current_key,
            "title": str(story.get("current_title", "") or member_articles[0].candidate.title)[:180],
            "article_ids": list(article_ids),
            "current_articles": [
                {
                    "id": article.candidate.id,
                    "headline": article.candidate.title[:180],
                    "source": article.candidate.source[:100],
                }
                for article in member_articles
            ],
            "current_facts": [claim.payload() for claim in current_claims],
            "prior_story_key": prior_key,
            "retrieved_candidate_count": len(baselines),
            "relevant_prior_facts": [claim.payload() for claim in prior_claims],
            "identity": identity,
            "proposed_operations": [item.payload() for item in validation.operations],
            "operation_validation": {
                "safe_to_apply": validation.safe_to_apply,
                "structurally_valid_references": validation.structurally_valid_references,
                "diagnostics": list(validation.diagnostics),
            },
            "operation_signal": publication_policy_for_operations(validation, request),
            "change_type": _change_type_for_operations(identity, validation),
            "editorial_signals": [
                {
                    "article_id": article.candidate.id,
                    "priority": round(float(article.decision.score), 4),
                    "novelty": round(float(article.decision.novelty), 4),
                    "novelty_basis": str(getattr(article.decision, "novelty_basis", "") or ""),
                    "impact": round(float(article.decision.impact), 4),
                    "impact_basis": str(getattr(article.decision, "impact_basis", "") or ""),
                    "urgency": round(float(article.decision.urgency), 4),
                    "urgency_basis": str(getattr(article.decision, "urgency_basis", "") or ""),
                }
                for article in member_articles
            ],
        }

    def _select_identity(
        self,
        *,
        current_key: str,
        story: Dict[str, Any],
        current_claims: tuple[Any, ...],
        baselines: List[Dict[str, Any]],
        label: str,
    ) -> Dict[str, Any]:
        if not baselines:
            return {
                "relationship": "distinct_story",
                "prior_story_key": "",
                "confidence": 1.0,
                "basis": "No historical candidate was retrieved.",
            }
        candidates = [
            {
                "story_key": str(item.get("story_key", "") or ""),
                "title": str(item.get("title", "") or "")[:180],
                "last_seen": str(item.get("last_seen", "") or "")[:32],
                "candidate_score": item.get("candidate_score", 0.0),
                "facts": [claim.payload() for claim in prior_claim_evidence(item, max_claims=8)],
            }
            for item in baselines[: max(1, min(3, int(self.config.max_prior_reports)))]
        ]
        packet = {
            "current": {
                "story_key": current_key,
                "title": str(story.get("current_title", "") or "")[:180],
                "articles": list(story.get("current_articles", []))[:8],
                "facts": [claim.payload() for claim in current_claims],
            },
            "prior_candidates": candidates,
        }
        try:
            raw = self._complete_json(
                STORY_IDENTITY_SYSTEM,
                STORY_IDENTITY_USER.format(identity_packet=compact_json(packet)),
                label=label,
                schema=STORY_IDENTITY_JSON_SCHEMA,
                max_new_tokens=192,
            )
        except Exception as exc:
            self.warnings.append(f"{label} failed ({type(exc).__name__}: {exc}); identity left uncertain.")
            return {
                "relationship": "uncertain",
                "prior_story_key": "",
                "confidence": 0.0,
                "basis": "Identity request failed.",
            }
        allowed = {item["story_key"] for item in candidates}
        relationship = str(raw.get("relationship", "uncertain") or "uncertain").strip()
        prior_key = str(raw.get("prior_story_key", "") or "").strip()
        confidence = _bounded_float(raw.get("confidence"), 0.0)
        basis = str(raw.get("basis", "") or "").strip()[:160]
        if relationship == "same_story" and prior_key in allowed and confidence >= 0.5:
            return {
                "relationship": "same_story",
                "prior_story_key": prior_key,
                "confidence": confidence,
                "basis": basis,
            }
        if relationship == "distinct_story" and not prior_key and confidence >= 0.5:
            return {
                "relationship": "distinct_story",
                "prior_story_key": "",
                "confidence": confidence,
                "basis": basis,
            }
        return {
            "relationship": "uncertain",
            "prior_story_key": "",
            "confidence": min(confidence, 0.49),
            "basis": basis or "Identity choice was invalid or below confidence threshold.",
        }

    def _edit_story_cards(
        self,
        cards: List[Dict[str, Any]],
        memory: UserMemory,
        brief_goal: str,
        brief_name: str,
    ) -> List[Dict[str, Any]]:
        prompt, submitted_cards, compacted = self._bounded_editor_prompt(
            cards,
            memory,
            brief_goal,
        )
        submitted_ids = {str(card["card_id"]) for card in submitted_cards}
        omitted_count = len(cards) - len(submitted_cards)
        self.debug.set_metric("analysis.delta.editor.cards", len(cards))
        self.debug.set_metric("analysis.delta.editor.cards_submitted", len(submitted_cards))
        self.debug.set_metric("analysis.delta.editor.cards_omitted", omitted_count)
        self.debug.set_metric("analysis.delta.editor.context_compacted", compacted)

        raw: Dict[str, Any] = {}
        if not submitted_cards:
            self.warnings.append(
                "story selection editor skipped because even minimal card context exceeded its input budget; "
                "conservative coverage decisions were applied."
            )
        else:
            if compacted:
                self.warnings.append(
                    "story selection editor used compact card context to stay within its input budget."
                )
            if omitted_count:
                self.warnings.append(
                    f"story selection editor omitted {omitted_count} lower-priority card(s) from its prompt; "
                    "conservative coverage decisions were applied to them."
                )
            try:
                raw = self._complete_json(
                    STORY_EDITOR_SYSTEM,
                    prompt,
                    label=f"story selection editor ({brief_name or 'brief'})",
                    schema=STORY_EDITOR_JSON_SCHEMA,
                    max_new_tokens=max(256, int(self.config.max_new_tokens)),
                )
            except Exception as exc:
                self.warnings.append(
                    f"story selection editor failed ({type(exc).__name__}: {exc}); "
                    "conservative coverage decisions were applied."
                )

        raw_decisions = raw.get("decisions", []) if isinstance(raw, dict) else []
        filtered_raw = {
            "decisions": [
                row for row in raw_decisions
                if isinstance(row, dict) and str(row.get("card_id", "") or "") in submitted_ids
            ]
        }
        return _validated_editor_decisions(filtered_raw, cards)

    def _bounded_editor_prompt(
        self,
        cards: List[Dict[str, Any]],
        memory: UserMemory,
        brief_goal: str,
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        ranked_cards = sorted(
            enumerate(cards),
            key=lambda item: (*_story_card_priority(item[1]), -item[0]),
            reverse=True,
        )
        article_limit = max(1, int(self.config.max_articles))
        submitted_cards: List[Dict[str, Any]] = []
        submitted_articles = 0
        for _, card in ranked_cards:
            article_count = max(1, len(card.get("article_ids", [])))
            if submitted_cards and submitted_articles + article_count > article_limit:
                break
            submitted_cards.append(card)
            submitted_articles += article_count

        max_text_chars = max(120, int(self.config.max_article_chars))
        variants = [
            (max_text_chars, None, None),
            (min(max_text_chars, 240), max_text_chars * 2, max_text_chars),
            (0, max_text_chars, max(80, max_text_chars // 2)),
        ]
        for index, (fact_chars, memory_chars, goal_chars) in enumerate(variants):
            prompt = _story_editor_prompt(
                submitted_cards,
                memory=_short_text(memory.to_prompt(), memory_chars),
                brief_goal=_short_text(brief_goal, goal_chars),
                fact_chars=fact_chars,
            )
            if self._editor_prompt_fits(prompt):
                return prompt, submitted_cards, index > 0

        while submitted_cards:
            submitted_cards = submitted_cards[:-1]
            prompt = _story_editor_prompt(
                submitted_cards,
                memory=_short_text(memory.to_prompt(), max_text_chars),
                brief_goal=_short_text(brief_goal, max(80, max_text_chars // 2)),
                fact_chars=0,
            )
            if submitted_cards and self._editor_prompt_fits(prompt):
                return prompt, submitted_cards, True
        return "", [], True

    def _editor_prompt_fits(self, prompt: str) -> bool:
        max_new_tokens = max(256, int(self.config.max_new_tokens))
        try:
            budget = resolve_client_token_budget(
                self.client,
                input_tokens=max(256, int(self.config.max_input_tokens)),
                output_tokens=max_new_tokens,
            )
            input_limit = budget.input_tokens
        except (AttributeError, TypeError, ValueError):
            input_limit = max(256, int(self.config.max_input_tokens))
        safe_limit = max(64, int(input_limit * EDITOR_PROMPT_SAFETY_RATIO))
        request_text = f"System:\n{STORY_EDITOR_SYSTEM}\n\nUser:\n{prompt}\n\nAssistant:\n"
        estimator = getattr(self.client, "estimate_tokens", None)
        estimated_tokens = (
            int(estimator(request_text))
            if callable(estimator)
            else max(1, (len(request_text) + 3) // 4)
        )
        return estimated_tokens <= safe_limit

    def _complete_json(
        self,
        system: str,
        user: str,
        *,
        label: str,
        schema: Any,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        fingerprint = {
            "v": 1,
            "stage": schema.name,
            "backend": getattr(self.client.config, "backend", ""),
            "model": getattr(self.client.config, "effective_model_label", ""),
            "system": system,
            "user": user,
        }
        cache_key = JSONCache.make_key(compact_json(fingerprint))
        if self.cache:
            cached = self.cache.get(cache_key, max_age_seconds=self.cache_ttl_seconds)
            if isinstance(cached, dict):
                return cached
        result = self.client.complete_json(
            system,
            user,
            label=label,
            max_new_tokens=max_new_tokens,
            input_token_limit=self.config.max_input_tokens,
            json_schema=schema,
        )
        if self.cache:
            self.cache.put(cache_key, result)
        return result


def _change_type_for_operations(
    identity: Dict[str, Any],
    validation: FactOperationValidation,
) -> str:
    if identity.get("relationship") == "uncertain" or not validation.safe_to_apply:
        return "uncertain"
    if identity.get("relationship") == "distinct_story":
        return "new"
    operations = {item.operation for item in validation.operations}
    if operations == {"repeat"}:
        return "unchanged"
    if "replace" in operations:
        return "correction"
    if "resolve" in operations:
        return "resolved"
    if "add" in operations:
        return "incremental"
    return "uncertain"


def _story_editor_prompt(
    cards: List[Dict[str, Any]],
    *,
    memory: str,
    brief_goal: str,
    fact_chars: int,
) -> str:
    return STORY_EDITOR_USER.format(
        memory=memory,
        brief_goal=brief_goal,
        story_cards=compact_json([
            _compact_editor_card(card, fact_chars=fact_chars)
            for card in cards
        ]),
    )


def _compact_editor_card(card: Dict[str, Any], *, fact_chars: int) -> Dict[str, Any]:
    current_facts = {
        str(item.get("claim_id", "") or ""): str(item.get("text", "") or "")
        for item in card.get("current_facts", [])
        if isinstance(item, dict) and str(item.get("claim_id", "") or "")
    }
    prior_facts = {
        str(item.get("claim_id", "") or ""): str(item.get("text", "") or "")
        for item in card.get("relevant_prior_facts", [])
        if isinstance(item, dict) and str(item.get("claim_id", "") or "")
    }
    operations = [
        item for item in card.get("proposed_operations", [])
        if isinstance(item, dict)
    ]
    operation_counts: Dict[str, int] = {}
    fact_deltas: List[Dict[str, Any]] = []
    referenced_text_count = 0
    for operation in operations:
        name = str(operation.get("operation", "uncertain") or "uncertain")
        operation_counts[name] = operation_counts.get(name, 0) + 1
        current_id = str(operation.get("current_evidence_id", "") or "")
        prior_texts = [
            prior_facts[str(prior_id)]
            for prior_id in operation.get("prior_fact_ids", [])
            if str(prior_id) in prior_facts
        ]
        current_text = current_facts.get(current_id, "")
        referenced_text_count += bool(current_text) + len(prior_texts)
        fact_deltas.append({
            "operation": name,
            "current": current_text,
            "prior": prior_texts,
        })
    if not fact_deltas:
        fact_deltas = [
            {"operation": "uncertain", "current": text, "prior": []}
            for text in current_facts.values()
        ]
        referenced_text_count = len(fact_deltas)

    text_limit = max(0, int(fact_chars))
    per_fact_chars = min(180, text_limit // max(1, referenced_text_count))
    bounded_fact_deltas = [
        {
            "operation": row["operation"],
            "current": _short_text(row["current"], per_fact_chars),
            "prior": [
                _short_text(text, per_fact_chars)
                for text in row["prior"]
                if _short_text(text, per_fact_chars)
            ],
        }
        for row in fact_deltas
        if per_fact_chars > 0
    ]
    identity = card.get("identity", {}) if isinstance(card.get("identity"), dict) else {}
    validation = (
        card.get("operation_validation", {})
        if isinstance(card.get("operation_validation"), dict)
        else {}
    )
    return {
        "card_id": str(card.get("card_id", "")),
        "title": _short_text(card.get("title", ""), 180),
        "change_type": str(card.get("change_type", "uncertain") or "uncertain"),
        "identity": {
            "relationship": str(identity.get("relationship", "uncertain") or "uncertain"),
            "confidence": _bounded_float(identity.get("confidence"), 0.0),
        },
        "safe_to_apply": bool(validation.get("safe_to_apply", False)),
        "diagnostics": [
            _short_text(item, 120)
            for item in validation.get("diagnostics", [])[:3]
            if _short_text(item, 120)
        ],
        "operation_signal": str(card.get("operation_signal", "visible_fail_open") or "visible_fail_open"),
        "operation_counts": operation_counts,
        "fact_deltas": bounded_fact_deltas,
        "headline_scores": _story_card_score_payload(card),
    }


def _story_card_score_payload(card: Dict[str, Any]) -> Dict[str, float]:
    signals = [
        item for item in card.get("editorial_signals", [])
        if isinstance(item, dict)
    ]
    return {
        name: round(max((_nonnegative_float(item.get(name)) for item in signals), default=0.0), 4)
        for name in ("priority", "novelty", "impact", "urgency")
    }


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _story_card_priority(card: Dict[str, Any]) -> tuple[float, float, float, float]:
    scores = _story_card_score_payload(card)
    return (
        scores["priority"],
        scores["impact"],
        scores["urgency"],
        scores["novelty"],
    )


def _validated_editor_decisions(
    value: Dict[str, Any],
    cards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_rows = value.get("decisions", []) if isinstance(value, dict) else []
    by_card: Dict[str, List[Dict[str, Any]]] = {}
    for raw in raw_rows if isinstance(raw_rows, list) else []:
        if not isinstance(raw, dict):
            continue
        card_id = str(raw.get("card_id", "") or "").strip()
        if card_id:
            by_card.setdefault(card_id, []).append(raw)

    output: List[Dict[str, Any]] = []
    for card in cards:
        card_id = str(card["card_id"])
        candidates = by_card.get(card_id, [])
        raw = candidates[0] if len(candidates) == 1 else {}
        disposition = str(raw.get("disposition", "full_report") or "full_report").strip()
        if disposition not in {"full_report", "continuing_bullet", "omit"}:
            disposition = "full_report"
        operation_validation = card.get("operation_validation", {})
        unsafe = not bool(operation_validation.get("safe_to_apply", False))
        uncertain_identity = card.get("identity", {}).get("relationship") == "uncertain"
        if disposition == "omit" and (unsafe or uncertain_identity):
            disposition = "full_report"
        try:
            materiality_level = max(0, min(3, int(raw.get("materiality", 0))))
        except (TypeError, ValueError):
            materiality_level = 0
        output.append(
            {
                "card_id": card_id,
                "disposition": disposition,
                "materiality": materiality_level,
                "summary": str(raw.get("summary", "") or card.get("title", "")).strip()[:200],
                "basis": str(
                    raw.get("basis", "")
                    or "Fail-open coverage because the editor decision was missing or invalid."
                ).strip()[:200],
            }
        )
    full_report_indexes = [
        index for index, decision in enumerate(output)
        if decision["disposition"] == "full_report"
    ]
    if len(full_report_indexes) <= MAX_EDITOR_FULL_REPORTS:
        return output
    keep = set(sorted(
        full_report_indexes,
        key=lambda index: (
            int(output[index]["materiality"]),
            *_story_card_priority(cards[index]),
            -index,
        ),
        reverse=True,
    )[:MAX_EDITOR_FULL_REPORTS])
    for index in full_report_indexes:
        if index in keep:
            continue
        output[index]["disposition"] = "continuing_bullet"
        output[index]["basis"] = _short_text(
            f"{output[index]['basis']} Kept as a continuing bullet by the five-story full-report cap.",
            200,
        )
    return output


def _story_delta_packet(
    cards: List[Dict[str, Any]],
    editor_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    editor_by_card = {str(item["card_id"]): item for item in editor_decisions}
    packet: Dict[str, Any] = {
        "story_delta_version": "story-cards.v1",
        "baseline_coverage_note": (
            "Each grouped current story was linked to at most one validated prior baseline; "
            "the editor compared the bounded card set, and prompt-excluded cards failed open."
        ),
        "new": [],
        "escalated": [],
        "weakened": [],
        "reframed": [],
        "unchanged_but_important": [],
        "evidence_gaps": [],
        "story_decisions": [],
        "story_cards": cards,
        "editorial_selection": editor_decisions,
    }
    for card in cards:
        editor = editor_by_card[str(card["card_id"])]
        identity = card["identity"]
        relationship = str(identity["relationship"])
        validation = card["operation_validation"]
        change_type = str(card["change_type"])
        confidence = _bounded_float(identity.get("confidence"), 0.0)
        if not validation.get("safe_to_apply"):
            confidence = min(confidence, 0.49)
        operations = list(card.get("proposed_operations", []))
        prior_ids = list(dict.fromkeys(
            str(prior_id)
            for operation in operations if isinstance(operation, dict)
            for prior_id in operation.get("prior_fact_ids", [])
            if str(prior_id)
        ))
        superseded = list(dict.fromkeys(
            str(prior_id)
            for operation in operations if isinstance(operation, dict)
            if operation.get("operation") in {"replace", "resolve"}
            for prior_id in operation.get("prior_fact_ids", [])
            if str(prior_id)
        ))
        row = {
            "story_key": str(card["story_key"]),
            "article_ids": list(card["article_ids"]),
            "prior_story_key": str(card.get("prior_story_key", "")),
            "relationship": relationship,
            "change_type": change_type,
            "materiality": round(float(editor["materiality"]) / 3.0, 4),
            "confidence": confidence,
            "disposition": str(editor["disposition"]),
            "summary": str(editor["summary"]),
            "bullet": str(editor["summary"]),
            "reason": str(editor["basis"]),
            "knowns": [],
            "unknowns": list(validation.get("diagnostics", [])),
            "watch_signals": [],
            "current_evidence_ids": [
                str(item.get("claim_id", ""))
                for item in card.get("current_facts", []) if isinstance(item, dict)
            ],
            "prior_evidence_ids": prior_ids,
            "superseded_prior_evidence_ids": superseded,
            "operations": operations,
            "editor_safe_to_defer": bool(
                validation.get("safe_to_apply") and relationship != "uncertain"
            ),
        }
        packet["story_decisions"].append(row)
        entry = {
            "item": str(card.get("title", "")),
            "summary": str(editor["summary"]),
            "article_ids": list(card["article_ids"]),
        }
        if change_type in {"new", "incremental"}:
            packet["new"].append(entry)
        elif change_type in {"correction", "resolved"}:
            packet["reframed"].append(entry)
        elif change_type == "unchanged" and editor["disposition"] != "omit":
            packet["unchanged_but_important"].append(entry)
        elif change_type == "uncertain":
            packet["evidence_gaps"].append(
                {
                    "gap": str(editor["summary"]),
                    "why_it_matters": "Identity or delta validation was uncertain; story was kept visible.",
                    "article_ids": list(card["article_ids"]),
                }
            )

    return packet


def _bounded_float(value: Any, default: float) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return default
