from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


IDENTITY_GATE_POLICY_VERSION = "candidate-gated.v1"


@dataclass(frozen=True)
class _ArticleIdentityContext:
    article_id: str
    current_story_key: str
    current_title: str
    allowed_prior_keys: tuple[str, ...]


def enforce_candidate_identity_gate(
    delta_packet: Dict[str, Any] | None,
    story_memory: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Validate model identity decisions against explicitly supplied candidates.

    The model may classify or explain a candidate link, but it cannot create a
    link. With no supplied prior candidate the current item is a new story. A
    same-story decision with an absent or unknown prior key fails open as
    uncertain and cannot suppress or merge the item.
    """

    packet = dict(delta_packet) if isinstance(delta_packet, dict) else {}
    contexts = _article_contexts(story_memory)
    if not contexts:
        packet["identity_gate"] = {
            "policy_version": IDENTITY_GATE_POLICY_VERSION,
            "articles": 0,
            "accepted_links": 0,
            "forced_new_without_candidate": 0,
            "rejected_links": 0,
            "synthesized_decisions": 0,
        }
        return packet

    source_rows = packet.get("story_decisions", [])
    if not isinstance(source_rows, list):
        source_rows = []
    by_article: Dict[str, List[Dict[str, Any]]] = {article_id: [] for article_id in contexts}
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        article_ids = row.get("article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for value in article_ids:
            article_id = str(value or "").strip()
            if article_id in by_article:
                by_article[article_id].append(row)

    gated_rows: List[Dict[str, Any]] = []
    forced_new_ids: set[str] = set()
    current_new_ids: set[str] = set()
    uncertain_ids: set[str] = set()
    accepted_links = 0
    rejected_links = 0
    synthesized = 0
    for article_id, context in contexts.items():
        candidates = by_article.get(article_id, [])
        raw, ambiguous = _select_source_decision(candidates)
        if raw is None:
            synthesized += 1
            raw = {}
        gated, outcome = _gate_one(raw, context, ambiguous=ambiguous)
        gated_rows.append(gated)
        if outcome == "forced_new_without_candidate":
            forced_new_ids.add(article_id)
            current_new_ids.add(article_id)
        elif outcome == "accepted_candidate_link":
            accepted_links += 1
        elif outcome in {"accepted_distinct_story", "accepted_related_theme"}:
            current_new_ids.add(article_id)
        elif outcome in {"rejected_candidate_link", "accepted_uncertain"}:
            if outcome == "rejected_candidate_link":
                rejected_links += 1
            uncertain_ids.add(article_id)

    packet["story_decisions"] = gated_rows
    if current_new_ids or uncertain_ids:
        _reconcile_editorial_lists(packet, new_ids=current_new_ids, uncertain_ids=uncertain_ids)
    packet["identity_gate"] = {
        "policy_version": IDENTITY_GATE_POLICY_VERSION,
        "articles": len(contexts),
        "accepted_links": accepted_links,
        "forced_new_without_candidate": len(forced_new_ids),
        "rejected_links": rejected_links,
        "synthesized_decisions": synthesized,
    }
    return packet


def _article_contexts(story_memory: Dict[str, Any] | None) -> Dict[str, _ArticleIdentityContext]:
    stories = story_memory.get("stories", []) if isinstance(story_memory, dict) else []
    if not isinstance(stories, list):
        return {}
    output: Dict[str, _ArticleIdentityContext] = {}
    for story in stories:
        if not isinstance(story, dict):
            continue
        current_key = str(story.get("story_key", "") or "").strip()
        current_title = str(story.get("current_title", "") or "").strip()
        baselines = story.get("prior_baselines", [])
        allowed = tuple(
            dict.fromkeys(
                str(item.get("story_key", "") or "").strip()
                for item in baselines if isinstance(item, dict)
                if str(item.get("story_key", "") or "").strip()
            )
        ) if isinstance(baselines, list) else ()
        article_ids = story.get("current_article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for value in article_ids:
            article_id = str(value or "").strip()
            if not article_id:
                continue
            output[article_id] = _ArticleIdentityContext(
                article_id=article_id,
                current_story_key=current_key or f"article:{article_id}",
                current_title=current_title,
                allowed_prior_keys=allowed,
            )
    return output


def _select_source_decision(
    rows: List[Dict[str, Any]],
) -> tuple[Dict[str, Any] | None, bool]:
    if not rows:
        return None, False
    signatures = {
        (
            str(row.get("relationship", "") or "").strip(),
            str(row.get("prior_story_key", "") or "").strip(),
            str(row.get("change_type", "") or "").strip(),
            str(row.get("disposition", "") or "").strip(),
        )
        for row in rows
    }
    selected = max(rows, key=lambda row: _bounded(row.get("confidence"), 0.0))
    return dict(selected), len(signatures) > 1


def _gate_one(
    raw: Dict[str, Any],
    context: _ArticleIdentityContext,
    *,
    ambiguous: bool,
) -> tuple[Dict[str, Any], str]:
    row = dict(raw)
    row["article_ids"] = [context.article_id]
    row["story_key"] = context.current_story_key
    relationship = str(row.get("relationship", "uncertain") or "uncertain").strip()
    prior_key = str(row.get("prior_story_key", "") or "").strip()
    allowed = set(context.allowed_prior_keys)

    if not allowed:
        row.update(
            {
                "prior_story_key": "",
                "relationship": "distinct_story",
                "change_type": "new",
                "materiality": 1.0,
                "confidence": max(0.8, _bounded(row.get("confidence"), 0.0)),
                "disposition": "full_report",
                "reason": _append_reason(
                    row.get("reason"),
                    "No prior candidate was supplied; architecture treats this as a new story.",
                ),
            }
        )
        outcome = "forced_new_without_candidate"
    elif ambiguous or (relationship == "same_story" and prior_key not in allowed):
        row.update(
            {
                "prior_story_key": "",
                "relationship": "uncertain",
                "change_type": "uncertain",
                "materiality": max(0.5, _bounded(row.get("materiality"), 0.0)),
                "confidence": min(0.49, _bounded(row.get("confidence"), 0.0)),
                "disposition": "full_report",
                "reason": _append_reason(
                    row.get("reason"),
                    "Same-story link was absent, ambiguous, or outside the supplied candidate set.",
                ),
            }
        )
        outcome = "rejected_candidate_link"
    elif relationship == "same_story":
        row["story_key"] = prior_key
        row["prior_story_key"] = prior_key
        outcome = "accepted_candidate_link"
    elif relationship == "distinct_story":
        row.update(
            {
                "prior_story_key": "",
                "change_type": "new",
                "materiality": max(0.7, _bounded(row.get("materiality"), 0.0)),
                "disposition": "full_report",
            }
        )
        outcome = "accepted_distinct_story"
    elif relationship == "related_theme":
        row.update(
            {
                "prior_story_key": "",
                "change_type": "new",
                "disposition": "full_report",
            }
        )
        outcome = "accepted_related_theme"
    else:
        row.update(
            {
                "prior_story_key": "",
                "relationship": "uncertain",
                "change_type": "uncertain",
                "materiality": max(0.5, _bounded(row.get("materiality"), 0.0)),
                "disposition": "full_report",
            }
        )
        outcome = "accepted_uncertain"

    row.setdefault("summary", "")
    row.setdefault("bullet", context.current_title)
    row.setdefault("reason", "")
    row.setdefault("knowns", [])
    row.setdefault("unknowns", [])
    row.setdefault("watch_signals", [])
    row["identity_gate"] = {
        "policy_version": IDENTITY_GATE_POLICY_VERSION,
        "outcome": outcome,
        "allowed_prior_story_keys": list(context.allowed_prior_keys),
    }
    return row, outcome


def _reconcile_editorial_lists(
    packet: Dict[str, Any],
    *,
    new_ids: set[str],
    uncertain_ids: set[str],
) -> None:
    affected = new_ids.union(uncertain_ids)
    for name in ("new", "escalated", "weakened", "reframed", "unchanged_but_important"):
        rows = packet.get(name, [])
        if not isinstance(rows, list):
            packet[name] = []
            continue
        packet[name] = [row for row in rows if not _row_references_any(row, affected)]

    current_rows = {str(row.get("article_ids", [""])[0]): row for row in packet["story_decisions"]}
    new_rows = packet.setdefault("new", [])
    for article_id in sorted(new_ids):
        decision = current_rows.get(article_id, {})
        new_rows.append(
            {
                "item": str(decision.get("bullet", "") or decision.get("summary", "") or article_id),
                "summary": str(decision.get("reason", "") or "First source-backed observation."),
                "article_ids": [article_id],
            }
        )
    if uncertain_ids:
        gaps = packet.setdefault("evidence_gaps", [])
        if not isinstance(gaps, list):
            gaps = []
            packet["evidence_gaps"] = gaps
        gaps.append(
            {
                "gap": f"Identity could not be validated for {len(uncertain_ids)} article(s).",
                "why_it_matters": "Unvalidated links remain visible and cannot merge or suppress a story.",
                "article_ids": sorted(uncertain_ids),
            }
        )


def _row_references_any(row: Any, article_ids: Iterable[str]) -> bool:
    if not isinstance(row, dict):
        return False
    values = row.get("article_ids", [])
    if not isinstance(values, list):
        return False
    targets = set(article_ids)
    return any(str(value or "").strip() in targets for value in values)


def _append_reason(value: Any, addition: str) -> str:
    current = " ".join(str(value or "").split()).strip()
    return f"{current} {addition}".strip()


def _bounded(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
