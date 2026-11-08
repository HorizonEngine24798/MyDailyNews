from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any, Dict, List

from mydailynews.ai.prompts import (
    PERSPECTIVES_PLANNER_RETRY_USER,
    PERSPECTIVES_PLANNER_SYSTEM,
    PERSPECTIVES_PLANNER_USER,
)
from mydailynews.ai.token_budget import resolve_client_token_budget
from mydailynews.app.models import BriefOutput
from mydailynews.common.utils import compact_json, normalize_whitespace, stable_id
from mydailynews.common.warnings import extend_warnings
from mydailynews.perspectives.rendering import (
    _article_summary,
    _as_list,
    _empty_plan,
    _normalize_anchor_groups,
    _short_text,
    _story_article_ids,
    _story_identity,
    _string_list,
    _text_list,
    _unique_text,
    normalize_name,
)
from mydailynews.perspectives.query_alignment import MULTILINGUAL_ASSIST_LOOKUP
from mydailynews.perspectives.schemas import (
    COUNTRY_ALIASES,
    EVIDENCE_TYPES,
    LANGUAGE_ALIASES,
    MAX_CANONICAL_QUERIES,
    MAX_PLANNER_STORIES_PER_CALL,
    PERSPECTIVES_PLANNER_SCHEMA,
    REGION_ALIASES,
    STRUCTURED_BRIEF_NAMES,
    TAG_RE,
)
from mydailynews.perspectives.sources import load_source_registry


def collect_perspectives_inputs(
    *,
    output_dir: Path,
    date: str,
    source_outputs: List[BriefOutput] | None = None,
    allow_disk_fallback: bool = True,
) -> Dict[str, Any]:
    warnings: List[str] = []
    brief_payloads: List[Dict[str, Any]] = []
    brief_articles: List[Dict[str, Any]] = []
    source_briefs: List[str] = []
    outputs_by_name = {
        str(output.name or "").strip().lower(): output
        for output in source_outputs or []
        if str(output.name or "").strip().lower() in STRUCTURED_BRIEF_NAMES
    }
    for brief_name in STRUCTURED_BRIEF_NAMES:
        current_output = outputs_by_name.get(brief_name)
        path = Path(current_output.json_path) if current_output and str(current_output.json_path or "").strip() else None
        if path is None and allow_disk_fallback:
            path = output_dir / f"{date}_{brief_name}_brief.json"
        if path is None or not path.exists():
            continue
        payload = _load_json(path, warnings, f"{brief_name} brief")
        if not payload:
            continue
        brief_payloads.append(payload)
        source_briefs.append(brief_name)
        for item in _as_list(payload.get("selected_articles")):
            article = _article_from_selected(item)
            if article["id"] or article["source"]:
                article["source_briefs"] = sorted({*article["source_briefs"], brief_name})
                brief_articles.append(article)

    articles = _dedupe_articles(brief_articles)
    stories = _stories_from_briefs(brief_payloads, articles)
    _attach_claims_to_stories(stories, brief_payloads, warnings)
    return {
        "mode": "briefs",
        "path": "",
        "source_briefs": source_briefs,
        "articles": articles,
        "stories": stories,
        "warnings": warnings,
    }


def plan_perspectives_queries(
    orchestrator,
    *,
    date: str,
    inputs: Dict[str, Any],
    config: Any,
    source_registry: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    stories = [story for story in _as_list(inputs.get("stories")) if isinstance(story, dict)]
    failed = {_story_identity(story): _empty_plan(story, "planner_failed", ["planner was not run"]) for story in stories}
    ai_client = getattr(orchestrator, "summary_ai_client", None)
    if ai_client is None:
        warnings.append("perspectives_report: planner failed because no summary AI client is available.")
        return failed
    if not source_registry:
        warnings.append("perspectives_report: planner skipped because the active source registry is empty.")
        return failed

    budget = resolve_client_token_budget(ai_client, output_tokens=2400)
    batches = [stories[index : index + MAX_PLANNER_STORIES_PER_CALL] for index in range(0, len(stories), MAX_PLANNER_STORIES_PER_CALL)]
    plans: Dict[str, Dict[str, Any]] = {}
    for batch_index, batch_stories in enumerate(batches, start=1):
        prompt = _planner_prompt(
            date=date,
            inputs={**inputs, "stories": batch_stories},
            source_registry=source_registry,
            config=config,
        )
        batch_label = "perspectives planner" if len(batches) == 1 else f"perspectives planner batch {batch_index}/{len(batches)}"
        last_plans = {_story_identity(story): failed[_story_identity(story)] for story in batch_stories}
        last_warnings: List[str] = []
        for attempt in range(1, 3):
            user_prompt = prompt if attempt == 1 else PERSPECTIVES_PLANNER_RETRY_USER.format(prompt=prompt)
            try:
                raw = ai_client.complete_json(
                    PERSPECTIVES_PLANNER_SYSTEM,
                    user_prompt,
                    label=batch_label,
                    max_new_tokens=budget.output_tokens,
                    input_token_limit=budget.input_tokens,
                    json_schema=PERSPECTIVES_PLANNER_SCHEMA,
                )
            except Exception as exc:
                warnings.append(f"perspectives_report: {batch_label} failed after AI error: {type(exc).__name__}: {exc}")
                break
            last_warnings = []
            last_plans = _validate_planner_response(
                raw,
                stories=batch_stories,
                source_registry=source_registry,
                warnings=last_warnings,
            )
            if not _planner_needs_tag_retry(last_plans):
                break
        extend_warnings(warnings, last_warnings)
        plans.update(last_plans)
    return plans


def _planner_prompt(*, date: str, inputs: Dict[str, Any], source_registry: List[Dict[str, Any]], config: Any) -> str:
    articles_by_id = {article["id"]: article for article in inputs.get("articles", []) if article.get("id")}
    payload = {
        "report_date": date,
        "lookback_days": int(getattr(config, "coverage_timespan_days", 7) or 7),
        "stories": [_story_planner_payload(story, articles_by_id) for story in _as_list(inputs.get("stories")) if isinstance(story, dict)],
        "tag_options": _planner_tag_options(source_registry),
        "multilingual_assist": MULTILINGUAL_ASSIST_LOOKUP,
    }
    return PERSPECTIVES_PLANNER_USER.format(date=date, data=compact_json(payload))


def _validate_planner_response(
    raw: Dict[str, Any],
    *,
    stories: List[Dict[str, Any]],
    source_registry: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    plans = {_story_identity(story): _empty_plan(story, "planner_failed", ["missing planner line"]) for story in stories}
    planner_items = _planner_items(raw, warnings)
    if not planner_items:
        warnings.append("perspectives_report: planner returned no plans.")
        return plans

    story_by_id = {_story_identity(story): story for story in stories}
    for plan_number, item in planner_items:
        story_id = str(item.get("story_id") or "").strip()
        if story_id not in story_by_id:
            warnings.append(f"perspectives_report: planner plan {plan_number} used unknown story_id {story_id!r}; discarded.")
            continue
        plans[story_id] = _normalize_planner_item(item, story_by_id[story_id], source_registry)
        if plans[story_id].get("status") != "ok":
            warnings.append(
                f"perspectives_report: planner plan {plan_number} for {story_id!r} was incomplete: "
                + "; ".join(_string_list(plans[story_id].get("diagnostics")))
            )
    return plans


def _planner_items(raw: Dict[str, Any], warnings: List[str]) -> List[tuple[int, Dict[str, Any]]]:
    if not isinstance(raw, dict):
        return []
    raw_plans = raw.get("plans")
    if not isinstance(raw_plans, list):
        return []
    items = []
    for plan_number, item in enumerate(raw_plans, start=1):
        if not isinstance(item, dict):
            warnings.append(f"perspectives_report: planner plan {plan_number} was not an object.")
            continue
        items.append((plan_number, item))
    return items


def _planner_needs_tag_retry(plans: Dict[str, Dict[str, Any]]) -> bool:
    return any(
        str(diagnostic).startswith("missing required target tags") or str(diagnostic) == "no usable target tags"
        for plan in plans.values()
        for diagnostic in _string_list(plan.get("diagnostics"))
    )


def _normalize_planner_item(item: Dict[str, Any], story: Dict[str, Any], source_registry: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_tags = item.get("target_tags") if isinstance(item.get("target_tags"), dict) else {}
    normalized_tags, rejected_tags = _normalize_target_tags(raw_tags, source_registry)
    diagnostics = []
    if any(rejected_tags.values()):
        diagnostics.append(f"unmatched target tags: {compact_json(rejected_tags)}")

    queries = _text_list(item.get("queries"), max_items=MAX_CANONICAL_QUERIES, max_chars=180)
    anchor_groups = _normalize_anchor_groups(item.get("anchor_groups"))
    if not anchor_groups:
        anchor_groups = _seed_anchor_groups(story)
    if not queries:
        diagnostics.append("no usable canonical query")
    story_loci = []
    for raw_locus in _as_list(item.get("story_loci"))[:3]:
        if not isinstance(raw_locus, dict):
            continue
        label = _short_text(raw_locus.get("label"), 100)
        kind = str(raw_locus.get("kind") or "").strip().lower()
        confidence = str(raw_locus.get("confidence") or "").strip().lower()
        if not label or kind not in {"event_site", "affected_area"} or confidence not in {"high", "medium", "low"}:
            continue
        story_loci.append(
            {
                "label": label,
                "country": str(raw_locus.get("country") or "").strip().upper()[:3],
                "kind": kind,
                "confidence": confidence,
                "reason": _short_text(raw_locus.get("reason"), 240),
            }
        )
    missing_tag_groups = [name for name in ("countries", "regions") if not normalized_tags.get(name)]
    if missing_tag_groups:
        diagnostics.append("missing required target tags: " + ", ".join(missing_tag_groups))
    if len(missing_tag_groups) == 2:
        diagnostics.append("no usable target tags")
    known_claims = {str(claim.get("claim_id") or ""): claim for claim in _as_list(story.get("claims")) if isinstance(claim, dict)}
    verification_targets: List[Dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    for raw_target in _as_list(item.get("verification_targets"))[:2]:
        if not isinstance(raw_target, dict):
            continue
        claim_id = str(raw_target.get("claim_id") or "").strip()
        if claim_id not in known_claims or claim_id in seen_claim_ids:
            if claim_id:
                diagnostics.append(f"unknown or duplicate verification claim_id: {claim_id}")
            continue
        target_queries = []
        for raw_query in _as_list(raw_target.get("queries"))[:2]:
            if not isinstance(raw_query, dict):
                continue
            query = _short_text(raw_query.get("query"), 180)
            evidence_type = str(raw_query.get("evidence_type") or "").strip().lower()
            if query and evidence_type in EVIDENCE_TYPES:
                target_queries.append({"query": query, "evidence_type": evidence_type})
        required_evidence_types = [value for value in _string_list(raw_target.get("required_evidence_types")) if value in EVIDENCE_TYPES]
        missing_evidence_types = set(required_evidence_types).difference(
            str(query.get("evidence_type") or "") for query in target_queries
        )
        if missing_evidence_types:
            diagnostics.append(f"verification target {claim_id} lacked required query types: {', '.join(sorted(missing_evidence_types))}")
            continue
        if not target_queries:
            diagnostics.append(f"verification target {claim_id} had no usable targeted query")
            continue
        seen_claim_ids.add(claim_id)
        verification_targets.append(
            {
                "claim_id": claim_id,
                "importance_reason": normalize_whitespace(str(raw_target.get("importance_reason") or "")),
                "required_evidence_types": required_evidence_types,
                "queries": target_queries,
            }
        )
    raw_aligned = item.get("aligned_queries") if isinstance(item.get("aligned_queries"), dict) else {}
    aligned_queries = {
        str(k).lower(): _text_list(v, max_items=MAX_CANONICAL_QUERIES, max_chars=180)
        for k, v in raw_aligned.items()
        if isinstance(v, list) and _text_list(v, max_items=MAX_CANONICAL_QUERIES, max_chars=180)
    }
    return {
        "story_id": _story_identity(story),
        "status": "ok" if queries and not missing_tag_groups else "planner_failed",
        "queries": queries,
        "aligned_queries": aligned_queries,
        "anchor_groups": anchor_groups,
        "story_loci": story_loci,
        "target_tags": {
            "raw": {
                "countries": _string_list(raw_tags.get("countries") if isinstance(raw_tags, dict) else []),
                "regions": _string_list(raw_tags.get("regions") if isinstance(raw_tags, dict) else []),
                "languages": _string_list(raw_tags.get("languages") if isinstance(raw_tags, dict) else []),
            },
            "normalized": normalized_tags,
            "rejected": rejected_tags,
        },
        "diagnostics": diagnostics,
        "verification_targets": verification_targets,
    }


def _normalize_target_tags(raw_tags: Dict[str, Any], sources: List[Dict[str, Any]]) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    countries = {str(source.get("country") or "").upper() for source in sources if source.get("country")}
    languages = {str(source.get("language") or "").lower() for source in sources if source.get("language")}
    regions = {str(region or "").lower() for source in sources for region in source.get("regions") or [] if str(region or "").strip()}
    normalized_countries, rejected_countries = _normalize_tag_values(_string_list(raw_tags.get("countries")), countries, COUNTRY_ALIASES, upper=True)
    normalized_languages, rejected_languages = _normalize_tag_values(_string_list(raw_tags.get("languages")), languages, LANGUAGE_ALIASES)
    normalized_regions, rejected_regions = _normalize_tag_values(_string_list(raw_tags.get("regions")), regions, REGION_ALIASES)
    return (
        {
            "countries": normalized_countries,
            "languages": normalized_languages,
            "regions": normalized_regions,
        },
        {
            "countries": rejected_countries,
            "languages": rejected_languages,
            "regions": rejected_regions,
        },
    )


def _config_fallback_tags(config: Any, sources: List[Dict[str, Any]]) -> tuple[Dict[str, List[str]], List[str]]:
    raw = {
        "countries": _string_list(getattr(config, "coverage_scope", [])),
        "languages": [],
        "regions": _string_list(getattr(config, "coverage_regions", [])),
    }
    if not any(raw.values()):
        return {"countries": [], "languages": [], "regions": []}, []
    normalized, rejected = _normalize_target_tags(raw, sources)
    diagnostics = ["config coverage tags had unmatched values: " + compact_json(rejected)] if any(rejected.values()) else []
    return normalized, diagnostics


def _normalize_tag_values(
    values: List[str],
    known_values: set[str],
    aliases: Dict[str, str],
    *,
    upper: bool = False,
) -> tuple[List[str], List[str]]:
    known_map = {_tag_key(value): (value.upper() if upper else value.lower()) for value in known_values if value}
    alias_map = {key: (value.upper() if upper else value.lower()) for key, value in aliases.items()}
    output: List[str] = []
    rejected: List[str] = []
    for value in values:
        key = _tag_key(value)
        match = alias_map.get(key) or known_map.get(key) or _fuzzy_tag_match(key, {**known_map, **alias_map})
        if match:
            if match not in output:
                output.append(match)
        else:
            rejected.append(value)
    return output, rejected


def _fuzzy_tag_match(key: str, candidates: Dict[str, str]) -> str:
    if not key:
        return ""
    scored = sorted(
        ((SequenceMatcher(None, key, candidate).ratio(), candidate) for candidate in candidates),
        reverse=True,
    )
    if not scored or scored[0][0] < 0.9:
        return ""
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.04:
        return ""
    return candidates[scored[0][1]]


def _tag_key(value: Any) -> str:
    return TAG_RE.sub("_", str(value or "").strip().lower()).strip("_")


def _source_metadata(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": str(source.get("source_id") or ""),
        "name": str(source.get("name") or ""),
        "country": str(source.get("country") or "").upper(),
        "language": str(source.get("language") or "").lower(),
        "regions": _string_list(source.get("regions")),
        "category": str(source.get("category") or ""),
        "tags": _string_list(source.get("tags")),
    }


def _planner_tag_options(sources: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    enabled = [source for source in sources if bool(source.get("enabled", True))]
    return {
        "countries": sorted({str(source.get("country") or "").upper() for source in enabled if source.get("country")}),
        "regions": sorted({str(region or "").lower() for source in enabled for region in source.get("regions") or [] if str(region or "").strip()}),
    }


def _seed_anchor_groups(story: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = _text_list(story.get("entities"), max_items=6, max_chars=100)
    return [{"kind": "entity", "terms": entities}] if entities else []


def _story_planner_payload(story: Dict[str, Any], articles_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    seed_articles = [articles_by_id[article_id] for article_id in _story_article_ids(story) if article_id in articles_by_id]
    return {
        "story_id": _story_identity(story),
        "title": str(story.get("story_title") or story.get("title") or ""),
        "summary": _short_text(story.get("summary") or _article_summary(seed_articles), 800),
        "entities": _string_list(story.get("entities")),
        "seed_headlines": [_short_text(article.get("headline"), 180) for article in seed_articles[:8]],
        "seed_sources": _unique_text([str(article.get("source") or "") for article in seed_articles if article.get("source")])[:8],
        "claims": [
            {
                "claim_id": str(claim.get("claim_id") or ""),
                "claim": _short_text(claim.get("claim"), 320),
                "claimant": _short_text(claim.get("claimant"), 120),
                "claim_type": str(claim.get("claim_type") or "other"),
                "confidence": str(claim.get("confidence") or ""),
            }
            for claim in _as_list(story.get("claims"))
            if isinstance(claim, dict)
        ],
        "confirmed_facts": _as_list(story.get("confirmed_facts")),
        "conflicting_claims": _as_list(story.get("conflicting_claims")),
        "open_questions": _as_list(story.get("open_questions")),
    }


def _seed_sources(articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    grouped: Dict[str, Dict[str, str]] = {}
    for article in articles:
        source = str(article.get("source") or "Unknown source")
        key = normalize_name(source) or "unknown"
        grouped.setdefault(key, {"source_key": key, "source": source, "article_count": 0})
        grouped[key]["article_count"] = str(int(grouped[key]["article_count"]) + 1)
    return list(grouped.values())


def _load_coverage_registry() -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        return load_source_registry(), []
    except Exception as exc:
        return [], [f"perspectives_report: failed to load source registry ({type(exc).__name__}: {exc})."]


def _load_json(path: Path, warnings: List[str], label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        warnings.append(f"perspectives_report: failed to load {label} JSON ({type(exc).__name__}: {exc}).")
        return {}
    return payload if isinstance(payload, dict) else {}


def _attach_claims_to_stories(
    stories: List[Dict[str, Any]],
    brief_payloads: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    candidates: List[tuple[set[str], Dict[str, Any]]] = []
    for payload in brief_payloads:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        evidence = analysis.get("evidence_packet") if isinstance(analysis.get("evidence_packet"), dict) else {}
        for cluster in _as_list(evidence.get("story_clusters")):
            if not isinstance(cluster, dict):
                continue
            boundary = set(_string_list(cluster.get("article_ids")))
            for raw_claim in _as_list(cluster.get("key_claims")):
                if not isinstance(raw_claim, dict):
                    continue
                claim_text = normalize_whitespace(str(raw_claim.get("claim") or ""))
                support_ids = [item for item in _string_list(raw_claim.get("support_article_ids")) if item in boundary]
                if not claim_text or not support_ids:
                    continue
                origin_ids = [item for item in _string_list(raw_claim.get("origin_article_ids")) if item in support_ids]
                claim = {
                    "claim_id": f"claim-{stable_id(claim_text.lower())}",
                    "claim": claim_text,
                    "claimant": normalize_whitespace(str(raw_claim.get("claimant") or "")),
                    "claim_type": normalize_whitespace(str(raw_claim.get("claim_type") or "other")) or "other",
                    "support_article_ids": support_ids,
                    "origin_article_ids": origin_ids,
                    "confidence": normalize_whitespace(str(raw_claim.get("confidence") or "")),
                }
                candidates.append((boundary | set(support_ids), claim))

    matched_ids: set[str] = set()
    for story in stories:
        story_ids = set(_story_article_ids(story))
        by_key: Dict[str, Dict[str, Any]] = {}
        for boundary, claim in candidates:
            overlap = story_ids.intersection(boundary).intersection(claim["support_article_ids"])
            if not overlap:
                continue
            key = normalize_name(claim["claim"])
            current = by_key.get(key)
            if current is None:
                current = dict(claim)
                current["support_article_ids"] = []
                current["origin_article_ids"] = []
                by_key[key] = current
            current["support_article_ids"] = _unique_text([*current["support_article_ids"], *[item for item in claim["support_article_ids"] if item in story_ids]])
            current["origin_article_ids"] = _unique_text([*current["origin_article_ids"], *[item for item in claim["origin_article_ids"] if item in story_ids]])
            matched_ids.add(claim["claim_id"])
        story["claims"] = list(by_key.values())
    unmatched = {claim["claim_id"] for _boundary, claim in candidates}.difference(matched_ids)
    if unmatched:
        warnings.append(f"perspectives_report: discarded {len(unmatched)} evidence claim(s) with no supporting-article overlap.")


def _stories_from_briefs(
    brief_payloads: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {article["id"]: article for article in articles if article.get("id")}
    known_ids = set(by_id)
    stories: List[Dict[str, Any]] = []
    used_story_ids: set[str] = set()

    def _merge_or_append(
        *,
        raw_id: str,
        raw_title: str,
        raw_summary: str,
        article_ids: List[str],
        confirmed_facts: List[Any] | None = None,
        conflicting_claims: List[Any] | None = None,
        open_questions: List[Any] | None = None,
    ) -> None:
        if not article_ids:
            return
        id_set = set(article_ids)
        for existing in stories:
            if id_set.intersection(existing["article_ids"]):
                existing["article_ids"] = _unique_text([*existing["article_ids"], *article_ids])
                if not existing.get("summary") and raw_summary:
                    existing["summary"] = raw_summary
                existing["confirmed_facts"] = _unique_text([*_string_list(existing.get("confirmed_facts")), *_string_list(confirmed_facts)])
                existing["conflicting_claims"] = _unique_text([*_string_list(existing.get("conflicting_claims")), *_string_list(conflicting_claims)])
                existing["open_questions"] = _unique_text([*_string_list(existing.get("open_questions")), *_string_list(open_questions)])
                return
        lead = by_id.get(article_ids[0], {})
        story_id = raw_id or str(lead.get("story_key") or lead.get("id") or f"story-{len(stories) + 1}")
        if story_id in used_story_ids:
            story_id = f"{story_id}-{len(stories) + 1}"
        used_story_ids.add(story_id)
        story_articles = [by_id[aid] for aid in article_ids if aid in by_id]
        story_title = raw_title or str(lead.get("headline") or lead.get("topic") or "Story")
        summary = raw_summary or _articles_summary(story_articles)
        stories.append(
            {
                "story_id": story_id,
                "story_title": story_title,
                "summary": summary,
                "entities": [],
                "article_ids": article_ids,
                "confirmed_facts": _as_list(confirmed_facts),
                "conflicting_claims": _as_list(conflicting_claims),
                "open_questions": _as_list(open_questions),
            }
        )

    for payload in brief_payloads:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        evidence = analysis.get("evidence_packet") if isinstance(analysis.get("evidence_packet"), dict) else {}
        for cluster in _as_list(evidence.get("story_clusters")):
            if not isinstance(cluster, dict):
                continue
            article_ids = [item for item in _string_list(cluster.get("article_ids")) if item in known_ids]
            if not article_ids:
                continue
            story_articles = [by_id[aid] for aid in article_ids if aid in by_id]
            _merge_or_append(
                raw_id=str(cluster.get("story_id") or cluster.get("cluster_id") or "").strip(),
                raw_title=str(cluster.get("story_title") or cluster.get("label") or "").strip(),
                raw_summary=_thread_summary(cluster) or _articles_summary(story_articles),
                article_ids=article_ids,
                confirmed_facts=_as_list(cluster.get("confirmed_facts") or cluster.get("consensus_points")),
                conflicting_claims=_as_list(cluster.get("conflicting_claims") or cluster.get("contested_points")),
                open_questions=_as_list(cluster.get("open_questions") or cluster.get("known_unknowns")),
            )

    for payload in brief_payloads:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        delta = analysis.get("delta_packet") if isinstance(analysis.get("delta_packet"), dict) else {}
        decisions_by_key = {
            str(item.get("story_key") or "").strip(): item
            for item in _as_list(delta.get("story_decisions"))
            if isinstance(item, dict) and str(item.get("story_key") or "").strip()
        }
        for card in _as_list(delta.get("story_cards")):
            if not isinstance(card, dict):
                continue
            article_ids = [item for item in _string_list(card.get("article_ids")) if item in known_ids]
            if not article_ids:
                continue
            story_key = str(card.get("story_key") or card.get("current_story_key") or "").strip()
            decision = decisions_by_key.get(story_key, {})
            story_articles = [by_id[aid] for aid in article_ids if aid in by_id]
            _merge_or_append(
                raw_id=story_key or str(card.get("card_id") or "").strip(),
                raw_title=str(card.get("title") or "").strip(),
                raw_summary=_short_text(decision.get("summary"), 900) or _articles_summary(story_articles),
                article_ids=article_ids,
                confirmed_facts=_as_list(decision.get("knowns")),
                open_questions=_as_list(decision.get("unknowns")),
            )

    if not stories:
        return _fallback_stories(articles)
    covered_ids = {aid for story in stories for aid in story.get("article_ids", [])}
    uncovered = [article for article in articles if article.get("id") not in covered_ids]
    if uncovered:
        for fallback in _fallback_stories(uncovered):
            _merge_or_append(
                raw_id=str(fallback.get("story_id") or ""),
                raw_title=str(fallback.get("story_title") or ""),
                raw_summary=str(fallback.get("summary") or ""),
                article_ids=_string_list(fallback.get("article_ids")),
            )
    return stories


def _articles_summary(articles: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for article in articles[:4]:
        headline = normalize_whitespace(str(article.get("headline") or ""))
        snippet = normalize_whitespace(str(article.get("snippet") or ""))
        if snippet and snippet.lower() != headline.lower():
            parts.append(f"{headline}. {snippet}" if headline and not snippet.startswith(headline[:30]) else snippet)
        elif headline:
            parts.append(headline)
    return _short_text(" ".join(parts), 900)


def _thread_summary(thread: Dict[str, Any]) -> str:
    direct = _short_text(thread.get("summary"), 900)
    if direct:
        return direct
    claims = []
    for item in _as_list(thread.get("key_claims"))[:4]:
        if isinstance(item, dict):
            claims.append(str(item.get("claim") or item.get("summary") or ""))
        else:
            claims.append(str(item or ""))
    if not claims:
        for item in _as_list(thread.get("internal_articles"))[:3]:
            if isinstance(item, dict):
                claims.append(str(item.get("summary") or item.get("title") or ""))
    return _short_text(" ".join(claims), 900)


def _fallback_stories(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    grouped_articles: Dict[str, List[Dict[str, Any]]] = {}
    for article in articles:
        key = str(article.get("story_key") or article.get("story_family_key") or article.get("topic") or article["id"])
        group = grouped.setdefault(
            key,
            {
                "story_id": key,
                "story_title": article.get("headline") or article.get("topic") or "Selected Articles",
                "summary": "",
                "entities": [],
                "article_ids": [],
            },
        )
        group["article_ids"].append(article["id"])
        grouped_articles.setdefault(key, []).append(article)
    for key, group in grouped.items():
        group["summary"] = _articles_summary(grouped_articles.get(key, []))
    return list(grouped.values())


def _article_from_selected(item: Any) -> Dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    article_id = str(item.get("id") or candidate.get("id") or decision.get("candidate_id") or "").strip()
    headline = str(item.get("headline") or item.get("title") or candidate.get("title") or "").strip()
    source = str(item.get("source") or candidate.get("source") or "").strip()
    url = str(item.get("url") or candidate.get("url") or "").strip()
    if not article_id:
        article_id = normalize_name(f"{source} {headline}") or normalize_name(url)
    return {
        "id": article_id,
        "headline": headline,
        "source": source,
        "url": url,
        "snippet": str(item.get("snippet") or candidate.get("snippet") or ""),
        "published_at": str(item.get("published_at") or candidate.get("published_at") or ""),
        "topic": str(item.get("topic") or decision.get("topic") or metadata.get("topic_name") or "").strip(),
        "story_key": str(item.get("story_key") or metadata.get("memory_story_key") or "").strip(),
        "story_family_key": str(item.get("story_family_key") or metadata.get("memory_story_family_key") or "").strip(),
        "source_briefs": _string_list(metadata.get("source_briefs") or item.get("source_briefs")),
    }


def _dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for article in articles:
        existing = by_id.get(article["id"])
        if existing is None:
            by_id[article["id"]] = article
            continue
        existing["source_briefs"] = sorted({*existing.get("source_briefs", []), *article.get("source_briefs", [])})
    return list(by_id.values())
