from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List

from mydailynews.ai.base import AIClient, AIJsonError
from mydailynews.ai.prompts import NARRATIVE_BRIEF_SYSTEM, NARRATIVE_BRIEF_USER
from mydailynews.ai.schemas import NARRATIVE_BRIEF_JSON_SCHEMA
from mydailynews.ai.token_budget import TokenBudget, resolve_client_token_budget
from mydailynews.app.models import UserMemory
from mydailynews.briefing.output import write_json
from mydailynews.common.utils import compact_json
from mydailynews.diagnostics.debug import DebugLogger


_URL_TEXT_RE = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_CLAIM_MARKER_RE = re.compile(r"<<(\d+)>>")
_LINK_KEYS = {
    "href",
    "json_path",
    "link",
    "links",
    "markdown_path",
    "path",
    "resolved_url",
    "uri",
    "url",
    "urls",
}
_NOISY_METADATA_KEYS = {
    "analysis_rollout",
    "candidate_count",
    "composite_ranking_enabled",
    "memory",
    "model",
    "prior_reports_count",
    "selected_count",
    "selection_reason_codes",
    "warnings",
}


@dataclass(frozen=True)
class NarrativeSourceBrief:
    name: str
    json_path: str
    brief: Dict[str, Any]


class NarrativeBriefGenerator:
    def __init__(
        self,
        client: AIClient,
        *,
        input_token_limit: int | None = None,
        max_new_tokens: int | None = None,
        target_words: int = 1800,
        editorial_style: str = "",
        debug: DebugLogger | None = None,
    ) -> None:
        self.client = client
        self.input_token_limit = input_token_limit
        self.max_new_tokens = max_new_tokens
        self.requested_target_words = max(100, int(target_words))
        output_tokens = self._request_budget().output_tokens
        self.target_words = min(self.requested_target_words, max(100, (output_tokens - 256) * 3 // 4))
        self.editorial_style = str(editorial_style or "").strip()
        self.debug = debug or DebugLogger(False)
        self.warnings: List[str] = []

    def generate(
        self,
        source_briefs: List[NarrativeSourceBrief],
        memory: UserMemory,
        *,
        date: str,
        enrichment_payload: Dict[str, Any] | None = None,
        enrichment_json_path: str = "",
        recall_packet: Dict[str, Any] | None = None,
        claim_cards: List[Dict[str, Any]] | None = None,
        perspectives_payload: Dict[str, Any] | None = None,
        perspectives_json_path: str = "",
    ) -> Dict[str, Any]:
        self.warnings = []
        if self.target_words < self.requested_target_words:
            self.warnings.append(
                f"narrative: target_words reduced from {self.requested_target_words} to {self.target_words} "
                "to fit the available output budget."
            )
        prompt_cards = [card for card in (claim_cards or []) if isinstance(card, dict)][:12]
        perspectives_context = _compact_perspectives_payload(perspectives_payload or {})
        perspectives_used = bool(perspectives_context.get("stories") or prompt_cards)
        prompt = self._render_prompt(
            source_briefs,
            memory,
            date=date,
            enrichment_payload=enrichment_payload,
            recall_packet=recall_packet or {},
            claim_cards=prompt_cards,
            perspectives_context=perspectives_context,
        )
        source_names = [brief.name for brief in source_briefs]
        enrichment_used = isinstance(enrichment_payload, dict) and bool(enrichment_payload)
        budget = self._request_budget()
        self.debug.log(
            "narrative_brief.ai",
            "synthesizing",
            source_briefs=",".join(source_names),
            enrichment_used=enrichment_used,
            prompt_chars=len(prompt),
            max_input_tokens=budget.input_tokens,
            max_new_tokens=budget.output_tokens,
        )
        try:
            result = self.client.complete_json(
                NARRATIVE_BRIEF_SYSTEM,
                prompt,
                label="narrative brief generation",
                max_new_tokens=budget.output_tokens,
                input_token_limit=budget.input_tokens,
                json_schema=NARRATIVE_BRIEF_JSON_SCHEMA,
            )
            normalized = self._normalize_result(
                result,
                date=date,
                source_briefs=source_names,
                enrichment_used=enrichment_used,
                enrichment_json_path=enrichment_json_path,
                recall_guidance_used=bool((recall_packet or {}).get("coverage_guidance")),
                perspectives_used=perspectives_used,
                perspectives_json_path=perspectives_json_path,
            )
            normalized = normalize_narrative_claim_markers(normalized, prompt_cards)
            if prompt_cards and not normalized.get("claim_context_cards"):
                self.warnings.append("narrative: claim-context cards were available but no permitted markers were used.")
        except AIJsonError as exc:
            markdown = str(exc.raw_response or "").strip()
            if not markdown:
                raise
            self.warnings.append("narrative: accepted raw Markdown after JSON parsing failed.")
            normalized = {
                "title": _markdown_title(markdown) or f"Narrative Daily Brief - {date}",
                "markdown": _CLAIM_MARKER_RE.sub("", markdown),
                "segments": [],
                "metadata": {
                    "schema_version": "narrative_brief.v1",
                    "generated_at": datetime.now().astimezone().isoformat(),
                    "date": date,
                    "source_briefs": source_names,
                    "enrichment_used": enrichment_used,
                    "enrichment_json_path": enrichment_json_path if enrichment_used else "",
                    "recall_guidance_used": bool((recall_packet or {}).get("coverage_guidance")),
                    "perspectives_used": perspectives_used,
                    "perspectives_json_path": perspectives_json_path if perspectives_used else "",
                },
            }
            self.debug.log(
                "narrative_brief.ai",
                "accepted_raw_markdown",
                response_chars=len(markdown),
            )
        self.debug.log(
            "narrative_brief.ai",
            "complete",
            markdown_chars=len(render_narrative_markdown(normalized)),
        )
        return normalized

    def _request_budget(self) -> TokenBudget:
        return resolve_client_token_budget(
            self.client,
            input_tokens=self.input_token_limit,
            output_tokens=self.max_new_tokens,
        )

    def _render_prompt(
        self,
        source_briefs: List[NarrativeSourceBrief],
        memory: UserMemory,
        *,
        date: str,
        enrichment_payload: Dict[str, Any] | None = None,
        recall_packet: Dict[str, Any] | None = None,
        claim_cards: List[Dict[str, Any]] | None = None,
        perspectives_context: Dict[str, Any] | None = None,
    ) -> str:
        payload = [
            {
                "name": source.name,
                "brief": _compact_source_brief(source.brief),
            }
            for source in source_briefs
        ]
        enrichment_context = "none"
        if isinstance(enrichment_payload, dict) and enrichment_payload:
            enrichment_context = compact_json(strip_source_links(_compact_enrichment_payload(enrichment_payload)))
        return NARRATIVE_BRIEF_USER.format(
            memory=memory.to_prompt(),
            date=date,
            editorial_style=(
                self.editorial_style
                or "Write like a sharp human news editor, not a consultant memo. Use clear narrative paragraphs, "
                "concrete verbs, and varied sentence rhythm. Avoid repeated Status/Impact/Operational Implication "
                "labels. Do not address the reader as an operator. Use bullets sparingly, only for genuinely "
                "scannable watch items. Prefer 'what changed, why it matters, what remains uncertain' woven into prose."
            ),
            target_length=(
                f"Roughly {self.target_words} words when the source material supports it; "
                "preserve material developments rather than forcing brevity."
            ),
            source_briefs=compact_json(payload),
            enrichment_context=enrichment_context,
            perspectives_context=compact_json(perspectives_context or {}),
            recall_packet=compact_json(recall_packet or {}),
            claim_cards=compact_json(_compact_claim_cards(claim_cards or [])),
        )

    @staticmethod
    def _normalize_result(
        result: Dict[str, Any],
        *,
        date: str,
        source_briefs: List[str],
        enrichment_used: bool = False,
        enrichment_json_path: str = "",
        recall_guidance_used: bool = False,
        perspectives_used: bool = False,
        perspectives_json_path: str = "",
    ) -> Dict[str, Any]:
        normalized = dict(result or {})
        normalized["title"] = _clean_text(normalized.get("title")) or f"Narrative Daily Brief - {date}"
        normalized["lede"] = _clean_markdownish_text(normalized.get("lede"))
        normalized["segments"] = _normalize_segments(normalized.get("segments", []))
        normalized["closing"] = _clean_markdownish_text(normalized.get("closing"))
        if not normalized["lede"] and not normalized["segments"]:
            raise ValueError("narrative brief generation: missing narrative content")
        normalized["metadata"] = {
            "schema_version": "narrative_brief.v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "date": date,
            "source_briefs": source_briefs,
            "enrichment_used": bool(enrichment_used),
            "enrichment_json_path": enrichment_json_path if enrichment_used else "",
            "recall_guidance_used": bool(recall_guidance_used),
            "perspectives_used": bool(perspectives_used),
            "perspectives_json_path": perspectives_json_path if perspectives_used else "",
        }
        return normalized


def load_source_brief(name: str, path: Path) -> NarrativeSourceBrief:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Narrative source brief is not a JSON object: {path}")
    return NarrativeSourceBrief(name=name, json_path=str(path), brief=payload)


def load_enrichment_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Narrative enrichment source is not a JSON object: {path}")
    return payload


def strip_source_links(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return _strip_dict(value, parent_key=parent_key)
    if isinstance(value, list):
        return [strip_source_links(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        return _URL_TEXT_RE.sub("", value).strip()
    return value


def write_narrative_outputs(output_dir: Path, date: str, narrative_brief: Dict[str, Any]) -> tuple[Path, Path]:
    markdown_path = output_dir / f"{date}_narrative_brief.md"
    json_path = output_dir / f"{date}_narrative_brief.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_narrative_markdown(narrative_brief), encoding="utf-8")
    write_json(json_path, narrative_brief)
    return markdown_path, json_path


def normalize_narrative_claim_markers(
    narrative_brief: Dict[str, Any],
    permitted_cards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = dict(narrative_brief)
    cards = {index: dict(card) for index, card in enumerate(permitted_cards, 1) if isinstance(card, dict)}
    text_fields: List[tuple[Any, Any]] = [(normalized, "lede")]
    for segment in normalized.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text_fields.append((segment, "body"))
        for key in ("key_points", "what_to_watch"):
            for index, _item in enumerate(segment.get(key, []) if isinstance(segment.get(key), list) else []):
                text_fields.append((segment[key], index))
    text_fields.append((normalized, "closing"))

    used: List[int] = []
    for owner, key in text_fields:
        for match in _CLAIM_MARKER_RE.finditer(str(owner[key] or "")):
            ref = int(match.group(1))
            if ref in cards and ref not in used:
                used.append(ref)
    renumber = {old: new for new, old in enumerate(used, 1)}

    def replace(value: Any) -> str:
        text = _CLAIM_MARKER_RE.sub(
            lambda match: f"<<{renumber[int(match.group(1))]}>>" if int(match.group(1)) in renumber else "",
            str(value or ""),
        )
        return "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.split("\n")).strip()

    for owner, key in text_fields:
        owner[key] = replace(owner[key])
    normalized["claim_context_cards"] = [{**cards[old], "ref": renumber[old]} for old in used]
    return normalized


def _compact_enrichment_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    story_threads = []
    raw_threads = payload.get("story_threads", [])
    if isinstance(raw_threads, list):
        for raw in raw_threads[:10]:
            if not isinstance(raw, dict):
                continue
            internal_articles = []
            for item in raw.get("internal_articles", [])[:3]:
                if not isinstance(item, dict):
                    continue
                internal_articles.append(
                    {
                        "title": _clean_text(item.get("title"))[:160],
                        "summary": _clean_text(item.get("summary"))[:420],
                        "what_it_adds": _clean_text(item.get("what_it_adds"))[:240],
                        "confidence": _clean_text(item.get("confidence"))[:24],
                    }
                )
            story_threads.append(
                {
                    "story_id": _clean_text(raw.get("story_id"))[:80],
                    "story_title": _clean_text(raw.get("story_title"))[:180],
                    "article_ids": [str(item)[:80] for item in raw.get("article_ids", [])[:8]],
                    "status": _clean_text(raw.get("status"))[:40],
                    "internal_articles": internal_articles,
                    "confirmed_facts": _compact_named_items(raw.get("confirmed_facts"), "fact", 6, 320),
                    "conflicting_claims": _compact_named_items(raw.get("conflicting_claims"), "claim", 4, 320),
                    "open_questions": _compact_named_items(raw.get("open_questions"), "question", 4, 260),
                }
            )
    context_notes = []
    context_by_article = payload.get("context_sources_by_article", {})
    if isinstance(context_by_article, dict):
        for article_id, sources in list(context_by_article.items())[:12]:
            if not isinstance(sources, list):
                continue
            for source in sources[:2]:
                if not isinstance(source, dict):
                    continue
                context_notes.append(
                    {
                        "article_id": str(article_id)[:80],
                        "title": _clean_text(source.get("title"))[:160],
                        "source": _clean_text(source.get("source"))[:80],
                        "summary": _clean_text(source.get("summary"))[:420],
                    }
                )
    return {
        "schema_version": payload.get("schema_version", ""),
        "date": payload.get("date", ""),
        "source_briefs": payload.get("source_briefs", []),
        "story_threads": story_threads,
        "context_notes": context_notes[:16],
    }


def _compact_source_brief(brief: Dict[str, Any]) -> Dict[str, Any]:
    material_fields = (
        "title",
        "lead",
        "knowns",
        "unknowns",
        "watch_signals",
        "topic_reports",
        "sections",
        "major_headlines",
    )
    return strip_source_links({key: brief[key] for key in material_fields if brief.get(key) not in (None, "", [])})


def _compact_perspectives_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    stories = []
    raw_stories = payload.get("stories", [])
    if isinstance(raw_stories, list):
        for raw in raw_stories[:10]:
            if not isinstance(raw, dict):
                continue
            framing = raw.get("framing_report") if isinstance(raw.get("framing_report"), dict) else {}
            story = {
                "story_title": _clean_text(raw.get("story_title"))[:180],
                "synthesis": _clean_text(framing.get("synthesis"))[:700],
                "shared_facts": _compact_named_items(framing.get("shared_facts"), "text", 3, 280),
                "repetition_without_independent_support": _compact_named_items(
                    framing.get("repetition_without_independent_support"), "text", 2, 280
                ),
                "verified_or_independently_supported_claims": _compact_named_items(
                    framing.get("verified_or_independently_supported_claims"), "text", 2, 280
                ),
                "qualified_disputed_or_unresolved_claims": _compact_named_items(
                    framing.get("qualified_disputed_or_unresolved_claims"), "text", 2, 280
                ),
                "country_source_comparison": _compact_named_items(
                    framing.get("country_source_comparison"), "text", 3, 320
                ),
                "coverage_limitations": _compact_text_items(framing.get("coverage_limitations"), 3, 240),
            }
            if any(value for key, value in story.items() if key != "story_title"):
                stories.append({key: value for key, value in story.items() if value})
    return {"stories": stories}


def _compact_claim_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compacted = []
    for index, card in enumerate(cards[:12], 1):
        sources = card.get("sources", []) if isinstance(card.get("sources"), list) else []
        compacted.append(
            {
                "ref": index,
                "claim_id": _clean_text(card.get("claim_id"))[:120],
                "claim": _clean_text(card.get("claim") or card.get("title"))[:420],
                "who_says": _clean_text(card.get("who_says"))[:240],
                "reporting_summary": _clean_text(card.get("reporting_summary"))[:420],
                "evidence_check": _clean_text(card.get("evidence_check"))[:420],
                "verification_verdict": _clean_text(card.get("verification_verdict"))[:40],
                "verdict_scope": _clean_text(card.get("verdict_scope"))[:60],
                "qualification": _clean_text(card.get("qualification"))[:360],
                "limitations": _clean_text(card.get("limitations"))[:300],
                "evidence_sources": [
                    {
                        "outlet": _clean_text(source.get("outlet"))[:100],
                        "headline": _clean_text(source.get("headline"))[:180],
                    }
                    for source in sources[:4]
                    if isinstance(source, dict) and (_clean_text(source.get("outlet")) or _clean_text(source.get("headline")))
                ],
            }
        )
    return [{key: value for key, value in card.items() if value not in ("", [])} for card in compacted]


def _compact_named_items(value: Any, text_key: str, max_items: int, max_chars: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return [
        text[:max_chars]
        for item in value[:max_items]
        if isinstance(item, dict) and (text := _clean_text(item.get(text_key)))
    ]


def _compact_text_items(value: Any, max_items: int, max_chars: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return [text[:max_chars] for item in value[:max_items] if (text := _clean_text(item))]


def render_narrative_markdown(narrative_brief: Dict[str, Any]) -> str:
    raw_markdown = str(narrative_brief.get("markdown", "")).strip()
    if raw_markdown:
        return raw_markdown + "\n"

    lines = [f"# {narrative_brief.get('title', 'Narrative Daily Brief')}", ""]
    metadata = narrative_brief.get("metadata", {})
    if metadata:
        lines.append(f"_Generated: {metadata.get('generated_at', '')}_")
        source_briefs = metadata.get("source_briefs", [])
        if source_briefs:
            lines.append(f"_Source briefs: {', '.join(str(item) for item in source_briefs)}_")
        lines.append("")
    lede = str(narrative_brief.get("lede", "")).strip()
    if lede:
        lines.extend(_paragraphs(lede))
        lines.append("")
    for segment in narrative_brief.get("segments", []):
        if not isinstance(segment, dict):
            continue
        heading = _clean_text(segment.get("heading")) or "Update"
        lines.append(f"## {heading}")
        lines.append("")
        body = str(segment.get("body", "")).strip()
        if body:
            lines.extend(_paragraphs(body))
            lines.append("")
        key_points = _clean_list(segment.get("key_points", []))
        if key_points:
            lines.append("Key points:")
            for item in key_points:
                lines.append(f"- {item}")
            lines.append("")
        watch_items = _clean_list(segment.get("what_to_watch", []))
        if watch_items:
            lines.append("What to watch:")
            for item in watch_items:
                lines.append(f"- {item}")
            lines.append("")
    closing = str(narrative_brief.get("closing", "")).strip()
    if closing:
        lines.append("## Closing")
        lines.append("")
        lines.extend(_paragraphs(closing))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _strip_dict(value: Dict[Any, Any], *, parent_key: str) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    parent = _normalized_key(parent_key)
    for raw_key, raw_item in value.items():
        key = str(raw_key)
        normalized = _normalized_key(key)
        if _is_link_key(normalized):
            continue
        if parent == "metadata" and normalized in _NOISY_METADATA_KEYS:
            continue
        cleaned[key] = strip_source_links(raw_item, parent_key=key)
    return cleaned


def _is_link_key(normalized_key: str) -> bool:
    return (
        normalized_key in _LINK_KEYS
        or normalized_key.endswith("_url")
        or normalized_key.endswith("_urls")
        or normalized_key.endswith("_uri")
        or normalized_key.endswith("_uris")
    )


def _normalized_key(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clean_markdownish_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()).strip() for line in text.split("\n")]
    compacted: List[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and compacted:
                compacted.append("")
            previous_blank = True
            continue
        compacted.append(line)
        previous_blank = False
    return "\n".join(compacted).strip()


def _markdown_title(markdown: str) -> str:
    for line in str(markdown or "").splitlines():
        text = line.strip()
        if text.startswith("#"):
            return _clean_text(text.lstrip("#").strip())
    return ""


def _normalize_segments(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    segments: List[Dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        heading = _clean_text(raw.get("heading"))
        body = _clean_markdownish_text(raw.get("body"))
        key_points = _clean_list(raw.get("key_points", []))
        what_to_watch = _clean_list(raw.get("what_to_watch", []))
        if heading or body or key_points or what_to_watch:
            segments.append(
                {
                    "heading": heading,
                    "body": body,
                    "key_points": key_points,
                    "what_to_watch": what_to_watch,
                }
            )
    return segments


def _clean_list(value: Any) -> List[str]:
    raw_items = value if isinstance(value, list) else []
    items: List[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        text = _clean_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _paragraphs(text: str) -> List[str]:
    return [block.strip() for block in str(text or "").split("\n") if block.strip()]
