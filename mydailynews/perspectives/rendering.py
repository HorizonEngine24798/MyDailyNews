from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from mydailynews.briefing.output import write_json
from mydailynews.common.utils import normalize_whitespace


def write_perspectives_report_outputs(output_dir: Path, date: str, payload: Dict[str, Any]) -> tuple[Path, Path]:
    markdown_path = output_dir / f"{date}_perspectives_report.md"
    json_path = output_dir / f"{date}_perspectives_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_perspectives_report_markdown(payload), encoding="utf-8")
    write_json(json_path, payload)
    return markdown_path, json_path


def render_perspectives_report_markdown(payload: Dict[str, Any]) -> str:
    date = str(payload.get("date", "") or "")
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    lines = [
        f"# Perspectives Report - {date}",
        "",
        f"_Stories: {metadata.get('story_count', 0)} | Coverage articles: {metadata.get('coverage_article_count', 0)} | Source countries: {metadata.get('coverage_source_country_count', 0)}_",
        "",
    ]
    for story in _as_list(payload.get("stories")):
        if not isinstance(story, dict):
            continue
        title = str(story.get("story_title") or story.get("story_id") or "Story")
        lines.extend([f"## {title}", ""])
        plan = story.get("planner", {}) if isinstance(story.get("planner"), dict) else {}
        coverage_quality = story.get("coverage_quality", {}) if isinstance(story.get("coverage_quality"), dict) else {}
        lines.append(
            f"_Planner: {plan.get('status', 'unknown')} | Coverage: {coverage_quality.get('status', 'unknown')} | Providers: {_count_text(story.get('coverage_provider_counts') or {}) or 'none'}_"
        )
        lines.append("")
        framing = story.get("framing_report", {}) if isinstance(story.get("framing_report"), dict) else {}
        synthesis = normalize_whitespace(str(framing.get("synthesis") or ""))
        if synthesis:
            lines.extend(["### Bottom line", synthesis, ""])
        _append_evidence_section(lines, "What coverage agrees on", framing.get("shared_facts"))
        _append_evidence_section(lines, "Repeated claims without independent support", framing.get("repetition_without_independent_support"))
        _append_evidence_section(lines, "Claims supported by retrieved evidence", framing.get("verified_or_independently_supported_claims"))
        _append_evidence_section(lines, "Qualified, disputed, or unresolved claims", framing.get("qualified_disputed_or_unresolved_claims"))
        _append_evidence_section(lines, "Where the framing differs", framing.get("country_source_comparison"))
        limitations = _string_list(framing.get("coverage_limitations")) + _string_list(coverage_quality.get("thin_reasons"))
        if limitations:
            lines.append("### What the coverage cannot establish")
            for item in _unique_text(limitations):
                lines.append(f"- {item}")
            lines.append("")
    warnings = _string_list(payload.get("warnings"))
    if warnings:
        lines.append("## Diagnostics")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    references = []
    seen: set[str] = set()
    for story in _as_list(payload.get("stories")):
        if not isinstance(story, dict):
            continue
        for article in _as_list(story.get("coverage_articles")):
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "").strip()
            source = str(article.get("source_name") or article.get("domain") or "").strip()
            url = str(article.get("canonical_url") or article.get("url") or "").strip()
            key = url or f"{title}|{source}"
            if not key or key in seen:
                continue
            seen.add(key)
            references.append((title, source, url))
        for document in _as_list(story.get("verification_documents")):
            if not isinstance(document, dict):
                continue
            title = str(document.get("title") or "").strip()
            source = str(document.get("source") or "").strip()
            url = str(document.get("url") or "").strip()
            key = url or f"{title}|{source}"
            if not key or key in seen:
                continue
            seen.add(key)
            references.append((title, source, url))
    if references:
        lines.append("## References")
        for title, source, url in references:
            if title and source:
                lines.append(f"- {title} ({source})")
            elif title:
                lines.append(f"- {title}")
            elif source:
                lines.append(f"- {source}")
            if url:
                lines.append(f"  {url}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _append_evidence_section(lines: List[str], title: str, items: Any) -> None:
    evidence = _evidence_items(items)
    if not evidence:
        return
    lines.append(f"### {title}")
    for item in evidence:
        lines.append(f"- {item['text']}")
    lines.append("")


def _evidence_items(
    value: Any,
    *,
    known_article_ids: set[str] | None = None,
    story_id: str = "",
    warnings: List[str] | None = None,
) -> List[Dict[str, Any]]:
    items = value if isinstance(value, list) else [value] if value else []
    output: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            text = normalize_whitespace(str(item.get("text") or item.get("summary") or item.get("point") or item.get("claim") or ""))
            article_ids = _validated_article_ids(
                item.get("article_ids") or item.get("evidence_article_ids"),
                known_article_ids,
                story_id,
                warnings,
            )
        else:
            text = normalize_whitespace(str(item or ""))
            article_ids = []
        if text:
            output.append({"text": text, "article_ids": article_ids})
    return output


def _validated_article_ids(
    value: Any,
    known_article_ids: set[str] | None,
    story_id: str,
    warnings: List[str] | None,
) -> List[str]:
    article_ids = _string_list(value)
    if known_article_ids is None:
        return article_ids
    unknown = [article_id for article_id in article_ids if article_id not in known_article_ids]
    if unknown and warnings is not None:
        warnings.append(f"perspectives_report: framing response used unknown article_ids for {story_id!r}: {', '.join(unknown)}")
    return [article_id for article_id in article_ids if article_id in known_article_ids]


def _text_items(value: Any) -> List[str]:
    items = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    output: List[str] = []
    for item in items:
        raw = item.get("text") if isinstance(item, dict) else item
        text = normalize_whitespace(str(raw or ""))
        if text:
            output.append(text)
    return output


def _count_text(counts: Dict[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in sorted(counts.items()) if value)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _unique_text(values: List[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output
