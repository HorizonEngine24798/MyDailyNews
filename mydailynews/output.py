from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, brief: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(brief), encoding="utf-8")


def render_markdown(brief: Dict[str, Any]) -> str:
    lines = [f"# {brief.get('title', 'Daily Brief')}", ""]
    metadata = brief.get("metadata", {})
    if metadata:
        lines.append(f"_Generated: {metadata.get('generated_at', '')}_")
        lines.append("")

    if brief.get("lead"):
        lines.append(str(brief["lead"]))
        lines.append("")

    knowns = brief.get("knowns", [])
    unknowns = brief.get("unknowns", [])
    watch_signals = brief.get("watch_signals", [])
    if knowns or unknowns or watch_signals:
        lines.append("## Signal Map")
        if knowns:
            lines.append("Knowns:")
            for item in knowns:
                lines.append(f"- {item}")
        if unknowns:
            lines.append("Unknowns:")
            for item in unknowns:
                lines.append(f"- {item}")
        if watch_signals:
            lines.append("Watch signals:")
            for item in watch_signals:
                lines.append(f"- {item}")
        lines.append("")

    topic_reports = brief.get("topic_reports", [])
    if topic_reports:
        lines.append("## Topic Reports")
        for report in topic_reports:
            if not isinstance(report, dict):
                continue
            lines.append(f"### {report.get('topic', 'Topic')}")
            why_it_matters = str(report.get("why_it_matters", "")).strip()
            what_changed = str(report.get("what_changed", "")).strip()
            narrative_summary = str(report.get("narrative_summary", "")).strip()
            who_is_affected = report.get("who_is_affected", [])
            if isinstance(who_is_affected, str):
                who_is_affected = [who_is_affected]
            if why_it_matters:
                lines.append(f"Why it matters: {why_it_matters}")
            if what_changed:
                lines.append(f"What changed: {what_changed}")
            if not why_it_matters and not what_changed and narrative_summary:
                lines.append(narrative_summary)
            if who_is_affected:
                lines.append("Who is affected:")
                for item in who_is_affected:
                    text = str(item).strip()
                    if text:
                        lines.append(f"- {text}")
            lines.append("")
            changes = report.get("narrative_changes", [])
            if changes:
                lines.append("Narrative changes:")
                for change in changes:
                    if not isinstance(change, dict):
                        continue
                    lines.append(
                        f"- {change.get('narrative', 'Narrative')}: "
                        f"{change.get('status', '')} - {change.get('summary', '')}"
                    )
            watch_items = report.get("what_to_watch", [])
            if watch_items:
                lines.append("What to watch:")
                for item in watch_items:
                    lines.append(f"- {item}")
            lines.append("")

    sections = brief.get("sections", [])
    if sections:
        lines.append("## Today's Shape")
        for section in sections:
            lines.append(f"### {section.get('heading', 'Section')}")
            lines.append(str(section.get("summary", "")))
            lines.append("")

    references = brief.get("references", [])
    if not isinstance(references, list) or not references:
        references = []
        for article in brief.get("selected_articles", []):
            if not isinstance(article, dict):
                continue
            references.append(
                {
                    "title": article.get("headline", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                }
            )
    if references:
        lines.append("## References")
        seen: set[str] = set()
        for item in references:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "").strip()
            source = str(item.get("source", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            key = url or f"{title}|{source}"
            if not key or key in seen:
                continue
            seen.add(key)
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
