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

    topic_reports = brief.get("topic_reports", [])
    if topic_reports:
        lines.append("## Topic Reports")
        for report in topic_reports:
            lines.append(f"### {report.get('topic', 'Topic')}")
            lines.append(str(report.get("narrative_summary", "")))
            lines.append("")
            changes = report.get("narrative_changes", [])
            if changes:
                lines.append("Narrative changes:")
                for change in changes:
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

    selected_articles = brief.get("selected_articles", [])
    if selected_articles:
        lines.append("## Source Notes")
        for article in selected_articles:
            lines.append(f"### {article.get('headline', 'Untitled')}")
            lines.append(f"- Source: {article.get('source', '')}")
            lines.append(f"- Score: {article.get('score', '')}")
            lines.append(f"- URL: {article.get('url', '')}")
            snippet = article.get("snippet", "")
            if snippet:
                lines.append(f"- Snippet: {snippet}")
            lines.append("")

    headlines = brief.get("major_headlines", [])
    if headlines:
        lines.append("## Major Headlines")
        for item in headlines:
            topic = item.get("topic")
            topic_text = f", {topic}" if topic else ""
            lines.append(f"- {item.get('headline', '')} ({item.get('source', '')}{topic_text}, score {item.get('score', '')})")
            if item.get("url"):
                lines.append(f"  {item['url']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
