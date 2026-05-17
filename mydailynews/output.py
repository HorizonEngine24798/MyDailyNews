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

    sections = brief.get("sections", [])
    if sections:
        lines.append("## Today's Shape")
        for section in sections:
            lines.append(f"### {section.get('heading', 'Section')}")
            lines.append(str(section.get("summary", "")))
            lines.append("")

    articles = brief.get("articles", [])
    if articles:
        lines.append("## Article Briefs")
        for article in articles:
            lines.append(f"### {article.get('headline', 'Untitled')}")
            lines.append(f"- Source: {article.get('source', '')}")
            lines.append(f"- Score: {article.get('score', '')}")
            lines.append(f"- URL: {article.get('url', '')}")
            lines.append(f"- Summary: {article.get('summary', '')}")
            lines.append(f"- Why it matters: {article.get('why_it_matters', '')}")
            context = article.get("key_context")
            if context:
                lines.append(f"- Context: {context}")
            lines.append("")

    headlines = brief.get("major_headlines", [])
    if headlines:
        lines.append("## Major Headlines")
        for item in headlines:
            lines.append(f"- {item.get('headline', '')} ({item.get('source', '')}, score {item.get('score', '')})")
            if item.get("url"):
                lines.append(f"  {item['url']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
