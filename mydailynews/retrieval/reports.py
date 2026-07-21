from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import PriorReport, PriorReportsSourceConfig
from mydailynews.common.utils import normalize_whitespace, stable_id


class PriorReportRetriever:
    """Load previous machine-readable briefs as narrative context."""

    def __init__(
        self,
        config: PriorReportsSourceConfig,
        default_output_dir: str,
        debug: DebugLogger | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir or default_output_dir)
        self.debug = debug or DebugLogger(False)
        self.errors: List[str] = []

    def fetch(self, today: date) -> List[PriorReport]:
        self.errors = []
        if not self.config.enabled:
            self.debug.log("prior_reports", "skipped_disabled")
            return []
        if not self.output_dir.exists():
            self.debug.log("prior_reports", "missing_output_dir", output_dir=self.output_dir)
            return []

        since = today - timedelta(days=self.config.days)
        paths_by_date: Dict[date, tuple[int, Path]] = {}
        for path in self.output_dir.glob("*_brief.json"):
            report_date = self._date_from_path(path)
            if not report_date or report_date >= today or report_date < since:
                continue
            priority = self._brief_priority(path)
            if priority < 0:
                continue
            current = paths_by_date.get(report_date)
            if current is None or priority > current[0]:
                paths_by_date[report_date] = (priority, path)

        reports: List[PriorReport] = []
        for report_date, (_priority, path) in sorted(paths_by_date.items(), reverse=True):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue

            reports.append(self._to_report(path, report_date, raw))
            if len(reports) >= self.config.max_reports:
                break

        self.debug.log("prior_reports", "complete", reports=len(reports), output_dir=self.output_dir)
        return reports

    @staticmethod
    def _brief_priority(path: Path) -> int:
        name = path.name.lower()
        if name.endswith("_narrative_brief.json"):
            return -1
        if name.endswith("_detailed_brief.json"):
            return 2
        if name.endswith("_general_brief.json"):
            return 1
        return 0

    def _to_report(self, path: Path, report_date: date, raw: Dict[str, Any]) -> PriorReport:
        topics = self._extract_topics(raw)
        summary = self._summarize(raw)
        return PriorReport(
            id=stable_id("prior_report", str(path), str(report_date)),
            date=report_date.isoformat(),
            title=str(raw.get("title", f"Daily Brief - {report_date.isoformat()}")),
            path=str(path),
            summary=summary[: self.config.max_chars_per_report],
            topics=topics,
            major_headlines=raw.get("major_headlines", []) if isinstance(raw.get("major_headlines"), list) else [],
            story_baselines=self._extract_story_baselines(raw),
        )

    @staticmethod
    def _date_from_path(path: Path) -> date | None:
        prefix = path.name[:10]
        try:
            return datetime.strptime(prefix, "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _extract_topics(raw: Dict[str, Any]) -> List[str]:
        topics: List[str] = []
        for report in raw.get("topic_reports", []) or []:
            topic = str(report.get("topic", "")).strip() if isinstance(report, dict) else ""
            if topic and topic not in topics:
                topics.append(topic)
        for article in raw.get("selected_articles", []) or []:
            topic = str(article.get("topic", "")).strip() if isinstance(article, dict) else ""
            if topic and topic not in topics:
                topics.append(topic)
        return topics

    @staticmethod
    def _summarize(raw: Dict[str, Any]) -> str:
        parts: List[str] = []
        if raw.get("lead"):
            parts.append(f"Lead: {raw['lead']}")

        for report in raw.get("topic_reports", []) or []:
            if not isinstance(report, dict):
                continue
            topic = report.get("topic", "Topic")
            narrative_summary = (
                report.get("narrative_summary")
                or report.get("summary")
                or report.get("why_it_matters")
                or report.get("what_changed")
                or ""
            )
            if narrative_summary:
                parts.append(f"{topic}: {narrative_summary}")
            for key in ("why_it_matters", "what_changed"):
                value = str(report.get(key, "")).strip()
                if value:
                    parts.append(f"{topic} {key.replace('_', ' ')}: {value}")
            for change in report.get("narrative_changes", []) or []:
                if isinstance(change, dict):
                    parts.append(f"{topic} change: {change.get('summary', '')}")

        for section in raw.get("sections", []) or []:
            if isinstance(section, dict) and section.get("summary"):
                parts.append(f"{section.get('heading', 'Section')}: {section['summary']}")

        headlines = []
        for item in raw.get("major_headlines", []) or []:
            if isinstance(item, dict) and item.get("headline"):
                headlines.append(str(item["headline"]))
        if headlines:
            parts.append("Major headlines: " + "; ".join(headlines[:10]))

        return normalize_whitespace("\n".join(parts))

    @staticmethod
    def _extract_story_baselines(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        analysis = raw.get("analysis", {})
        delta = analysis.get("delta_packet", {}) if isinstance(analysis, dict) else {}
        decisions = delta.get("story_decisions", []) if isinstance(delta, dict) else []
        if not isinstance(decisions, list):
            return []
        return [dict(item) for item in decisions if isinstance(item, dict) and str(item.get("story_key", "") or "").strip()]
