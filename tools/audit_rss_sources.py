"""Audit RSS source feeds for liveness and freshness, and optionally merge vetted candidates.

Examples:
    # Audit every enabled source in the bundled registry
    python tools/audit_rss_sources.py --report output/rss_audit.json

    # Audit a candidate file and append passing candidates to the registry
    python tools/audit_rss_sources.py --candidates candidates.json --merge

A feed passes when it returns HTTP < 400, parses into at least ``--min-entries``
items with title and link, and its newest dated item is within ``--max-age-days``.
A source passes when at least one of its feeds passes; on merge, failing feeds are
dropped from the candidate row.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mydailynews.app.config import DEFAULT_USER_AGENT  # noqa: E402
from mydailynews.common.cache import CachedHttpClient  # noqa: E402
from mydailynews.perspectives.sources import (  # noqa: E402
    DEFAULT_SOURCE_REGISTRY_PATH,
    validate_source_registry,
)
from mydailynews.retrieval.registry_rss import parse_feed_articles  # noqa: E402


def _parse_iso(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def audit_feed(http: CachedHttpClient, url: str, *, min_entries: int, max_age_days: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"url": url, "ok": False}
    response = http.get_text(url, timeout=20, allow_redirects=True, cache_mode=CachedHttpClient.NO_CACHE)
    result["status_code"] = response.status_code
    if not response.ok:
        result["reason"] = f"http_{response.status_code or 'error'}"
        return result
    articles = parse_feed_articles(response.text, feed_url=url, source={})
    dates = [d for d in (_parse_iso(a.get("published_at")) for a in articles) if d is not None]
    newest = max(dates) if dates else None
    age_days = (datetime.now(timezone.utc) - newest).total_seconds() / 86400 if newest else None
    snippet_ratio = (sum(1 for a in articles if a.get("snippet")) / len(articles)) if articles else 0.0
    result.update(
        {
            "entries": len(articles),
            "dated_entries": len(dates),
            "newest": newest.isoformat() if newest else "",
            "age_days": round(age_days, 2) if age_days is not None else None,
            "snippet_ratio": round(snippet_ratio, 2),
            "sample_title": articles[0]["title"][:120] if articles else "",
        }
    )
    if len(articles) < min_entries:
        result["reason"] = "too_few_entries"
    elif age_days is None:
        result["reason"] = "undated"
    elif age_days > max_age_days:
        result["reason"] = "stale"
    else:
        result["ok"] = True
    return result


def audit_sources(sources: List[Dict[str, Any]], *, min_entries: int, max_age_days: float, workers: int) -> List[Dict[str, Any]]:
    http = CachedHttpClient(user_agent=DEFAULT_USER_AGENT, cache=None, cache_mode=CachedHttpClient.NO_CACHE)
    jobs = [(source, url) for source in sources for url in source.get("feed_urls") or []]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        feed_results = list(
            pool.map(lambda job: audit_feed(http, str(job[1]), min_entries=min_entries, max_age_days=max_age_days), jobs)
        )
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for (source, _url), result in zip(jobs, feed_results):
        by_source.setdefault(str(source.get("source_id")), []).append(result)
    report = []
    for source in sources:
        feeds = by_source.get(str(source.get("source_id")), [])
        report.append(
            {
                "source_id": source.get("source_id"),
                "country": source.get("country"),
                "language": source.get("language"),
                "enabled": bool(source.get("enabled", True)),
                "ok": any(feed["ok"] for feed in feeds),
                "feeds": feeds,
            }
        )
    return report


def merge_candidates(registry_path: Path, candidates: List[Dict[str, Any]], report: List[Dict[str, Any]]) -> List[str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    existing = payload["sources"]
    existing_ids = {str(row.get("source_id")) for row in existing}
    existing_urls = {str(url) for row in existing for url in row.get("feed_urls") or []}
    results = {str(row["source_id"]): row for row in report}
    added: List[str] = []
    for candidate in candidates:
        source_id = str(candidate.get("source_id"))
        audit = results.get(source_id)
        if not audit or not audit["ok"] or source_id in existing_ids:
            continue
        good_urls = [feed["url"] for feed in audit["feeds"] if feed["ok"] and feed["url"] not in existing_urls]
        if not good_urls:
            continue
        row = dict(candidate)
        row["feed_urls"] = good_urls
        row["enabled"] = True
        existing.append(row)
        existing_ids.add(source_id)
        existing_urls.update(good_urls)
        added.append(source_id)
    errors = validate_source_registry(existing)
    if errors:
        raise SystemExit("merged registry failed validation: " + "; ".join(errors[:10]))
    registry_path.write_text(dump_registry(payload), encoding="utf-8", newline="\n")
    return added


def dump_registry(payload: Dict[str, Any]) -> str:
    """Serialize like the hand-maintained registry: 2-space indent, scalar lists inline,
    except multi-URL ``feed_urls``/``sitemap_urls`` lists which stay one URL per line."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    scalar = r"(?:\"(?:[^\"\\]|\\.)*\"|-?\d[\d.eE+-]*|true|false|null)"
    scalar_list = re.compile(r"\"(\w+)\": \[\s*\n(\s*" + scalar + r"(?:,\s*\n\s*" + scalar + r")*)\s*\n\s*\]")

    def collapse(match: re.Match) -> str:
        key, body = match.group(1), match.group(2)
        items = [item.strip() for item in body.split(",\n")]
        if key in {"feed_urls", "sitemap_urls"} and len(items) > 1:
            return match.group(0)
        return f"\"{key}\": [" + ", ".join(items) + "]"

    return scalar_list.sub(collapse, text) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=DEFAULT_SOURCE_REGISTRY_PATH)
    parser.add_argument("--candidates", type=Path, help="JSON list (or {sources: [...]}) of candidate source rows to audit.")
    parser.add_argument("--include-disabled", action="store_true", help="Also audit disabled registry sources.")
    parser.add_argument("--country", action="append", default=[], help="Restrict to these country codes (repeatable).")
    parser.add_argument("--min-entries", type=int, default=5)
    parser.add_argument("--max-age-days", type=float, default=7.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--report", type=Path, help="Write the full JSON audit report here.")
    parser.add_argument("--merge", action="store_true", help="Append passing candidates to the registry (requires --candidates).")
    parser.add_argument("--from-report", type=Path, help="Reuse a previous --report instead of re-fetching feeds.")
    args = parser.parse_args()

    if args.candidates:
        raw = json.loads(args.candidates.read_text(encoding="utf-8-sig"))
        sources = raw.get("sources") if isinstance(raw, dict) else raw
    else:
        sources = json.loads(args.registry.read_text(encoding="utf-8-sig"))["sources"]
        if not args.include_disabled:
            sources = [row for row in sources if row.get("enabled", True)]
    countries = {code.upper() for code in args.country}
    if countries:
        sources = [row for row in sources if str(row.get("country") or "").upper() in countries]

    if args.from_report:
        report = json.loads(args.from_report.read_text(encoding="utf-8"))
    else:
        report = audit_sources(sources, min_entries=args.min_entries, max_age_days=args.max_age_days, workers=args.workers)
    for row in report:
        status = "PASS" if row["ok"] else "FAIL"
        details = ", ".join(
            f"{feed.get('reason') or 'ok'}:{feed.get('entries', 0)}e/{feed.get('age_days')}d" for feed in row["feeds"]
        )
        print(f"{status} {row['country']} {row['source_id']} [{details}]")
    passed = sum(1 for row in report if row["ok"])
    print(f"\n{passed}/{len(report)} sources passed")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.merge:
        if not args.candidates:
            parser.error("--merge requires --candidates")
        added = merge_candidates(args.registry, sources, report)
        print(f"merged {len(added)} sources: {', '.join(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
