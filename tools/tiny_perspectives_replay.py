"""Replay a historical perspectives report through a small local model.

The original full report is much larger than a mobile model context window.  This
keeps the historical report's story boundaries and passes only its explicitly
recorded evidence/framing fields to the model for each story.  It is deliberately
an evaluation replay, not a claim that the model independently re-retrieved the
July sources.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


SYSTEM = """You are a careful news editor. Use only the supplied record. Do not add
facts, people, dates, motives, numbers, or sources. If the record is thin or has
no evidence, say so plainly. Write concise Markdown with these sections exactly:
Bottom line, What is supported, Uncertainty and gaps, Why it matters. Avoid generic
advice and do not make up cross-source disagreement."""


def limited(values: Any, count: int = 5) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values[:count]:
        if isinstance(value, dict):
            text = str(value.get("text") or value.get("claim") or "").strip()
        else:
            text = str(value).strip()
        if text:
            result.append(text)
    return result


def story_record(story: dict[str, Any]) -> dict[str, Any]:
    framing = story.get("framing_report") if isinstance(story.get("framing_report"), dict) else {}
    cards = story.get("claim_cards") if isinstance(story.get("claim_cards"), list) else []
    return {
        "title": story.get("story_title", "Untitled story"),
        "coverage_status": story.get("coverage_status", "unknown"),
        "coverage_gap": story.get("coverage_gap", ""),
        "recorded_summary": str(story.get("summary") or "")[:1200],
        "shared_facts": limited(framing.get("shared_facts"), 6),
        "verified_or_supported": limited(framing.get("verified_or_independently_supported_claims"), 5),
        "qualified_or_unresolved": limited(framing.get("qualified_disputed_or_unresolved_claims"), 5),
        "coverage_limitations": limited(framing.get("coverage_limitations"), 5),
        "claim_cards": [
            {
                "claim": str(card.get("claim") or "")[:450],
                "evidence_check": card.get("evidence_check", ""),
                "qualification": str(card.get("qualification") or "")[:350],
                "limitations": str(card.get("limitations") or "")[:350],
            }
            for card in cards[:4]
            if isinstance(card, dict)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit-stories", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    stories = payload.get("stories", [])
    if args.limit_stories:
        stories = stories[: args.limit_stories]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report: list[str] = ["# Tiny-model replay: historical perspectives report", ""]
    metrics: list[dict[str, Any]] = []
    for index, story in enumerate(stories, start=1):
        record = story_record(story)
        user = "Historical evidence record (not live retrieval):\n" + json.dumps(record, ensure_ascii=False)
        started = time.perf_counter()
        response = requests.post(
            args.base_url.rstrip("/") + "/chat/completions",
            json={
                "model": "Qwen3-1.7B-Q4_K_M",
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": args.max_tokens,
            },
            timeout=900,
        )
        response.raise_for_status()
        body = response.json()
        text = str(body["choices"][0]["message"].get("content") or "").strip()
        usage = body.get("usage", {})
        elapsed = time.perf_counter() - started
        report.extend([f"## {index}. {record['title']}", "", text or "_No response._", ""])
        metrics.append(
            {
                "story_id": story.get("story_id", ""),
                "title": record["title"],
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "response_chars": len(text),
            }
        )
        args.output.write_text("\n".join(report), encoding="utf-8")
        args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({"completed": index, "total": len(stories), **metrics[-1]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
