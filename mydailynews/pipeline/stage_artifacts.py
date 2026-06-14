from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

STAGE_ARTIFACT_SCHEMA_VERSION = "stage_artifact.replay.v1"

NEXT_STAGE_BY_STAGE = {
    "prior_reports": "snapshot",
    "snapshot": "shared_headline_scoring",
    "shared_headline_scoring": "candidate_prepare",
    "candidate_prepare": "headline_limit",
    "headline_limit": "headline_decisions",
    "headline_decisions": "headline_select",
    "headline_select": "article_fetch",
    "article_fetch": "enrichment",
    "enrichment": "evidence_distillation",
    "evidence_distillation": "delta_extraction",
    "delta_extraction": "final_brief",
    "final_brief": "write_output",
    "write_output": "",
}


def next_stage_after(stage: str) -> str:
    return NEXT_STAGE_BY_STAGE.get(str(stage or "").strip(), "")


def build_stage_payload(
    *,
    stage: str,
    brief: str,
    summary: Dict[str, Any],
    next_stage_input: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": STAGE_ARTIFACT_SCHEMA_VERSION,
        "brief": str(brief or "pipeline"),
        "stage": str(stage or ""),
        "summary": dict(summary or {}),
        "next_stage": next_stage_after(stage),
        "next_stage_input": dict(next_stage_input or {}),
    }


def build_stage_artifact(
    *,
    run_label: str,
    brief: str,
    stage: str,
    generated_at: str,
    summary: Dict[str, Any],
    next_stage_input: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = build_stage_payload(
        stage=stage,
        brief=brief,
        summary=summary,
        next_stage_input=next_stage_input,
    )
    return {
        **payload,
        "run_label": str(run_label or ""),
        "generated_at": str(generated_at or ""),
    }


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
