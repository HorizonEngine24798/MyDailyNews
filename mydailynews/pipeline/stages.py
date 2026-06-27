from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

PIPELINE_BRIEFS = ("general", "detailed")
PIPELINE_MODULES = ("briefs", "enrichment", "narrative_brief")
PIPELINE_MODULE_CHOICES = PIPELINE_MODULES + ("series",)

PIPELINE_STAGE_ORDER = (
    "prior_reports",
    "snapshot",
    "shared_headline_scoring",
)

BRIEF_STAGE_ORDER = (
    "candidate_prepare",
    "headline_limit",
    "headline_decisions",
    "headline_select",
    "article_fetch",
    "story_grouping",
    "evidence_distillation",
    "delta_extraction",
    "final_brief",
    "write_output",
    "write_handoff",
)

MODULE_STAGE_ORDER = (
    "enrichment",
    "narrative_brief",
)

ALL_STAGE_ORDER = PIPELINE_STAGE_ORDER + BRIEF_STAGE_ORDER + MODULE_STAGE_ORDER


def _normalize(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def normalize_stage_name(value: str) -> str:
    stage = _normalize(value)
    if not stage:
        return ""
    if stage not in ALL_STAGE_ORDER:
        supported = ", ".join(ALL_STAGE_ORDER)
        raise ValueError(f"Unsupported stage '{value}'. Supported stages: {supported}")
    return stage


def normalize_brief_selection(value: str) -> tuple[str, ...]:
    selected = _normalize(value)
    if not selected or selected == "both":
        return PIPELINE_BRIEFS
    if selected in PIPELINE_BRIEFS:
        return (selected,)
    supported = "general, detailed, both"
    raise ValueError(f"Unsupported brief '{value}'. Supported values: {supported}")


def normalize_module_name(value: str) -> str:
    module = _normalize(value)
    if not module:
        return "series"
    if module in PIPELINE_MODULE_CHOICES:
        return module
    supported = ", ".join(PIPELINE_MODULE_CHOICES)
    raise ValueError(f"Unsupported module '{value}'. Supported values: {supported}")


def normalize_run_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = date_type.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid --date value '{value}'. Expected YYYY-MM-DD.") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"Invalid --date value '{value}'. Expected YYYY-MM-DD.")
    return text


def validate_run_date_usage(module: str, date: str) -> None:
    normalized_module = normalize_module_name(module)
    normalized_date = normalize_run_date(date)
    if normalized_date and normalized_module in {"series", "briefs"}:
        raise ValueError("--date can only be used with standalone modules: enrichment, narrative_brief")


def normalize_skip_modules(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    modules: list[str] = []
    for value in values or ():
        module = _normalize(value)
        if module == "series":
            raise ValueError("--skip-module cannot skip 'series'")
        if module not in PIPELINE_MODULES:
            supported = ", ".join(PIPELINE_MODULES)
            raise ValueError(f"Unsupported module '{value}'. Supported values: {supported}")
        if module not in modules:
            modules.append(module)
    return tuple(modules)


@dataclass(frozen=True)
class PipelineRunOptions:
    briefs: tuple[str, ...] = PIPELINE_BRIEFS
    module: str = "series"
    date: str = ""
    skip_modules: tuple[str, ...] = ()
    stop_after_stage: str = ""
    save_intermediate: bool = False
    dump_stage_artifacts: bool = False
    stage_artifact_dir: str = ""

    @classmethod
    def from_cli(
        cls,
        *,
        brief: str,
        module: str = "series",
        date: str = "",
        skip_modules: tuple[str, ...] | list[str] | None = None,
        stop_after_stage: str = "",
        save_intermediate: bool = False,
        no_save_intermediate: bool = False,
        dump_stage_artifacts: bool = False,
        stage_artifact_dir: str = "",
    ) -> "PipelineRunOptions":
        normalized_module = normalize_module_name(module)
        normalized_date = normalize_run_date(date)
        validate_run_date_usage(normalized_module, normalized_date)
        normalized_stop = normalize_stage_name(stop_after_stage)
        should_save_intermediate = bool(save_intermediate or normalized_stop)
        if no_save_intermediate:
            should_save_intermediate = False
        return cls(
            briefs=normalize_brief_selection(brief),
            module=normalized_module,
            date=normalized_date,
            skip_modules=normalize_skip_modules(skip_modules),
            stop_after_stage=normalized_stop,
            save_intermediate=should_save_intermediate,
            dump_stage_artifacts=bool(dump_stage_artifacts),
            stage_artifact_dir=str(stage_artifact_dir or "").strip(),
        )
