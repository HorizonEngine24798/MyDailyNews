from __future__ import annotations

from dataclasses import dataclass

PIPELINE_BRIEFS = ("general", "detailed")

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
    "enrichment",
    "evidence_distillation",
    "delta_extraction",
    "final_brief",
    "write_output",
)

POST_BRIEF_STAGE_ORDER = (
    "narrative_brief",
)

ALL_STAGE_ORDER = PIPELINE_STAGE_ORDER + BRIEF_STAGE_ORDER + POST_BRIEF_STAGE_ORDER


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


@dataclass(frozen=True)
class PipelineRunOptions:
    briefs: tuple[str, ...] = PIPELINE_BRIEFS
    stop_after_stage: str = ""
    save_intermediate: bool = False
    dump_stage_artifacts: bool = False
    stage_artifact_dir: str = ""

    @classmethod
    def from_cli(
        cls,
        *,
        brief: str,
        stop_after_stage: str,
        save_intermediate: bool,
        no_save_intermediate: bool,
        dump_stage_artifacts: bool,
        stage_artifact_dir: str,
    ) -> "PipelineRunOptions":
        normalized_stop = normalize_stage_name(stop_after_stage)
        should_save_intermediate = bool(save_intermediate or normalized_stop)
        if no_save_intermediate:
            should_save_intermediate = False
        return cls(
            briefs=normalize_brief_selection(brief),
            stop_after_stage=normalized_stop,
            save_intermediate=should_save_intermediate,
            dump_stage_artifacts=bool(dump_stage_artifacts),
            stage_artifact_dir=str(stage_artifact_dir or "").strip(),
        )
