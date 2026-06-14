from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from mydailynews.common.booleans import parse_bool, parse_optional_bool
from mydailynews.app.models import (
    AnalysisConfig,
    AnalysisRolloutModeConfig,
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
)


ANALYSIS_ROLLOUT_PROFILE_DEFAULTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "safe_local": {
        "general": {
            "evidence_enabled": False,
            "delta_enabled": False,
        },
        "detailed": {
            "evidence_enabled": True,
            "delta_enabled": False,
        },
    },
    "balanced_local": {
        "general": {
            "evidence_enabled": False,
            "delta_enabled": False,
        },
        "detailed": {
            "evidence_enabled": True,
            "delta_enabled": True,
        },
    },
    "quality_focused": {
        "general": {
            "evidence_enabled": True,
            "delta_enabled": False,
        },
        "detailed": {
            "evidence_enabled": True,
            "delta_enabled": True,
        },
    },
}
ANALYSIS_ROLLOUT_PROFILE_NAMES = frozenset(ANALYSIS_ROLLOUT_PROFILE_DEFAULTS)


def _coerce_optional_int(value: Any, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    return max(minimum, int(value))


def _rollout_mode_from_object(raw: Any) -> AnalysisRolloutModeConfig:
    if raw is None:
        raw = {}
    if isinstance(raw, dict):
        getter = raw.get
    else:
        getter = lambda key, default=None: getattr(raw, key, default)
    return AnalysisRolloutModeConfig(
        evidence_enabled=parse_optional_bool(getter("evidence_enabled")),
        delta_enabled=parse_optional_bool(getter("delta_enabled")),
        evidence_max_input_tokens=_coerce_optional_int(getter("evidence_max_input_tokens"), minimum=256),
        evidence_max_new_tokens=_coerce_optional_int(getter("evidence_max_new_tokens"), minimum=64),
        evidence_max_articles=_coerce_optional_int(getter("evidence_max_articles"), minimum=1),
        evidence_max_articles_per_batch=_coerce_optional_int(getter("evidence_max_articles_per_batch"), minimum=1),
        evidence_max_articles_dropped_to_avoid_split=_coerce_optional_int(
            getter("evidence_max_articles_dropped_to_avoid_split"),
            minimum=0,
        ),
        evidence_max_article_chars=_coerce_optional_int(getter("evidence_max_article_chars"), minimum=120),
        delta_max_input_tokens=_coerce_optional_int(getter("delta_max_input_tokens"), minimum=256),
        delta_max_new_tokens=_coerce_optional_int(getter("delta_max_new_tokens"), minimum=64),
        delta_max_articles=_coerce_optional_int(getter("delta_max_articles"), minimum=1),
        delta_max_articles_per_batch=_coerce_optional_int(getter("delta_max_articles_per_batch"), minimum=1),
        delta_max_articles_dropped_to_avoid_split=_coerce_optional_int(
            getter("delta_max_articles_dropped_to_avoid_split"),
            minimum=0,
        ),
        delta_max_article_chars=_coerce_optional_int(getter("delta_max_article_chars"), minimum=120),
        delta_max_prior_reports=_coerce_optional_int(getter("delta_max_prior_reports"), minimum=1),
    )


def _merge_rollout_modes(base: AnalysisRolloutModeConfig, override: AnalysisRolloutModeConfig) -> AnalysisRolloutModeConfig:
    merged = AnalysisRolloutModeConfig(**vars(base))
    for key, value in vars(override).items():
        if value is None:
            continue
        setattr(merged, key, value)
    return merged


def _copy_evidence_config(raw: EvidenceDistillationConfig | None) -> EvidenceDistillationConfig:
    if raw is None:
        return EvidenceDistillationConfig()
    return replace(raw)


def _copy_delta_config(raw: DeltaExtractionConfig | None) -> DeltaExtractionConfig:
    if raw is None:
        return DeltaExtractionConfig()
    return replace(raw)


def _apply_rollout_mode_overrides(
    evidence_config: EvidenceDistillationConfig,
    delta_config: DeltaExtractionConfig,
    mode: AnalysisRolloutModeConfig,
) -> None:
    if mode.evidence_enabled is not None:
        evidence_config.enabled = parse_bool(mode.evidence_enabled, field_name="analysis.rollout.evidence_enabled")
    if mode.delta_enabled is not None:
        delta_config.enabled = parse_bool(mode.delta_enabled, field_name="analysis.rollout.delta_enabled")
    if mode.evidence_max_input_tokens is not None:
        evidence_config.max_input_tokens = min(
            int(evidence_config.max_input_tokens),
            int(mode.evidence_max_input_tokens),
        )
    if mode.evidence_max_new_tokens is not None:
        evidence_config.max_new_tokens = min(
            int(evidence_config.max_new_tokens),
            int(mode.evidence_max_new_tokens),
        )
    if mode.evidence_max_articles is not None:
        evidence_config.max_articles = min(
            int(evidence_config.max_articles),
            int(mode.evidence_max_articles),
        )
    if mode.evidence_max_articles_per_batch is not None:
        evidence_config.max_articles_per_batch = min(
            int(evidence_config.max_articles_per_batch),
            int(mode.evidence_max_articles_per_batch),
        )
    if mode.evidence_max_articles_dropped_to_avoid_split is not None:
        evidence_config.max_articles_dropped_to_avoid_split = min(
            int(evidence_config.max_articles_dropped_to_avoid_split),
            int(mode.evidence_max_articles_dropped_to_avoid_split),
        )
    if mode.evidence_max_article_chars is not None:
        evidence_config.max_article_chars = min(
            int(evidence_config.max_article_chars),
            int(mode.evidence_max_article_chars),
        )
    if mode.delta_max_input_tokens is not None:
        delta_config.max_input_tokens = min(
            int(delta_config.max_input_tokens),
            int(mode.delta_max_input_tokens),
        )
    if mode.delta_max_new_tokens is not None:
        delta_config.max_new_tokens = min(
            int(delta_config.max_new_tokens),
            int(mode.delta_max_new_tokens),
        )
    if mode.delta_max_articles is not None:
        delta_config.max_articles = min(
            int(delta_config.max_articles),
            int(mode.delta_max_articles),
        )
    if mode.delta_max_articles_per_batch is not None:
        delta_config.max_articles_per_batch = min(
            int(delta_config.max_articles_per_batch),
            int(mode.delta_max_articles_per_batch),
        )
    if mode.delta_max_articles_dropped_to_avoid_split is not None:
        delta_config.max_articles_dropped_to_avoid_split = min(
            int(delta_config.max_articles_dropped_to_avoid_split),
            int(mode.delta_max_articles_dropped_to_avoid_split),
        )
    if mode.delta_max_article_chars is not None:
        delta_config.max_article_chars = min(
            int(delta_config.max_article_chars),
            int(mode.delta_max_article_chars),
        )
    if mode.delta_max_prior_reports is not None:
        delta_config.max_prior_reports = min(
            int(delta_config.max_prior_reports),
            int(mode.delta_max_prior_reports),
        )


def resolve_analysis_stage_configs(analysis: AnalysisConfig, brief_name: str) -> tuple[EvidenceDistillationConfig, DeltaExtractionConfig, Dict[str, Any]]:
    evidence_config = _copy_evidence_config(getattr(analysis, "evidence_distillation", None))
    delta_config = _copy_delta_config(getattr(analysis, "delta_extraction", None))

    metadata: Dict[str, Any] = {
        "rollout_enabled": False,
        "rollout_profile": "",
        "rollout_mode": brief_name,
        "evidence_requested_enabled": bool(evidence_config.enabled),
        "delta_requested_enabled": bool(delta_config.enabled),
        "evidence_skip_reason": "disabled",
        "delta_skip_reason": "disabled",
    }
    rollout = getattr(analysis, "rollout", None)
    rollout_enabled = parse_bool(getattr(rollout, "enabled", False), default=False, field_name="analysis.rollout.enabled")
    if rollout is None or not rollout_enabled:
        metadata["evidence_effective_enabled"] = bool(evidence_config.enabled)
        metadata["delta_effective_enabled"] = bool(delta_config.enabled)
        metadata["evidence_skip_reason"] = "enabled" if evidence_config.enabled else "disabled"
        metadata["delta_skip_reason"] = "enabled" if delta_config.enabled else "disabled"
        return evidence_config, delta_config, metadata

    metadata["rollout_enabled"] = rollout_enabled
    profile = str(getattr(rollout, "profile", "safe_local")).strip().lower() or "safe_local"
    if profile not in ANALYSIS_ROLLOUT_PROFILE_NAMES:
        profile = "safe_local"
    metadata["rollout_profile"] = profile

    mode_key = "detailed" if brief_name == "detailed" else "general"
    profile_mode_raw = ANALYSIS_ROLLOUT_PROFILE_DEFAULTS.get(
        profile,
        ANALYSIS_ROLLOUT_PROFILE_DEFAULTS["safe_local"],
    ).get(mode_key, {})
    profile_mode = _rollout_mode_from_object(profile_mode_raw)
    explicit_mode = _rollout_mode_from_object(getattr(rollout, mode_key, None))
    effective_mode = _merge_rollout_modes(profile_mode, explicit_mode)

    _apply_rollout_mode_overrides(evidence_config, delta_config, effective_mode)
    metadata["evidence_effective_enabled"] = bool(evidence_config.enabled)
    metadata["delta_effective_enabled"] = bool(delta_config.enabled)
    metadata["evidence_skip_reason"] = "enabled" if evidence_config.enabled else "rollout_disabled"
    metadata["delta_skip_reason"] = "enabled" if delta_config.enabled else "rollout_disabled"
    return evidence_config, delta_config, metadata
