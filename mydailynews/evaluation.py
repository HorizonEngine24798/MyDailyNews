from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

DATASET_SCHEMA_VERSION = "baseline_eval_dataset.v1"
GUARDRAILS_SCHEMA_VERSION = "baseline_guardrails.v1"
REPORT_SCHEMA_VERSION = "baseline_eval_report.v1"
GUARDRAIL_RESULT_SCHEMA_VERSION = "baseline_eval_guardrail_result.v1"
METRIC_DEFINITIONS_VERSION = "baseline_metrics.v1"

INCIDENT_MARKERS = (
    "oom",
    "out of memory",
    "truncat",
    "dropped lower-ranked",
    "exceed budget",
    "prompt may exceed budget",
)


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_dataset(dataset_path: Path) -> Dict[str, Any]:
    dataset = load_json(dataset_path)
    schema_version = str(dataset.get("schema_version", "")).strip()
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema_version={schema_version!r}; expected {DATASET_SCHEMA_VERSION!r}"
        )
    profiles = dataset.get("profiles")
    cases = dataset.get("cases")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Dataset must include non-empty 'profiles' list.")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must include non-empty 'cases' list.")
    profile_ids = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Each profile must be an object.")
        profile_id = str(profile.get("id", "")).strip()
        if not profile_id:
            raise ValueError("Each profile must include non-empty 'id'.")
        if profile_id in profile_ids:
            raise ValueError(f"Duplicate profile id in dataset: {profile_id}")
        profile_ids.add(profile_id)
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object.")
        profile_id = str(case.get("profile_id", "")).strip()
        if profile_id not in profile_ids:
            raise ValueError(f"Case references unknown profile_id={profile_id!r}")
    return dataset


def load_guardrails(guardrails_path: Path) -> Dict[str, Any]:
    payload = load_json(guardrails_path)
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != GUARDRAILS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported guardrails schema_version={schema_version!r}; expected {GUARDRAILS_SCHEMA_VERSION!r}"
        )
    return payload


def build_evaluation_report(dataset_path: Path, artifact_root: Path) -> Dict[str, Any]:
    dataset = load_dataset(dataset_path)
    metric_definitions_version = str(
        dataset.get("metric_definitions_version") or METRIC_DEFINITIONS_VERSION
    ).strip() or METRIC_DEFINITIONS_VERSION
    cases = dataset["cases"]

    aggregate_top5_hits = 0
    aggregate_top5_total = 0
    aggregate_novelty_hits = 0
    aggregate_novelty_total = 0
    aggregate_alignment_hits = 0
    aggregate_alignment_total = 0
    aggregate_duplicate_count = 0
    aggregate_selected_total = 0
    aggregate_utility_scores: List[float] = []
    aggregate_latencies: List[float] = []
    aggregate_incidents = 0
    aggregate_warning_count = 0
    aggregate_missing_judgments = 0
    aggregate_prompt_tokens_by_stage: Dict[str, int] = {}
    aggregate_ai_input_tokens_by_bucket: Dict[str, int] = {}

    case_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    novelty_threshold = _safe_int(dataset.get("rubric", {}).get("thresholds", {}).get("novelty_high_min"), default=4)
    alignment_threshold = _safe_int(
        dataset.get("rubric", {}).get("thresholds", {}).get("personal_relevance_alignment_min"),
        default=4,
    )

    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        brief_name = str(case.get("brief_name", "")).strip() or "unknown"
        profile_id = str(case.get("profile_id", "")).strip() or "unknown"
        artifact_paths = case.get("artifact_paths", {})
        if not isinstance(artifact_paths, dict):
            raise ValueError(f"Case {case_id!r} has invalid artifact_paths")
        brief_path = artifact_root / str(artifact_paths.get("brief_json", "")).strip()
        analytics_path = artifact_root / str(artifact_paths.get("analytics_json", "")).strip()
        probe_path_text = str(artifact_paths.get("llm_probe_json", "")).strip()
        probe_path = artifact_root / probe_path_text if probe_path_text else None

        if not brief_path.exists():
            raise FileNotFoundError(f"Case {case_id}: missing brief artifact {brief_path}")
        if not analytics_path.exists():
            raise FileNotFoundError(f"Case {case_id}: missing analytics artifact {analytics_path}")

        brief_payload = load_json(brief_path)
        analytics_payload = load_json(analytics_path)
        probe_payload = load_json(probe_path) if probe_path and probe_path.exists() else {}

        selected_articles = _as_list(brief_payload.get("selected_articles"))
        selected_count = len(selected_articles)
        top_articles = selected_articles[:5]

        judgment_rows = _as_list(case.get("story_judgments"))
        judgments = _build_judgment_lookup(judgment_rows)
        brief_ratings = case.get("brief_ratings", {}) if isinstance(case.get("brief_ratings"), dict) else {}

        top_hits, top_total, missing_top = _top5_precision(top_articles, judgments)
        novelty_hits, novelty_total, missing_novelty = _ratio_for_dimension(
            selected_articles,
            judgments,
            dimension="novelty",
            threshold=novelty_threshold,
        )
        alignment_hits, alignment_total, missing_alignment = _ratio_for_dimension(
            selected_articles,
            judgments,
            dimension="personal_relevance",
            threshold=alignment_threshold,
        )
        duplication_count, duplication_leak_rate = _duplication_leak(selected_articles)
        utility_score = _utility_score(brief_ratings, selected_articles, judgments)
        latency_sec = _safe_float(
            analytics_payload.get("durations_sec", {}).get("pipeline.total"),
            default=0.0,
        )

        prompt_tokens_by_stage = _max_prompt_tokens_by_stage(probe_payload)
        ai_input_tokens_by_bucket = _max_ai_input_tokens_by_bucket(analytics_payload)
        warnings_for_case = _collect_case_warnings(brief_payload)
        incidents = _count_incidents(warnings_for_case, analytics_payload)

        if missing_top > 0:
            warnings.append(f"Case {case_id}: missing judgments for {missing_top} top-5 selected stories.")
        missing_total = max(missing_novelty, missing_alignment)
        if missing_total > 0:
            warnings.append(f"Case {case_id}: missing judgments for {missing_total} selected stories.")
        if probe_path and not probe_path.exists():
            warnings.append(f"Case {case_id}: llm_probe_json not found ({probe_path})")

        aggregate_top5_hits += top_hits
        aggregate_top5_total += top_total
        aggregate_novelty_hits += novelty_hits
        aggregate_novelty_total += novelty_total
        aggregate_alignment_hits += alignment_hits
        aggregate_alignment_total += alignment_total
        aggregate_duplicate_count += duplication_count
        aggregate_selected_total += selected_count
        if utility_score is not None:
            aggregate_utility_scores.append(utility_score)
        if latency_sec > 0:
            aggregate_latencies.append(latency_sec)
        aggregate_incidents += incidents
        aggregate_warning_count += len(warnings_for_case)
        aggregate_missing_judgments += missing_total
        _merge_max_dict(aggregate_prompt_tokens_by_stage, prompt_tokens_by_stage)
        _merge_max_dict(aggregate_ai_input_tokens_by_bucket, ai_input_tokens_by_bucket)

        case_rows.append(
            {
                "case_id": case_id,
                "profile_id": profile_id,
                "brief_name": brief_name,
                "selected_count": selected_count,
                "top5_story_value_precision": _ratio(top_hits, top_total),
                "novelty_ratio": _ratio(novelty_hits, novelty_total),
                "decision_context_alignment": _ratio(alignment_hits, alignment_total),
                "duplication_leak_rate": duplication_leak_rate,
                "brief_utility_score": utility_score,
                "latency_sec": round(latency_sec, 4),
                "warning_count": len(warnings_for_case),
                "oom_or_truncation_incidents": incidents,
                "missing_story_judgments": missing_total,
                "max_prompt_tokens_by_stage": dict(sorted(prompt_tokens_by_stage.items())),
                "max_ai_input_tokens_by_bucket": dict(sorted(ai_input_tokens_by_bucket.items())),
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "metric_definitions_version": metric_definitions_version,
        "dataset": {
            "path": str(dataset_path),
            "dataset_version": str(dataset.get("dataset_version", "")).strip(),
            "profiles_count": len(dataset.get("profiles", [])),
            "cases_count": len(cases),
        },
        "aggregate": {
            "top5_story_value_precision": _ratio(aggregate_top5_hits, aggregate_top5_total),
            "novelty_ratio": _ratio(aggregate_novelty_hits, aggregate_novelty_total),
            "duplication_leak_rate": _ratio(aggregate_duplicate_count, aggregate_selected_total),
            "decision_context_alignment": _ratio(aggregate_alignment_hits, aggregate_alignment_total),
            "brief_utility_score": round(mean(aggregate_utility_scores), 4) if aggregate_utility_scores else None,
            "latency_p95_sec": round(percentile(aggregate_latencies, 95), 4) if aggregate_latencies else None,
            "max_prompt_tokens_by_stage": dict(sorted(aggregate_prompt_tokens_by_stage.items())),
            "max_ai_input_tokens_by_bucket": dict(sorted(aggregate_ai_input_tokens_by_bucket.items())),
            "oom_or_truncation_incidents": aggregate_incidents,
            "warning_count": aggregate_warning_count,
            "missing_story_judgments": aggregate_missing_judgments,
            "cases_with_latency": len(aggregate_latencies),
        },
        "cases": case_rows,
        "warnings": warnings,
    }
    return report


def evaluate_guardrails(report: Dict[str, Any], guardrails: Dict[str, Any]) -> Dict[str, Any]:
    aggregate = report.get("aggregate", {}) if isinstance(report.get("aggregate"), dict) else {}
    checks: List[Dict[str, Any]] = []

    performance_limits = guardrails.get("performance_limits", {})
    if isinstance(performance_limits, dict):
        for metric_name, bound in performance_limits.items():
            if not isinstance(bound, dict):
                continue
            actual = _safe_float(aggregate.get(metric_name), default=None)
            min_bound = _safe_float(bound.get("min"), default=None)
            max_bound = _safe_float(bound.get("max"), default=None)
            if actual is None:
                checks.append(
                    {
                        "name": metric_name,
                        "status": "fail",
                        "reason": "missing_actual",
                        "actual": None,
                        "expected": {"min": min_bound, "max": max_bound},
                    }
                )
                continue
            passed = True
            if min_bound is not None and actual < min_bound:
                passed = False
            if max_bound is not None and actual > max_bound:
                passed = False
            checks.append(
                {
                    "name": metric_name,
                    "status": "pass" if passed else "fail",
                    "reason": "within_bounds" if passed else "out_of_bounds",
                    "actual": actual,
                    "expected": {"min": min_bound, "max": max_bound},
                }
            )

    prompt_limits = guardrails.get("prompt_token_limits_by_stage", {})
    prompt_actual = aggregate.get("max_prompt_tokens_by_stage", {})
    if isinstance(prompt_limits, dict):
        for stage_name, limit_value in prompt_limits.items():
            limit = _safe_int(limit_value, default=None)
            actual = _safe_int(prompt_actual.get(stage_name), default=None) if isinstance(prompt_actual, dict) else None
            if limit is None:
                continue
            passed = actual is not None and actual <= limit
            checks.append(
                {
                    "name": f"max_prompt_tokens_by_stage.{stage_name}",
                    "status": "pass" if passed else "fail",
                    "reason": "within_bounds" if passed else ("missing_actual" if actual is None else "out_of_bounds"),
                    "actual": actual,
                    "expected": {"max": limit},
                }
            )

    quality_tolerance = guardrails.get("quality_regression_tolerance", {})
    if isinstance(quality_tolerance, dict):
        for metric_name, settings in quality_tolerance.items():
            if not isinstance(settings, dict):
                continue
            baseline = _safe_float(settings.get("baseline"), default=None)
            max_drop = _safe_float(settings.get("max_drop"), default=0.0)
            actual = _safe_float(aggregate.get(metric_name), default=None)
            if baseline is None:
                continue
            min_allowed = baseline - float(max_drop or 0.0)
            passed = actual is not None and actual >= min_allowed
            checks.append(
                {
                    "name": metric_name,
                    "status": "pass" if passed else "fail",
                    "reason": "within_tolerance" if passed else ("missing_actual" if actual is None else "regressed"),
                    "actual": actual,
                    "expected": {
                        "baseline": baseline,
                        "max_drop": max_drop,
                        "min_allowed": min_allowed,
                    },
                }
            )

    passed = all(check.get("status") == "pass" for check in checks) if checks else True
    return {
        "schema_version": GUARDRAIL_RESULT_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "guardrails_version": str(guardrails.get("guardrails_version", "")).strip(),
        "passed": passed,
        "checks": checks,
    }


def render_report_markdown(report: Dict[str, Any], guardrail_result: Dict[str, Any] | None = None) -> str:
    dataset = report.get("dataset", {}) if isinstance(report.get("dataset"), dict) else {}
    aggregate = report.get("aggregate", {}) if isinstance(report.get("aggregate"), dict) else {}
    cases = report.get("cases", []) if isinstance(report.get("cases"), list) else []
    report_warnings = report.get("warnings", []) if isinstance(report.get("warnings"), list) else []

    lines: List[str] = []
    lines.append("# Baseline Evaluation Report")
    lines.append("")
    lines.append(f"- Generated at: {report.get('generated_at', '')}")
    lines.append(f"- Dataset version: {dataset.get('dataset_version', '') or 'n/a'}")
    lines.append(f"- Metric definitions version: {report.get('metric_definitions_version', '')}")
    lines.append(f"- Profiles: {dataset.get('profiles_count', 0)}")
    lines.append(f"- Cases: {dataset.get('cases_count', 0)}")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(f"- top5_story_value_precision: {_fmt_num(aggregate.get('top5_story_value_precision'))}")
    lines.append(f"- novelty_ratio: {_fmt_num(aggregate.get('novelty_ratio'))}")
    lines.append(f"- duplication_leak_rate: {_fmt_num(aggregate.get('duplication_leak_rate'))}")
    lines.append(f"- decision_context_alignment: {_fmt_num(aggregate.get('decision_context_alignment'))}")
    lines.append(f"- brief_utility_score: {_fmt_num(aggregate.get('brief_utility_score'))}")
    lines.append(f"- latency_p95_sec: {_fmt_num(aggregate.get('latency_p95_sec'))}")
    lines.append(f"- oom_or_truncation_incidents: {int(aggregate.get('oom_or_truncation_incidents') or 0)}")
    lines.append(f"- warning_count: {int(aggregate.get('warning_count') or 0)}")
    lines.append(f"- missing_story_judgments: {int(aggregate.get('missing_story_judgments') or 0)}")
    lines.append("")
    lines.append("### max_prompt_tokens_by_stage")
    prompt_by_stage = aggregate.get("max_prompt_tokens_by_stage", {})
    if isinstance(prompt_by_stage, dict) and prompt_by_stage:
        for stage_name, token_value in sorted(prompt_by_stage.items()):
            lines.append(f"- {stage_name}: {token_value}")
    else:
        lines.append("- n/a")
    lines.append("")
    lines.append("### max_ai_input_tokens_by_bucket")
    ai_bucket_tokens = aggregate.get("max_ai_input_tokens_by_bucket", {})
    if isinstance(ai_bucket_tokens, dict) and ai_bucket_tokens:
        for bucket_name, token_value in sorted(ai_bucket_tokens.items()):
            lines.append(f"- {bucket_name}: {token_value}")
    else:
        lines.append("- n/a")
    lines.append("")

    if guardrail_result is not None:
        lines.append("## Guardrails")
        lines.append("")
        lines.append(f"- Result: {'PASS' if guardrail_result.get('passed') else 'FAIL'}")
        checks = guardrail_result.get("checks", [])
        if isinstance(checks, list) and checks:
            for check in checks:
                if not isinstance(check, dict):
                    continue
                lines.append(
                    f"- [{str(check.get('status', '')).upper()}] {check.get('name')}: "
                    f"actual={_fmt_num(check.get('actual'))}, expected={json.dumps(check.get('expected', {}), ensure_ascii=False)}"
                )
        lines.append("")

    lines.append("## Case Metrics")
    lines.append("")
    for case in cases:
        if not isinstance(case, dict):
            continue
        lines.append(
            f"### {case.get('case_id', 'unknown')} "
            f"({case.get('profile_id', 'unknown')}/{case.get('brief_name', 'unknown')})"
        )
        lines.append(f"- selected_count: {int(case.get('selected_count') or 0)}")
        lines.append(f"- top5_story_value_precision: {_fmt_num(case.get('top5_story_value_precision'))}")
        lines.append(f"- novelty_ratio: {_fmt_num(case.get('novelty_ratio'))}")
        lines.append(f"- duplication_leak_rate: {_fmt_num(case.get('duplication_leak_rate'))}")
        lines.append(f"- decision_context_alignment: {_fmt_num(case.get('decision_context_alignment'))}")
        lines.append(f"- brief_utility_score: {_fmt_num(case.get('brief_utility_score'))}")
        lines.append(f"- latency_sec: {_fmt_num(case.get('latency_sec'))}")
        lines.append(f"- oom_or_truncation_incidents: {int(case.get('oom_or_truncation_incidents') or 0)}")
        lines.append(f"- warning_count: {int(case.get('warning_count') or 0)}")
        lines.append(f"- missing_story_judgments: {int(case.get('missing_story_judgments') or 0)}")
        lines.append("")

    if report_warnings:
        lines.append("## Evaluation Warnings")
        lines.append("")
        for item in report_warnings:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_report(output_dir: Path, report: Dict[str, Any], guardrail_result: Dict[str, Any] | None = None) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_eval_report.json"
    markdown_path = output_dir / "baseline_eval_report.md"

    payload = dict(report)
    if guardrail_result is not None:
        payload["guardrails"] = guardrail_result

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_report_markdown(report, guardrail_result=guardrail_result), encoding="utf-8")
    return json_path, markdown_path


def percentile(values: Iterable[float], q: float) -> float:
    points = sorted(float(value) for value in values if value is not None)
    if not points:
        return 0.0
    if len(points) == 1:
        return points[0]
    quantile = max(0.0, min(100.0, float(q))) / 100.0
    position = quantile * (len(points) - 1)
    low_index = int(math.floor(position))
    high_index = int(math.ceil(position))
    if low_index == high_index:
        return points[low_index]
    low_value = points[low_index]
    high_value = points[high_index]
    ratio = position - low_index
    return low_value + (high_value - low_value) * ratio


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_story_key_from_article(article: Dict[str, Any]) -> str:
    story_id = str(article.get("id", "")).strip()
    if story_id:
        return f"id:{story_id}"
    url = _normalize_url(str(article.get("url", "")))
    if url:
        return f"url:{url}"
    headline = _normalize_text(str(article.get("headline") or article.get("title") or ""))
    return f"headline:{headline}" if headline else ""


def _normalize_story_key_from_judgment(row: Dict[str, Any]) -> str:
    story_id = str(row.get("story_id", "")).strip()
    if story_id:
        return f"id:{story_id}"
    url = _normalize_url(str(row.get("url", "")))
    if url:
        return f"url:{url}"
    headline = _normalize_text(str(row.get("headline", "")))
    return f"headline:{headline}" if headline else ""


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _normalize_url(url: str) -> str:
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower().strip()
    path = (parsed.path or "").strip()
    if not host and not path:
        return ""
    return f"{host}{path}"


def _build_judgment_lookup(rows: List[Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _normalize_story_key_from_judgment(row)
        if not key:
            continue
        mapping[key] = row
    return mapping


def _top5_precision(top_articles: List[Any], judgments: Dict[str, Dict[str, Any]]) -> Tuple[int, int, int]:
    hits = 0
    total = len(top_articles)
    missing = 0
    for article in top_articles:
        if not isinstance(article, dict):
            continue
        key = _normalize_story_key_from_article(article)
        judgment = judgments.get(key)
        if judgment is None:
            missing += 1
            continue
        if bool(judgment.get("would_regret_missing")):
            hits += 1
    return hits, total, missing


def _ratio_for_dimension(
    selected_articles: List[Any],
    judgments: Dict[str, Dict[str, Any]],
    *,
    dimension: str,
    threshold: int,
) -> Tuple[int, int, int]:
    hits = 0
    total = 0
    missing = 0
    for article in selected_articles:
        if not isinstance(article, dict):
            continue
        key = _normalize_story_key_from_article(article)
        judgment = judgments.get(key)
        if judgment is None:
            missing += 1
            continue
        score = _safe_int(judgment.get(dimension), default=None)
        if score is None:
            missing += 1
            continue
        total += 1
        if score >= threshold:
            hits += 1
    return hits, total, missing


def _duplication_leak(selected_articles: List[Any]) -> Tuple[int, float]:
    selected_count = len(selected_articles)
    if selected_count == 0:
        return 0, 0.0
    cluster_keys: List[str] = []
    for article in selected_articles:
        if not isinstance(article, dict):
            continue
        cluster = article.get("event_cluster")
        if isinstance(cluster, dict):
            cluster_id = str(cluster.get("id", "")).strip()
            if cluster_id:
                cluster_keys.append(f"cluster:{cluster_id}")
                continue
        cluster_keys.append(_normalize_story_key_from_article(article))
    unique_count = len({key for key in cluster_keys if key})
    duplicates = max(0, selected_count - unique_count)
    return duplicates, _ratio(duplicates, selected_count)


def _utility_score(
    brief_ratings: Dict[str, Any],
    selected_articles: List[Any],
    judgments: Dict[str, Dict[str, Any]],
) -> float | None:
    explicit_utility = _safe_float(brief_ratings.get("utility"), default=None)
    if explicit_utility is not None:
        return round(explicit_utility, 4)

    utility_scores: List[float] = []
    for article in selected_articles:
        if not isinstance(article, dict):
            continue
        key = _normalize_story_key_from_article(article)
        judgment = judgments.get(key)
        if judgment is None:
            continue
        utility_value = _safe_float(judgment.get("utility"), default=None)
        if utility_value is not None:
            utility_scores.append(utility_value)
    if not utility_scores:
        return None
    return round(mean(utility_scores), 4)


def _max_prompt_tokens_by_stage(probe_payload: Dict[str, Any]) -> Dict[str, int]:
    stage_max: Dict[str, int] = {}
    records = probe_payload.get("records")
    if not isinstance(records, list):
        return stage_max
    for row in records:
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage", "")).strip()
        prompt_tokens = _safe_int(row.get("prompt_tokens"), default=None)
        if not stage or prompt_tokens is None:
            continue
        stage_max[stage] = max(stage_max.get(stage, 0), prompt_tokens)
    return stage_max


def _max_ai_input_tokens_by_bucket(analytics_payload: Dict[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    by_bucket = analytics_payload.get("ai", {}).get("by_bucket", {})
    if not isinstance(by_bucket, dict):
        return result
    for bucket, row in by_bucket.items():
        if not isinstance(row, dict):
            continue
        value = _safe_int(row.get("input_tokens"), default=None)
        if value is None:
            continue
        result[str(bucket)] = max(result.get(str(bucket), 0), value)
    return result


def _collect_case_warnings(brief_payload: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    metadata = brief_payload.get("metadata", {})
    if isinstance(metadata, dict):
        for item in _as_list(metadata.get("warnings")):
            text = str(item).strip()
            if text:
                warnings.append(text)
    for item in _as_list(brief_payload.get("warnings")):
        text = str(item).strip()
        if text:
            warnings.append(text)
    return warnings


def _count_incidents(warnings_for_case: List[str], analytics_payload: Dict[str, Any]) -> int:
    incidents = 0
    for warning in warnings_for_case:
        lowered = warning.lower()
        if any(marker in lowered for marker in INCIDENT_MARKERS):
            incidents += 1

    metrics = analytics_payload.get("metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            lowered_key = str(key).lower()
            if "oom" not in lowered_key:
                continue
            numeric = _safe_int(value, default=0) or 0
            incidents += max(0, numeric)
    return incidents


def _merge_max_dict(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = max(target.get(key, 0), int(value))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _fmt_num(value: Any) -> str:
    numeric = _safe_float(value, default=None)
    if numeric is None:
        return "n/a"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.4f}".rstrip("0").rstrip(".")
