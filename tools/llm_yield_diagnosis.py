from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple


KV_RE = re.compile(r"(\w+)=('(?:[^'\\]|\\.)*'|[^\s|]+)")
DEBUG_RE = re.compile(r"^\[debug\]\s+stage=([^|]+?)\s*\|\s*action=([^|]+?)(?:\s*\|\s*(.*))?$")
PROBE_RE = re.compile(r"^\[llm_probe\]\s+(call_start|call_done)\s+(.*)$")
HEADLINE_LABEL_RE = re.compile(r"headline scoring batch\s+(\d+)/(\d+)")


def _parse_kv_pairs(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for key, raw in KV_RE.findall(text or ""):
        value = raw
        if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
            value = value[1:-1]
        fields[key] = value
    return fields


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _latest_run_dir(output_root: Path) -> Path:
    candidates = []
    for path in output_root.glob("*_baseline"):
        if not path.is_dir():
            continue
        stdout_log = path / "llm_call_probe.stdout.txt"
        if stdout_log.exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No *_baseline directory with llm_call_probe.stdout.txt under {output_root}")
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_inputs(
    *,
    run_dir: str,
    stdout_log: str,
    probe_json: str,
    output_root: str,
) -> Tuple[Path, Path, Path | None]:
    if stdout_log:
        stdout_path = Path(stdout_log)
        base_dir = stdout_path.parent
    else:
        if run_dir:
            base_dir = Path(run_dir)
        else:
            base_dir = _latest_run_dir(Path(output_root))
        stdout_path = base_dir / "llm_call_probe.stdout.txt"

    if not stdout_path.exists():
        raise FileNotFoundError(f"stdout log not found: {stdout_path}")

    if probe_json:
        probe_path: Path | None = Path(probe_json)
    else:
        guessed = base_dir / "llm_probe.json"
        probe_path = guessed if guessed.exists() else None

    return base_dir, stdout_path, probe_path


def _parse_log(stdout_path: Path) -> Dict[str, Any]:
    lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()

    call_starts: List[Dict[str, str]] = []
    call_dones: List[Dict[str, str]] = []
    ai_requests: List[Dict[str, str]] = []
    ai_responses: List[Dict[str, str]] = []
    headline_batch_complete: List[Dict[str, str]] = []
    headline_batch_scoring: List[Dict[str, str]] = []
    headline_batch_invalid: List[Dict[str, str]] = []

    for line in lines:
        probe_match = PROBE_RE.match(line.strip())
        if probe_match:
            event = probe_match.group(1)
            fields = _parse_kv_pairs(probe_match.group(2))
            if event == "call_start":
                call_starts.append(fields)
            elif event == "call_done":
                call_dones.append(fields)
            continue

        debug_match = DEBUG_RE.match(line.strip())
        if not debug_match:
            continue
        stage = debug_match.group(1).strip()
        action = debug_match.group(2).strip()
        fields = _parse_kv_pairs(debug_match.group(3) or "")
        fields["stage"] = stage
        fields["action"] = action

        if stage == "ai.request":
            ai_requests.append(fields)
        elif stage == "ai.response":
            ai_responses.append(fields)
        elif stage == "headline.ai.batch" and action == "complete":
            headline_batch_complete.append(fields)
        elif stage == "headline.ai.batch" and action == "scoring":
            headline_batch_scoring.append(fields)
        elif stage == "headline.ai.batch" and action == "skipped_invalid_json":
            headline_batch_invalid.append(fields)

    return {
        "line_count": len(lines),
        "call_starts": call_starts,
        "call_dones": call_dones,
        "ai_requests": ai_requests,
        "ai_responses": ai_responses,
        "headline_batch_complete": headline_batch_complete,
        "headline_batch_scoring": headline_batch_scoring,
        "headline_batch_invalid": headline_batch_invalid,
    }


def _compute_metrics(parsed: Dict[str, Any], probe_path: Path | None) -> Dict[str, Any]:
    call_starts: List[Dict[str, str]] = parsed["call_starts"]
    call_dones: List[Dict[str, str]] = parsed["call_dones"]
    ai_requests: List[Dict[str, str]] = parsed["ai_requests"]
    ai_responses: List[Dict[str, str]] = parsed["ai_responses"]
    headline_batch_complete: List[Dict[str, str]] = parsed["headline_batch_complete"]
    headline_batch_scoring: List[Dict[str, str]] = parsed["headline_batch_scoring"]
    headline_batch_invalid: List[Dict[str, str]] = parsed["headline_batch_invalid"]

    start_by_stage = Counter(item.get("stage", "unknown") for item in call_starts)
    done_by_status = Counter(item.get("status", "unknown") for item in call_dones)
    done_errors = Counter(item.get("error_type", "") for item in call_dones if item.get("status") != "ok")

    response_by_status = Counter(item.get("status", "unknown") for item in ai_responses)
    request_attempt_counter = Counter(item.get("attempt", "unknown") for item in ai_requests)
    request_by_action = Counter(item.get("action", "unknown") for item in ai_requests)

    attempt_outcomes = sum(response_by_status.values())
    syntax_yield_attempt = _pct(response_by_status.get("ok", 0), attempt_outcomes)
    syntax_yield_call = _pct(done_by_status.get("ok", 0), len(call_dones))

    # Retry diagnostics by label/action.
    status_by_action: Dict[str, Dict[str, str]] = defaultdict(dict)
    for item in ai_responses:
        action = item.get("action", "unknown")
        attempt = item.get("attempt", "unknown")
        status = item.get("status", "unknown")
        status_by_action[action][attempt] = status

    retry_triggered = 0
    retry_salvaged = 0
    retry_failed = 0
    retry_missing_second = 0
    first_invalid = 0
    first_ok = 0
    for action, statuses in status_by_action.items():
        first = statuses.get("1/2")
        second = statuses.get("2/2")
        if first == "ok":
            first_ok += 1
        if first == "invalid_json":
            first_invalid += 1
        if second is not None:
            retry_triggered += 1
            if first == "invalid_json" and second == "ok":
                retry_salvaged += 1
            elif first == "invalid_json" and second == "invalid_json":
                retry_failed += 1
        elif first == "invalid_json":
            retry_missing_second += 1

    retry_salvage_rate = _pct(retry_salvaged, retry_triggered)

    # Headline semantic yield.
    headline_started_calls = sum(1 for item in call_starts if item.get("stage") == "headline_scoring")
    headline_completed_calls = sum(1 for item in call_dones if item.get("stage") == "headline_scoring")
    headline_call_elapsed = [
        _safe_float(item.get("elapsed_sec", 0.0))
        for item in call_dones
        if item.get("stage") == "headline_scoring"
    ]
    headline_total_elapsed_sec = sum(headline_call_elapsed)
    headline_elapsed_p50_sec = median(headline_call_elapsed) if headline_call_elapsed else 0.0
    headline_elapsed_avg_sec = (headline_total_elapsed_sec / len(headline_call_elapsed)) if headline_call_elapsed else 0.0

    ai_decisions = [_safe_int(item.get("ai_decisions", 0)) for item in headline_batch_complete]
    headline_decision_total = sum(ai_decisions)
    headline_decision_nonzero_batches = sum(1 for value in ai_decisions if value > 0)
    headline_batch_items = [_safe_int(item.get("items", 0)) for item in headline_batch_scoring]
    headline_batch_size_distribution = Counter(headline_batch_items)

    semantic_yield_per_started_call = (
        headline_decision_total / headline_started_calls if headline_started_calls else 0.0
    )
    semantic_yield_per_completed_call = (
        headline_decision_total / headline_completed_calls if headline_completed_calls else 0.0
    )
    decision_per_request = headline_decision_total / len(ai_requests) if ai_requests else 0.0
    decision_per_gpu_hour = (
        headline_decision_total / (headline_total_elapsed_sec / 3600.0) if headline_total_elapsed_sec > 0 else 0.0
    )

    # Truncation pressure diagnostics.
    prompt_over_limit = 0
    ratio_values: List[float] = []
    for item in call_starts:
        if item.get("stage") != "headline_scoring":
            continue
        prompt_tokens = _safe_int(item.get("prompt_tokens", 0))
        limit_tokens = _safe_int(item.get("planned_input_tokens", 0))
        if limit_tokens <= 0:
            continue
        if prompt_tokens > limit_tokens:
            prompt_over_limit += 1
        ratio_values.append(prompt_tokens / limit_tokens)
    truncation_pressure_pct = _pct(prompt_over_limit, headline_started_calls)
    truncation_ratio_avg = (sum(ratio_values) / len(ratio_values)) if ratio_values else 0.0

    denominator_counter = Counter()
    for item in call_starts:
        label = item.get("label", "")
        match = HEADLINE_LABEL_RE.search(label)
        if match:
            denominator_counter[int(match.group(2))] += 1

    probe_meta: Dict[str, Any] = {}
    if probe_path and probe_path.exists():
        try:
            probe_meta = json.loads(probe_path.read_text(encoding="utf-8")).get("meta", {})
        except Exception:
            probe_meta = {}

    return {
        "summary": {
            "call_start_total": len(call_starts),
            "call_done_total": len(call_dones),
            "in_flight_calls": len(call_starts) - len(call_dones),
            "ai_request_total": len(ai_requests),
            "ai_response_total": len(ai_responses),
            "attempts_per_started_call": round((len(ai_requests) / len(call_starts)) if call_starts else 0.0, 3),
            "attempts_per_completed_call": round((len(ai_requests) / len(call_dones)) if call_dones else 0.0, 3),
        },
        "syntax_yield": {
            "attempt_ok_pct": round(syntax_yield_attempt, 3),
            "call_ok_pct": round(syntax_yield_call, 3),
            "response_status_counts": dict(response_by_status),
            "done_status_counts": dict(done_by_status),
            "done_error_type_counts": dict(done_errors),
        },
        "retry_yield": {
            "first_attempt_ok": first_ok,
            "first_attempt_invalid_json": first_invalid,
            "retry_triggered": retry_triggered,
            "retry_salvaged": retry_salvaged,
            "retry_failed_invalid_json_again": retry_failed,
            "retry_missing_second_attempt": retry_missing_second,
            "retry_salvage_rate_pct": round(retry_salvage_rate, 3),
            "request_attempt_counts": dict(request_attempt_counter),
        },
        "semantic_yield_headline": {
            "headline_started_calls": headline_started_calls,
            "headline_completed_calls": headline_completed_calls,
            "headline_batch_scoring_count": len(headline_batch_scoring),
            "headline_batch_complete_count": len(headline_batch_complete),
            "headline_batch_invalid_json_count": len(headline_batch_invalid),
            "headline_batch_size_distribution": dict(headline_batch_size_distribution),
            "headline_total_ai_decisions": headline_decision_total,
            "headline_nonzero_decision_batches": headline_decision_nonzero_batches,
            "decisions_per_started_call": round(semantic_yield_per_started_call, 6),
            "decisions_per_completed_call": round(semantic_yield_per_completed_call, 6),
            "decisions_per_ai_request": round(decision_per_request, 6),
        },
        "cost_yield_headline": {
            "headline_total_elapsed_sec": round(headline_total_elapsed_sec, 3),
            "headline_total_elapsed_hours": round(headline_total_elapsed_sec / 3600.0, 4),
            "headline_elapsed_avg_sec": round(headline_elapsed_avg_sec, 3),
            "headline_elapsed_p50_sec": round(float(headline_elapsed_p50_sec), 3),
            "decisions_per_gpu_hour": round(decision_per_gpu_hour, 6),
        },
        "truncation_pressure_headline": {
            "headline_prompt_over_input_limit_count": prompt_over_limit,
            "headline_prompt_over_input_limit_pct": round(truncation_pressure_pct, 3),
            "headline_prompt_to_limit_ratio_avg": round(truncation_ratio_avg, 3),
        },
        "stage_mix": {
            "call_start_by_stage": dict(start_by_stage),
            "request_by_action_top_20": request_by_action.most_common(20),
            "headline_denominator_counts": dict(denominator_counter),
        },
        "probe_meta": probe_meta,
        "line_count": parsed["line_count"],
    }


def _print_human_report(run_dir: Path, stdout_path: Path, probe_path: Path | None, metrics: Dict[str, Any]) -> None:
    summary = metrics["summary"]
    syntax = metrics["syntax_yield"]
    retry = metrics["retry_yield"]
    semantic = metrics["semantic_yield_headline"]
    cost = metrics["cost_yield_headline"]
    trunc = metrics["truncation_pressure_headline"]

    print(f"run_dir={run_dir}")
    print(f"stdout_log={stdout_path}")
    print(f"probe_json={probe_path if probe_path else 'missing'}")
    print(f"line_count={metrics['line_count']}")
    print("")
    print("=== LLM Yield Diagnosis ===")
    print(
        "calls "
        f"started={summary['call_start_total']} completed={summary['call_done_total']} "
        f"in_flight={summary['in_flight_calls']} requests={summary['ai_request_total']} "
        f"responses={summary['ai_response_total']}"
    )
    print(
        "attempts "
        f"per_started_call={summary['attempts_per_started_call']} "
        f"per_completed_call={summary['attempts_per_completed_call']}"
    )
    print(
        "syntax_yield "
        f"attempt_ok_pct={syntax['attempt_ok_pct']} call_ok_pct={syntax['call_ok_pct']} "
        f"response_status={syntax['response_status_counts']}"
    )
    print(
        "retry_yield "
        f"first_ok={retry['first_attempt_ok']} first_invalid={retry['first_attempt_invalid_json']} "
        f"retried={retry['retry_triggered']} salvaged={retry['retry_salvaged']} "
        f"failed_again={retry['retry_failed_invalid_json_again']} salvage_rate_pct={retry['retry_salvage_rate_pct']}"
    )
    print(
        "semantic_yield_headline "
        f"started_calls={semantic['headline_started_calls']} completed_calls={semantic['headline_completed_calls']} "
        f"batch_scoring={semantic['headline_batch_scoring_count']} batch_complete={semantic['headline_batch_complete_count']} "
        f"batch_invalid_json={semantic['headline_batch_invalid_json_count']} total_ai_decisions={semantic['headline_total_ai_decisions']} "
        f"decisions_per_started_call={semantic['decisions_per_started_call']} "
        f"decisions_per_request={semantic['decisions_per_ai_request']}"
    )
    print(
        "cost_yield_headline "
        f"elapsed_hours={cost['headline_total_elapsed_hours']} avg_call_sec={cost['headline_elapsed_avg_sec']} "
        f"p50_call_sec={cost['headline_elapsed_p50_sec']} decisions_per_gpu_hour={cost['decisions_per_gpu_hour']}"
    )
    print(
        "truncation_pressure_headline "
        f"over_limit_count={trunc['headline_prompt_over_input_limit_count']} "
        f"over_limit_pct={trunc['headline_prompt_over_input_limit_pct']} "
        f"avg_prompt_to_limit_ratio={trunc['headline_prompt_to_limit_ratio_avg']}"
    )
    print(f"headline_batch_size_distribution={semantic['headline_batch_size_distribution']}")
    print(f"headline_denominator_counts={metrics['stage_mix']['headline_denominator_counts']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose LLM yield (syntax, retry, semantic, cost) from llm_call_probe logs.")
    parser.add_argument("--run-dir", default="", help="Path to *_baseline run directory.")
    parser.add_argument("--stdout-log", default="", help="Path to llm_call_probe.stdout.txt.")
    parser.add_argument("--probe-json", default="", help="Path to llm_probe.json (optional).")
    parser.add_argument("--output-root", default="output", help="Root output dir used when auto-selecting latest run.")
    parser.add_argument("--output-json", default="", help="Optional path to write full diagnosis JSON.")
    parser.add_argument("--print-json", action="store_true", help="Print full diagnosis JSON to stdout.")
    args = parser.parse_args()

    run_dir, stdout_path, probe_path = _resolve_inputs(
        run_dir=args.run_dir,
        stdout_log=args.stdout_log,
        probe_json=args.probe_json,
        output_root=args.output_root,
    )
    parsed = _parse_log(stdout_path)
    metrics = _compute_metrics(parsed, probe_path)
    _print_human_report(run_dir, stdout_path, probe_path, metrics)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"diagnosis_json={output_path}")

    if args.print_json:
        print("")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
