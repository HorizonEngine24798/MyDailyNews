from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except Exception:  # pragma: no cover - optional for non-transformers envs
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.ai.factory import create_ai_client
from mydailynews.config import load_config
from mydailynews.debug import DebugLogger


def _cuda_mb(value: float) -> float:
    return round(value / (1024 * 1024), 2)


def _cuda_snapshot() -> Dict[str, float]:
    if torch is None or not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": _cuda_mb(torch.cuda.memory_allocated()),
        "reserved_mb": _cuda_mb(torch.cuda.memory_reserved()),
    }


def _build_dummy_prompt(target_input_limit: int) -> str:
    return ("news-token-probe " * (target_input_limit * 8)).strip()


def _run_case(client, input_limit: int, max_new_tokens: int) -> Dict[str, Any]:
    prompt = _build_dummy_prompt(input_limit)
    result: Dict[str, Any] = {
        "input_limit": int(input_limit),
        "max_new_tokens": int(max_new_tokens),
        "gpu_before": _cuda_snapshot(),
    }
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    try:
        payload = client.complete_json(
            "Return exactly one JSON object with keys ok (boolean) and note (string).",
            prompt,
            label=f"oom sweep input={input_limit}",
            max_new_tokens=max_new_tokens,
            input_token_limit=input_limit,
        )
        result["status"] = "ok"
        result["response_keys"] = sorted(payload.keys()) if isinstance(payload, dict) else []
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        result["elapsed_sec"] = round(time.perf_counter() - started, 3)
        result["gpu_after"] = _cuda_snapshot()
        if torch is not None and torch.cuda.is_available():
            result["gpu_peak_allocated_mb"] = _cuda_mb(torch.cuda.max_memory_allocated())
            result["gpu_peak_reserved_mb"] = _cuda_mb(torch.cuda.max_memory_reserved())
    return result


def _print_case(case: Dict[str, Any]) -> None:
    parts = [
        f"input_limit={case.get('input_limit')}",
        f"max_new={case.get('max_new_tokens')}",
        f"status={case.get('status')}",
        f"elapsed_sec={case.get('elapsed_sec')}",
        f"peak_alloc_mb={case.get('gpu_peak_allocated_mb', 0)}",
        f"peak_reserved_mb={case.get('gpu_peak_reserved_mb', 0)}",
    ]
    if case.get("status") != "ok":
        parts.append(f"error_type={case.get('error_type')}")
    print("case " + " ".join(parts), flush=True)


def _refine_boundary(
    client,
    *,
    low_ok: int,
    high_fail: int,
    max_new_tokens: int,
    started: float,
    max_runtime_sec: float,
    rounds: int = 2,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    lo = low_ok
    hi = high_fail
    for _ in range(max(0, rounds)):
        if time.perf_counter() - started >= max_runtime_sec:
            break
        if hi - lo <= 512:
            break
        mid = (lo + hi) // 2
        case = _run_case(client, mid, max_new_tokens=max_new_tokens)
        _print_case(case)
        cases.append(case)
        if case.get("status") == "ok":
            lo = mid
        else:
            hi = mid
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Approximate failure threshold with dummy token sweeps.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument(
        "--ai-role",
        choices=("summary", "final"),
        default="summary",
        help="Which AI config section to sweep: ai_summary or ai_final.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation tokens for each sweep case.")
    parser.add_argument("--start-input", type=int, default=4000, help="First input token limit to test.")
    parser.add_argument("--step", type=int, default=2000, help="Coarse sweep increment.")
    parser.add_argument("--max-input", type=int, default=30000, help="Maximum input token limit to try.")
    parser.add_argument("--max-runtime-sec", type=int, default=900, help="Hard cap for total experiment runtime.")
    parser.add_argument("--refine-rounds", type=int, default=2, help="Extra midpoint checks after first failure.")
    parser.add_argument("--output", default="", help="Optional path to write raw JSON results.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2

    started = time.perf_counter()
    config = load_config(config_path)
    ai_config = config.ai_summary if args.ai_role == "summary" else config.ai_final
    client = create_ai_client(ai_config, DebugLogger(False))

    print("ai_role", args.ai_role, flush=True)
    print("backend", ai_config.backend, flush=True)
    print("model", ai_config.effective_model_label, flush=True)
    print("config_max_input", ai_config.max_input_tokens, "config_max_new", ai_config.max_new_tokens, flush=True)
    print("sweep_max_new", args.max_new_tokens, flush=True)
    print("gpu_snapshot", _cuda_snapshot(), flush=True)
    print(f"max_runtime_sec={args.max_runtime_sec}", flush=True)

    all_cases: List[Dict[str, Any]] = []
    first_fail: int | None = None
    last_ok: int | None = None

    current = max(512, int(args.start_input))
    step = max(256, int(args.step))
    max_input = max(current, int(args.max_input))

    while current <= max_input:
        if time.perf_counter() - started >= args.max_runtime_sec:
            print("stop_reason=max_runtime_reached", flush=True)
            break
        case = _run_case(client, current, max_new_tokens=max(16, int(args.max_new_tokens)))
        all_cases.append(case)
        _print_case(case)
        if case.get("status") == "ok":
            last_ok = current
            current += step
            continue
        first_fail = current
        break

    if first_fail is not None and last_ok is not None and last_ok < first_fail:
        refine_cases = _refine_boundary(
            client,
            low_ok=last_ok,
            high_fail=first_fail,
            max_new_tokens=max(16, int(args.max_new_tokens)),
            started=started,
            max_runtime_sec=float(args.max_runtime_sec),
            rounds=max(0, int(args.refine_rounds)),
        )
        all_cases.extend(refine_cases)

    elapsed = round(time.perf_counter() - started, 3)
    ok_limits = [int(item["input_limit"]) for item in all_cases if item.get("status") == "ok"]
    fail_limits = [int(item["input_limit"]) for item in all_cases if item.get("status") != "ok"]
    approx_safe = max(ok_limits) if ok_limits else 0
    approx_fail = min(fail_limits) if fail_limits else 0

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ai_role": args.ai_role,
        "backend": ai_config.backend,
        "model": ai_config.effective_model_label,
        "config_max_input_tokens": ai_config.max_input_tokens,
        "config_max_new_tokens": ai_config.max_new_tokens,
        "sweep_max_new_tokens": int(args.max_new_tokens),
        "elapsed_sec": elapsed,
        "approx_safe_input_tokens": approx_safe,
        "approx_first_fail_input_tokens": approx_fail,
        "cases": all_cases,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(config.output_dir) / f"oom_sweep_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("summary")
    print("elapsed_sec", elapsed)
    print("approx_safe_input_tokens", approx_safe)
    print("approx_first_fail_input_tokens", approx_fail)
    print("output", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
