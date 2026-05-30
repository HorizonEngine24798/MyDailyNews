from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except Exception:  # pragma: no cover - optional for non-transformers envs
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.ai.base import AIClient
from mydailynews.config import load_config
import mydailynews.orchestrator as orchestrator_module
from mydailynews.orchestrator import NewsOrchestrator

WORD_RE = re.compile(r"\S+")


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _cuda_stats() -> Dict[str, float]:
    if torch is None or not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
    }


def _stage_from_label(label: str) -> str:
    value = (label or "").lower()
    if value.startswith("headline scoring batch"):
        return "headline_scoring"
    if value.startswith("final brief generation"):
        return "final_brief_generation"
    return "other"


class _ProbedClient:
    def __init__(self, inner: AIClient, probe: "LLMCallProbe") -> None:
        self._inner = inner
        self._probe = probe
        self.config = inner.config

    @property
    def max_input_tokens(self) -> int:
        return self._inner.max_input_tokens

    @property
    def max_new_tokens(self) -> int:
        return self._inner.max_new_tokens

    def estimate_tokens(self, text: str) -> int:
        return self._inner.estimate_tokens(text)

    def unload(self) -> None:
        self._inner.unload()

    def complete_json(
        self,
        system: str,
        user: str,
        label: str = "ai.complete_json",
        *,
        max_new_tokens: int | None = None,
        input_token_limit: int | None = None,
        json_schema: Any = None,
    ) -> Dict[str, Any]:
        self._probe._call_counter += 1
        call_id = self._probe._call_counter

        prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
        planned_max_new = max(64, int(max_new_tokens or self.max_new_tokens))
        planned_input_limit = max(512, int(input_token_limit or self.max_input_tokens))

        record: Dict[str, Any] = {
            "call_id": call_id,
            "ts": datetime.now().astimezone().isoformat(),
            "stage": _stage_from_label(label),
            "label": label,
            "backend": self.config.backend,
            "model": self.config.effective_model_label,
            "system_chars": len(system),
            "user_chars": len(user),
            "prompt_chars": len(prompt),
            "system_words": _count_words(system),
            "user_words": _count_words(user),
            "prompt_words": _count_words(prompt),
            "system_tokens": self._safe_token_count(system),
            "user_tokens": self._safe_token_count(user),
            "prompt_tokens": self._safe_token_count(prompt),
            "planned_max_new_tokens": planned_max_new,
            "planned_input_token_limit": planned_input_limit,
            "planned_total_tokens": planned_max_new + planned_input_limit,
            "uses_json_schema": bool(json_schema),
            "enable_thinking": bool(getattr(self.config, "enable_thinking", False)),
            "gpu_before": _cuda_stats(),
        }
        self._emit_probe_line(
            "call_start",
            call_id=call_id,
            stage=record["stage"],
            label=label,
            prompt_tokens=record["prompt_tokens"],
            planned_input_tokens=planned_input_limit,
            planned_new_tokens=planned_max_new,
        )

        started = time.perf_counter()
        try:
            result = self._inner.complete_json(
                system,
                user,
                label=label,
                max_new_tokens=max_new_tokens,
                input_token_limit=input_token_limit,
                json_schema=json_schema,
            )
            record["status"] = "ok"
            return result
        except Exception as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            raise
        finally:
            record["elapsed_sec"] = round(time.perf_counter() - started, 3)
            record["gpu_after"] = _cuda_stats()
            self._probe.records.append(record)
            self._emit_probe_line(
                "call_done",
                call_id=call_id,
                stage=record["stage"],
                label=label,
                status=record.get("status", "unknown"),
                elapsed_sec=record["elapsed_sec"],
                error_type=record.get("error_type", ""),
            )

    def _safe_token_count(self, text: str) -> int:
        try:
            return int(self._inner.estimate_tokens(text))
        except Exception:
            return 0

    @staticmethod
    def _emit_probe_line(event: str, **fields: Any) -> None:
        parts = [f"[llm_probe] {event}"]
        for key, value in fields.items():
            text = str(value)
            if any(char.isspace() for char in text):
                text = repr(text)
            parts.append(f"{key}={text}")
        print(" ".join(parts), flush=True)


class LLMCallProbe:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._call_counter = 0
        self._orig_factory = orchestrator_module.create_ai_client

    def install(self) -> None:
        probe = self

        def wrapped_factory(config, debug=None):
            client = probe._orig_factory(config, debug)
            return _ProbedClient(client, probe)

        orchestrator_module.create_ai_client = wrapped_factory

    def uninstall(self) -> None:
        orchestrator_module.create_ai_client = self._orig_factory


def _stage_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "errors": 0,
            "max_prompt_tokens": 0,
            "max_prompt_words": 0,
            "max_planned_total_tokens": 0,
            "max_planned_input_tokens": 0,
            "max_planned_new_tokens": 0,
            "max_elapsed_sec": 0.0,
        }
    )
    for record in records:
        stage = record.get("stage", "other")
        item = summary[stage]
        item["calls"] += 1
        if record.get("status") != "ok":
            item["errors"] += 1
        item["max_prompt_tokens"] = max(item["max_prompt_tokens"], int(record.get("prompt_tokens", 0)))
        item["max_prompt_words"] = max(item["max_prompt_words"], int(record.get("prompt_words", 0)))
        item["max_planned_total_tokens"] = max(item["max_planned_total_tokens"], int(record.get("planned_total_tokens", 0)))
        item["max_planned_input_tokens"] = max(item["max_planned_input_tokens"], int(record.get("planned_input_token_limit", 0)))
        item["max_planned_new_tokens"] = max(item["max_planned_new_tokens"], int(record.get("planned_max_new_tokens", 0)))
        item["max_elapsed_sec"] = max(item["max_elapsed_sec"], float(record.get("elapsed_sec", 0.0)))
    return dict(summary)


def _print_report(records: List[Dict[str, Any]], run_elapsed_sec: float) -> None:
    print("")
    print("=== LLM Probe Summary ===")
    print(f"total_calls={len(records)} run_elapsed_sec={round(run_elapsed_sec, 2)}")
    summary = _stage_summary(records)
    if not summary:
        print("no llm calls were recorded")
        return

    print("")
    print("by_stage:")
    for stage, item in summary.items():
        print(
            "  "
            + " ".join(
                [
                    f"stage={stage}",
                    f"calls={item['calls']}",
                    f"errors={item['errors']}",
                    f"max_prompt_tokens={item['max_prompt_tokens']}",
                    f"max_prompt_words={item['max_prompt_words']}",
                    f"max_planned_input_tokens={item['max_planned_input_tokens']}",
                    f"max_planned_new_tokens={item['max_planned_new_tokens']}",
                    f"max_planned_total_tokens={item['max_planned_total_tokens']}",
                    f"max_elapsed_sec={round(item['max_elapsed_sec'], 2)}",
                ]
            )
        )

    failed = [item for item in records if item.get("status") != "ok"]
    if failed:
        print("")
        print("failed_calls:")
        for item in failed:
            print(
                "  "
                + " ".join(
                    [
                        f"call_id={item.get('call_id')}",
                        f"stage={item.get('stage')}",
                        f"label={item.get('label')!r}",
                        f"backend={item.get('backend')}",
                        f"model={item.get('model')}",
                        f"prompt_tokens={item.get('prompt_tokens')}",
                        f"planned_input={item.get('planned_input_token_limit')}",
                        f"planned_new={item.get('planned_max_new_tokens')}",
                        f"error_type={item.get('error_type')}",
                    ]
                )
            )


def _build_output_path(output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"llm_probe_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe every LLM call with token/word/memory telemetry.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--no-enrichment", action="store_true", help="Disable enrichment during probe run.")
    parser.add_argument("--output", default="", help="Optional path to write raw probe JSON.")
    parser.add_argument("--debug", action="store_true", help="Enable normal pipeline debug logs in addition to probe logs.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2

    config = load_config(config_path)
    if args.no_enrichment:
        config.enrichment.enabled = False

    output_path = Path(args.output) if args.output else _build_output_path(Path(config.output_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probe = LLMCallProbe()
    probe.install()

    pipeline_error: Exception | None = None
    started = time.perf_counter()
    try:
        result = NewsOrchestrator(config, debug=args.debug).run()
        print(
            "pipeline_outputs="
            + ", ".join(
                [
                    f"{output.name}:{output.selected_count}/{output.candidate_count}"
                    for output in result.outputs
                ]
            )
        )
        if result.warnings:
            print(f"pipeline_warnings={len(result.warnings)}")
    except Exception as exc:
        pipeline_error = exc
    finally:
        probe.uninstall()

    run_elapsed = time.perf_counter() - started
    payload = {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "run_elapsed_sec": round(run_elapsed, 3),
            "config": str(config_path),
            "summary_backend": config.ai_summary.backend,
            "summary_model": config.ai_summary.effective_model_label,
            "summary_max_input_tokens": config.ai_summary.max_input_tokens,
            "summary_max_new_tokens": config.ai_summary.max_new_tokens,
            "final_backend": config.ai_final.backend,
            "final_model": config.ai_final.effective_model_label,
            "final_max_input_tokens": config.ai_final.max_input_tokens,
            "final_max_new_tokens": config.ai_final.max_new_tokens,
            "enrichment_enabled": config.enrichment.enabled,
            "error": str(pipeline_error) if pipeline_error else "",
            "error_type": type(pipeline_error).__name__ if pipeline_error else "",
        },
        "records": probe.records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"probe_output={output_path}")
    _print_report(probe.records, run_elapsed)

    if pipeline_error:
        print("")
        print("pipeline_exception:")
        print("".join(traceback.format_exception(type(pipeline_error), pipeline_error, pipeline_error.__traceback__)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
