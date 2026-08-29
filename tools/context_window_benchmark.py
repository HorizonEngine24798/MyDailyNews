"""Replay a real failed final-brief prompt against a local llama.cpp server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.ai.schemas import FINAL_BRIEF_JSON_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model", default="Qwen3-1.7B-Q4_K_M")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    system = str(artifact["system_prompt_original"])
    user = str(artifact["user_prompt_original"])
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": FINAL_BRIEF_JSON_SCHEMA.name, "schema": FINAL_BRIEF_JSON_SCHEMA.schema},
        },
        "json_schema": FINAL_BRIEF_JSON_SCHEMA.schema,
    }
    started = time.perf_counter()
    response = requests.post(args.base_url.rstrip("/") + "/chat/completions", json=payload, timeout=600)
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    raw = response.json()
    choice = raw["choices"][0]
    content = str(choice.get("message", {}).get("content") or "")
    try:
        parsed = json.loads(content)
        valid_json = isinstance(parsed, dict)
    except json.JSONDecodeError:
        valid_json = False
    result = {
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason", ""),
        "valid_json": valid_json,
        "prompt_tokens": raw.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": raw.get("usage", {}).get("completion_tokens"),
        "timings": raw.get("timings", {}),
        "response": content,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
