from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator
from urllib.parse import urlparse

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.local.json"
DEFAULT_WRITE = REPO_ROOT / "config.recommended.json"
DEFAULT_MODEL_CATALOG = REPO_ROOT / "profiles" / "model_catalog.json"
STORY_ENRICHMENT_BUDGET_KEYS = (
    "max_context_chars_per_article",
    "max_story_threads",
    "planner_max_questions_per_story",
    "search_results_per_query",
    "max_fetched_research_pages_per_story",
    "max_selected_article_excerpt_chars",
    "max_research_excerpt_chars",
    "cache_ttl_seconds",
)
REMOVED_FILTERING_KEYS = (
    "max_selected_per_event_cluster",
    "prefer_multi_source_clusters",
    "multi_source_cluster_bonus",
    "event_cluster_time_window_hours",
)
REMOVED_CACHE_KEYS = ("wikipedia_retention_days",)


@dataclass(frozen=True)
class HardwareInfo:
    os_name: str
    system_ram_gb: float | None
    gpu_name: str
    gpu_vendor: str
    vram_gb: float | None


@dataclass(frozen=True)
class ProbeReport:
    version_ok: bool = False
    version_output: str = ""
    server_ready: bool = False
    json_probe_ok: bool = False
    log_path: str = ""
    warnings: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe llama.cpp and write a local MyDailyNews config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Input local config JSON.")
    parser.add_argument("--write", default=str(DEFAULT_WRITE), help="Output recommended config JSON.")
    parser.add_argument("--model-catalog", default=str(DEFAULT_MODEL_CATALOG), help="Model catalog JSON.")
    parser.add_argument("--download-dir", default="", help="Directory for prompted model downloads.")
    parser.add_argument("--detect-only", action="store_true", help="Print detected hardware and recommendation only.")
    parser.add_argument("--print-launch-command", action="store_true", help="Print the llama-server command for the written config.")
    parser.add_argument("--no-hardware-detect", action="store_true", help="Skip hardware detection and use conservative CPU settings.")
    parser.add_argument("--no-server-probe", action="store_true", help="Do not launch or probe llama-server.")
    parser.add_argument("--no-json-probe", action="store_true", help="Do not run the JSON completion probe.")
    parser.add_argument("--no-download-prompt", action="store_true", help="Do not prompt to download a recommended GGUF model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    write_path = Path(args.write)
    catalog_path = Path(args.model_catalog)

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Create one first, for example: copy config.example.json config.local.json")
        return 1
    if not catalog_path.exists():
        print(f"Model catalog not found: {catalog_path}")
        return 1

    source_config = load_json(config_path)
    catalog = load_json(catalog_path)
    hardware = conservative_hardware() if args.no_hardware_detect else detect_hardware()
    tier = choose_tier(catalog, hardware)
    model = model_for_tier(catalog, tier)
    recommended = build_recommended_config(source_config, tier, model)

    print_detection(hardware, tier, model)

    download_dir = Path(args.download_dir) if args.download_dir else Path(catalog.get("download_dir", "models"))
    model_path = existing_model_path(source_config)
    if model_path:
        set_model_path(recommended, model_path, str(source_config.get("ai_summary", {}).get("server_model") or model["model_label"]))
    elif not args.no_download_prompt:
        downloaded = maybe_prompt_download(model, download_dir)
        if downloaded:
            set_model_path(recommended, str(downloaded), str(model["model_label"]))
    else:
        print("No local model path found; leaving server_model_path unchanged.")

    if args.detect_only:
        return 0

    report = ProbeReport()
    if not args.no_server_probe:
        report = probe_config(recommended, run_json_probe=not args.no_json_probe)
        print_probe_report(report)

    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(recommended, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recommended config written to {write_path}")
    if args.print_launch_command:
        print("Launch command:")
        print(" ".join(build_launch_command(recommended["ai_summary"])))
    return 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def conservative_hardware() -> HardwareInfo:
    return HardwareInfo(
        os_name=platform.platform(),
        system_ram_gb=detect_system_ram_gb(),
        gpu_name="",
        gpu_vendor="cpu",
        vram_gb=None,
    )


def detect_hardware() -> HardwareInfo:
    nvidia = detect_nvidia_gpu()
    if nvidia is not None:
        gpu_name, vram_gb = nvidia
        return HardwareInfo(platform.platform(), detect_system_ram_gb(), gpu_name, "nvidia", vram_gb)
    generic = detect_generic_gpu()
    if generic is not None:
        gpu_name, vram_gb = generic
        return HardwareInfo(platform.platform(), detect_system_ram_gb(), gpu_name, "gpu", vram_gb)
    return conservative_hardware()


def detect_system_ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024**3), 1)
    except Exception:
        pass
    if platform.system().lower() == "windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024**3), 1)
        except Exception:
            return None
    return None


def detect_nvidia_gpu() -> tuple[str, float] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        return parts[0], round(float(parts[1]) / 1024.0, 1)
    except ValueError:
        return parts[0], None


def detect_generic_gpu() -> tuple[str, float | None] | None:
    if platform.system().lower() != "windows":
        return None
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        return None
    try:
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM | ConvertTo-Json",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    name = str(payload.get("Name") or "")
    raw_ram = payload.get("AdapterRAM")
    vram_gb = None
    try:
        if raw_ram:
            vram_gb = round(float(raw_ram) / (1024**3), 1)
    except Exception:
        pass
    return (name, vram_gb) if name else None


def choose_tier(catalog: dict[str, Any], hardware: HardwareInfo) -> dict[str, Any]:
    tiers = catalog.get("tiers", [])
    vram = hardware.vram_gb
    if vram is None:
        return next(tier for tier in tiers if tier["id"] == "cpu_small")
    for tier in tiers:
        min_vram = tier.get("min_vram_gb")
        max_vram = tier.get("max_vram_gb")
        if min_vram is not None and vram < float(min_vram):
            continue
        if max_vram is not None and vram > float(max_vram):
            continue
        return tier
    return tiers[-1]


def model_for_tier(catalog: dict[str, Any], tier: dict[str, Any]) -> dict[str, Any]:
    model_id = tier["recommended_model_id"]
    for model in catalog.get("models", []):
        if model.get("id") == model_id:
            return model
    raise ValueError(f"Catalog tier references unknown model id: {model_id}")


def build_recommended_config(config: dict[str, Any], tier: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    settings = tier["settings"]
    for section in ("ai_summary", "ai_final"):
        ai = updated.setdefault(section, {})
        ai["backend"] = "llama_cpp_server"
        ai["server_model"] = model["model_label"]
        ai["context_window_tokens"] = settings["context_window_tokens"]
        ai["max_input_tokens"] = settings["max_input_tokens"]
        ai["max_new_tokens"] = settings["max_new_tokens"]
        ai["request_timeout_seconds"] = settings["request_timeout_seconds"]
        ai["server_arguments"] = settings["server_arguments"]
        ai["manage_server"] = bool(ai.get("manage_server", True))
        ai["server_auto_stop"] = bool(ai.get("server_auto_stop", True))

    _apply_filtering(updated.setdefault("general_filtering", {}), settings, general=True)
    _apply_filtering(updated.setdefault("filtering", {}), settings, general=False)
    _apply_narrative_briefing(updated.setdefault("narrative_briefing", {}))
    _apply_story_enrichment_budget(updated.setdefault("enrichment", {}), settings)
    _apply_analysis(updated.setdefault("analysis", {}), settings)
    _apply_runtime(updated.setdefault("runtime", {}))
    _apply_cache(updated.setdefault("cache", {}))
    return updated


def _apply_filtering(section: dict[str, Any], settings: dict[str, Any], *, general: bool) -> None:
    for key in REMOVED_FILTERING_KEYS:
        section.pop(key, None)
    section["max_candidates_for_ai"] = settings["max_candidates_for_ai"]
    section["max_headlines_per_ai_batch"] = settings["max_headlines_per_ai_batch"]
    section["headline_max_input_tokens"] = settings["headline_max_input_tokens"]
    section["headline_max_new_tokens"] = settings["headline_max_new_tokens"]
    section["headline_single_replay_max_new_tokens"] = settings["headline_single_replay_max_new_tokens"]
    section["max_selected_articles"] = (
        settings["general_max_selected_articles"] if general else settings["detailed_max_selected_articles"]
    )
    section["article_text_max_chars"] = settings["article_text_max_chars"]


def _apply_narrative_briefing(section: dict[str, Any]) -> None:
    section["enabled"] = bool(section.get("enabled", True))
    section.setdefault("max_input_tokens", None)
    section.setdefault("max_new_tokens", None)
    section.setdefault("target_words", 1800)
    section.setdefault(
        "editorial_style",
        "Write like a sharp human news editor, not a consultant memo. Use clear narrative paragraphs, "
        "concrete verbs, and varied sentence rhythm. Avoid repeated Status/Impact/Operational Implication "
        "labels. Do not address the reader as an operator. Use bullets sparingly, only for genuinely scannable "
        "watch items. Prefer 'what changed, why it matters, what remains uncertain' woven into prose.",
    )


def _apply_runtime(section: dict[str, Any]) -> None:
    section.pop("max_enrichment_workers", None)
    section.setdefault("max_http_workers", 1)
    section.setdefault("max_article_workers", 1)
    section.setdefault("use_shared_snapshot", True)


def _apply_cache(section: dict[str, Any]) -> None:
    for key in REMOVED_CACHE_KEYS:
        section.pop(key, None)


def _apply_story_enrichment_budget(section: dict[str, Any], settings: dict[str, Any]) -> None:
    budget = settings.get("story_enrichment_budget")
    if not isinstance(budget, dict):
        raise ValueError("Catalog tier settings must include story_enrichment_budget.")

    enabled = bool(section.get("enabled", False))
    mode = str(section.get("mode") or "story_llm").strip().lower()
    target_mode = "disabled" if mode == "disabled" else "story_llm"
    section.clear()
    section["enabled"] = bool(enabled and target_mode != "disabled")
    section["mode"] = target_mode
    for key in STORY_ENRICHMENT_BUDGET_KEYS:
        if key not in budget:
            raise ValueError(f"Catalog story_enrichment_budget is missing required key: {key}")
        section[key] = budget[key]


def _apply_analysis(analysis: dict[str, Any], settings: dict[str, Any]) -> None:
    evidence = analysis.setdefault("evidence_distillation", {})
    evidence["max_input_tokens"] = settings["evidence_max_input_tokens"]
    evidence["max_new_tokens"] = settings["evidence_max_new_tokens"]
    evidence["max_articles"] = settings["evidence_max_articles"]
    evidence["max_articles_per_batch"] = settings["evidence_max_articles_per_batch"]
    evidence["max_article_chars"] = min(int(evidence.get("max_article_chars", 1200)), settings["article_text_max_chars"])

    delta = analysis.setdefault("delta_extraction", {})
    delta["max_input_tokens"] = settings["delta_max_input_tokens"]
    delta["max_new_tokens"] = settings["delta_max_new_tokens"]
    delta["max_articles"] = settings["delta_max_articles"]
    delta["max_articles_per_batch"] = settings["delta_max_articles_per_batch"]

    rollout = analysis.setdefault("rollout", {})
    rollout["enabled"] = bool(rollout.get("enabled", True))
    for mode in ("general", "detailed"):
        mode_config = rollout.setdefault(mode, {})
        mode_config["evidence_max_input_tokens"] = settings["evidence_max_input_tokens"]
        mode_config["evidence_max_new_tokens"] = settings["evidence_max_new_tokens"]
        mode_config["evidence_max_articles"] = settings["evidence_max_articles"]
        mode_config["evidence_max_articles_per_batch"] = settings["evidence_max_articles_per_batch"]
        mode_config["delta_max_input_tokens"] = settings["delta_max_input_tokens"]
        mode_config["delta_max_new_tokens"] = settings["delta_max_new_tokens"]
        mode_config["delta_max_articles"] = settings["delta_max_articles"]
        mode_config["delta_max_articles_per_batch"] = settings["delta_max_articles_per_batch"]


def existing_model_path(config: dict[str, Any]) -> str:
    for section in ("ai_summary", "ai_final"):
        path = str(config.get(section, {}).get("server_model_path") or "").strip()
        if path and not looks_like_placeholder(path) and Path(os.path.expandvars(os.path.expanduser(path))).exists():
            return path
    return ""


def set_model_path(config: dict[str, Any], model_path: str, model_label: str) -> None:
    for section in ("ai_summary", "ai_final"):
        ai = config.setdefault(section, {})
        ai["server_model_path"] = model_path
        ai["server_model"] = model_label


def maybe_prompt_download(model: dict[str, Any], download_dir: Path) -> Path | None:
    print(f"Recommended model: {model['name']} ({model['repo_id']})")
    print(f"Download URL: {model['url']}")
    if not sys.stdin.isatty():
        print("Non-interactive terminal detected; skipping download prompt.")
        return None
    answer = input(f"Download {model['filename']} into {download_dir}? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        return None
    return download_model(model, download_dir)


def download_model(model: dict[str, Any], download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / str(model["filename"])
    if target.exists():
        print(f"Model already exists: {target}")
        return target
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(str(model["url"]), stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    try:
        os.replace(partial, target)
    except PermissionError:
        shutil.copyfile(partial, target)
        try:
            partial.unlink(missing_ok=True)
        except PermissionError:
            pass
    return target


def probe_config(config: dict[str, Any], *, run_json_probe: bool) -> ProbeReport:
    warnings: list[str] = []
    ai_config = config.get("ai_summary", {})
    version_ok, version_output = check_llama_version(str(ai_config.get("server_executable", "")))
    if not version_ok:
        warnings.append(version_output or "Could not run llama-server --version.")

    with managed_probe_server(ai_config) as server:
        ready = endpoint_ready(str(ai_config.get("base_url", "http://127.0.0.1:8080/v1")))
        json_ok = False
        if ready and run_json_probe:
            json_ok = run_json_completion_probe(ai_config)
            if not json_ok:
                warnings.append("JSON completion probe failed.")
        if not ready:
            warnings.append("Server readiness probe failed.")
        return ProbeReport(
            version_ok=version_ok,
            version_output=version_output,
            server_ready=ready,
            json_probe_ok=json_ok,
            log_path=server.get("log_path", ""),
            warnings=tuple(warnings),
        )


def check_llama_version(executable: str) -> tuple[bool, str]:
    resolved = resolve_executable(executable)
    if not resolved:
        return False, f"Could not resolve llama-server executable: {executable}"
    try:
        result = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=15, check=False)
    except Exception as exc:
        return False, f"Could not run llama-server --version: {exc}"
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output


@contextmanager
def managed_probe_server(ai_config: dict[str, Any]) -> Iterator[dict[str, str]]:
    base_url = str(ai_config.get("base_url", "http://127.0.0.1:8080/v1"))
    if endpoint_ready(base_url):
        yield {"attached": "true", "log_path": ""}
        return
    if not ai_config.get("manage_server", False):
        yield {"attached": "false", "log_path": ""}
        return
    command = build_launch_command(ai_config)
    if not command or not Path(str(ai_config.get("server_model_path", ""))).exists():
        yield {"attached": "false", "log_path": ""}
        return

    log_dir = Path(str(ai_config.get("server_log_dir", "output/diagnostics/llama_server"))) / "autoconfig"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_probe.log"
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT)
        try:
            wait_for_endpoint_or_exit(
                base_url,
                process,
                timeout=max(10, int(ai_config.get("server_startup_timeout_seconds", 180))),
            )
            yield {"attached": "false", "log_path": str(log_path)}
        finally:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


def build_launch_command(ai_config: dict[str, Any]) -> list[str]:
    executable = resolve_executable(str(ai_config.get("server_executable", "")))
    model_path = str(ai_config.get("server_model_path", "")).strip()
    if not executable or not model_path or looks_like_placeholder(model_path):
        return []
    host, port = host_port_from_base_url(str(ai_config.get("base_url", "http://127.0.0.1:8080/v1")))
    return [
        executable,
        "-m",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        *[str(item) for item in ai_config.get("server_arguments", [])],
    ]


def resolve_executable(raw_path: str) -> str:
    value = str(raw_path or "").strip()
    if not value or looks_like_placeholder(value):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.parent != Path("."):
        if path.name.lower() in {"llama-cli", "llama-cli.exe"}:
            sibling = path.with_name("llama-server.exe" if path.name.endswith(".exe") else "llama-server")
            if sibling.exists():
                return str(sibling)
        return str(path) if path.exists() else ""
    found = shutil.which(expanded)
    if found:
        return found
    if expanded in {"llama-cli", "llama-cli.exe"}:
        return shutil.which("llama-server.exe" if expanded.endswith(".exe") else "llama-server") or ""
    return ""


def endpoint_ready(base_url: str) -> bool:
    for url in probe_urls(base_url):
        try:
            response = requests.get(url, timeout=(0.5, 1.2))
        except requests.RequestException:
            continue
        if response.status_code in {200, 401, 403}:
            return True
    return False


def wait_for_endpoint(base_url: str, *, timeout: int) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if endpoint_ready(base_url):
            return True
        time.sleep(0.5)
    return False


def wait_for_endpoint_or_exit(base_url: str, process: subprocess.Popen, *, timeout: int) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if endpoint_ready(base_url):
            return True
        if process.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def run_json_completion_probe(ai_config: dict[str, Any]) -> bool:
    base_url = str(ai_config.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/")
    payload = {
        "model": ai_config.get("server_model") or ai_config.get("model_id") or "local-model",
        "messages": [
            {"role": "system", "content": "Return compact JSON only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
        "max_tokens": 64,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=(2, 45))
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return bool(parsed.get("ok"))
    except Exception:
        return False


def probe_urls(base_url: str) -> list[str]:
    base = str(base_url or "http://127.0.0.1:8080/v1").rstrip("/")
    parsed = urlparse(base)
    root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else base
    return [f"{base}/models", f"{base}/health", f"{root}/health", f"{root}/v1/health"]


def host_port_from_base_url(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url or "http://127.0.0.1:8080/v1")
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        return host, int(parsed.port)
    if parsed.scheme == "https":
        return host, 443
    return host, 80


def looks_like_placeholder(value: str) -> bool:
    upper = str(value or "").upper()
    return "PATH/TO/" in upper or "PATH\\TO\\" in upper or "REPLACE_ME" in upper or "YOUR_" in upper


def print_detection(hardware: HardwareInfo, tier: dict[str, Any], model: dict[str, Any]) -> None:
    ram = f"{hardware.system_ram_gb} GB" if hardware.system_ram_gb is not None else "unknown"
    vram = f"{hardware.vram_gb} GB" if hardware.vram_gb is not None else "unknown"
    gpu = hardware.gpu_name or "none detected"
    print(f"Detected OS: {hardware.os_name}")
    print(f"Detected system RAM: {ram}")
    print(f"Detected GPU: {gpu} ({hardware.gpu_vendor}, VRAM: {vram})")
    print(f"Selected tier: {tier['label']}")
    print(f"Recommended model class: {model['model_class']}")


def print_probe_report(report: ProbeReport) -> None:
    print(f"llama-server --version: {'ok' if report.version_ok else 'not confirmed'}")
    print(f"Server readiness probe: {'ok' if report.server_ready else 'failed'}")
    print(f"JSON completion probe: {'ok' if report.json_probe_ok else 'not confirmed'}")
    if report.log_path:
        print(f"Probe log: {report.log_path}")
    for warning in report.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
