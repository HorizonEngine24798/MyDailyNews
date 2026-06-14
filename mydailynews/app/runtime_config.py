from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Iterable
from urllib.parse import urlparse

from mydailynews.app.models import AIConfig, AppConfig


PLACEHOLDER_MARKERS = (
    "PATH/TO/",
    "PATH\\TO\\",
    "<",
    ">",
    "REPLACE_ME",
    "YOUR_",
)


@dataclass(frozen=True)
class RuntimeConfigIssue:
    section: str
    field: str
    message: str
    severity: str = "error"


def find_runtime_config_issues(config: AppConfig) -> list[RuntimeConfigIssue]:
    issues: list[RuntimeConfigIssue] = []
    for section_name, ai_config in (
        ("ai_summary", config.ai_summary),
        ("ai_final", config.ai_final),
    ):
        issues.extend(_ai_runtime_issues(section_name, ai_config))
    return _dedupe_issues(issues)


def format_runtime_config_issues(issues: Iterable[RuntimeConfigIssue]) -> str:
    lines = []
    for issue in issues:
        lines.append(f"- {issue.section}.{issue.field}: {issue.message}")
    return "\n".join(lines)


def _ai_runtime_issues(section_name: str, config: AIConfig) -> list[RuntimeConfigIssue]:
    issues: list[RuntimeConfigIssue] = []
    context = _effective_context_window(config)
    if context > 0 and int(config.max_input_tokens) + int(config.max_new_tokens) > context:
        issues.append(
            RuntimeConfigIssue(
                section=section_name,
                field="max_input_tokens/max_new_tokens",
                message=(
                    "max_input_tokens + max_new_tokens exceeds the configured context window "
                    f"({config.max_input_tokens} + {config.max_new_tokens} > {context})."
                ),
            )
        )

    if not config.manage_server:
        return issues

    executable = str(config.server_executable or "").strip()
    if not executable:
        issues.append(
            RuntimeConfigIssue(
                section=section_name,
                field="server_executable",
                message="server_executable is required when manage_server=true.",
            )
        )
    elif _looks_like_placeholder(executable):
        issues.append(
            RuntimeConfigIssue(
                section=section_name,
                field="server_executable",
                message="replace the placeholder with your llama-server executable path.",
            )
        )
    elif not _executable_is_resolvable(executable):
        issues.append(
            RuntimeConfigIssue(
                section=section_name,
                field="server_executable",
                message=f"could not resolve llama-server executable: {executable}",
            )
        )

    model_path = str(config.server_model_path or "").strip()
    if not model_path:
        issues.append(
            RuntimeConfigIssue(
                section=section_name,
                field="server_model_path",
                message="server_model_path is required when manage_server=true.",
            )
        )
    elif _looks_like_placeholder(model_path):
        issues.append(
            RuntimeConfigIssue(
                section=section_name,
                field="server_model_path",
                message="replace the placeholder with a local GGUF model path or run tools/autoconfig.py.",
            )
        )
    else:
        expanded_model_path = Path(os.path.expandvars(os.path.expanduser(model_path)))
        if not expanded_model_path.exists():
            issues.append(
                RuntimeConfigIssue(
                    section=section_name,
                    field="server_model_path",
                    message=f"model file does not exist: {expanded_model_path}",
                )
            )

    return issues


def _effective_context_window(config: AIConfig) -> int:
    if int(config.context_window_tokens or 0) > 0:
        return int(config.context_window_tokens)
    args = [str(item) for item in (config.server_arguments or [])]
    for index, arg in enumerate(args):
        if arg in {"-c", "--ctx-size", "--ctx_size", "--context", "--context-size"} and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except ValueError:
                return 0
        if arg.startswith("--ctx-size=") or arg.startswith("--ctx_size="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return 0
    return 0


def _looks_like_placeholder(value: str) -> bool:
    upper = value.upper()
    return any(marker in upper for marker in PLACEHOLDER_MARKERS)


def _executable_is_resolvable(raw_path: str) -> bool:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    parsed = urlparse(expanded)
    if parsed.scheme and parsed.scheme != "file":
        return False
    path = Path(expanded)
    if path.parent != Path("."):
        return path.exists()
    if shutil.which(expanded):
        return True
    if expanded in {"llama-cli", "llama-cli.exe"}:
        sibling = "llama-server.exe" if expanded.endswith(".exe") else "llama-server"
        return shutil.which(sibling) is not None
    return False


def _dedupe_issues(issues: list[RuntimeConfigIssue]) -> list[RuntimeConfigIssue]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[RuntimeConfigIssue] = []
    for issue in issues:
        key = (issue.section, issue.field, issue.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped
