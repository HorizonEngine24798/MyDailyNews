from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from typing import Any, Dict, List

from mydailynews.app.config import load_config


RUN_KINDS = {"series", "briefs", "enrichment", "narrative_brief", "tts", "memory"}
BRIEF_CHOICES = {"general", "detailed", "both"}
MEMORY_RUN_ACTIONS = {"inspect", "prune", "export"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TAIL_LIMIT = 20000
RUN_ID_LENGTH = 12
OUTPUT_DISCOVERY_GRACE_SECONDS = 2.0
OUTPUT_DISCOVERY_LIMIT = 40


@dataclass
class RunJob:
    id: str
    kind: str
    label: str
    command: List[str]
    status: str
    started_at: str
    finished_at: str = ""
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    output_paths: List[str] = field(default_factory=list)
    error: str = ""


class GuiRunManager:
    def __init__(self, *, root: Path | str, config_path: Path | str) -> None:
        self.root = Path(root).resolve()
        self.config_path = Path(config_path).resolve()
        self._jobs: Dict[str, RunJob] = {}
        self._processes: Dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def list_runs(self) -> Dict[str, Any]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)
            return {"runs": [self._public_job(job) for job in jobs]}

    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(run_id or ""))
            if job is None:
                raise FileNotFoundError("Run not found.")
            return {"run": self._public_job(job)}

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        command, kind, label = self._build_command(payload)
        job = RunJob(
            id=uuid.uuid4().hex[:RUN_ID_LENGTH],
            kind=kind,
            label=label,
            command=command,
            status="running",
            started_at=_now_iso(),
        )
        with self._lock:
            self._jobs[job.id] = job
        try:
            process = subprocess.Popen(
                command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.finished_at = _now_iso()
                job.returncode = -1
                job.error = f"{type(exc).__name__}: {exc}"
            return {"run": self._public_job(job)}

        with self._lock:
            self._processes[job.id] = process
        thread = threading.Thread(target=self._watch_process, args=(job.id, process), daemon=True)
        thread.start()
        return {"run": self._public_job(job)}

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(run_id or ""))
            process = self._processes.get(str(run_id or ""))
            if job is None:
                raise FileNotFoundError("Run not found.")
            if job.status != "running" or process is None:
                return {"run": self._public_job(job)}
            job.status = "canceling"
        process.terminate()
        return self.get_run(run_id)

    def _watch_process(self, run_id: str, process: subprocess.Popen[str]) -> None:
        stdout = ""
        stderr = ""
        error = ""
        try:
            stdout, stderr = process.communicate()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                process.kill()
            except OSError:
                pass
        finished_at = _now_iso()
        returncode = process.returncode
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return
            status = "canceled" if job.status == "canceling" else ("completed" if returncode == 0 and not error else "failed")
            job.status = status
            job.finished_at = finished_at
            job.returncode = returncode
            job.stdout_tail = _tail(stdout)
            job.stderr_tail = _tail(stderr)
            job.error = error
            job.output_paths = self._discover_output_paths(job)
            self._processes.pop(run_id, None)

    def _build_command(self, payload: Dict[str, Any]) -> tuple[List[str], str, str]:
        if not isinstance(payload, dict):
            raise ValueError("Run payload must be an object.")
        kind = _clean(payload.get("kind")).lower() or "series"
        if kind not in RUN_KINDS:
            raise ValueError(f"Unsupported run kind: {kind}")
        command = [
            sys.executable,
            "-B",
            str(self.root / "main.py"),
            "--config",
            str(self.config_path),
        ]
        label = kind
        if kind == "series":
            command.extend(["--module", "series"])
            label = "Run default series"
        elif kind == "briefs":
            brief = _clean(payload.get("brief")).lower() or "both"
            if brief not in BRIEF_CHOICES:
                raise ValueError(f"Unsupported brief mode: {brief}")
            command.extend(["--module", "briefs", "--brief", brief])
            label = f"Run briefs ({brief})"
        elif kind in {"enrichment", "narrative_brief", "tts"}:
            date = _clean(payload.get("date"))
            if date and DATE_RE.match(date) is None:
                raise ValueError("Date must use YYYY-MM-DD.")
            command.extend(["--module", kind])
            if date:
                command.extend(["--date", date])
            markdown_path = _clean(payload.get("markdown_path"))
            if markdown_path and kind == "tts":
                command.extend(["--markdown-path", markdown_path])
            label = f"Run {kind.replace('_', ' ')}"
        elif kind == "memory":
            memory_action = _clean(payload.get("memory_action")).lower() or "inspect"
            if memory_action not in MEMORY_RUN_ACTIONS:
                raise ValueError(f"Unsupported memory action: {memory_action}")
            command.extend(["--memory", memory_action])
            label = f"Memory {memory_action}"
        if payload.get("debug") is True and kind != "memory":
            command.append("--debug")
        return command, kind, label

    def _discover_output_paths(self, job: RunJob) -> List[str]:
        try:
            config = load_config(self.config_path)
            output_dir = _resolve_inside_root(self.root, config.output_dir)
        except Exception:
            output_dir = self.root / "output"
        started = _timestamp(job.started_at)
        paths: List[str] = []
        for directory in [output_dir, output_dir / "handoff", output_dir / "diagnostics"]:
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".md", ".json", ".txt", ".log", ".wav"}:
                    continue
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    continue
                if modified + OUTPUT_DISCOVERY_GRACE_SECONDS < started:
                    continue
                paths.append(str(path))
                if len(paths) >= OUTPUT_DISCOVERY_LIMIT:
                    return sorted(paths)
        return sorted(paths)

    def _public_job(self, job: RunJob) -> Dict[str, Any]:
        payload = asdict(job)
        payload["command_display"] = _command_display(job.command)
        return payload


def _resolve_inside_root(root: Path, raw_path: Path | str) -> Path:
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    resolved.relative_to(root)
    return resolved


def _tail(text: str) -> str:
    value = str(text or "")
    return value[-TAIL_LIMIT:]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _command_display(command: List[str]) -> str:
    return " ".join(_quote_part(part) for part in command)


def _quote_part(part: str) -> str:
    text = str(part)
    if not text:
        return '""'
    if any(char.isspace() for char in text):
        return '"' + text.replace('"', '\\"') + '"'
    return text
