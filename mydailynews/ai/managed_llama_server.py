from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import threading
import time
from typing import BinaryIO, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.app.models import AIConfig


@dataclass(frozen=True)
class _ServerKey:
    executable: str
    model_path: str
    host: str
    port: int
    args: Tuple[str, ...]


@dataclass
class _ManagedServerState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: Optional[subprocess.Popen] = None
    log_handle: Optional[BinaryIO] = None
    log_path: str = ""
    spawned_by_us: bool = False
    ref_count: int = 0


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: Dict[_ServerKey, _ManagedServerState] = {}


class ManagedLlamaServerLease:
    """Reference-counted lifecycle wrapper for a local llama-server process."""

    def __init__(self, config: AIConfig, base_url: str, debug: DebugLogger) -> None:
        self.config = config
        self.base_url = (base_url or "").rstrip("/")
        self.debug = debug
        self.enabled = bool(config.backend == "llama_cpp_server" and config.manage_server)
        self._released = False
        self._key: _ServerKey | None = None
        self._state: _ManagedServerState | None = None
        self._host, self._port = self._host_port_from_base_url(self.base_url)
        self._poll_interval_seconds = 0.4
        if not self.enabled:
            return

        model_path = self._resolved_model_path(config.server_model_path)
        args = tuple(str(item).strip() for item in (config.server_arguments or []) if str(item).strip())
        key = _ServerKey(
            executable=self._resolved_executable(config.server_executable),
            model_path=model_path,
            host=self._host,
            port=self._port,
            args=args,
        )

        if not key.executable:
            raise RuntimeError("Managed llama.cpp server requires ai.server_executable to be set.")
        if not key.model_path:
            raise RuntimeError("Managed llama.cpp server requires ai.server_model_path (GGUF path) to be set.")

        with _REGISTRY_LOCK:
            state = _REGISTRY.get(key)
            if state is None:
                state = _ManagedServerState()
                _REGISTRY[key] = state
            state.ref_count += 1
        self._key = key
        self._state = state
        self.debug.log(
            "ai.server",
            "lease_acquired",
            endpoint=self.base_url,
            host=self._host,
            port=self._port,
            refs=state.ref_count,
        )

    def ensure_running(self) -> None:
        if not self.enabled:
            return
        if self._state is None or self._key is None:
            raise RuntimeError("Managed llama.cpp server lease is not initialized.")

        with self._state.lock:
            if self._endpoint_is_ready():
                return

            process = self._state.process
            if process is not None and process.poll() is not None:
                self.debug.log(
                    "ai.server",
                    "process_exited",
                    endpoint=self.base_url,
                    exit_code=process.returncode,
                    log_path=self._state.log_path,
                )
                self._state.process = None
                self._state.spawned_by_us = False
                self._close_log_locked()

            if self._state.process is None:
                # If a compatible endpoint is already live, attach without owning lifecycle.
                if self._endpoint_is_ready():
                    self.debug.log("ai.server", "attached_existing", endpoint=self.base_url)
                    return
                self._start_process_locked()

            self._wait_until_ready_locked()

    def release(self) -> None:
        if not self.enabled or self._released:
            return
        self._released = True
        if self._state is None or self._key is None:
            return

        should_stop = False
        with _REGISTRY_LOCK:
            state = _REGISTRY.get(self._key)
            if state is None:
                return
            state.ref_count = max(0, state.ref_count - 1)
            refs = state.ref_count
            should_stop = refs == 0 and bool(self.config.server_auto_stop)
            if refs == 0:
                _REGISTRY.pop(self._key, None)
        self.debug.log(
            "ai.server",
            "lease_released",
            endpoint=self.base_url,
            refs=refs,
            auto_stop=bool(self.config.server_auto_stop),
        )
        if should_stop:
            with self._state.lock:
                self._stop_process_locked(reason="last_release")

    def _start_process_locked(self) -> None:
        if self._state is None or self._key is None:
            raise RuntimeError("Managed llama.cpp server lease is not initialized.")
        model_path = Path(self._key.model_path)
        if not model_path.exists():
            raise RuntimeError(f"Managed llama.cpp server model path does not exist: {model_path}")

        cmd = [
            self._key.executable,
            "-m",
            self._key.model_path,
            "--host",
            self._key.host,
            "--port",
            str(self._key.port),
            *self._key.args,
        ]

        executable_parent = Path(self._key.executable).parent
        cwd = str(executable_parent) if executable_parent != Path(".") and executable_parent.exists() else None
        creationflags = self._popen_creationflags()
        log_handle = self._open_log_locked(cmd=cmd, cwd=cwd, creationflags=creationflags)

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            self._close_log_locked()
            raise RuntimeError(
                f"Managed llama.cpp server executable not found: {self._key.executable}"
            ) from exc
        except Exception as exc:
            self._close_log_locked()
            raise RuntimeError(f"Managed llama.cpp server failed to start: {type(exc).__name__}: {exc}") from exc

        self._state.process = process
        self._state.spawned_by_us = True
        self.debug.log(
            "ai.server",
            "spawned",
            endpoint=self.base_url,
            pid=process.pid,
            command=" ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""),
            log_path=self._state.log_path,
        )

    def _wait_until_ready_locked(self) -> None:
        if self._state is None:
            raise RuntimeError("Managed llama.cpp server lease is not initialized.")
        timeout = max(10, int(self.config.server_startup_timeout_seconds))
        deadline = time.perf_counter() + float(timeout)
        last_exit_code: int | None = None
        last_log_path = self._state.log_path

        while time.perf_counter() < deadline:
            if self._endpoint_is_ready():
                self.debug.log("ai.server", "ready", endpoint=self.base_url)
                return

            process = self._state.process
            if process is not None and process.poll() is not None:
                last_exit_code = process.returncode
                last_log_path = self._state.log_path
                self._state.process = None
                self._state.spawned_by_us = False
                self._close_log_locked()
                break
            time.sleep(self._poll_interval_seconds)

        self._stop_process_locked(reason="startup_failed")
        if last_exit_code is not None:
            exit_hex = f"0x{last_exit_code & 0xFFFFFFFF:08x}"
            raise RuntimeError(
                "Managed llama.cpp server exited during startup "
                f"(exit code {last_exit_code} / {exit_hex})."
                + (f" Server log: {last_log_path}" if last_log_path else "")
            )
        raise RuntimeError(
            f"Managed llama.cpp server did not become ready within {timeout}s at {self.base_url}."
            + (f" Server log: {self._state.log_path}" if self._state.log_path else "")
        )

    def _stop_process_locked(self, reason: str) -> None:
        if self._state is None:
            return
        process = self._state.process
        if process is None:
            return
        if not self._state.spawned_by_us:
            self._state.process = None
            self._state.spawned_by_us = False
            return

        timeout = max(1, int(self.config.server_shutdown_timeout_seconds))
        self.debug.log(
            "ai.server",
            "stopping",
            endpoint=self.base_url,
            pid=process.pid,
            reason=reason,
            log_path=self._state.log_path,
        )
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=max(1, timeout // 2))
            except Exception:
                pass
        finally:
            self.debug.log(
                "ai.server",
                "stopped",
                endpoint=self.base_url,
                pid=process.pid,
                return_code=process.returncode,
                log_path=self._state.log_path,
            )
            self._state.process = None
            self._state.spawned_by_us = False
            self._close_log_locked()

    def _open_log_locked(self, *, cmd: list[str], cwd: str | None, creationflags: int) -> BinaryIO:
        if self._state is None or self._key is None:
            raise RuntimeError("Managed llama.cpp server lease is not initialized.")
        self._close_log_locked()
        raw_dir = str(self.config.server_log_dir or "output/diagnostics/llama_server").strip()
        log_dir = Path(os.path.expandvars(os.path.expanduser(raw_dir)))
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}"
        model_label = self._safe_filename(Path(self._key.model_path).stem)
        path = log_dir / f"{stamp}_llama_server_{self._key.port}_{model_label}.log"
        handle = path.open("ab")
        handle.write(
            (
                f"\n--- llama-server launch {stamp} ---\n"
                f"endpoint={self.base_url}\n"
                f"executable={self._key.executable}\n"
                f"model={self._key.model_path}\n"
                f"args={' '.join(self._key.args)}\n\n"
                f"cwd={cwd or ''}\n"
                f"creationflags={creationflags}\n"
                f"command={' '.join(cmd)}\n\n"
            ).encode("utf-8", errors="replace")
        )
        handle.flush()
        self._state.log_handle = handle
        self._state.log_path = str(path)
        return handle

    def _close_log_locked(self) -> None:
        if self._state is None:
            return
        handle = self._state.log_handle
        self._state.log_handle = None
        if handle is not None:
            try:
                handle.flush()
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
        return safe[:120] or "model"

    @staticmethod
    def _popen_creationflags() -> int:
        # CUDA llama.cpp builds can fail before logging with 0xc0000022 when
        # launched with CREATE_NO_WINDOW. stdout/stderr are already redirected,
        # so the normal console launch path is the more reliable Windows mode.
        return 0

    def _endpoint_is_ready(self) -> bool:
        for url in self._probe_urls():
            try:
                response = requests.get(url, timeout=(0.5, 1.2))
            except requests.RequestException:
                continue
            if response.status_code == 200:
                return True
            if response.status_code in {401, 403}:
                return True
        return False

    def _probe_urls(self) -> list[str]:
        base = self.base_url.rstrip("/")
        parsed = urlparse(base)
        root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else base
        candidates = [
            f"{base}/models",
            f"{base}/health",
            f"{root}/health",
            f"{root}/v1/health",
        ]
        seen: set[str] = set()
        urls: list[str] = []
        for url in candidates:
            normalized = url.rstrip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
        return urls

    @staticmethod
    def _resolved_model_path(raw_path: str) -> str:
        text = str(raw_path or "").strip()
        if not text:
            return ""
        expanded = os.path.expandvars(os.path.expanduser(text))
        return str(Path(expanded))

    @staticmethod
    def _resolved_executable(raw_path: str) -> str:
        text = str(raw_path or "").strip()
        if not text:
            return ""
        expanded = os.path.expandvars(os.path.expanduser(text))
        path = Path(expanded)
        if path.name.lower() == "llama-cli.exe":
            sibling = path.with_name("llama-server.exe")
            if sibling.exists():
                return str(sibling)
        if path.name.lower() == "llama-cli":
            sibling = path.with_name("llama-server")
            if sibling.exists():
                return str(sibling)
        return str(path) if path.parent != Path(".") else expanded

    @staticmethod
    def _host_port_from_base_url(base_url: str) -> tuple[str, int]:
        parsed = urlparse(base_url or "http://127.0.0.1:8080/v1")
        host = parsed.hostname or "127.0.0.1"
        if parsed.port is not None:
            return host, int(parsed.port)
        if parsed.scheme == "https":
            return host, 443
        return host, 80
