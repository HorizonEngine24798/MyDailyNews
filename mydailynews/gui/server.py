from __future__ import annotations

from functools import partial
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from mydailynews.gui.data import GuiDataService


STATIC_DIR = Path(__file__).resolve().parent / "static"


def serve_gui(
    *,
    root: Path | str,
    config_path: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> int:
    service = GuiDataService(root=root, config_path=config_path)
    handler_class = partial(GuiRequestHandler, service=service)
    server = ThreadingHTTPServer((host, int(port)), handler_class)
    url = f"http://{host}:{int(port)}"
    print(f"MyDailyNews GUI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
        print("Stopping MyDailyNews GUI.")
    finally:
        server.server_close()
    return 0


class GuiRequestHandler(BaseHTTPRequestHandler):
    server_version = "MyDailyNewsGUI/1.0"

    def __init__(self, *args: Any, service: GuiDataService, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        self._handle(lambda: self._route_get())

    def do_POST(self) -> None:
        self._handle(lambda: self._route_post())

    def do_PUT(self) -> None:
        self._handle(lambda: self._route_put())

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _route_get(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_static(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            self._send_static(_static_path(path))
            return
        if path == "/api/state":
            self._send_json(self.service.app_state())
            return
        if path == "/api/config":
            self._send_json(self.service.read_config())
            return
        if path == "/api/reports":
            self._send_json(self.service.list_reports())
            return
        if path.startswith("/api/reports/"):
            report_id = unquote(path[len("/api/reports/") :])
            self._send_json(self.service.report_detail(report_id))
            return
        if path == "/api/memory":
            self._send_json(self.service.memory_snapshot())
            return
        if path == "/api/learned-preferences":
            self._send_json(self.service.learned_preferences())
            return
        if path == "/api/runs":
            self._send_json(self.service.list_runs())
            return
        if path.startswith("/api/runs/"):
            run_id = unquote(path[len("/api/runs/") :])
            self._send_json(self.service.run_detail(run_id))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Route not found.")

    def _route_post(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/feedback":
            self._send_json(self.service.record_feedback(self._read_json()))
            return
        if path == "/api/memory/prune":
            self._send_json(self.service.prune_memory())
            return
        if path == "/api/memory/repair":
            self._send_json(self.service.repair_memory(self._read_json()))
            return
        if path == "/api/autoconfig":
            self._send_json(self.service.run_autoconfig(self._read_json(default={})))
            return
        if path == "/api/previews/user-memory":
            self._send_json(self.service.preview_user_memory(self._read_json()))
            return
        if path == "/api/previews/learned-preferences":
            self._send_json(self.service.preview_learned_preferences(self._read_json()))
            return
        if path == "/api/runs":
            self._send_json(self.service.start_run(self._read_json()))
            return
        if path.startswith("/api/runs/") and path.endswith("/cancel"):
            run_id = unquote(path[len("/api/runs/") : -len("/cancel")])
            self._send_json(self.service.cancel_run(run_id))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Route not found.")

    def _route_put(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/config":
            self._send_json(self.service.save_config(self._read_json()))
            return
        if path.startswith("/api/config/section/"):
            section = unquote(path[len("/api/config/section/") :])
            self._send_json(self.service.save_config_section(section, self._read_json()))
            return
        if path == "/api/memory/story-index":
            self._send_json(self.service.save_story_index(self._read_json()))
            return
        if path == "/api/learned-preferences":
            self._send_json(self.service.save_learned_preferences(self._read_json()))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Route not found.")

    def _handle(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc) or "Not found.")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except subprocess.TimeoutExpired as exc:
            self._send_error(HTTPStatus.REQUEST_TIMEOUT, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _read_json(self, default: Any | None = None) -> Any:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            if default is not None:
                return default
            raise ValueError("Request body must be JSON.")
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_static(self, path: Path) -> None:
        resolved = path.resolve()
        try:
            resolved.relative_to(STATIC_DIR)
        except ValueError as exc:
            raise FileNotFoundError("Static file not found.") from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError("Static file not found.")
        content_type = _content_type(resolved)
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    return "application/octet-stream"


def _static_path(url_path: str) -> Path:
    relative = unquote(url_path[len("/static/") :])
    parts = [part for part in relative.split("/") if part]
    if not parts or any(part in {".", ".."} or "\\" in part for part in parts):
        raise FileNotFoundError("Static file not found.")
    return STATIC_DIR.joinpath(*parts)
