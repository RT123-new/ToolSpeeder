"""Real local async tools: SQLite executor, Subprocess Sandbox, Local File I/O, and Mock HTTP Server/Client."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from toolspeed.adapters.base import BaseToolAdapter, ToolSchema
from toolspeed.core.types import ToolCall, ToolResult


class AsyncSQLiteTool(BaseToolAdapter):
    """Real local asynchronous SQLite database query executor using parameterized queries in threadpool."""

    def __init__(self, db_path: str = ":memory:", name: str = "sqlite_executor"):
        self._db_path = db_path
        self._name = name
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description="Execute SQL queries against a local SQLite database.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "params": {"type": "array", "description": "Query parameters list"},
                },
                "required": ["query"],
            },
            is_side_effect=True,
            cost_usd=0.0001,
        )

    def _sync_execute(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        conn = self._conn
        should_close = False
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            should_close = True

        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            if query.strip().upper().startswith(("SELECT", "PRAGMA", "EXPLAIN")):
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
            else:
                conn.commit()
                result = [{"rows_affected": cursor.rowcount, "last_row_id": cursor.lastrowid}]
            return result
        finally:
            cursor.close()
            if should_close:
                conn.close()

    async def execute(self, call: ToolCall) -> ToolResult:
        start_ns = time.perf_counter_ns()
        query = call.arguments.get("query", "")
        params = call.arguments.get("params", [])

        if not query:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error="Missing required 'query' argument.",
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        try:
            async with self._lock:
                result = await asyncio.to_thread(self._sync_execute, query, params)
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=result,
                error=None,
                is_error=False,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.0001,
            )
        except Exception as ex:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error=str(ex),
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.0001,
            )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


class SafeSubprocessSandbox(BaseToolAdapter):
    """Local safe subprocess executor with strict timeout, cwd isolation, and output capture."""

    def __init__(
        self,
        name: str = "subprocess_sandbox",
        sandbox_dir: str | None = None,
        default_timeout_s: float = 10.0,
        max_output_bytes: int = 100_000,
    ):
        self._name = name
        self._sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="toolspeed_sandbox_")
        self._default_timeout_s = default_timeout_s
        self._max_output_bytes = max_output_bytes
        os.makedirs(self._sandbox_dir, exist_ok=True)

    @property
    def sandbox_dir(self) -> str:
        return self._sandbox_dir

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description="Execute commands safely in an isolated sandbox directory.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command or script to run"},
                    "timeout_s": {"type": "number", "description": "Execution timeout in seconds"},
                },
                "required": ["command"],
            },
            is_side_effect=True,
            cost_usd=0.0005,
        )

    def _sync_run(self, command: str, timeout_s: float) -> dict[str, Any]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "PYTHONUNBUFFERED": "1",
        }
        res = subprocess.run(
            command,
            shell=True,
            cwd=self._sandbox_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        stdout = res.stdout[: self._max_output_bytes]
        stderr = res.stderr[: self._max_output_bytes]
        return {
            "exit_code": res.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": len(res.stdout) > self._max_output_bytes,
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        start_ns = time.perf_counter_ns()
        command = call.arguments.get("command", "")
        timeout_s = float(call.arguments.get("timeout_s", self._default_timeout_s))

        if not command:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error="Missing required 'command' argument.",
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        try:
            res_dict = await asyncio.to_thread(self._sync_run, command, timeout_s)
            is_error = res_dict["exit_code"] != 0
            error_msg = res_dict["stderr"] if is_error else None
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=res_dict,
                error=error_msg,
                is_error=is_error,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.0005,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error=f"Command timed out after {timeout_s}s",
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.0005,
            )
        except Exception as ex:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error=str(ex),
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.0005,
            )


class AsyncLocalFileIOTool(BaseToolAdapter):
    """Sandboxed asynchronous local filesystem operations tool."""

    def __init__(
        self,
        base_dir: str | None = None,
        name: str = "file_io",
    ):
        self._name = name
        self._base_dir = Path(base_dir or tempfile.mkdtemp(prefix="toolspeed_fileio_")).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _resolve_safe(self, rel_path: str) -> Path:
        target = (self._base_dir / rel_path).resolve()
        if not str(target).startswith(str(self._base_dir)):
            raise ValueError(f"Path traversal detected: '{rel_path}' is outside sandbox base dir.")
        return target

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description="Sandboxed file I/O: read, write, append, list, delete files.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "append", "list", "delete", "stat"]},
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Content for write/append"},
                },
                "required": ["action", "path"],
            },
            is_side_effect=True,
            cost_usd=0.00005,
        )

    def _sync_op(self, action: str, path_str: str, content: str | None) -> Any:
        target = self._resolve_safe(path_str)
        if action == "read":
            if not target.exists() or not target.is_file():
                raise FileNotFoundError(f"File not found: {path_str}")
            return target.read_text(encoding="utf-8")
        elif action == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
            return {"status": "written", "bytes": len(content or "")}
        elif action == "append":
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(content or "")
            return {"status": "appended", "bytes": len(content or "")}
        elif action == "list":
            if not target.exists() or not target.is_dir():
                raise NotADirectoryError(f"Directory not found: {path_str}")
            return [str(p.relative_to(self._base_dir)) for p in target.iterdir()]
        elif action == "delete":
            if target.exists():
                if target.is_file():
                    target.unlink()
                elif target.is_dir():
                    import shutil

                    shutil.rmtree(target)
                return {"status": "deleted"}
            return {"status": "not_found"}
        elif action == "stat":
            if not target.exists():
                raise FileNotFoundError(f"Path not found: {path_str}")
            stat = target.stat()
            return {
                "size_bytes": stat.st_size,
                "is_file": target.is_file(),
                "is_dir": target.is_dir(),
                "mtime": stat.st_mtime,
            }
        else:
            raise ValueError(f"Unknown file action: {action}")

    async def execute(self, call: ToolCall) -> ToolResult:
        start_ns = time.perf_counter_ns()
        action = call.arguments.get("action", "read")
        path_str = call.arguments.get("path", "")
        content = call.arguments.get("content")

        try:
            res = await asyncio.to_thread(self._sync_op, action, path_str, content)
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=res,
                error=None,
                is_error=False,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.00005,
            )
        except Exception as ex:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error=str(ex),
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.00005,
            )


class _MockHTTPHandler(BaseHTTPRequestHandler):
    """Internal HTTP handler for MockHTTPServer."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard logging to keep output clean
        pass

    def _send_response_json(self, status: int, data: Any, delay_s: float = 0.0) -> None:
        if delay_s > 0:
            time.sleep(delay_s)
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        server: MockHTTPServer = self.server.wrapper  # type: ignore
        route_info = server.routes.get(("GET", self.path))
        if route_info:
            status, resp, delay = route_info
            self._send_response_json(status, resp, delay)
        else:
            self._send_response_json(200, {"path": self.path, "method": "GET", "status": "ok"})

    def do_POST(self) -> None:
        server: MockHTTPServer = self.server.wrapper  # type: ignore
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
        try:
            parsed_json = json.loads(body) if body else {}
        except Exception:
            parsed_json = {"raw": body}

        route_info = server.routes.get(("POST", self.path))
        if route_info:
            status, resp, delay = route_info
            self._send_response_json(status, resp, delay)
        else:
            self._send_response_json(200, {"path": self.path, "method": "POST", "received": parsed_json})


class MockHTTPServer:
    """Lightweight in-process HTTP server for testing live HTTP calls and network latencies."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.routes: dict[
            tuple[str, str], tuple[int, Any, float]
        ] = {}  # (method, path) -> (status, response_body, delay_s)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def add_route(self, method: str, path: str, response: Any, status_code: int = 200, delay_s: float = 0.0) -> None:
        self.routes[(method.upper(), path)] = (status_code, response, delay_s)

    def start(self) -> str:
        self._server = HTTPServer((self.host, self.port), _MockHTTPHandler)
        self._server.wrapper = self  # type: ignore
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None


class AsyncHTTPClientTool(BaseToolAdapter):
    """Asynchronous HTTP request tool communicating with live or mock HTTP services."""

    def __init__(self, base_url: str = "", name: str = "http_client"):
        self._base_url = base_url.rstrip("/")
        self._name = name

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description="Send HTTP requests (GET, POST, PUT, DELETE) to endpoints.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL or relative path"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
                    "body": {"type": "object", "description": "JSON request body"},
                    "headers": {"type": "object", "description": "Request headers"},
                },
                "required": ["url"],
            },
            is_side_effect=True,
            cost_usd=0.0002,
        )

    def _sync_request(
        self, url: str, method: str, body: dict[str, Any] | None, headers: dict[str, str]
    ) -> dict[str, Any]:
        target_url = url if url.startswith("http") else f"{self._base_url}/{url.lstrip('/')}"
        data_bytes = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(target_url, data=data_bytes, method=method)
        req.add_header("Content-Type", "application/json")
        for k, v in headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                resp_bytes = resp.read()
                resp_text = resp_bytes.decode("utf-8")
                try:
                    resp_json = json.loads(resp_text)
                except Exception:
                    resp_json = resp_text
                return {
                    "status_code": resp.status,
                    "body": resp_json,
                    "headers": dict(resp.headers),
                }
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_body)
            except Exception:
                err_json = err_body
            return {
                "status_code": he.code,
                "body": err_json,
                "error": str(he),
            }

    async def execute(self, call: ToolCall) -> ToolResult:
        start_ns = time.perf_counter_ns()
        url = call.arguments.get("url", "")
        method = call.arguments.get("method", "GET").upper()
        body = call.arguments.get("body")
        headers = call.arguments.get("headers", {})

        if not url:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error="Missing required 'url' argument.",
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
            )

        try:
            res_dict = await asyncio.to_thread(self._sync_request, url, method, body, headers)
            status_code = res_dict.get("status_code", 200)
            is_error = status_code >= 400
            error_msg = res_dict.get("error") if is_error else None
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=res_dict["body"],
                error=error_msg,
                is_error=is_error,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.0002,
                metadata={"status_code": status_code},
            )
        except Exception as ex:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                name=self._name,
                result=None,
                error=str(ex),
                is_error=True,
                execution_time_ns=time.perf_counter_ns() - start_ns,
                cost_usd=0.0002,
            )
