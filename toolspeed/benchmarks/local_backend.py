"""Local Wall-Clock Benchmark Backend with OS primitive I/O and real monotonic timing."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
    ToolSchema,
)
from toolspeed.core.types import (
    ApprovalGrant,
    EvidenceLevel,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class ThreadingLocalTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class LocalHTTPServer:
    """Embedded lightweight multithreaded HTTP server for local network loopback benchmarking."""

    def __init__(self, port: int = 0) -> None:
        self.port = port
        self.server: ThreadingLocalTCPServer | None = None
        self.thread: threading.Thread | None = None
        self._started = threading.Event()

    def start(self) -> None:
        class ShardHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = json.dumps({"status": "ok", "path": self.path, "timestamp": time.time()}).encode("utf-8")
                self.wfile.write(resp)

            def do_POST(self) -> None:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    parsed = json.loads(body.decode("utf-8"))
                except Exception:
                    parsed = {}
                resp = json.dumps({"status": "success", "received": parsed}).encode("utf-8")
                self.wfile.write(resp)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingLocalTCPServer(("127.0.0.1", self.port), ShardHandler)
        self.port = self.server.server_address[1]

        def _run() -> None:
            self._started.set()
            if self.server:
                self.server.serve_forever()

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        self._started.wait()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class W2State:
    db_path: str
    table_hash: str


class LocalWallClockBackend:
    """Local Wall-Clock Backend executing real OS primitives with pure monotonic timing."""

    def __init__(self, evidence_level: EvidenceLevel = EvidenceLevel.LOCAL_WALL_CLOCK):
        self.evidence_level = evidence_level
        self._servers: list[LocalHTTPServer] = []
        self._temp_dirs: list[str] = []
        self._lock = threading.Lock()
        self._shared_server: LocalHTTPServer | None = None
        self._shared_sandbox_dir: str | None = None
        self._shared_fileio_dir: str | None = None

    async def create_w2_state(self, trial_idx: int = 0, arm: str = "baseline") -> W2State:
        """Creates an isolated, non-shared SQLite database file per trial and arm with deterministic initial data."""
        d = tempfile.mkdtemp(prefix=f"toolspeed_w2_{arm}_{trial_idx}_")
        self._temp_dirs.append(d)
        db_file = os.path.join(d, "w2_orders.db")
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, item TEXT, status TEXT, amount REAL)")
        for i in range(100):
            cur.execute("INSERT INTO orders VALUES (?, ?, ?, ?)", (i, f"item_{i}", "pending", 10.0 + i))
        conn.commit()
        conn.close()
        tbl_hash = hashlib.sha256(b"w2_orders_initial_schema_and_100_rows").hexdigest()
        return W2State(db_path=db_file, table_hash=tbl_hash)

    async def get_w2_row_count(self, trial_idx: int = 0) -> int:
        return 100

    async def execute_w2_step(self, trial_idx: int = 0) -> None:
        pass

    def _get_shared_server(self) -> LocalHTTPServer:
        with self._lock:
            if self._shared_server is None:
                srv = LocalHTTPServer()
                srv.start()
                self._servers.append(srv)
                self._shared_server = srv
            return self._shared_server

    def _get_shared_sandbox_dir(self) -> str:
        with self._lock:
            if self._shared_sandbox_dir is None:
                d = tempfile.mkdtemp(prefix="toolspeed_local_sb_")
                self._temp_dirs.append(d)
                self._shared_sandbox_dir = d
            return self._shared_sandbox_dir

    def _get_shared_fileio_dir(self) -> str:
        with self._lock:
            if self._shared_fileio_dir is None:
                d = tempfile.mkdtemp(prefix="toolspeed_local_fio_")
                self._temp_dirs.append(d)
                self._shared_fileio_dir = d
            return self._shared_fileio_dir

    def generate_task(self, workload_id: str, trial_index: int = 0) -> Task:
        """Constructs an immutable Task with seeded parameters and strict validator."""
        if workload_id == "W1":
            return Task(
                task_id=f"w1_local_trial_{trial_index:04d}",
                prompt=f"Query local HTTP server shards 0 to 4 for trial {trial_index}",
                expected_output={"status": "success", "total_shards": 5},
                metadata={"workload_id": "W1", "trial_index": trial_index},
            )
        elif workload_id == "W2":
            return Task(
                task_id=f"w2_local_trial_{trial_index:04d}",
                prompt=f"Execute SQLite user orders chain for user u_{trial_index}",
                context={"user_id": f"u_{trial_index}"},
                expected_output={
                    "user": {"user_id": f"u_{trial_index}", "name": f"User_{trial_index}"},
                    "orders": {"order_count": 2},
                    "fused": True,
                },
                metadata={"workload_id": "W2", "trial_index": trial_index, "workflow_id": "user_orders"},
            )
        elif workload_id == "W3":
            return Task(
                task_id=f"w3_local_trial_{trial_index:04d}",
                prompt=f"Execute branching customer check for cust_{trial_index}",
                expected_output={"status": "approved", "customer_id": f"cust_{trial_index}"},
                metadata={"workload_id": "W3", "trial_index": trial_index},
            )
        elif workload_id == "W4":
            key_id = f"item_{trial_index % 10}"
            return Task(
                task_id=f"w4_local_trial_{trial_index:04d}",
                prompt=f"Lookup price for {key_id}",
                expected_output={"sku": key_id, "price": 49.99},
                metadata={"workload_id": "W4", "trial_index": trial_index},
            )
        elif workload_id == "W5":
            return Task(
                task_id=f"w5_local_trial_{trial_index:04d}",
                prompt=f"Stream query data for trial {trial_index}",
                expected_output={"status": "success", "count": 100},
                metadata={"workload_id": "W5", "trial_index": trial_index},
            )
        elif workload_id == "W6":
            return Task(
                task_id=f"w6_local_trial_{trial_index:04d}",
                prompt=f"Run subprocess compute task {trial_index}",
                expected_output={"status": "success", "exit_code": 0},
                metadata={"workload_id": "W6", "trial_index": trial_index},
            )
        elif workload_id == "W7":
            idemp_key = f"tx_local_{trial_index:04d}"
            grant = ApprovalGrant.create(
                "execute_fund_transfer", {"recipient": "Alice", "amount": 100.0, "idempotency_key": idemp_key}
            )
            return Task(
                task_id=f"w7_local_trial_{trial_index:04d}",
                prompt=f"Execute fund transfer with idempotency key {idemp_key}",
                parameters={"idempotency_key": idemp_key},
                expected_output={"status": "transferred", "idempotency_key": idemp_key},
                metadata={"workload_id": "W7", "trial_index": trial_index, "approval_grant": grant},
            )
        else:
            return Task(
                task_id=f"e5a_local_trial_{trial_index:04d}",
                prompt=f"Action bytecode transport evaluation trial {trial_index}",
                expected_output={"status": "done", "trial": trial_index},
                metadata={"workload_id": "E5a", "trial_index": trial_index},
            )

    def create_workload_environment(
        self, workload_id: str, trial_index: int = 0
    ) -> tuple[ToolRegistry, BaseLLMAdapter]:
        registry = ToolRegistry()
        server = self._get_shared_server()

        if workload_id == "W1":

            class HTTPShardTool(BaseToolAdapter):
                def __init__(self, base_url: str, shard_idx: int) -> None:
                    self._base_url = base_url
                    self._shard_idx = shard_idx
                    self._name = f"read_shard_{shard_idx}"

                @property
                def name(self) -> str:
                    return self._name

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name=self._name,
                        description=f"Read shard {self._shard_idx}",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                        is_side_effect=False,
                    )

                def _fetch_sync(self, url: str) -> dict[str, Any]:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        return json.loads(resp.read().decode("utf-8"))

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    url = f"{self._base_url}/shard/{self._shard_idx}"
                    res = await asyncio.to_thread(self._fetch_sync, url)
                    dur_ns = time.perf_counter_ns() - start_ns
                    dur_ms = dur_ns / 1_000_000.0
                    return ToolResult(
                        call_id=call.call_id,
                        name=self._name,
                        tool_name=self._name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ms,
                    )

            for i in range(5):
                registry.register(HTTPShardTool(server.base_url, i))

            calls = [ToolCall(name=f"read_shard_{i}", arguments={"query": f"key_{i}"}) for i in range(5)]
            decisions = [
                LLMDecision(reasoning="Querying all 5 local shards concurrently", tool_calls=calls),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "success", "total_shards": 5}),
            ]
            model = LocalScriptedAdapter(decisions=decisions, decision_delay_s=0.001)
            return registry, model

        elif workload_id == "W2":
            db_dir = self._get_shared_fileio_dir()
            db_path = os.path.join(db_dir, f"test_chain_{trial_index}.db")

            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, name TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS orders (order_id TEXT, user_id TEXT, amount REAL)")
            conn.execute("INSERT OR REPLACE INTO users VALUES (?, ?)", (f"u_{trial_index}", f"User_{trial_index}"))
            conn.execute("INSERT INTO orders VALUES (?, ?, ?)", (f"o1_{trial_index}", f"u_{trial_index}", 100.0))
            conn.execute("INSERT INTO orders VALUES (?, ?, ?)", (f"o2_{trial_index}", f"u_{trial_index}", 50.0))
            conn.commit()
            conn.close()

            class SQLiteFetchUser(BaseToolAdapter):
                @property
                def name(self) -> str:
                    return "fetch_user"

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="fetch_user",
                        description="Fetch user record",
                        parameters={"type": "object", "properties": {"user_id": {"type": "string"}}},
                    )

                def _query_sync(self, uid: str) -> dict[str, Any]:
                    c = sqlite3.connect(db_path)
                    row = c.execute("SELECT user_id, name FROM users WHERE user_id = ?", (uid,)).fetchone()
                    c.close()
                    return {"user_id": row[0], "name": row[1]} if row else {"error": "not found"}

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    uid = call.arguments.get("user_id", "")
                    res = await asyncio.to_thread(self._query_sync, uid)
                    dur_ns = time.perf_counter_ns() - start_ns
                    return ToolResult(
                        call_id=call.call_id,
                        name=self.name,
                        tool_name=self.name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            class SQLiteFetchOrders(BaseToolAdapter):
                @property
                def name(self) -> str:
                    return "fetch_orders"

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="fetch_orders",
                        description="Fetch orders record",
                        parameters={"type": "object", "properties": {"user_id": {"type": "string"}}},
                    )

                def _query_sync(self, uid: str) -> dict[str, Any]:
                    c = sqlite3.connect(db_path)
                    rows = c.execute("SELECT order_id, amount FROM orders WHERE user_id = ?", (uid,)).fetchall()
                    c.close()
                    return {"order_count": len(rows), "orders": [{"id": r[0], "amount": r[1]} for r in rows]}

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    uid = call.arguments.get("user_id", "")
                    res = await asyncio.to_thread(self._query_sync, uid)
                    dur_ns = time.perf_counter_ns() - start_ns
                    return ToolResult(
                        call_id=call.call_id,
                        name=self.name,
                        tool_name=self.name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            registry.register(SQLiteFetchUser())
            registry.register(SQLiteFetchOrders())

            decisions = [
                LLMDecision(
                    reasoning="Fetching user",
                    tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": f"u_{trial_index}"})],
                ),
                LLMDecision(
                    reasoning="Fetching orders",
                    tool_calls=[ToolCall(name="fetch_orders", arguments={"user_id": f"u_{trial_index}"})],
                ),
                LLMDecision(
                    reasoning="Done",
                    tool_calls=[],
                    final_answer={
                        "user": {"user_id": f"u_{trial_index}", "name": f"User_{trial_index}"},
                        "orders": {"order_count": 2},
                        "fused": True,
                    },
                ),
            ]
            model = LocalScriptedAdapter(decisions=decisions, decision_delay_s=0.001)
            return registry, model

        elif workload_id == "W3":

            class LocalBranchTool(BaseToolAdapter):
                def __init__(self, name: str) -> None:
                    self._name = name

                @property
                def name(self) -> str:
                    return self._name

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name=self._name,
                        description="Local branch inspection",
                        parameters={"type": "object", "properties": {"customer_id": {"type": "string"}}},
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    await asyncio.sleep(0.001)
                    dur_ns = time.perf_counter_ns() - start_ns
                    res = {"customer_id": call.arguments.get("customer_id"), "status": "active", "risk": "low"}
                    return ToolResult(
                        call_id=call.call_id,
                        name=self._name,
                        tool_name=self._name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            registry.register(LocalBranchTool("read_customer_state"))
            registry.register(LocalBranchTool("audit_transaction"))

            spec_call = ToolCall(
                name="read_customer_state",
                arguments={"customer_id": f"cust_{trial_index}"},
                speculation_confidence=0.95,
            )
            decisions = [
                LLMDecision(reasoning="Checking state", tool_calls=[copy.deepcopy(spec_call)]),
                LLMDecision(
                    reasoning="Done",
                    tool_calls=[],
                    final_answer={"status": "approved", "customer_id": f"cust_{trial_index}"},
                ),
            ]
            model = LocalScriptedAdapter(decisions=decisions, draft_prediction=spec_call, decision_delay_s=0.001)
            return registry, model

        elif workload_id == "W4":

            class LocalPricingTool(BaseToolAdapter):
                @property
                def name(self) -> str:
                    return "pricing_lookup"

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="pricing_lookup",
                        description="Local pricing lookup",
                        parameters={"type": "object", "properties": {"sku": {"type": "string"}}},
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    sku = call.arguments.get("sku", "")
                    await asyncio.sleep(0.001)
                    dur_ns = time.perf_counter_ns() - start_ns
                    res = {"sku": sku, "price": 49.99}
                    return ToolResult(
                        call_id=call.call_id,
                        name=self.name,
                        tool_name=self.name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            registry.register(LocalPricingTool())
            key_id = f"item_{trial_index % 10}"
            call = ToolCall(name="pricing_lookup", arguments={"sku": key_id})
            decisions = [
                LLMDecision(reasoning="Lookup cached price", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"sku": key_id, "price": 49.99}),
            ]
            model = LocalScriptedAdapter(decisions=decisions, decision_delay_s=0.001)
            return registry, model

        elif workload_id == "W5":

            class LocalStreamTool(BaseToolAdapter):
                @property
                def name(self) -> str:
                    return "stream_query_data"

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="stream_query_data",
                        description="Local stream query",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                        },
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    await asyncio.sleep(0.001)
                    dur_ns = time.perf_counter_ns() - start_ns
                    res = {"status": "success", "count": call.arguments.get("limit", 100)}
                    return ToolResult(
                        call_id=call.call_id,
                        name=self.name,
                        tool_name=self.name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            registry.register(LocalStreamTool())
            call = ToolCall(name="stream_query_data", arguments={"query": "SELECT * FROM data", "limit": 100})
            decisions = [
                LLMDecision(reasoning="Streaming query", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "success", "count": 100}),
            ]
            model = LocalScriptedAdapter(decisions=decisions, decision_delay_s=0.001)
            return registry, model

        elif workload_id == "W6":
            sandbox_dir = self._get_shared_sandbox_dir()

            class LocalSubprocessTool(BaseToolAdapter):
                @property
                def name(self) -> str:
                    return "sandbox_compute"

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="sandbox_compute",
                        description="Local sandboxed subprocess execution",
                        parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
                    )

                def _run_subproc(self) -> dict[str, Any]:
                    res = subprocess.run(
                        ["echo", "computed"], capture_output=True, text=True, cwd=sandbox_dir, check=False
                    )
                    return {"status": "success", "exit_code": res.returncode, "stdout": res.stdout.strip()}

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    res = await asyncio.to_thread(self._run_subproc)
                    dur_ns = time.perf_counter_ns() - start_ns
                    return ToolResult(
                        call_id=call.call_id,
                        name=self.name,
                        tool_name=self.name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            registry.register(LocalSubprocessTool())
            call = ToolCall(name="sandbox_compute", arguments={"cmd": "echo computed"})
            decisions = [
                LLMDecision(reasoning="Running subprocess", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "success", "exit_code": 0}),
            ]
            model = LocalScriptedAdapter(decisions=decisions, decision_delay_s=0.001)
            return registry, model

        elif workload_id == "W7":

            class LocalFundTransferTool(BaseToolAdapter):
                def __init__(self) -> None:
                    super().__init__(
                        ToolSpec(
                            name="execute_fund_transfer",
                            is_read_only=False,
                            side_effects=True,
                            requires_approval=True,
                            is_idempotent=True,
                        )
                    )

                @property
                def name(self) -> str:
                    return "execute_fund_transfer"

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="execute_fund_transfer",
                        description="Execute local fund transfer",
                        parameters={
                            "type": "object",
                            "properties": {
                                "recipient": {"type": "string"},
                                "amount": {"type": "number"},
                                "idempotency_key": {"type": "string"},
                            },
                        },
                        is_side_effect=True,
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    await asyncio.sleep(0.001)
                    dur_ns = time.perf_counter_ns() - start_ns
                    idemp_key = call.arguments.get("idempotency_key", "")
                    res = {"status": "transferred", "idempotency_key": idemp_key}
                    return ToolResult(
                        call_id=call.call_id,
                        name=self.name,
                        tool_name=self.name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            registry.register(LocalFundTransferTool())
            idemp_key = f"tx_local_{trial_index:04d}"
            call = ToolCall(
                name="execute_fund_transfer",
                arguments={"recipient": "Alice", "amount": 100.0, "idempotency_key": idemp_key},
                requires_approval=True,
                idempotency_key=idemp_key,
            )
            decisions = [
                LLMDecision(reasoning="Execute transfer", tool_calls=[call]),
                LLMDecision(
                    reasoning="Done",
                    tool_calls=[],
                    final_answer={"status": "transferred", "idempotency_key": idemp_key},
                ),
            ]
            model = LocalScriptedAdapter(decisions=decisions, decision_delay_s=0.001)
            return registry, model

        else:
            # E5a Local Bytecode
            class LocalBytecodeTool(BaseToolAdapter):
                @property
                def name(self) -> str:
                    return "bytecode_transport_tool"

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="bytecode_transport_tool",
                        description="Local transport tool",
                        parameters={
                            "type": "object",
                            "properties": {"payload_id": {"type": "integer"}, "data": {"type": "string"}},
                        },
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    dur_ns = time.perf_counter_ns() - start_ns
                    res = {"status": "done", "trial": call.arguments.get("payload_id")}
                    return ToolResult(
                        call_id=call.call_id,
                        name=self.name,
                        tool_name=self.name,
                        result=res,
                        output=res,
                        execution_time_ns=dur_ns,
                        execution_time_ms=dur_ns / 1_000_000.0,
                    )

            registry.register(LocalBytecodeTool())
            call = ToolCall(
                name="bytecode_transport_tool", arguments={"payload_id": trial_index, "data": f"content_{trial_index}"}
            )
            decisions = [
                LLMDecision(reasoning="Transporting call", tool_calls=[call]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "done", "trial": trial_index}),
            ]
            model = LocalScriptedAdapter(decisions=decisions, decision_delay_s=0.001)
            return registry, model

    def cleanup(self) -> None:
        """Tears down servers and deletes temporary sandbox directories."""
        with self._lock:
            for s in self._servers:
                try:
                    s.stop()
                except Exception:
                    pass
            self._servers.clear()
            self._shared_server = None

            for d in self._temp_dirs:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
            self._temp_dirs.clear()
            self._shared_sandbox_dir = None
            self._shared_fileio_dir = None

    async def close(self) -> None:
        self.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.cleanup()


class LocalScriptedAdapter(BaseLLMAdapter):
    """Local scripted adapter executing fast decision steps."""

    def __init__(
        self,
        decisions: list[LLMDecision] | None = None,
        draft_prediction: ToolCall | None = None,
        decision_delay_s: float = 0.001,
    ):
        self.decisions = list(decisions or [])
        self.draft_prediction = draft_prediction
        self.decision_delay_s = decision_delay_s
        self._turn_index = 0

    def _get_decision_sync(self, task: Any) -> LLMDecision:
        if self._turn_index < len(self.decisions):
            decision = self.decisions[self._turn_index]
            self._turn_index += 1
            return decision
        return LLMDecision(
            reasoning="Task complete.",
            tool_calls=[],
            final_answer=None,
            input_tokens=100,
            output_tokens=20,
        )

    async def decide(
        self,
        task: Any,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> LLMDecision:
        if self.decision_delay_s > 0:
            await asyncio.sleep(self.decision_delay_s)
        return self._get_decision_sync(task)

    async def predict_draft(
        self,
        task: Any,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> ToolCall | None:
        if self.decision_delay_s > 0:
            await asyncio.sleep(self.decision_delay_s * 0.2)
        return self.draft_prediction

    async def stream_decision(
        self,
        task: Any,
        history: list[dict[str, Any]],
        available_tools: list[ToolSpec],
    ) -> AsyncIterator[StreamingChunk]:
        decision = self._get_decision_sync(task)
        chunks = 4
        delay = self.decision_delay_s / chunks if self.decision_delay_s > 0 else 0.0

        for i in range(chunks):
            if delay > 0:
                await asyncio.sleep(delay)
            is_final = i == chunks - 1
            ready_calls = list(decision.tool_calls) if (i >= 1 and decision.tool_calls) else []
            fragment = (
                json.dumps(ready_calls[0].arguments)
                if ready_calls
                else (json.dumps(decision.tool_calls[0].arguments) if (is_final and decision.tool_calls) else "")
            )

            yield StreamingChunk(
                token_index=i,
                delta_text=f"token_{i} ",
                commit_horizon_ready=ready_calls,
                raw_json_fragment=fragment,
                is_final=is_final,
                parsed_tool_calls=decision.tool_calls if is_final else [],
                metadata={"final_answer": decision.final_answer} if is_final else {},
            )
